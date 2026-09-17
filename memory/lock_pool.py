"""记忆并发锁 — thread 私有锁池 + agent 级跨进程互斥锁。

设计要点：
- 不同 ``thread_id`` 完全并行，不加锁
- 同一 ``thread_id`` 的写入串行化（防止读改写竞态）
- 内部全局锁保护锁字典自身的增删
- agent 级长期记忆（namespace ``(agent_key, "global_facts")``）跨进程共享，
  由 :class:`AgentMemoryLock` 保护：进程内退化为 ``asyncio.Lock``，指定路径时
  升级为 OS 级文件锁实现**跨进程**互斥

集群部署时，thread 级锁可替换为 Redis 分布式锁 ``memory:lock:{thread_id}``。
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from types import TracebackType
from typing import Self

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

logger = logging.getLogger(__name__)


class ThreadMemoryLockPool:
    """per-thread ``asyncio.Lock`` 池。

    使用方式::

        pool = ThreadMemoryLockPool()
        lock = await pool.get(thread_id)
        async with lock:
            # 同一 thread 的写入串行化
            await memory_store.save_fact(thread_id, item)

    线程安全性：
    - ``_dict_lock`` 保护 ``_locks`` 字典的增删，确保并发 ``get`` 不会创建重复锁
    - 单个 ``asyncio.Lock`` 实例保护同一 thread 的写入流程
    """

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._dict_lock = asyncio.Lock()

    async def get(self, thread_id: str) -> asyncio.Lock:
        """获取指定 thread 的锁（不存在则创建）。

        Args:
            thread_id: 会话线程 ID

        Returns:
            该 thread 专属的 ``asyncio.Lock`` 实例
        """
        async with self._dict_lock:
            if thread_id not in self._locks:
                self._locks[thread_id] = asyncio.Lock()
                logger.debug("为 thread %s 创建记忆锁", thread_id)
            return self._locks[thread_id]

    async def cleanup(self, thread_id: str) -> None:
        """thread 销毁时清理锁缓存，防止内存泄漏。

        Args:
            thread_id: 已销毁的会话线程 ID
        """
        async with self._dict_lock:
            removed = self._locks.pop(thread_id, None)
            if removed is not None:
                logger.debug("清理 thread %s 的记忆锁", thread_id)


class AgentMemoryLock:
    """agent 级长期记忆的跨进程互斥锁。

    两种工作模式：

    - ``path is None``：退化为进程内 ``asyncio.Lock``，只在当前进程内串行化
      （供测试与内存/非 SQLite 场景使用）。
    - ``path`` 指定：使用 OS 级文件锁实现**跨进程**互斥。Windows 走
      ``msvcrt.locking``（``LK_NBLCK``），POSIX 走 ``fcntl.flock``
      （``LOCK_EX | LOCK_NB``）。每次 ``__aenter__`` 都重新 ``os.open`` 一个
      独立的文件描述符，因此同一进程内的两个协程也会互相冲突，而非误判为可重入。

    为什么用 OS 级文件锁：锁在进程崩溃时由操作系统自动释放，不会留下陈旧锁文件，
    也不会出现"锁文件存在即视为被占用"的伪死锁。锁文件路径由调用方传入，通常与
    SQLite 库文件同目录；该文件仅作锁目标，不承载任何数据。

    Args:
        path: 锁文件路径；为 None 时退化为进程内 ``asyncio.Lock``
        timeout: 获取锁的最长等待秒数，超时抛 ``TimeoutError``
        poll_interval: 竞争失败后两次重试之间的休眠秒数（避免阻塞事件循环）
    """

    def __init__(
        self,
        path: str | None = None,
        timeout: float = 10.0,
        poll_interval: float = 0.05,
    ) -> None:
        self._path = path
        self._timeout = timeout
        self._poll_interval = poll_interval
        self._async_lock: asyncio.Lock | None = asyncio.Lock() if path is None else None
        self._fd: int | None = None

    async def __aenter__(self) -> Self:
        """获取锁：进程内直接 acquire，跨进程循环尝试 OS 文件锁。"""
        if self._path is None:
            assert self._async_lock is not None
            await self._async_lock.acquire()
            return self

        deadline = time.monotonic() + self._timeout
        while True:
            fd = os.open(
                self._path,
                os.O_CREAT | os.O_RDWR | getattr(os, "O_BINARY", 0),
                0o644,
            )
            # 先保证文件至少有 1 字节，供 Windows 字节区间锁使用
            try:
                if os.fstat(fd).st_size == 0:
                    os.write(fd, b"\0")
                os.lseek(fd, 0, os.SEEK_SET)
            except OSError:
                os.close(fd)
                raise

            try:
                if sys.platform == "win32":
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                else:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                os.close(fd)
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"获取 agent 级记忆锁超时（{self._timeout}s）: {self._path}"
                    ) from None
                await asyncio.sleep(self._poll_interval)
                continue

            self._fd = fd
            return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """释放锁；无论如何都关闭文件描述符。"""
        if self._path is None:
            if self._async_lock is not None:
                self._async_lock.release()
            return

        fd = self._fd
        self._fd = None
        if fd is None:
            return
        try:
            if sys.platform == "win32":
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


__all__ = ["AgentMemoryLock", "ThreadMemoryLockPool"]
