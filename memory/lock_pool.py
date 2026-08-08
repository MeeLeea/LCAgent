"""Thread 私有并发锁池 — 保护同一 thread 内部并发写。

设计要点：
- 不同 ``thread_id`` 完全并行，不加锁
- 同一 ``thread_id`` 的写入串行化（防止读改写竞态）
- 内部全局锁保护锁字典自身的增删

集群部署时，可替换为 Redis 分布式锁 ``memory:lock:{thread_id}``。
"""
from __future__ import annotations

import asyncio
import logging

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



__all__ = ["ThreadMemoryLockPool"]
