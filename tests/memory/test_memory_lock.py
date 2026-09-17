"""AgentMemoryLock 单元测试

覆盖：
- path=None 时退化为进程内锁，协程间严格串行
- path 指定时基于 OS 级文件锁互斥：第二个获取者超时
- 释放后可再次获取
- 跨进程互斥（子进程尝试获取同一锁文件应超时）

测试风格与 tests/memory/test_memory_store.py 一致：
模块级 ``def test_x(): asyncio.run(run())``，内部 ``async def run()``。
"""
from __future__ import annotations

import asyncio
import os
import sys

import pytest

from memory.lock_pool import AgentMemoryLock

try:
    import subprocess  # noqa: F401  # 仅用于可用性探测

    _HAS_SUBPROCESS = True
except ImportError:  # pragma: no cover - 平台极端裁剪场景
    _HAS_SUBPROCESS = False


# ════════════════════════════════════════════════════════════════════════
#  进程内锁（path=None）
# ════════════════════════════════════════════════════════════════════════


def test_in_process_lock_serializes():
    """path=None 时两个协程不得交叉执行（记录顺序必须首尾相邻）。"""

    async def run():
        lock = AgentMemoryLock(path=None)
        order: list[str] = []

        async def worker(name: str) -> None:
            async with lock:
                order.append(f"{name}-start")
                # 在临界区内让出事件循环，若锁失效则另一协程会插入
                await asyncio.sleep(0)
                order.append(f"{name}-end")

        await asyncio.gather(worker("a"), worker("b"))

        assert order in (
            ["a-start", "a-end", "b-start", "b-end"],
            ["b-start", "b-end", "a-start", "a-end"],
        )

    asyncio.run(run())


# ════════════════════════════════════════════════════════════════════════
#  文件锁（path 指定）
# ════════════════════════════════════════════════════════════════════════


def test_file_lock_excludes_second_acquirer(tmp_path):
    """已持锁时，第二个获取者应在短超时后抛 TimeoutError。"""

    async def run():
        lock_path = str(tmp_path / "agent_memory.lock")
        holder = AgentMemoryLock(path=lock_path)
        async with holder:
            contender = AgentMemoryLock(
                path=lock_path, timeout=0.2, poll_interval=0.05
            )
            with pytest.raises(TimeoutError):
                async with contender:
                    pass

    asyncio.run(run())


def test_file_lock_released_after_exit(tmp_path):
    """释放后应能再次成功获取（同一进程、同一路径）。"""

    async def run():
        lock_path = str(tmp_path / "agent_memory.lock")
        async with AgentMemoryLock(path=lock_path):
            pass

        acquired = False
        async with AgentMemoryLock(path=lock_path, timeout=1.0, poll_interval=0.05):
            acquired = True
        assert acquired is True

    asyncio.run(run())


def test_file_lock_works_across_processes(tmp_path):
    """父进程持锁期间，子进程以短超时尝试获取同一锁文件应报 TIMEOUT。

    这是证明跨进程互斥真实生效的关键用例。
    """
    if not _HAS_SUBPROCESS:  # pragma: no cover - 极端裁剪环境
        pytest.skip("subprocess 不可用")

    async def run():
        lock_path = str(tmp_path / "agent_memory.lock")
        child_script = (
            "import asyncio, sys\n"
            "from memory.lock_pool import AgentMemoryLock\n"
            "\n"
            "async def main() -> None:\n"
            "    lock = AgentMemoryLock(path=sys.argv[1], timeout=0.3, poll_interval=0.05)\n"
            "    try:\n"
            "        async with lock:\n"
            "            print('ACQUIRED')\n"
            "    except TimeoutError:\n"
            "        print('TIMEOUT')\n"
            "\n"
            "asyncio.run(main())\n"
        )

        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")

        holder = AgentMemoryLock(path=lock_path)
        async with holder:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                child_script,
                lock_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            stdout, stderr = await proc.communicate()

        assert b"TIMEOUT" in stdout, f"stderr={stderr.decode(errors='replace')}"
        assert b"ACQUIRED" not in stdout

    asyncio.run(run())
