"""SessionManager 门面测试 - 验证删除会话时清理 thread 级长期记忆。

运行：
  pytest tests/session/test_session_manager.py -v
"""
import asyncio
from typing import Any

from memory.lock_pool import ThreadMemoryLockPool
from memory.manager import MemoryManager
from memory.models import MemoryCategory, ThreadFactItem
from memory.store import ThreadMemoryStore
from session.manager import SessionManager


class _FakeRegistry:
    """最小 SessionRegistry mock，仅记录 adelete_session 调用。"""

    def __init__(self) -> None:
        self.deleted: str | None = None

    async def adelete_session(self, session_id: str) -> bool:
        self.deleted = session_id
        return True


class _FakeAgent:
    """最小 AgentCore mock，提供 SessionManager 所需的 session 引用。"""

    def __init__(self, registry: _FakeRegistry) -> None:
        self.session = registry


def _make_manager() -> MemoryManager:
    store = ThreadMemoryStore()
    lock_pool = ThreadMemoryLockPool()
    return MemoryManager(
        memory_store=store,
        lock_pool=lock_pool,
        llm_getter=lambda: None,
    )


def test_adelete_session_clears_thread_memory():
    """删除会话应先清 thread 级长期记忆（agent 级保留），再删会话。"""
    async def run() -> None:
        mgr = _make_manager()
        await mgr.store.save_fact("t1", ThreadFactItem(content="thread-a"))
        await mgr.store.save_agent_fact(
            ThreadFactItem(content="global-a", category="user_fact")
        )

        reg = _FakeRegistry()
        sm = SessionManager(agent=_FakeAgent(reg), memory=mgr)

        result = await sm.adelete_session("t1")
        assert result is True
        assert reg.deleted == "t1"
        # thread 级记忆被清，agent 级跨会话记忆保留
        assert await mgr.count_facts("t1") == 0
        assert await mgr.count_agent_facts() == 1

    asyncio.run(run())


def test_adelete_session_releases_lock_pool():
    """release_lock=True 时锁池缓存应被释放。"""
    async def run() -> None:
        mgr = _make_manager()
        await mgr.store.save_fact("t1", ThreadFactItem(content="a"))

        reg = _FakeRegistry()
        sm = SessionManager(agent=_FakeAgent(reg), memory=mgr)

        await sm.adelete_session("t1")
        assert mgr._lock_pool._locks.get("t1") is None

    asyncio.run(run())


def test_adelete_session_without_memory_still_deletes():
    """记忆未注入（memory=None）时删除会话不应报错。"""
    async def run() -> None:
        reg = _FakeRegistry()
        sm = SessionManager(agent=_FakeAgent(reg), memory=None)

        result = await sm.adelete_session("t1")
        assert result is True
        assert reg.deleted == "t1"

    asyncio.run(run())
