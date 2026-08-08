"""
并发多会话测试套件
==================

覆盖特征点：
  - agent_core._invoke_config 的 thread_id 处理
  - 所有会话共享同一编译图（无 per-thread 缓存）
  - _pending_interrupts 的 per-thread 中断状态隔离
  - manually_compact(thread_id=) 针对指定线程压缩
  - server._thread_lock 的 per-thread 锁隔离

运行：
  pytest tests/test_concurrent_sessions.py -v
"""
import asyncio
from types import SimpleNamespace
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


# --------------------------------------------------------------------------- #
# agent_core._invoke_config
# --------------------------------------------------------------------------- #
def test_invoke_config_respects_thread_id():
    from agent.agent_core import AgentCore

    core = object.__new__(AgentCore)
    core._initial_thread_id = "fallback"
    core.max_iterations = 25

    assert core._invoke_config() == {
        "configurable": {"thread_id": "fallback"},
        "recursion_limit": 25,
    }
    assert core._invoke_config("thread-x") == {
        "configurable": {"thread_id": "thread-x"},
        "recursion_limit": 25,
    }


# --------------------------------------------------------------------------- #
# 所有会话共享同一编译图（无状态化验证）
# --------------------------------------------------------------------------- #
def test_shared_executor_no_per_thread_cache():
    """AgentCore 不再维护 per-thread 编译图缓存，所有会话共享 agent_executor。"""
    from agent.agent_core import AgentCore

    core = object.__new__(AgentCore)
    shared_executor = SimpleNamespace(tag="shared")
    core.agent_executor = shared_executor

    # 不存在 _executors 属性（已移除）
    assert not hasattr(core, "_executors")
    assert not hasattr(core, "_MAX_THREAD_EXECUTORS")
    # agent_executor 即为唯一编译图
    assert core.agent_executor is shared_executor


def test_no_mutable_system_message():
    """AgentCore 不再持有可变 _system_message，技能注入由中间件完成。"""
    from agent.agent_core import AgentCore

    core = object.__new__(AgentCore)
    assert not hasattr(core, "_system_message")
    assert not hasattr(core, "active_skills")


# --------------------------------------------------------------------------- #
# pending_interrupts per-session 隔离（SessionStore）
# --------------------------------------------------------------------------- #
def test_pending_interrupt_capture_and_clear_per_thread():
    """_acapture / _aclear 通过 SessionStore 实现 per-session 中断隔离。"""
    from agent.agent_core import AgentCore
    from session.store import SessionStore

    core = object.__new__(AgentCore)
    core._session_store = SessionStore()

    async def _run():
        await core._acapture_pending_interrupt(
            {"configurable": {"thread_id": "t-a"}}, "run"
        )
        await core._acapture_pending_interrupt(
            {"configurable": {"thread_id": "t-b"}}, "chat"
        )

        assert await core._get_store().aget_interrupt_mode("t-a") == "run"
        assert await core._get_store().aget_interrupt_mode("t-b") == "chat"

        # 只清理 t-a，不影响 t-b
        await core._aclear_pending_interrupt("t-a")
        assert await core._get_store().aget_interrupt_mode("t-a") is None
        assert await core._get_store().aget_interrupt_mode("t-b") == "chat"

        # 清理 t-b
        await core._aclear_pending_interrupt("t-b")
        assert await core._get_store().aget_interrupt_mode("t-b") is None

    asyncio.run(_run())


def test_pending_interrupt_lazy_init_on_object_new():
    """object.__new__ 未初始化 _session_store 时 _get_store 惰性创建。"""
    from agent.agent_core import AgentCore

    core = object.__new__(AgentCore)

    async def _run():
        await core._acapture_pending_interrupt(
            {"configurable": {"thread_id": "t-x"}}, "chat"
        )
        assert await core._get_store().aget_interrupt_mode("t-x") == "chat"

    asyncio.run(_run())


def test_pending_interrupt_lazy_clear_no_crash():
    from agent.agent_core import AgentCore

    core = object.__new__(AgentCore)

    async def _run():
        # _aclear_pending_interrupt 在无 SessionStore 时通过 _get_store 惰性创建
        await core._aclear_pending_interrupt("t-x")  # 不应抛异常

    asyncio.run(_run())


# --------------------------------------------------------------------------- #
# manually_compact(thread_id=) 针对指定线程
# --------------------------------------------------------------------------- #
def test_manually_compact_uses_target_thread():
    from agent.agent_core import AgentCore

    thread_id = "thread-compact"
    state_updates = []
    fetched_session_ids = []

    class _FakeSessionContext:
        def __init__(self, sid: str):
            self.config = {
                "configurable": {"thread_id": sid},
                "recursion_limit": 25,
            }

    class _FakeSession:
        """模拟 SessionRegistry：记录 aget_messages 的 session_id 参数。"""

        current_session_id = "default"

        def get_context(self, session_id: str | None = None):
            return _FakeSessionContext(session_id or "default")

        async def aget_messages(self, session_id: str | None = None):
            fetched_session_ids.append(session_id)
            return [HumanMessage(content=f"m{idx}") for idx in range(8)]

    class FakeExecutor:
        async def aget_state(self, config):
            return SimpleNamespace(values={"summary": ""})

        async def aupdate_state(self, config, values):
            state_updates.append((config, values))

    core = object.__new__(AgentCore)
    core.verbose = False
    core.max_iterations = 25
    core.agent_executor = FakeExecutor()
    core._state_lock = asyncio.Lock()
    core._session_registry = _FakeSession()

    # 压缩中间件最小桩：直接产出更新
    class _FakeCompaction:
        async def arun_compaction(self, msgs, existing_summary="", force=False):
            return {
                "summary": "sum",
                "messages": [SimpleNamespace()],
            }

    core._compaction_middleware = _FakeCompaction()

    result = asyncio.run(core.manually_compact(thread_id=thread_id))

    assert fetched_session_ids == [thread_id]
    assert result is not None
    assert result["summary"] == "sum"
    assert len(state_updates) == 1
    assert state_updates[0][0]["configurable"]["thread_id"] == thread_id


# --------------------------------------------------------------------------- #
# server._thread_lock per-thread 锁隔离
# --------------------------------------------------------------------------- #
def test_server_thread_locks_isolate_threads():
    from api import server

    lock_a1 = server._thread_lock("thread-a")
    lock_a2 = server._thread_lock("thread-a")
    lock_b = server._thread_lock("thread-b")

    # 同一线程返回同一锁；不同线程不同锁
    assert lock_a1 is lock_a2
    assert lock_a1 is not lock_b

    async def run():
        # 线程 a 持有锁时，线程 b 仍可加锁（互不影响）
        async with lock_a1:
            acquired = await asyncio.wait_for(
                lock_b.acquire(), timeout=0.5
            )
            lock_b.release()
            return acquired

    assert asyncio.run(run()) is True
