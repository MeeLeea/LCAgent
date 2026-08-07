"""
并发多会话测试套件
==================

覆盖特征点：
  - memory.get_config 显式 thread_id 隔离
  - agent_core._invoke_config 的 thread_id 处理
  - 所有会话共享同一编译图（无 per-thread 缓存）
  - _pending_interrupts 的 per-thread 中断状态隔离
  - manually_compact(thread_id=) 针对指定线程压缩
  - StreamHandler 流式对话的 thread_id 透传
  - server._thread_lock 的 per-thread 锁隔离

运行：
  pytest tests/test_concurrent_sessions.py -v
"""
import asyncio
from types import SimpleNamespace
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


# --------------------------------------------------------------------------- #
# memory.get_config thread_id 隔离
# --------------------------------------------------------------------------- #
class FakeMemoryBase:
    def __init__(self):
        self.thread_id = "current-default"


def test_get_config_thread_id_isolation():
    from agent.memory import AgentMemory

    memory = object.__new__(AgentMemory)
    memory.thread_id = "current-default"

    assert memory.get_config() == {"configurable": {"thread_id": "current-default"}}
    assert memory.get_config(thread_id="thread-a") == {
        "configurable": {"thread_id": "thread-a"}
    }
    assert memory.get_config(thread_id="thread-b") == {
        "configurable": {"thread_id": "thread-b"}
    }


# --------------------------------------------------------------------------- #
# agent_core._invoke_config
# --------------------------------------------------------------------------- #
def test_invoke_config_respects_thread_id():
    from agent.agent_core import AgentCore

    calls = []

    class FakeMemory:
        def get_config(self, thread_id: str | None = None):
            calls.append(thread_id)
            return {"configurable": {"thread_id": thread_id or "fallback"}}

    core = object.__new__(AgentCore)
    core.memory = FakeMemory()
    core.max_iterations = 25

    assert core._invoke_config() == {
        "configurable": {"thread_id": "fallback"},
        "recursion_limit": 25,
    }
    assert core._invoke_config("thread-x") == {
        "configurable": {"thread_id": "thread-x"},
        "recursion_limit": 25,
    }
    assert calls == [None, "thread-x"]


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
    from agent.session.store import SessionStore

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
# StreamHandler thread_id 透传
# --------------------------------------------------------------------------- #
class _FakeMetrics:
    def extract_and_record_llm_usage(self, output, provider="", model=""):
        pass


class _FakeStreamExecutor:
    """模拟共享编译图。"""

    def __init__(self):
        self.used: list[str | None] = []

    async def astream_events(self, *args: Any, **kwargs: Any):
        config = kwargs.get("config") or {}
        tid = config.get("configurable", {}).get("thread_id")
        self.used.append(tid)
        yield {
            "event": "on_chat_model_end",
            "metadata": {},
            "data": {"output": AIMessage(content="你好", id="msg-1")},
        }

    async def aget_state(self, config):
        return None


class _FakeStreamAgent:
    """模拟 StreamHandler 依赖的 agent 接口（无状态模式）。"""

    def __init__(self):
        self.verbose = False
        self.metrics = _FakeMetrics()
        self._state_lock = asyncio.Lock()
        self.cleared_threads: list[str | None] = []
        self.agent_executor = _FakeStreamExecutor()

    def _invoke_config(self, thread_id: str | None = None):
        tid = thread_id or "default"
        return {"configurable": {"thread_id": tid, "recursion_limit": 25}}

    def _thread_id_from_config(self, config: dict[str, Any]) -> str | None:
        configurable = config.get("configurable")
        if isinstance(configurable, dict):
            tid = configurable.get("thread_id")
            if isinstance(tid, str):
                return tid
        return None

    async def _aclear_pending_interrupt(self, thread_id: str | None = None) -> None:
        self.cleared_threads.append(thread_id)

    async def _arepair_rejected_tool_calls(self, config):
        pass


def test_stream_handler_thread_id_plumbing():
    from agent.message_utils import StreamHandler

    agent = _FakeStreamAgent()
    handler = StreamHandler(agent)

    async def collect():
        return [ev async for ev in handler.astream_chat("你好", thread_id="t-sse")]

    events = asyncio.run(collect())

    # 共享 executor 被用于事件流，config 中携带 thread_id
    assert agent.agent_executor.used == ["t-sse"]
    # 中断清理带 thread_id
    assert agent.cleared_threads == ["t-sse"]
    # 无中断，正常完成
    assert events[-1] == {"type": "done"}


def test_stream_handler_default_thread_when_none():
    from agent.message_utils import StreamHandler

    agent = _FakeStreamAgent()
    handler = StreamHandler(agent)

    async def collect():
        return [ev async for ev in handler.astream_chat("你好")]

    events = asyncio.run(collect())

    # config 解析后的默认线程 id（_invoke_config(None) → "default"）
    assert agent.agent_executor.used == ["default"]
    assert agent.cleared_threads == ["default"]
    assert events[-1] == {"type": "done"}


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
