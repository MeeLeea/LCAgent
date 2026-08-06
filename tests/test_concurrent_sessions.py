"""
并发多会话测试套件
==================

覆盖特征点：
  - memory.get_config 显式 thread_id 隔离
  - agent_core._config_for / _invoke_config 的 thread_id 处理
  - _executor_for 的 per-thread 编译图缓存与 LRU 淘汰
  - _update_system_prompt 的 per-thread 提示词隔离
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
# agent_core._config_for / _invoke_config
# --------------------------------------------------------------------------- #
def test_config_for_respects_thread_id():
    from agent.agent_core import AgentCore

    calls = []

    class FakeMemory:
        def get_config(self, thread_id: str | None = None):
            calls.append(thread_id)
            return {"configurable": {"thread_id": thread_id or "fallback"}}

    core = object.__new__(AgentCore)
    core.memory = FakeMemory()
    core.max_iterations = 25

    assert core._config_for() == {
        "configurable": {"thread_id": "fallback"},
        "recursion_limit": 25,
    }
    assert core._config_for("thread-x") == {
        "configurable": {"thread_id": "thread-x"},
        "recursion_limit": 25,
    }
    assert calls == [None, "thread-x"]


def test_config_for_no_arg_invoke_override_keeps_working():
    """兼容测试中 `core._invoke_config = lambda: {...}` 的无参覆盖。"""
    from agent.agent_core import AgentCore

    core = object.__new__(AgentCore)
    core._invoke_config = lambda: {"configurable": {"thread_id": "override"}}
    assert core._config_for() == {"configurable": {"thread_id": "override"}}


# --------------------------------------------------------------------------- #
# _executor_for per-thread 缓存与 LRU 淘汰
# --------------------------------------------------------------------------- #
class _FakeThreadExecutor:
    """模拟 per-thread 编译图。"""

    def __init__(self, tag: str):
        self.tag = tag

    async def astream_events(self, *args: Any, **kwargs: Any):
        yield {}


def test_executor_for_caches_per_thread():
    from collections import OrderedDict

    from agent.agent_core import AgentCore

    core = object.__new__(AgentCore)
    core.agent_executor = _FakeThreadExecutor("base")
    core._state_lock = asyncio.Lock()
    core._executors = OrderedDict()
    core._MAX_THREAD_EXECUTORS = 50

    # 构建线程图，返回独立 executor
    def _create_agent_executor(system_message: SystemMessage | None = None):
        return _FakeThreadExecutor(f"thread-{len(core._executors)}")

    core._create_agent_executor = _create_agent_executor
    core._get_system_prompt = lambda skill_block="": "system prompt"

    async def run():
        e1 = await core._executor_for("t1")
        e2 = await core._executor_for("t2")
        e1_again = await core._executor_for("t1")
        return e1, e2, e1_again

    e1, e2, e1_again = asyncio.run(run())

    # 不同线程不同图；同一线程复用缓存
    assert e1 is not e2
    assert e1 is e1_again
    assert list(core._executors.keys()) == ["t2", "t1"]  # t1 最近访问，移到最后
    # 线程图缓存了独立的 SystemMessage
    assert core._executors["t1"][1] is not core._executors["t2"][1]
    # 未指定 thread_id 时回退 base
    assert asyncio.run(core._executor_for(None)) is core.agent_executor


def test_executor_for_evicts_least_recently_used():
    from collections import OrderedDict

    from agent.agent_core import AgentCore

    core = object.__new__(AgentCore)
    core.agent_executor = _FakeThreadExecutor("base")
    core._state_lock = asyncio.Lock()
    core._executors = OrderedDict()
    core._MAX_THREAD_EXECUTORS = 2

    def _create_agent_executor(system_message: SystemMessage | None = None):
        return _FakeThreadExecutor(f"thread-{len(core._executors)}")

    core._create_agent_executor = _create_agent_executor
    core._get_system_prompt = lambda skill_block="": "system prompt"

    async def run():
        await core._executor_for("t1")
        await core._executor_for("t2")
        await core._executor_for("t3")  # 触发淘汰 t1

    asyncio.run(run())

    assert "t1" not in core._executors
    assert "t2" in core._executors
    assert "t3" in core._executors


# --------------------------------------------------------------------------- #
# _update_system_prompt per-thread 隔离
# --------------------------------------------------------------------------- #
def test_update_system_prompt_only_touches_target_thread():
    from collections import OrderedDict

    from agent.agent_core import AgentCore

    core = object.__new__(AgentCore)
    core.agent_core_prompt = "base prompt"
    core._state_lock = asyncio.Lock()
    core._executors = OrderedDict()
    core._MAX_THREAD_EXECUTORS = 50
    core._compute_skill_block = lambda task: f"[skill:{task}]"

    def _create_agent_executor(system_message: SystemMessage | None = None):
        sys_msg = system_message or SystemMessage(content="")
        return _FakeThreadExecutor("x"), sys_msg

    # 预置两个线程的执行器缓存，模拟已构建的线程图
    sys_a = SystemMessage(content="initial-a")
    sys_b = SystemMessage(content="initial-b")
    core._executors["t-a"] = (_FakeThreadExecutor("a"), sys_a)
    core._executors["t-b"] = (_FakeThreadExecutor("b"), sys_b)

    core._update_system_prompt("alpha", "t-a")

    assert "alpha" in sys_a.content
    assert "alpha" not in sys_b.content  # t-b 提示词不受影响
    assert "alpha" not in getattr(core, "_system_message", SystemMessage(content="")).content


# --------------------------------------------------------------------------- #
# _pending_interrupts per-thread 隔离
# --------------------------------------------------------------------------- #
def _build_core_with_pending() -> Any:
    from agent.agent_core import AgentCore

    core = object.__new__(AgentCore)
    core._pending_interrupts = {}
    return core


def test_pending_interrupt_capture_and_clear_per_thread():
    core = _build_core_with_pending()

    core._capture_pending_interrupt(
        {"configurable": {"thread_id": "t-a"}}, "run"
    )
    core._capture_pending_interrupt(
        {"configurable": {"thread_id": "t-b"}}, "chat"
    )

    assert core._pending_interrupts == {"t-a": "run", "t-b": "chat"}

    # 只清理 t-a，不影响 t-b
    core._clear_pending_interrupt("t-a")
    assert core._pending_interrupts == {"t-b": "chat"}

    # 未指定线程时清空全部
    core._clear_pending_interrupt()
    assert core._pending_interrupts == {}


def test_pending_interrupt_lazy_init_on_object_new():
    """object.__new__ 未初始化 _pending_interrupts 时自动创建。"""
    from agent.agent_core import AgentCore

    core = object.__new__(AgentCore)
    core._capture_pending_interrupt(
        {"configurable": {"thread_id": "t-x"}}, "chat"
    )
    assert core._pending_interrupts == {"t-x": "chat"}


def test_pending_interrupt_lazy_clear_no_crash():
    from agent.agent_core import AgentCore

    core = object.__new__(AgentCore)
    core._clear_pending_interrupt("t-x")  # 不应抛异常


# --------------------------------------------------------------------------- #
# manually_compact(thread_id=) 针对指定线程
# --------------------------------------------------------------------------- #
def test_manually_compact_uses_target_thread():
    from collections import OrderedDict

    from agent.agent_core import AgentCore

    thread_id = "thread-compact"
    state_updates = []
    fetched_thread_ids = []

    class FakeMemory:
        def get_config(self, thread_id: str | None = None):
            return {"configurable": {"thread_id": thread_id or "default"}}

        async def aget_messages(self, thread_id: str | None = None):
            fetched_thread_ids.append(thread_id)
            return [HumanMessage(content=f"m{idx}") for idx in range(8)]

    class FakeExecutor:
        async def aget_state(self, config):
            return SimpleNamespace(values={"summary": ""})

        async def aupdate_state(self, config, values):
            state_updates.append((config, values))

    core = object.__new__(AgentCore)
    core.memory = FakeMemory()
    core.verbose = False
    core.max_iterations = 25
    core._executors = OrderedDict({thread_id: (FakeExecutor(), SystemMessage(content=""))})
    core._MAX_THREAD_EXECUTORS = 50
    core._state_lock = asyncio.Lock()

    # 压缩中间件最小桩：直接产出更新
    class _FakeCompaction:
        async def arun_compaction(self, msgs, existing_summary="", force=False):
            return {
                "summary": "sum",
                "messages": [SimpleNamespace()],
            }

    core._compaction_middleware = _FakeCompaction()

    result = asyncio.run(core.manually_compact(thread_id=thread_id))

    assert fetched_thread_ids == [thread_id]
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


class _FakeStreamAgent:
    """模拟 StreamHandler 依赖的 agent 接口。"""

    def __init__(self):
        self.verbose = False
        self.metrics = _FakeMetrics()
        self._state_lock = asyncio.Lock()
        self.updated_threads: list[str | None] = []
        self.cleared_threads: list[str | None] = []
        self.executor_used: list[str | None] = []

    def _config_for(self, thread_id: str | None = None):
        tid = thread_id or "default"
        return {"configurable": {"thread_id": tid, "recursion_limit": 25}}

    async def _executor_for(self, thread_id: str | None = None):
        return _FakeStreamExecutor(thread_id, self.executor_used)

    def _update_system_prompt(self, task: str = "", thread_id: str | None = None) -> None:
        self.updated_threads.append(thread_id)

    def _thread_id_from_config(self, config: dict[str, Any]) -> str | None:
        configurable = config.get("configurable")
        if isinstance(configurable, dict):
            tid = configurable.get("thread_id")
            if isinstance(tid, str):
                return tid
        return None

    def _clear_pending_interrupt(self, thread_id: str | None = None) -> None:
        self.cleared_threads.append(thread_id)


class _FakeStreamExecutor:
    def __init__(self, thread_id: str | None, used: list[str | None]):
        self.thread_id = thread_id
        self.used = used

    async def astream_events(self, *args: Any, **kwargs: Any):
        self.used.append(self.thread_id)
        yield {
            "event": "on_chat_model_end",
            "metadata": {},
            "data": {"output": AIMessage(content="你好", id="msg-1")},
        }

    async def aget_state(self, config):
        return None


def test_stream_handler_thread_id_plumbing():
    from agent.message_utils import StreamHandler

    agent = _FakeStreamAgent()
    handler = StreamHandler(agent)

    async def collect():
        return [ev async for ev in handler.astream_chat("你好", thread_id="t-sse")]

    events = asyncio.run(collect())

    # thread executor 被用于事件流；提示词与中断清理都带 thread_id
    assert agent.executor_used == ["t-sse"]
    assert agent.updated_threads == ["t-sse"]
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

    assert agent.executor_used == [None]
    assert agent.updated_threads == [None]
    # config 解析后的默认线程 id（_config_for(None) → "default"）
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
