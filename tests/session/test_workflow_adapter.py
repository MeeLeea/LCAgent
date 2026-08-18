"""
测试 WorkflowAdapter（session/workflow_adapter.py）

覆盖：
- SessionAgent 协议属性：session / checkpoint_info / _current_sid / set_current_session
- arun_events 节点事件流（NODE_START → NODE_END → DONE）与 thread_id/is_important
- 非 workflow 会话拒绝执行（ValueError）
- 执行前 raw_context 注入：长期记忆 recall + checkpoint 历史消息
- aresume_events 不支持（NotImplementedError）
- aget_execution_history / aclear_history（复用 SessionStore）
- manually_compact（无消息/非 workflow 会话/正常压缩）
- aclose 幂等
"""
from __future__ import annotations

import asyncio
import types
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from session.workflow_adapter import WorkflowAdapter
from utils.events import EventType


class FakeNode:
    id = "node_a"


class FakeGraph:
    """模拟编译后的 workflow 图：记录入参并触发 NodeTrackingHandler 回调。"""

    def __init__(self, result: dict | None = None) -> None:
        self.result = result or {"final_answer": "最终完成"}
        self.last_state: dict | None = None
        self.last_config: dict | None = None
        self.update_calls: list = []

    def get_graph(self):
        return types.SimpleNamespace(
            nodes={
                "__start__": types.SimpleNamespace(id="__start__"),
                "node_a": FakeNode(),
            }
        )

    async def ainvoke(self, state, config):
        self.last_state = state
        self.last_config = config
        handler = config["callbacks"][0]
        handler.on_chain_start(
            {}, {}, run_id="r1", metadata={"langgraph_node": "node_a"}
        )
        handler.on_chain_end({}, run_id="r1")
        return dict(self.result)

    async def aupdate_state(self, config, update) -> None:
        self.update_calls.append((config, update))


class FakeGraphToken(FakeGraph):
    """在节点执行期间额外触发 on_chat_model_stream 的图(TOKEN 事件)。

    模拟 TeamAgent.astream 透传 callbacks 后,LLM token 增量到达
    NodeTrackingHandler.on_chat_model_stream 的场景;含一个空 content 块,
    验证空块被过滤不产生事件。
    """

    async def ainvoke(self, state, config):
        self.last_state = state
        self.last_config = config
        handler = config["callbacks"][0]
        handler.on_chain_start(
            {}, {}, run_id="r1", metadata={"langgraph_node": "node_a"}
        )
        handler.on_chat_model_stream(types.SimpleNamespace(content="增量文本"))
        handler.on_chat_model_stream(types.SimpleNamespace(content=""))
        handler.on_chain_end({}, run_id="r1")
        return dict(self.result)


class FakeStore:
    """模拟 SessionStore（执行历史按会话隔离）。"""

    def __init__(self) -> None:
        self.history: list = []
        self.cleared: list[str] = []

    async def aget_history(self, sid: str) -> list:
        return self.history

    async def aclear_history(self, sid: str) -> None:
        self.cleared.append(sid)


class FakeMemory:
    """模拟 MemoryManager：仅记录 recall 调用。"""

    def __init__(self, recalled: str = "【长期记忆】\n- [事实] 用户偏好中文\n") -> None:
        self.recalled = recalled
        self.recall_calls: list[str] = []

    async def recall_text(self, thread_id: str, limit=None) -> str:
        self.recall_calls.append(thread_id)
        return self.recalled


def _make_registry(workflow_name: str = "simple") -> MagicMock:
    """构造 SessionRegistry 的替身（覆盖 adapter 用到的全部接口）。"""
    reg = MagicMock()
    reg.checkpointer = None
    reg.current_session_id = "workflow-simple-thread-1"
    reg.workflow_name_of.return_value = workflow_name
    reg.awarm_workspace = AsyncMock()
    reg.aget_messages = AsyncMock(return_value=[])
    reg.get_context.return_value = types.SimpleNamespace(workspace_path=None)
    reg._store = FakeStore()
    return reg


def _make_adapter(reg=None, memory=None) -> WorkflowAdapter:
    return WorkflowAdapter(reg or _make_registry(), memory=memory)


def _collect(adapter, task="测试任务", thread_id=None, is_run_mode=True) -> list:
    """收集 arun_events 事件流（asyncio.run 风格，与项目测试一致）。"""
    events = []

    async def _run():
        async for ev in adapter.arun_events(
            task, thread_id=thread_id, is_run_mode=is_run_mode
        ):
            events.append(ev)

    asyncio.run(_run())
    return events


# ────────────── SessionAgent 协议：属性 ──────────────


def test_session_property_returns_registry():
    """session 属性返回构造时注入的 SessionRegistry。"""
    reg = _make_registry()
    adapter = _make_adapter(reg)
    assert adapter.session is reg


def test_current_sid_explicit_beats_default():
    """显式 thread_id 优先于 registry 当前会话。"""
    reg = _make_registry()
    adapter = _make_adapter(reg)
    assert adapter._current_sid() == "workflow-simple-thread-1"
    assert adapter._current_sid("workflow-pipline-thread-9") == "workflow-pipline-thread-9"


def test_set_current_session_delegates_to_registry():
    """set_current_session 委托 registry.current_session_id。"""
    reg = _make_registry()
    adapter = _make_adapter(reg)
    adapter.set_current_session("workflow-simple-thread-2")
    assert reg.current_session_id == "workflow-simple-thread-2"


def test_checkpoint_info_sqlite():
    """checkpoint 后端为 Sqlite 时上报 sqlite + 文件路径。"""
    reg = _make_registry()

    class _FakeSqliteCheckpointer:
        checkpoint_file = "/tmp/wf.db"

    adapter = WorkflowAdapter(reg, checkpointer=_FakeSqliteCheckpointer())
    info = adapter.checkpoint_info
    assert info["checkpoint_backend"] == "sqlite"
    assert info["checkpoint_file"] == "/tmp/wf.db"


def test_checkpoint_info_memory_fallback():
    """无 checkpointer 时上报内存后端。"""
    adapter = _make_adapter()
    info = adapter.checkpoint_info
    assert info["checkpoint_backend"] == "memory"
    assert info["checkpoint_file"] == "(内存)"


# ────────────── arun_events 执行 ──────────────


def test_arun_events_node_flow(monkeypatch):
    """事件流顺序 NODE_START → NODE_END → DONE，DONE 携带 final_answer。"""
    fake_graph = FakeGraph()
    monkeypatch.setattr(
        "graph.registry.build_workflow",
        lambda name, checkpointer=None: (fake_graph, {"manager": object()}),
    )
    adapter = _make_adapter()

    events = _collect(adapter, thread_id="workflow-simple-thread-1")

    assert [ev.event_type for ev in events] == [
        EventType.NODE_START,
        EventType.NODE_END,
        EventType.DONE,
    ]
    assert events[0].node == "node_a"
    assert events[1].node == "node_a"
    done = events[-1]
    assert done.content == "最终完成"
    assert done.thread_id == "workflow-simple-thread-1"
    assert done.is_important is True


def test_arun_events_emits_token_stream(monkeypatch):
    """节点执行期间 LLM token 增量 → TOKEN 事件;空块被过滤。

    模拟 TeamAgent.astream 透传 callbacks 后,LLM 的 on_chat_model_stream
    事件被 NodeTrackingHandler 捕获转发为 TOKEN 事件。
    """
    fake_graph = FakeGraphToken()
    monkeypatch.setattr(
        "graph.registry.build_workflow",
        lambda name, checkpointer=None: (fake_graph, {"manager": object()}),
    )
    adapter = _make_adapter()

    events = _collect(adapter, thread_id="workflow-simple-thread-1")

    tokens = [ev for ev in events if ev.event_type == EventType.TOKEN]
    assert [t.content for t in tokens] == ["增量文本"]
    assert tokens[0].thread_id == "workflow-simple-thread-1"
    assert tokens[0].role == "assistant"
    # 事件顺序保持: NODE_START → TOKEN → NODE_END → DONE
    assert [ev.event_type for ev in events] == [
        EventType.NODE_START,
        EventType.TOKEN,
        EventType.NODE_END,
        EventType.DONE,
    ]


def test_arun_events_not_workflow_session():
    """非 workflow 会话（workflow_name_of 返回空）抛 ValueError。"""
    reg = _make_registry(workflow_name=None)
    adapter = _make_adapter(reg)
    try:
        _collect(adapter)
    except ValueError as error:
        assert "不是 workflow 会话" in str(error)
    else:
        raise AssertionError("应抛出 ValueError")


def test_arun_events_recall_injected(monkeypatch):
    """执行前 recall 的长期记忆注入 initial_state.raw_context。"""
    fake_graph = FakeGraph()
    monkeypatch.setattr(
        "graph.registry.build_workflow",
        lambda name, checkpointer=None: (fake_graph, {"manager": object()}),
    )
    memory = FakeMemory()
    adapter = _make_adapter(memory=memory)

    _collect(adapter, thread_id="workflow-simple-thread-1")

    assert memory.recall_calls == ["workflow-simple-thread-1"]
    assert "用户偏好中文" in fake_graph.last_state["raw_context"]


def test_arun_events_history_injected(monkeypatch):
    """checkpoint 历史 AIMessage 注入 raw_context（含【历史执行记录】标记）。"""
    fake_graph = FakeGraph()
    monkeypatch.setattr(
        "graph.registry.build_workflow",
        lambda name, checkpointer=None: (fake_graph, {"manager": object()}),
    )
    reg = _make_registry()
    reg.aget_messages = AsyncMock(
        return_value=[
            AIMessage(content="上一轮的设计方案 A"),
            HumanMessage(content="用户新任务"),
        ]
    )
    adapter = _make_adapter(reg)

    _collect(adapter)

    raw_context = fake_graph.last_state["raw_context"]
    assert "【历史执行记录】" in raw_context
    assert "上一轮的设计方案 A" in raw_context
    # HumanMessage 不进入历史预览
    assert "用户新任务" not in raw_context


def test_arun_events_done_is_important_false_by_default(monkeypatch):
    """is_run_mode=False（对话模式）时 DONE 事件 is_important=False。"""
    fake_graph = FakeGraph()
    monkeypatch.setattr(
        "graph.registry.build_workflow",
        lambda name, checkpointer=None: (fake_graph, {"manager": object()}),
    )
    adapter = _make_adapter()

    events = _collect(adapter, is_run_mode=False)

    assert events[-1].event_type == EventType.DONE
    assert events[-1].is_important is False


def test_arun_events_injects_workspace_path(monkeypatch):
    """会话绑定 workspace 时,workspace_path 写入 config.configurable(Worker 工具隔离)。"""
    fake_graph = FakeGraph()
    monkeypatch.setattr(
        "graph.registry.build_workflow",
        lambda name, checkpointer=None: (fake_graph, {"manager": object()}),
    )
    reg = _make_registry()
    reg.get_context.return_value = types.SimpleNamespace(workspace_path="C:/ws")
    adapter = _make_adapter(reg)

    _collect(adapter, thread_id="workflow-simple-thread-1")

    configurable = fake_graph.last_config["configurable"]
    assert configurable["thread_id"] == "workflow-simple-thread-1"
    assert configurable["workspace_path"] == "C:/ws"


def test_arun_events_no_workspace_path_by_default(monkeypatch):
    """会话未绑定 workspace 时不注入 workspace_path(兼容旧会话)。"""
    fake_graph = FakeGraph()
    monkeypatch.setattr(
        "graph.registry.build_workflow",
        lambda name, checkpointer=None: (fake_graph, {"manager": object()}),
    )
    adapter = _make_adapter()

    _collect(adapter, thread_id="workflow-simple-thread-1")

    configurable = fake_graph.last_config["configurable"]
    assert "workspace_path" not in configurable


# ────────────── 不支持能力 / 历史 / 压缩 / 生命周期 ──────────────


def test_aresume_events_not_implemented():
    """workflow 无 HITL 中断语义，aresume_events 抛 NotImplementedError。"""
    adapter = _make_adapter()
    try:
        asyncio.run(adapter.aresume_events({}))
    except NotImplementedError:
        pass
    else:
        raise AssertionError("应抛出 NotImplementedError")


def test_aget_execution_history_reuses_store():
    """执行历史委托 SessionStore 按会话隔离。"""
    reg = _make_registry()
    reg._store.history = [{"turn": 1}]
    adapter = _make_adapter(reg)

    async def _run():
        return await adapter.aget_execution_history("workflow-simple-thread-1")

    assert asyncio.run(_run()) == [{"turn": 1}]


def test_aclear_history_reuses_store():
    """清空历史委托 SessionStore。"""
    reg = _make_registry()
    adapter = _make_adapter(reg)

    asyncio.run(adapter.aclear_history("workflow-simple-thread-1"))

    assert reg._store.cleared == ["workflow-simple-thread-1"]


def test_manually_compact_no_messages_returns_none():
    """无 checkpoint 消息时手动压缩返回 None。"""
    reg = _make_registry()
    reg.aget_messages = AsyncMock(return_value=[])
    adapter = _make_adapter(reg)

    assert asyncio.run(adapter.manually_compact()) is None


def test_manually_compact_non_workflow_session_returns_none():
    """非 workflow 会话手动压缩返回 None。"""
    reg = _make_registry(workflow_name=None)
    reg.aget_messages = AsyncMock(return_value=[AIMessage(content="历史")])
    adapter = _make_adapter(reg)

    assert asyncio.run(adapter.manually_compact()) is None


def test_manually_compact_applies_update(monkeypatch):
    """有消息 + 可压缩时返回摘要统计并写回 checkpoint。"""
    reg = _make_registry()
    reg.aget_messages = AsyncMock(
        return_value=[AIMessage(content="历史") for _ in range(3)]
    )

    class _FakeCheckpointer:
        async def aget_tuple(self, config):
            return types.SimpleNamespace(
                checkpoint={"channel_values": {"summary": "旧摘要"}}
            )

    fake_graph = FakeGraph()

    class _FakeMW:
        def __init__(self) -> None:
            self.seen_summary = None

        async def arun_compaction(self, msgs, existing_summary="", force=False):
            self.seen_summary = existing_summary
            return {"summary": "新摘要", "messages": [1, 2, 3]}

    fake_mw = _FakeMW()
    monkeypatch.setattr(
        "graph.registry.build_workflow",
        lambda name, checkpointer=None: (fake_graph, {"manager": object()}),
    )
    monkeypatch.setattr(
        "session.workflow_adapter._build_compaction_middleware", lambda agent, cfg: fake_mw
    )
    adapter = WorkflowAdapter(reg, checkpointer=_FakeCheckpointer())

    result = asyncio.run(adapter.manually_compact(force=True))

    assert result == {"summary": "新摘要", "messages_before": 3, "messages_after": 2}
    assert fake_mw.seen_summary == "旧摘要"
    assert len(fake_graph.update_calls) == 1


def test_close_idempotent():
    """aclose 幂等，关闭后再执行抛异常。"""
    adapter = _make_adapter()

    async def _run():
        await adapter.aclose()
        await adapter.aclose()  # 重复关闭应静默成功

    asyncio.run(_run())
    # 关闭后 arun_events 拒绝执行
    try:
        _collect(adapter)
    except RuntimeError as error:
        assert "已关闭" in str(error)
    else:
        raise AssertionError("关闭后应拒绝执行")