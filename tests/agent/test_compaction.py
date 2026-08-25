"""Compaction 中间件测试：增量摘要 + 工具输出 Prune + 安全切割 + 手动触发"""
import asyncio
from types import SimpleNamespace

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from agent.compaction import (
    CompactionConfig,
    LCAgentCompactionMiddleware,
    LCAgentState,
)

# ============ 测试用 FakeModel ============


class FakeModel:
    """模拟 LLM，记录调用并返回预设响应"""

    def __init__(self, response: str = "摘要内容", fail: bool = False):
        self.response = response
        self.fail = fail
        self.invoke_calls: list[str] = []
        self.ainvoke_calls: list[str] = []

    def invoke(self, prompt: str):
        self.invoke_calls.append(prompt)
        if self.fail:
            raise RuntimeError("LLM 不可用")
        return SimpleNamespace(text=self.response)

    async def ainvoke(self, prompt: str):
        self.ainvoke_calls.append(prompt)
        if self.fail:
            raise RuntimeError("LLM 不可用")
        return SimpleNamespace(text=self.response)


def _build_messages(count: int, tool_output_len: int = 0) -> list:
    """构建 N 条消息，可选在偶数位置放长工具输出"""
    msgs = []
    for i in range(count):
        if i % 4 == 0:
            msgs.append(HumanMessage(content=f"用户消息-{i}"))
        elif i % 4 == 1:
            msgs.append(AIMessage(content=f"AI回复-{i}"))
        elif i % 4 == 2:
            msgs.append(AIMessage(content="", tool_calls=[{"name": "search", "args": {"q": f"query-{i}"}, "id": f"call-{i}"}]))
        else:
            content = "x" * tool_output_len if tool_output_len else f"工具结果-{i}"
            msgs.append(ToolMessage(content=content, tool_call_id=f"call-{i-1}", name="search"))
    return msgs


# ============ 不触发压缩 ============


def test_no_compaction_when_below_threshold():
    """消息数不超过阈值时不压缩"""
    model = FakeModel()
    mw = LCAgentCompactionMiddleware(model, CompactionConfig(max_messages=10, keep_recent=4))
    state = {"messages": _build_messages(5), "summary": ""}
    result = mw.before_model(state, runtime=None)
    assert result is None
    assert len(model.invoke_calls) == 0


def test_no_compaction_async_when_below_threshold():
    """异步：消息数不超过阈值时不压缩"""
    model = FakeModel()
    mw = LCAgentCompactionMiddleware(model, CompactionConfig(max_messages=10, keep_recent=4))

    async def run():
        state = {"messages": _build_messages(5), "summary": ""}
        return await mw.abefore_model(state, runtime=None)

    result = asyncio.run(run())
    assert result is None


# ============ 增量摘要 ============


def test_incremental_summary_uses_existing_summary():
    """有已有摘要时，prompt 包含已有摘要内容"""
    model = FakeModel(response="更新后的摘要")
    mw = LCAgentCompactionMiddleware(model, CompactionConfig(max_messages=5, keep_recent=2))
    msgs = _build_messages(8)
    state = {"messages": msgs, "summary": "之前的摘要内容"}

    result = mw.before_model(state, runtime=None)

    assert result is not None
    assert result["summary"] == "更新后的摘要"
    # prompt 中应包含已有摘要
    assert "之前的摘要内容" in model.invoke_calls[0]
    assert "【已有摘要】" in model.invoke_calls[0]


def test_first_time_summary_has_no_existing():
    """无已有摘要时，prompt 不包含已有摘要段"""
    model = FakeModel(response="首次摘要")
    mw = LCAgentCompactionMiddleware(model, CompactionConfig(max_messages=5, keep_recent=2))
    msgs = _build_messages(8)
    state = {"messages": msgs, "summary": ""}

    result = mw.before_model(state, runtime=None)

    assert result is not None
    assert result["summary"] == "首次摘要"
    assert "【已有摘要】" not in model.invoke_calls[0]


def test_compaction_summary_failure_returns_none():
    """LLM 调用失败时不压缩，保留原消息"""
    model = FakeModel(fail=True)
    mw = LCAgentCompactionMiddleware(model, CompactionConfig(max_messages=5, keep_recent=2))
    msgs = _build_messages(8)
    state = {"messages": msgs, "summary": "旧摘要"}

    result = mw.before_model(state, runtime=None)

    assert result is None  # 不压缩


def test_compaction_summary_failure_returns_empty_string(caplog):
    """摘要失败时返回空串并记录 warning。"""
    model = FakeModel(fail=True)
    mw = LCAgentCompactionMiddleware(model, CompactionConfig())
    msgs = _build_messages(4)

    with caplog.at_level("WARNING"):
        sync_result = mw._create_summary_sync("旧摘要", msgs)

    assert sync_result == ""
    assert "增量摘要生成失败，跳过压缩" in caplog.text


def test_async_compaction_summary_failure_returns_empty_string(caplog):
    """异步摘要失败时返回空串并记录 warning。"""
    model = FakeModel(fail=True)
    mw = LCAgentCompactionMiddleware(model, CompactionConfig())
    msgs = _build_messages(4)

    async def run():
        with caplog.at_level("WARNING"):
            return await mw._aincremental_summary("旧摘要", msgs)

    result = asyncio.run(run())

    assert result == ""
    assert "增量摘要生成失败，跳过压缩" in caplog.text


# ============ 工具输出 Prune ============


def test_prune_replaces_long_tool_output():
    """超长工具输出被替换为占位符"""
    long_output = "A" * 500
    msgs = [
        HumanMessage(content="问题"),
        AIMessage(content="", tool_calls=[{"name": "read", "args": {}, "id": "c1"}]),
        ToolMessage(content=long_output, tool_call_id="c1", name="read"),
    ]
    model = FakeModel()
    mw = LCAgentCompactionMiddleware(model, CompactionConfig(max_tool_output_chars=200, tool_prune_preview=50))

    pruned = mw._prune_tool_outputs(msgs)

    assert len(pruned) == 3
    assert isinstance(pruned[2], ToolMessage)
    assert "[工具输出已裁剪 500→50字符]" in pruned[2].content
    assert pruned[2].content.endswith("...")
    assert len(pruned[2].content) < 100  # 远小于原始 500


def test_prune_preserves_short_tool_output():
    """短工具输出不被裁剪"""
    short_output = "短结果"
    msgs = [
        ToolMessage(content=short_output, tool_call_id="c1", name="calc"),
    ]
    model = FakeModel()
    mw = LCAgentCompactionMiddleware(model, CompactionConfig(max_tool_output_chars=200))

    pruned = mw._prune_tool_outputs(msgs)

    assert pruned[0].content == "短结果"


def test_prune_preserves_tool_call_id_and_name():
    """Prune 后 tool_call_id 和 name 保持不变"""
    msgs = [
        ToolMessage(content="B" * 300, tool_call_id="call-xyz", name="file_read", status="success"),
    ]
    model = FakeModel()
    mw = LCAgentCompactionMiddleware(model, CompactionConfig(max_tool_output_chars=100, tool_prune_preview=30))

    pruned = mw._prune_tool_outputs(msgs)

    assert pruned[0].tool_call_id == "call-xyz"
    assert pruned[0].name == "file_read"
    assert pruned[0].status == "success"


# ============ 安全切割 ============


def test_safe_cutoff_does_not_split_ai_tool_pair():
    """切割点不拆开 AIMessage(tool_calls) + ToolMessage 对"""
    msgs = []
    for i in range(10):
        msgs.append(HumanMessage(content=f"msg-{i}"))
    # 在末尾放置 AI+Tool 对
    msgs.append(AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "c1"}]))
    msgs.append(ToolMessage(content="result", tool_call_id="c1", name="t"))
    msgs.append(HumanMessage(content="latest"))

    mw = LCAgentCompactionMiddleware(FakeModel(), CompactionConfig(max_messages=5, keep_recent=4))
    cutoff = mw._find_safe_cutoff(msgs)

    # msgs 有 13 条，keep 4 -> target=9
    # msgs[9] 是 HumanMessage，不是 ToolMessage，直接返回 9
    assert cutoff == 9
    # 切割点左侧不含 ToolMessage（不含未配对的工具结果）
    assert not isinstance(msgs[cutoff], ToolMessage)


def test_safe_cutoff_retreats_from_tool_message():
    """切割点落在 ToolMessage 上时向前回退到 AIMessage"""
    msgs = _build_messages(12, tool_output_len=0)
    # keep=2 -> target=10, msgs[10] 是 ToolMessage -> 回退
    mw = LCAgentCompactionMiddleware(FakeModel(), CompactionConfig(max_messages=5, keep_recent=2))
    cutoff = mw._find_safe_cutoff(msgs)

    assert cutoff > 0
    assert not isinstance(msgs[cutoff], ToolMessage)


def test_safe_cutoff_returns_zero_when_too_few():
    """消息数不足时返回 0"""
    mw = LCAgentCompactionMiddleware(FakeModel(), CompactionConfig(max_messages=50, keep_recent=20))
    assert mw._find_safe_cutoff(_build_messages(5)) == 0


# ============ 消息重建 ============


def test_compaction_produces_summary_system_message():
    """压缩后消息列表头部是 SystemMessage（摘要）"""
    model = FakeModel(response="压缩摘要")
    mw = LCAgentCompactionMiddleware(model, CompactionConfig(max_messages=5, keep_recent=2))
    msgs = _build_messages(8)
    state = {"messages": msgs, "summary": ""}

    result = mw.before_model(state, runtime=None)

    assert result is not None
    new_messages = result["messages"]
    # 第一条是 RemoveMessage(REMOVE_ALL_MESSAGES)
    from langchain_core.messages import RemoveMessage
    assert isinstance(new_messages[0], RemoveMessage)
    # 第二条是 SystemMessage（摘要）
    assert isinstance(new_messages[1], SystemMessage)
    assert "压缩摘要" in new_messages[1].content
    assert "【历史对话摘要" in new_messages[1].content


# ============ 手动触发 ============


def test_manual_compaction_returns_update_dict():
    """手动压缩返回 {messages, summary} 字典"""
    model = FakeModel(response="手动摘要")
    mw = LCAgentCompactionMiddleware(model, CompactionConfig(max_messages=5, keep_recent=2))
    msgs = _build_messages(8)

    async def run():
        return await mw.arun_compaction(msgs, existing_summary="")

    result = asyncio.run(run())

    assert result is not None
    assert result["summary"] == "手动摘要"
    assert "messages" in result
    assert "summary" in result


def test_manual_compaction_returns_none_when_below_threshold():
    """消息不足时手动压缩返回 None（force=False 默认行为）"""
    mw = LCAgentCompactionMiddleware(FakeModel(), CompactionConfig(max_messages=50, keep_recent=10))

    async def run():
        return await mw.arun_compaction(_build_messages(5), existing_summary="")

    result = asyncio.run(run())
    assert result is None


def test_manual_compaction_force_bypasses_threshold():
    """force=True 时跳过阈值检查，消息数未超阈值也能压缩

    场景：max_messages=50（阈值高），消息只有 25 条，
    但 keep_recent=10，force=True 允许压缩前 15 条。
    """
    model = FakeModel(response="强制摘要")
    mw = LCAgentCompactionMiddleware(model, CompactionConfig(max_messages=50, keep_recent=10))
    msgs = _build_messages(25)

    async def run():
        return await mw.arun_compaction(msgs, existing_summary="", force=True)

    result = asyncio.run(run())

    assert result is not None
    assert result["summary"] == "强制摘要"
    # RemoveMessage(1) + summary SystemMessage(1) + 保留消息
    # cutoff 因 ToolMessage 安全对齐可能 > keep_recent，故用范围断言
    assert 11 <= len(result["messages"]) <= 13


def test_manual_compaction_force_returns_none_when_too_few():
    """force=True 时消息数 <= keep_recent 仍返回 None（无法安全切割）"""
    mw = LCAgentCompactionMiddleware(FakeModel(), CompactionConfig(max_messages=50, keep_recent=20))

    async def run():
        return await mw.arun_compaction(_build_messages(5), existing_summary="", force=True)

    result = asyncio.run(run())
    assert result is None


def test_manual_compaction_uses_existing_summary():
    """手动压缩时传入已有摘要，prompt 包含它"""
    model = FakeModel(response="合并摘要")
    mw = LCAgentCompactionMiddleware(model, CompactionConfig(max_messages=5, keep_recent=2))

    async def run():
        return await mw.arun_compaction(_build_messages(8), existing_summary="旧摘要文本")

    result = asyncio.run(run())

    assert result is not None
    assert "旧摘要文本" in model.ainvoke_calls[0]


# ============ per-thread 隔离（通过 state 天然隔离） ============


def test_state_isolation_via_checkpoint():
    """验证 summary 存在 state 中，不同 thread 的 state 独立

    这里验证 LCAgentState 有 summary 字段，
    实际的 per-thread 隔离由 LangGraph checkpoint 保证：
    每个 thread_id 有独立的 state 快照。
    """
    # LCAgentState 是 AgentState 的子类，添加了 summary 字段
    assert "summary" in LCAgentState.__annotations__

    # 模拟两个 thread 的 state
    state_a = {"messages": [], "summary": "Thread A 的摘要"}
    state_b = {"messages": [], "summary": "Thread B 的摘要"}

    # 两个 state 完全独立
    assert state_a["summary"] != state_b["summary"]
    assert state_a["summary"] == "Thread A 的摘要"
    assert state_b["summary"] == "Thread B 的摘要"


# ============ CompactionConfig ============


def test_config_from_kwargs_uses_defaults_when_zero():
    """max_context_messages=0 时使用默认值"""
    cfg = CompactionConfig.from_kwargs(max_context_messages=0, context_trim_keep=12)
    assert cfg.max_messages == 50  # 默认值
    assert cfg.keep_recent == 12


def test_config_from_kwargs_uses_provided_values():
    """使用提供的值"""
    cfg = CompactionConfig.from_kwargs(max_context_messages=30, context_trim_keep=8)
    assert cfg.max_messages == 30
    assert cfg.keep_recent == 8


def test_config_from_kwargs_enforces_min_keep():
    """keep_recent 最少为 4"""
    cfg = CompactionConfig.from_kwargs(max_context_messages=30, context_trim_keep=2)
    assert cfg.keep_recent == 4
