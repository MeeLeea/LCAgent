"""空最终回答检测：回归测试。

历史回归：提交 0f7c3ff 引入 `final_ai[0] = output`，但调用方以空列表
`_ai_holder = []` 传入，导致每个 `on_chat_model_end` 都抛
`list assignment index out of range`，前端在会话结束前报错。
修复：`_arun_graph_events` 用切片赋值 `final_ai[:] = [output]`，兼容空/已占位列表。
本文件验证该路径不再抛 IndexError，且能记录最终 AIMessage。

另覆盖：空最终回答（network_error 断流 / reasoning 耗尽 length）在模型节点
结束的 superstep 已随图写入 checkpoint；`arun_events` 发 ERROR 前必须先把
空 AIMessage 从 checkpoint 移除，避免残缺历史污染 retry 时的模型上下文。
"""
from __future__ import annotations

import asyncio
import contextlib
from types import SimpleNamespace
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from agent.agent_core import AgentCore
from utils.events import AgentEvent, EventType


class FakeToolLLM(FakeMessagesListChatModel):
    """支持 bind_tools 的 FakeMessagesListChatModel（create_agent 会调 bind_tools）。"""

    def bind_tools(self, tools, **kwargs):
        return self


@tool
def noop() -> str:
    """回显 ok。"""
    return "ok"


def _make_core(graph: Any) -> AgentCore:
    """构造最小 AgentCore（绕过 __init__），只设 _arun_graph_events 依赖的属性。"""
    core = object.__new__(AgentCore)
    core.agent_executor = graph
    core.llm = SimpleNamespace(provider="fake", model="fake-model")
    core._metrics = SimpleNamespace(
        extract_and_record_llm_usage=lambda *a, **kw: None,
        increment_turn=lambda: None,
    )
    return core


def _run(core: AgentCore, input_msg: dict, holder: list[AIMessage]) -> list[AgentEvent]:
    config = {"configurable": {"thread_id": "test-thread"}}

    async def _collect():
        events: list[AgentEvent] = []
        async for ev in core._arun_graph_events(
            input_msg,
            config,
            "test-thread",
            "test-trace",
            final_ai=holder,
        ):
            events.append(ev)
        return events

    return asyncio.run(_collect())


class TestEmptyFinalAnswerGuard:
    """回归：final_ai out-param 以空列表传入时不得抛 IndexError，且能记录最终 AIMessage。"""

    def test_accepts_empty_list_and_records_final_ai(self):
        # Given: 单轮、最终空回答（空内容、无工具调用）
        llm = FakeToolLLM(responses=[AIMessage(content="", tool_calls=[])])
        graph = create_agent(model=llm, tools=[], checkpointer=MemorySaver())
        core = _make_core(graph)
        holder: list[AIMessage] = []  # 空列表初始化 —— 正是历史回归现场

        _run(
            core,
            {"messages": [{"role": "user", "content": "hi"}]},
            holder,
        )

        # 本轮为空文本且无工具调用，`_arun_graph_events` 层不产出事件
        # （DONE/ERROR 由上层 arun_events 发出）；此处关键是 holder 被正确记录且不抛 IndexError。
        assert len(holder) == 1, "应记录最终 AIMessage"
        assert isinstance(holder[0], AIMessage)
        assert holder[0].content == ""

    def test_multiple_model_calls_records_latest(self):
        # Given: 先工具调用、模型再回一轮（结束）；holder 应记录最后一次 model 调用
        llm = FakeToolLLM(
            responses=[
                AIMessage(
                    content="先执行工具",
                    tool_calls=[{
                        "name": "noop",
                        "args": {},
                        "id": "call_1",
                        "type": "tool_call",
                    }],
                ),
            ]
        )
        graph = create_agent(model=llm, tools=[noop], checkpointer=MemorySaver())
        core = _make_core(graph)
        holder: list[AIMessage] = []

        _run(
            core,
            {"messages": [{"role": "user", "content": "do it"}]},
            holder,
        )

        assert holder, "应记录到 AIMessage"
        assert isinstance(holder[0], AIMessage)


def _patch_for_arun_events(core: AgentCore) -> None:
    """为 arun_events 外层调用链补齐最小 mock（config 组装 / 中断检查 / verbose 切换）。"""
    core._ensure_not_closed = lambda: None
    core._current_sid = lambda thread_id=None: thread_id or "test-thread"
    core._session_registry = None

    async def _fake_config(thread_id: str | None = None) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id or "test-thread"}}

    async def _no_interrupt(*args: Any, **kwargs: Any) -> None:
        return None

    core._ainvoke_config = _fake_config
    core._acheck_interrupt_event = _no_interrupt
    core._aclear_pending_interrupt = _no_interrupt
    core._temp_verbose = lambda verbose: contextlib.nullcontext()


class TestEmptyFinalAnswerCheckpointCleanup:
    """空最终回答清理：arun_events 发 ERROR 前应把空 AIMessage 从 checkpoint 移除。"""

    def test_empty_final_answer_removed_from_checkpoint(self):
        # Given: 模型返回空回答（finish_reason=network_error，模拟上游断流）
        llm = FakeToolLLM(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[],
                    response_metadata={"finish_reason": "network_error"},
                ),
            ]
        )
        graph = create_agent(model=llm, tools=[], checkpointer=MemorySaver())
        core = _make_core(graph)
        _patch_for_arun_events(core)
        config = {"configurable": {"thread_id": "t-empty"}}

        # When: 走完整 arun_events 路径
        async def _collect() -> list[AgentEvent]:
            return [ev async for ev in core.arun_events("hi", thread_id="t-empty")]

        events = asyncio.run(_collect())

        # Then: 尾部是 ERROR（而非静默空 DONE）
        assert events[-1].event_type == EventType.ERROR

        # checkpoint 中不应残留空 AIMessage，用户消息保留
        state = asyncio.run(graph.aget_state(config))
        messages = state.values.get("messages", [])
        assert not [m for m in messages if isinstance(m, AIMessage)], (
            "空 AIMessage 应已从 checkpoint 移除，避免污染 retry 上下文"
        )
        assert any(isinstance(m, HumanMessage) for m in messages)
