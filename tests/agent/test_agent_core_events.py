"""工具异常事件流测试：验证 _arun_graph_events 正确处理 on_tool_error 事件。

覆盖场景：
1. 单工具崩 → 发 tool_result（content 含错误）+ error 事件
2. 并行多工具崩 → 第一个崩的发 tool_result，其余孤儿补发失败 tool_result + error
3. 正常工具调用不受影响（on_tool_end 正常发 tool_result）

设计：
- 用 FakeToolLLM（支持 bind_tools）+ 真实 create_agent 构造最小 graph
- 用 object.__new__(AgentCore) 绕过 __init__，只设必要属性
- 直接调 _arun_graph_events 收集 AgentEvent 列表

注意：工具错误的 LLM 反思纠错（转 ToolMessage(status="error")）由
ToolExecutionErrorMW 负责（见 test_tool_error_mw.py），
本文件只覆盖事件流映射。
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from agent.agent_core import AgentCore
from utils.events import AgentEvent, EventType


class FakeToolLLM(FakeMessagesListChatModel):
    """支持 bind_tools 的 FakeMessagesListChatModel。

    create_agent 的 model_node 会调 model.bind_tools(tools)，
    原 FakeMessagesListChatModel 抛 NotImplementedError。这里重写返回 self。
    """

    def bind_tools(self, tools, **kwargs):
        return self


@tool
def crash_tool(x: str) -> str:
    """故意抛 KeyError 的工具，复现 server-thread-1c0242a9 的真实异常。"""
    raise KeyError("dst_path")


@tool
def ok_tool(x: str) -> str:
    """正常工具，返回输入的 echo。"""
    return f"echo: {x}"


def _make_core(graph: Any) -> AgentCore:
    """构造最小 AgentCore（绕过 __init__），只设 _arun_graph_events 依赖的属性。"""
    core = object.__new__(AgentCore)
    core.agent_executor = graph
    core.llm = SimpleNamespace(provider="fake", model="fake-model")
    # metrics 是 property，底层字段 _metrics
    core._metrics = SimpleNamespace(
        extract_and_record_llm_usage=lambda *a, **kw: None,
        increment_turn=lambda: None,
    )
    return core


def _collect_events(core: AgentCore, input_msg: dict, thread_id: str = "test-thread") -> list[AgentEvent]:
    """跑 _arun_graph_events 收集所有 AgentEvent。"""
    config = {"configurable": {"thread_id": thread_id}}
    trace_id = "test-trace"

    async def _run():
        events: list[AgentEvent] = []
        async for ev in core._arun_graph_events(input_msg, config, thread_id, trace_id):
            events.append(ev)
        return events

    return asyncio.run(_run())


# ============ 测试 ============


class TestSingleToolError:
    """单工具崩：on_tool_error → tool_result（含错误）+ error 事件。"""

    def test_single_tool_error_yields_tool_result_and_error(self):
        # Given: LLM 第一轮发起 crash_tool 调用
        llm = FakeToolLLM(responses=[
            AIMessage(
                content="调用崩工具",
                tool_calls=[{
                    "name": "crash_tool",
                    "args": {"x": "1"},
                    "id": "call_1",
                    "type": "tool_call",
                }],
            ),
        ])
        graph = create_agent(
            model=llm,
            tools=[crash_tool],
            checkpointer=MemorySaver(),
        )
        core = _make_core(graph)

        # When: 跑 _arun_graph_events
        events = _collect_events(core, {"messages": [("user", "go")]})

        # Then: 收到 tool_result（content 含错误信息）
        tool_results = [e for e in events if e.event_type == EventType.TOOL_RESULT]
        assert len(tool_results) >= 1, f"应至少有 1 个 tool_result，实际 {len(tool_results)}: {[e.event_type.value for e in events]}"
        assert "[工具执行失败]" in tool_results[0].content
        assert "dst_path" in tool_results[0].content

        # And: 收到 error 事件（异常逃逸到 except Exception）
        errors = [e for e in events if e.event_type == EventType.ERROR]
        assert len(errors) == 1, f"应有 1 个 error 事件，实际 {len(errors)}"


class TestParallelToolError:
    """并行多工具崩：第一个崩的发 tool_result，其余孤儿补发失败 tool_result。"""

    def test_parallel_tool_error_supplements_orphan_tool_results(self):
        # Given: LLM 第一轮发起 5 个并发 crash_tool 调用
        llm = FakeToolLLM(responses=[
            AIMessage(
                content="并发调用 5 个崩工具",
                tool_calls=[
                    {"name": "crash_tool", "args": {"x": "1"}, "id": "c1", "type": "tool_call"},
                    {"name": "crash_tool", "args": {"x": "2"}, "id": "c2", "type": "tool_call"},
                    {"name": "crash_tool", "args": {"x": "3"}, "id": "c3", "type": "tool_call"},
                    {"name": "crash_tool", "args": {"x": "4"}, "id": "c4", "type": "tool_call"},
                    {"name": "crash_tool", "args": {"x": "5"}, "id": "c5", "type": "tool_call"},
                ],
            ),
        ])
        graph = create_agent(
            model=llm,
            tools=[crash_tool],
            checkpointer=MemorySaver(),
        )
        core = _make_core(graph)

        # When: 跑 _arun_graph_events
        events = _collect_events(core, {"messages": [("user", "go")]})

        # Then: 收到多个 tool_result（至少 2 个：1 个 on_tool_error + 1 个孤儿补发）
        tool_results = [e for e in events if e.event_type == EventType.TOOL_RESULT]
        assert len(tool_results) >= 2, f"应至少有 2 个 tool_result（1 on_tool_error + 孤儿补发），实际 {len(tool_results)}"

        # And: 每个 tool_result 的 content 都含 [工具执行失败]
        for tr in tool_results:
            assert "[工具执行失败]" in tr.content, f"tool_result content 应含错误标记: {tr.content}"

        # And: 收到 error 事件
        errors = [e for e in events if e.event_type == EventType.ERROR]
        assert len(errors) == 1


class TestNormalToolUnaffected:
    """正常工具调用不受影响：on_tool_end 正常发 tool_result。"""

    def test_normal_tool_yields_tool_result_without_error(self):
        # Given: LLM 第一轮发起 ok_tool 调用，第二轮返回纯文本
        llm = FakeToolLLM(responses=[
            AIMessage(
                content="调用正常工具",
                tool_calls=[{
                    "name": "ok_tool",
                    "args": {"x": "hello"},
                    "id": "call_ok_1",
                    "type": "tool_call",
                }],
            ),
            AIMessage(content="完成"),
        ])
        graph = create_agent(
            model=llm,
            tools=[ok_tool],
            checkpointer=MemorySaver(),
        )
        core = _make_core(graph)

        # When: 跑 _arun_graph_events
        events = _collect_events(core, {"messages": [("user", "go")]})

        # Then: 收到 tool_result（正常结果，不含 [工具执行失败]）
        tool_results = [e for e in events if e.event_type == EventType.TOOL_RESULT]
        assert len(tool_results) >= 1
        assert "echo: hello" in tool_results[0].content
        assert "[工具执行失败]" not in tool_results[0].content

        # And: 无 error 事件
        errors = [e for e in events if e.event_type == EventType.ERROR]
        assert len(errors) == 0, f"正常调用不应有 error 事件，实际 {len(errors)}"
