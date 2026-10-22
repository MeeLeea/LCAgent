"""SessionManager.adelete_turn（删除会话内某一轮）测试。

覆盖：
- 删除中间轮：仅移除该轮（用户消息 + AI 回复），其余轮次保留且顺序不变
- 删除含工具调用链的轮：AIMessage(tool_calls) 与 ToolMessage 整块移除
- 删除最后一轮 / 第一轮
- 轮序号越界、空会话 → ValueError
- 删除后继续对话：新消息接在剩余历史之后（checkpoint 语义正确）

运行：
  pytest tests/session/test_delete_turn.py -v
"""

import asyncio
from typing import Annotated, Any, TypedDict

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    ToolMessage,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from session import SessionRegistry, SessionStore
from session.manager import SessionManager

# --------------------------------------------------------------------------- #
# 辅助：最小真实图 + FakeAgent
# --------------------------------------------------------------------------- #


class _S(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def _echo(state: _S) -> dict[str, Any]:
    """模拟一轮 AI 回复：回显最后一条用户消息。"""
    return {"messages": [AIMessage(content=f"echo:{state['messages'][-1].content}")]}


def _build_graph(checkpointer):
    builder = StateGraph(_S)
    builder.add_node("echo", _echo)
    builder.add_edge(START, "echo")
    builder.add_edge("echo", END)
    return builder.compile(checkpointer=checkpointer)


class _FakeAgent:
    """提供 SessionManager 依赖的最小 agent 表面（executor + registry）。"""

    def __init__(self, graph, registry: SessionRegistry) -> None:
        self.agent_executor = graph
        self.session = registry

    def _current_sid(self, thread_id: str | None) -> str:
        return thread_id or self.session.current_session_id


def _make_manager() -> tuple[SessionManager, Any, str]:
    checkpointer = MemorySaver()
    graph = _build_graph(checkpointer)
    registry = SessionRegistry(checkpointer, SessionStore(), process_type="server")
    agent = _FakeAgent(graph, registry)
    return SessionManager(agent), graph, "thread-test"


async def _append(graph, thread_id: str, msgs: list[AnyMessage]) -> None:
    """以 echo 节点视角向 checkpoint 追加消息（走 add_messages 通道，id 自动生成）。"""
    await graph.aupdate_state(
        {"configurable": {"thread_id": thread_id}},
        {"messages": msgs},
        as_node="echo",
    )


async def _run_turn(graph, thread_id: str, text: str) -> None:
    """模拟一轮对话：用户消息进图，节点追加 AI 回复。"""
    await graph.ainvoke(
        {"messages": [HumanMessage(content=text)]},
        {"configurable": {"thread_id": thread_id}},
    )


def _texts(messages: list[AnyMessage]) -> list[str]:
    return [str(m.content) for m in messages]


# --------------------------------------------------------------------------- #
# 测试
# --------------------------------------------------------------------------- #


def test_delete_middle_turn():
    """删除中间轮：仅移除该轮，其余轮次保留且顺序不变。"""
    async def run():
        manager, graph, tid = _make_manager()
        for q in ("q1", "q2", "q3"):
            await _run_turn(graph, tid, q)

        removed = await manager.adelete_turn(tid, 1)
        assert removed == 2  # HumanMessage(q2) + AIMessage(echo:q2)

        remaining = _texts(await manager.aget_messages(tid))
        assert remaining == ["q1", "echo:q1", "q3", "echo:q3"]

    asyncio.run(run())


def test_delete_turn_with_tool_chain():
    """删除含工具调用链的轮：AI(tool_calls) + ToolMessage + 最终 AI 整块移除。"""
    async def run():
        manager, graph, tid = _make_manager()
        await _run_turn(graph, tid, "q1")
        # 第二轮带工具链：用户消息 → AI 发起调用 → 工具结果 → 最终回复
        await _append(graph, tid, [HumanMessage(content="q2")])
        await _append(
            graph,
            tid,
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"id": "call_1", "name": "calculator", "args": {"expression": "1+1"}}
                    ],
                ),
                ToolMessage(content="2", tool_call_id="call_1", name="calculator"),
                AIMessage(content="final:2"),
            ],
        )
        await _run_turn(graph, tid, "q3")

        # 删除第 1 轮（q2 锚定）：q2 + AI(tool_calls) + ToolMessage + final 全部移除
        removed = await manager.adelete_turn(tid, 1)
        assert removed == 4
        assert _texts(await manager.aget_messages(tid)) == [
            "q1", "echo:q1", "q3", "echo:q3",
        ]

    asyncio.run(run())


def test_delete_last_turn():
    """删除最后一轮。"""
    async def run():
        manager, graph, tid = _make_manager()
        for q in ("q1", "q2"):
            await _run_turn(graph, tid, q)

        removed = await manager.adelete_turn(tid, 1)
        assert removed == 2
        assert _texts(await manager.aget_messages(tid)) == ["q1", "echo:q1"]

    asyncio.run(run())


def test_delete_turn_out_of_range():
    """轮序号越界 → ValueError。"""
    async def run():
        manager, graph, tid = _make_manager()
        await _run_turn(graph, tid, "q1")

        for bad in (1, -1, 99):
            try:
                await manager.adelete_turn(tid, bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"turn_index={bad} 应抛 ValueError")

    asyncio.run(run())


def test_delete_turn_empty_session():
    """空会话（无 checkpoint 消息）→ ValueError。"""
    async def run():
        manager, _, tid = _make_manager()
        try:
            await manager.adelete_turn(tid, 0)
        except ValueError:
            pass
        else:
            raise AssertionError("空会话应抛 ValueError")

    asyncio.run(run())


def test_continue_after_delete():
    """删除后继续对话：新消息接在剩余历史之后，无残留消息。"""
    async def run():
        manager, graph, tid = _make_manager()
        for q in ("q1", "q2"):
            await _run_turn(graph, tid, q)
        await manager.adelete_turn(tid, 0)

        await _run_turn(graph, tid, "q3")
        remaining = _texts(await manager.aget_messages(tid))
        assert remaining == ["q2", "echo:q2", "q3", "echo:q3"]

    asyncio.run(run())
