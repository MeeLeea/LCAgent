"""节点级 compaction 包装回归测试(TDD 锁 graph/common/node_spec.py 新机制)。

覆盖 5 个用例:
1. ``compaction_mw=None`` → 节点返回值不含 RemoveMessage, 其它字段原样透传
2. 消息数 <= max_messages(arun_compaction 返回 None) → 与未包装时逐字相同
3. 消息数 > max_messages 且摘要模型可用 → summary 非空、messages 含 SystemMessage、
   且 messages[-1] 仍是节点自身产出
4. 节点返回的其它字段(out)在压缩后仍保留(守护 ``{**result, **update}``)
5. config 注入回归: 包装后的节点仍收到 LangGraph 注入的 config(workspace_path 可达)
   —— 守护 ``config`` 注解必须是 ``Optional[RunnableConfig]``(写成 ``Any`` 会静默失效)
"""
from __future__ import annotations

import asyncio
from typing import Annotated, Any, Optional, TypedDict

import pytest
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agent.compaction import CompactionConfig, LCAgentCompactionMiddleware
from graph.common import NodeSpec, register_nodes

_SUMMARY_TEXT = "STUB-SUMMARY"


class _Resp:
    """stub LLM 响应: 仅暴露 ``.text``(compaction 读取该属性)。"""

    def __init__(self, text: str) -> None:
        self.text = text


class _StubModel:
    """stub 摘要模型: ``ainvoke`` 为 async, 返回带 ``.text`` 的对象, 不联网。"""

    def __init__(self, text: str = _SUMMARY_TEXT) -> None:
        self._text = text

    async def ainvoke(self, prompt: Any) -> _Resp:
        return _Resp(self._text)


def _make_mw(max_messages: int, keep_recent: int = 1) -> LCAgentCompactionMiddleware:
    """构造带 stub 摘要模型的 compaction 中间件(非 force, 仅超阈值时触发)。"""
    return LCAgentCompactionMiddleware(
        model=_StubModel(),
        config=CompactionConfig(max_messages=max_messages, keep_recent=keep_recent),
    )


class _WfState(TypedDict, total=False):
    """工作流 state: ``messages`` 必须是 add_messages 通道, 否则 RemoveMessage 不被消费。"""

    out: str
    messages: Annotated[list[AnyMessage], add_messages]
    summary: str


async def _node(
    state: dict,
    agent: Any = None,
    injector: Any = None,
    config: Optional[RunnableConfig] = None,  # noqa: UP045
) -> dict:
    """节点: 返回 ``out`` 字段 + 一条自身产出(固定 id 便于逐字比较)。"""
    return {"out": "x", "messages": [AIMessage(content="node-out", id="node-out")]}


def _run_graph(
    mw: Any,
    node_fn: Any,
    state_in: dict,
    config: dict | None = None,
    checkpointer: Any = None,
) -> dict:
    """以 5 位置参数注册节点(用户文件写法)并运行, 返回终态。"""
    builder = StateGraph(_WfState)
    register_nodes(builder, {"r": None}, None, mw, [NodeSpec("n", node_fn, role="r")])
    builder.add_edge(START, "n")
    builder.add_edge("n", END)
    compiled = builder.compile(checkpointer=checkpointer)
    return asyncio.run(compiled.ainvoke(state_in, config=config))


def test_no_compaction_mw_passes_through():
    """用例 1: ``compaction_mw=None`` → 无 RemoveMessage, out/messages 原样透传。"""
    out = _run_graph(None, _node, {"messages": [AIMessage(content="m0", id="m0")]})

    assert out["out"] == "x"
    assert [m.content for m in out["messages"]] == ["m0", "node-out"]
    assert not any(type(m).__name__ == "RemoveMessage" for m in out["messages"])
    assert not any(isinstance(m, SystemMessage) for m in out["messages"])
    assert not out.get("summary")


def test_below_threshold_is_identical_to_unwrapped():
    """用例 2: 消息数 <= max_messages(arun_compaction 返回 None) → 与未包装逐字相同。"""
    state_in = {"messages": [AIMessage(content="m0", id="m0")]}

    wrapped = _run_graph(_make_mw(max_messages=10, keep_recent=5), _node, dict(state_in))
    plain = _run_graph(None, _node, dict(state_in))

    assert wrapped["out"] == plain["out"]
    assert wrapped["messages"] == plain["messages"]
    assert not any(isinstance(m, SystemMessage) for m in wrapped["messages"])


def test_above_threshold_compacts_and_keeps_node_output():
    """用例 3: 超阈值压缩后 summary 非空、含 SystemMessage、末条仍是节点产出。"""
    state_in = {"messages": [AIMessage(content=f"m{i}", id=f"m{i}") for i in range(4)]}

    out = _run_graph(_make_mw(max_messages=2, keep_recent=1), _node, state_in)

    assert out.get("summary"), "压缩后 summary 必须非空"
    assert any(isinstance(m, HumanMessage) for m in out["messages"]), "缺少摘要 HumanMessage"
    assert not any(isinstance(m, SystemMessage) for m in out["messages"]), (
        "摘要不得使用 system 角色（避免 payload 出现多条 system 消息）"
    )
    assert out["messages"][-1].content == "node-out", "节点自身产出被压缩吞掉"


def test_other_node_fields_survive_compaction():
    """用例 4: 压缩后节点返回的其它字段(out)仍保留(守护 ``{**result, **update}``)。"""
    state_in = {"messages": [AIMessage(content=f"m{i}", id=f"m{i}") for i in range(4)]}

    out = _run_graph(_make_mw(max_messages=2, keep_recent=1), _node, state_in)

    assert out.get("summary"), "前置条件: 本用例必须触发压缩"
    assert out["out"] == "x", "节点返回的 out 字段在压缩后丢失(合并语义被改坏)"


def test_wrapped_node_still_receives_injected_config():
    """用例 5: 包装后的节点仍收到 LangGraph 注入的 config(workspace_path 可达)。"""
    received: dict[str, Any] = {}

    async def config_node(
        state: dict,
        agent: Any = None,
        injector: Any = None,
        config: Optional[RunnableConfig] = None,  # noqa: UP045
    ) -> dict:
        received["config"] = config
        return {"out": "done", "messages": [AIMessage(content="done", id="done")]}

    builder = StateGraph(_WfState)
    register_nodes(
        builder,
        {"r": None},
        None,
        compaction_mw=_make_mw(max_messages=50, keep_recent=20),
        specs=[NodeSpec("n", config_node, role="r")],
    )
    builder.add_edge(START, "n")
    builder.add_edge("n", END)
    compiled = builder.compile(checkpointer=MemorySaver())

    asyncio.run(
        compiled.ainvoke(
            {"messages": []},
            {"configurable": {"thread_id": "t1", "workspace_path": "C:/ws"}},
        )
    )

    assert received["config"] is not None, "LangGraph 未注入 config(注解可能不是 Optional[RunnableConfig])"
    assert received["config"]["configurable"]["thread_id"] == "t1"
    assert received["config"]["configurable"]["workspace_path"] == "C:/ws"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
