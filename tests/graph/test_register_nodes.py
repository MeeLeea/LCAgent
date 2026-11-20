"""
测试 graph/common.node_spec.register_nodes 与 NodeSpec 声明式节点注册

覆盖:
- 批量注册多个 NodeSpec, builder.nodes 出现全部节点名
- 注册的节点函数原样调用
- 按 role 从 agents 字典取实例并经 agent= kwarg 注入节点
- injector 透传到节点函数
- bound method 形态(角色实例的节点方法): self 已绑定 agent, 只补绑 injector
- 独立函数与 bound method 混合注册
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from graph.common import NodeSpec, register_nodes


async def _record_node(state: dict, agent, injector=None, config=None) -> dict:
    """模拟节点:记录被调用时收到的 agent/injector,追加一条 AIMessage。"""
    seq = state.get("__seq", 0)
    agent.calls.append(seq)
    return {"__seq": seq, "output": getattr(agent, "name", "?"), "messages": [AIMessage(content="x")]}


def test_register_nodes_batch_registers_all():
    """批量注册多个 NodeSpec, builder.nodes 出现全部节点名。"""
    agent_a = SimpleNamespace(name="A", calls=[])
    agent_b = SimpleNamespace(name="B", calls=[])
    agents = {"a": agent_a, "b": agent_b}
    builder = StateGraph(dict)

    register_nodes(
        builder,
        agents,
        injector=None,
        specs=[
            NodeSpec("n1", _record_node, role="a"),
            NodeSpec("n2", _record_node, role="b"),
            NodeSpec("n3", _record_node, role="a"),
        ],
    )
    builder.add_edge(START, "n1")
    builder.add_edge("n1", END)
    compiled = builder.compile()

    node_names = set(compiled.get_graph().nodes.keys())
    assert {"n1", "n2", "n3"}.issubset(node_names)


def test_register_nodes_calls_without_compaction():
    """注册的节点函数原样调用,无 compaction 包装。"""
    agent = SimpleNamespace(name="A", calls=[])
    agents = {"a": agent}
    builder = StateGraph(dict)

    register_nodes(
        builder,
        agents,
        injector=None,
        specs=[NodeSpec("n1", _record_node, role="a")],
    )
    builder.add_edge(START, "n1")
    builder.add_edge("n1", END)
    compiled = builder.compile()

    result = asyncio.run(compiled.ainvoke({"__seq": 7}, config={"configurable": {"thread_id": "t1"}}))
    assert agent.calls == [7]
    assert result["output"] == "A"


def test_register_nodes_role_indexes_agents_dict():
    """多个 role 共用同一节点函数时, 各自绑定到对应 agent 实例。"""
    agent_a = SimpleNamespace(name="A", calls=[])
    agent_b = SimpleNamespace(name="B", calls=[])
    agents = {"role_a": agent_a, "role_b": agent_b}
    builder = StateGraph(dict)

    register_nodes(
        builder,
        agents,
        injector=None,
        specs=[
            NodeSpec("n1", _record_node, role="role_a"),
            NodeSpec("n2", _record_node, role="role_b"),
        ],
    )
    builder.add_edge(START, "n1")
    builder.add_edge("n1", "n2")
    builder.add_edge("n2", END)
    compiled = builder.compile()

    asyncio.run(compiled.ainvoke({"__seq": 5}, config={"configurable": {"thread_id": "t3"}}))
    assert agent_a.calls == [5]
    assert agent_b.calls == [5]


def test_register_nodes_injector_passed_through():
    """injector 经 kwarg 注入节点函数(非 None 时)。"""
    received: dict = {}

    async def _capturing_node(state, agent, injector=None, config=None) -> dict:
        received["agent"] = agent
        received["injector"] = injector
        return {"output": "ok", "messages": [AIMessage(content="x")]}

    agent = SimpleNamespace(name="A")
    agents = {"a": agent}
    injector_obj = SimpleNamespace(tag="INJ")
    builder = StateGraph(dict)

    register_nodes(
        builder,
        agents,
        injector=injector_obj,
        specs=[NodeSpec("n1", _capturing_node, role="a")],
    )
    builder.add_edge(START, "n1")
    builder.add_edge("n1", END)
    compiled = builder.compile()

    asyncio.run(compiled.ainvoke({}, config={"configurable": {"thread_id": "t4"}}))
    assert received["agent"] is agent
    assert received["injector"] is injector_obj


# ──────────────────────────────────────────────
# bound method 形态: 节点实现为角色实例的 async 方法
# ──────────────────────────────────────────────


class _AgentWithNodeMethod:
    """携带 bound-method 节点的模拟角色: self 即 agent, 签名无 agent 参数。"""

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[int] = []
        self.received_injector: object = None

    async def record_node(self, state: dict, injector=None, config=None) -> dict:
        seq = state.get("__seq", 0)
        self.calls.append(seq)
        self.received_injector = injector
        return {"__seq": seq, "output": self.name, "messages": [AIMessage(content="x")]}


def test_register_nodes_bound_method_binds_self_as_agent():
    """bound method 注册: self 已绑定 agent 实例, 不再注入 agent= kwarg。

    回归保护: 若 register_nodes 对 bound method 仍注入 agent=..., 会因
    方法签名无该参数而 TypeError; 且 LangGraph 必须将其识别为异步节点
    并 await(否则 InvalidUpdateError)。
    """
    agent = _AgentWithNodeMethod("M")
    agents = {"m": agent}
    injector_obj = SimpleNamespace(tag="INJ")
    builder = StateGraph(dict)

    register_nodes(
        builder,
        agents,
        injector=injector_obj,
        specs=[NodeSpec("n1", agent.record_node, role="m")],
    )
    builder.add_edge(START, "n1")
    builder.add_edge("n1", END)
    compiled = builder.compile()

    result = asyncio.run(compiled.ainvoke({"__seq": 3}, config={"configurable": {"thread_id": "t5"}}))
    assert agent.calls == [3]
    assert result["output"] == "M"
    assert agent.received_injector is injector_obj


def test_register_nodes_bound_method_and_plain_function_coexist():
    """同一批 specs 中 bound method 与独立函数混合注册, 各自绑定正确。"""
    agent = _AgentWithNodeMethod("M")
    plain_agent = SimpleNamespace(name="P", calls=[])
    agents = {"m": agent, "p": plain_agent}
    builder = StateGraph(dict)

    register_nodes(
        builder,
        agents,
        injector=None,
        specs=[
            NodeSpec("n1", agent.record_node, role="m"),
            NodeSpec("n2", _record_node, role="p"),
        ],
    )
    builder.add_edge(START, "n1")
    builder.add_edge("n1", "n2")
    builder.add_edge("n2", END)
    compiled = builder.compile()

    result = asyncio.run(compiled.ainvoke({"__seq": 9}, config={"configurable": {"thread_id": "t6"}}))
    assert agent.calls == [9]
    assert plain_agent.calls == [9]
    assert result["output"] == "P"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
