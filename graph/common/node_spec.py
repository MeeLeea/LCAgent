"""声明式节点规格与批量注册工具。

NodeSpec 将「节点名 + 节点函数 + 绑定角色」声明成数据，
register_nodes 批量执行 partial 绑定 + add_node。
节点级 compaction 包装已移除，压缩统一由 before_model 中间件处理。
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

from langgraph.graph import StateGraph


@dataclass
class NodeSpec:
    """声明式节点规格:把「节点名 + 节点函数 + 绑定哪个角色」声明成数据。

    Attributes:
        name: 节点名(LangGraph builder.add_node 的第一参数)
        fn: 节点函数(未绑定,统一签名 ``async def fn(state, agent, injector=None, config=None)``)
        role: agents 字典中该角色实例的键名
    """

    name: str
    fn: Callable
    role: str


def register_nodes(
    builder: StateGraph,
    agents: dict[str, Any],
    injector: Any,
    specs: list[NodeSpec],
) -> None:
    """批量注册节点。

    将 partial 绑定 agent/injector → add_node 两步合一，
    避免每个节点重复写模板代码。

    Args:
        builder: 待注册节点的 LangGraph StateGraph 构造器
        agents: 角色实例字典
        injector: 技能注入器(SkillInjector)
        specs: 节点规格列表
    """
    for spec in specs:
        bound = partial(spec.fn, agent=agents[spec.role], injector=injector)
        builder.add_node(spec.name, bound)
