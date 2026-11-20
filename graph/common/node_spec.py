"""声明式节点规格与批量注册工具。

NodeSpec 将「节点名 + 节点函数 + 绑定角色」声明成数据，
register_nodes 批量执行 partial 绑定 + add_node。
消息通道压缩统一由 LCAgentCompactionMiddleware.before_model 中间件在
模型调用前自动触发（见 agent/compaction.py）；节点级压缩已移除，
register_nodes 不再接受 compaction_mw，也不再对节点返回值做任何压缩。

节点函数两种形态（register_nodes 自动识别）:
- 独立函数: ``async def fn(state, agent, injector=None, config=None)``,
  由 partial 绑定 ``agent=agents[spec.role]`` 与 injector。
- 角色实例的 bound method: ``async def ClassName.node(self, state, injector=None,
  config=None)``, self 即 agent, register_nodes 只补绑 injector
  (经 ``inspect.ismethod`` 检测, 不再注入 agent= 避免 TypeError)。
"""
from __future__ import annotations

import inspect
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
        fn: 节点函数。支持两种形态:
            未绑定独立函数 ``async def fn(state, agent, injector=None, config=None)``;
            或角色实例的 bound method ``agent.xxx_node``(self 已绑定,签名无 agent)。
        role: agents 字典中该角色实例的键名
    """

    name: str
    fn: Callable
    role: str


def register_nodes(
    builder: StateGraph,
    agents: dict[str, Any],
    injector: Any,
    specs: list[NodeSpec] | None = None,
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
    for spec in specs or []:
        # bound method(角色实例的节点方法): self 已绑定 agent, 只补 injector;
        # 独立函数: partial 绑定 agent 实例 + injector。两者 partial 均保留
        # coroutine 特征(LangGraph 据此判定节点为异步并 await)。
        if inspect.ismethod(spec.fn) and getattr(spec.fn, "__self__", None) is not None:
            bound = partial(spec.fn, injector=injector)
        else:
            bound = partial(spec.fn, agent=agents[spec.role], injector=injector)
        builder.add_node(spec.name, bound)
