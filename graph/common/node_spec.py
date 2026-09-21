"""声明式节点规格与批量注册工具。

NodeSpec 将「节点名 + 节点函数 + 绑定角色」声明成数据，
register_nodes 批量执行 partial 绑定 + add_node。
可选 compaction_mw 开启节点级压缩：节点返回后调用 arun_compaction，
仅在消息数 > max_messages（默认 50）时触发（非 force），
把 messages/summary 合并进节点返回值。
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
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


def _wrap_node_with_compaction(fn: Callable, mw: Any) -> Callable:
    """把节点函数包一层节点级 compaction。

    节点执行完成后, 用「state 中的历史消息 + 节点本次产出的消息」调用
    ``mw.arun_compaction(...)``；仅在消息数超过阈值时返回非 None 更新。

    Args:
        fn: 已绑定 agent/injector 的节点函数(partial)
        mw: 提供 ``arun_compaction(messages, existing_summary=...)`` 的中间件

    Returns:
        包装后的异步节点函数
    """

    async def wrapped(
        state: dict[str, Any],
        config: Optional[RunnableConfig] = None,  # noqa: UP045 - LangGraph 注解判定仅接受该字符串形态
    ) -> dict[str, Any]:
        # config 注解必须是 Optional[RunnableConfig](与 node_factory.py 一致):
        # LangGraph 按注解白名单决定是否注入 config, 写成 Any 会静默不注入,
        # 导致 workspace_path 隔离 / TOKEN 流式 / output_files 解析失效。
        result = await fn(state, config=config)
        msgs = result.get("messages")
        if not msgs:
            return result
        accumulated = list(state.get("messages", []) or []) + list(msgs)
        update = await mw.arun_compaction(
            accumulated, existing_summary=state.get("summary", "") or ""
        )
        if update is None:
            return result
        # 刻意用 {**result, **update}: update["messages"](RemoveAll + 摘要 + kept,
        # kept 已含节点新产出)取代 result["messages"], 不丢不重。禁止改成相加。
        return {**result, **update}

    return wrapped


def register_nodes(
    builder: StateGraph,
    agents: dict[str, Any],
    injector: Any,
    compaction_mw: Any | None = None,
    specs: list[NodeSpec] | None = None,
) -> None:
    """批量注册节点。

    将 partial 绑定 agent/injector → add_node 两步合一，
    避免每个节点重复写模板代码。

    Args:
        builder: 待注册节点的 LangGraph StateGraph 构造器
        agents: 角色实例字典
        injector: 技能注入器(SkillInjector)
        compaction_mw: 可选的节点级压缩中间件; 非 None 时节点返回后调用
            ``arun_compaction`` 并合并 messages/summary
        specs: 节点规格列表
    """
    for spec in specs or []:
        bound = partial(spec.fn, agent=agents[spec.role], injector=injector)
        if compaction_mw is None:
            builder.add_node(spec.name, bound)
        else:
            builder.add_node(spec.name, _wrap_node_with_compaction(bound, compaction_mw))
