"""工作流构图公共 setup — 聚合各 build_*_workflow 的雷同构图片段。

simple / rtl_graph 两个 build_*_workflow 在设置注入器与 State schema 两个步骤
上逐字雷同:先按 ``skills_dir`` / ``auto_match_skills`` 构造 SkillInjector,再以
``compose_state_schema(agents[, extra_channels])`` 动态组合出 State schema 并
包进 ``StateGraph``。本模块抽出这段样板,两图只保留节点表、边/条件路由、
``extra_channels`` 等各自差异。

本函数供 graph/ 两个内置工作流构建器复用,属内部构图辅助,不面向
「扩展工作流」的外部作者(它们在 build_xxx 内直接构造注入器、更自由),
故不加入 graph.common 的 ``__all__`` 导出面。
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langgraph.graph import StateGraph

from graph.common.schema import compose_state_schema
from skmng.injector import SkillInjector


def build_graph_scaffold(
    agents: Mapping[str, Any],
    *,
    skills_dir: str | None,
    auto_match_skills: bool,
    extra_channels: Mapping[str, Any] | None = None,
) -> tuple[StateGraph, SkillInjector]:
    """构造构图所需的基础(builder + 技能注入器)。

    将「构建技能注入器 + 依据 agents 的 state_fields 动态组合 State schema 并
    包进 StateGraph」这段两图共用的样板收敛为一步:

    Args:
        agents: 角色名 -> Agent 实例 的字典。仅读取其 ``state_fields`` 类属性
            用于拼装 schema,不修改其运行期状态。
        skills_dir: 技能目录路径,为 None 时使用默认目录(.agents/skills)。
        auto_match_skills: 是否在节点渲染 prompt 时按任务自动匹配注入技能。
        extra_channels: 工作流级额外编排通道(如 RTL 的 round/max_rounds 等),
            传 None 表示不扩展(与 simple 工作流的 ``compose_state_schema(agents)``
            等价);通道名冲突时由 ``compose_state_schema`` 抛 ``ValueError``。

    Returns:
        ``(builder, injector)`` 二元组:builder 供调用方 register_nodes 与
        添加边/条件路由后编译;injector 需透传给 register_nodes。
    """
    injector = SkillInjector(
        skills_dir=skills_dir,
        auto_match=auto_match_skills,
    )
    builder = StateGraph(compose_state_schema(agents, extra_channels=extra_channels))
    return builder, injector