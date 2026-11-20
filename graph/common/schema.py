"""graph.common.schema — 工作流 State schema 的动态组合器。

按各 Agent 声明的 ``state_fields`` 类属性与底座通道动态拼装工作流 State
的 ``TypedDict``，避免每新增一个角色都需手写 State schema。

设计约定（见 .omo/plans/workflow-state-composition.md）:
- 底座通道固定包含 ``task`` / ``raw_context`` / ``messages`` / ``active_skills``，
  **不含** ``summary`` 与 ``context_summary``（前者 T3 移除，后者归 ManagerAgent）。
- 各角色经 ``state_fields: ClassVar[dict[str, type]]`` 声明自己产出的通道，
  composer 遍历 ``sorted(agents)`` 按确定性顺序合并。
- 任一通道名冲突（底座 vs 角色 / 角色 vs 角色 / 与 extra_channels）→ 抛
  ``ValueError``，避免 schema 静默覆盖。
"""
from __future__ import annotations

import logging
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

logger = logging.getLogger(__name__)

# 底座通道：所有工作流 State 共同拥有的基础字段。
# 注意：刻意不含 ``summary``（工作流级概述已移除，见计划 T3）与
# ``context_summary``（该通道归 ManagerAgent 通过 ``state_fields`` 声明，见 T2）。
_BASE_CHANNELS: dict[str, Any] = {
    "task": str,
    "raw_context": str,
    "messages": Annotated[list[AnyMessage], add_messages],
    "active_skills": list[str],
}


def compose_state_schema(
    agents: dict[str, Any],
    extra_channels: dict[str, Any] | None = None,
) -> type[TypedDict]:
    """依据各角色的 ``state_fields`` 声明动态拼装工作流 State schema。

    流程:
    1. 以 ``_BASE_CHANNELS`` 的拷贝为起点；
    2. 遍历 ``sorted(agents)``（确定性顺序），读取 ``type(agents[role]).state_fields``
       （ClassVar，缺省 ``{}``）并合并；
    3. 合并 ``extra_channels``（工作流级编排通道，如 RTL 的控制流/校验通道）；
    4. 任一通道名冲突即抛 ``ValueError``；
    5. 返回 ``TypedDict("WorkflowState", channels, total=False)``。

    Args:
        agents: 角色名 -> Agent 实例 的字典。仅读取每个角色类的
            ``state_fields`` 类属性，不实例化也不读取运行期状态。
        extra_channels: 额外的工作流级通道（如 RTL 的 round / sim_status 等），
            冲突时报错且错误信息不含角色名。

    Returns:
        动态构造的工作流 State TypedDict。

    Raises:
        ValueError: 任一通道名与既有通道冲突时。
    """
    channels: dict[str, Any] = dict(_BASE_CHANNELS)

    # 逐个合并各角色声明的通道，冲突即抛（信息含角色名，便于定位）。
    for role in sorted(agents):
        agent = agents[role]
        cls = type(agent)
        state_fields: dict[str, type] = getattr(cls, "state_fields", {})
        for name, field_type in state_fields.items():
            if name in channels:
                raise ValueError(f"state 通道冲突: {name}（角色 {role}）")
            channels[name] = field_type

    # 合并额外通道（工作流级编排字段），冲突报错（此时无角色归属，故不含角色名）。
    if extra_channels:
        for name, field_type in extra_channels.items():
            if name in channels:
                raise ValueError(f"state 通道冲突: {name}（extra_channels）")
            channels[name] = field_type

    logger.debug("最终工作流 State 通道集合: %s", sorted(channels))
    return TypedDict("WorkflowState", channels, total=False)
