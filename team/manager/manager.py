"""
Manager Agent - 负责拆解任务并生成执行计划
"""
from typing import ClassVar

from graph.common import register_agent
from team.base import TeamAgent


@register_agent("manager", tools=None)
class ManagerAgent(TeamAgent):
    """管理者 Agent。工作流提示词见 AGENT.md 的 `## workflow:*` 小节,LLM 参数见 team_agents.json。"""

    # 本角色产出的 workflow state 通道（供 graph 动态组合 State schema）
    state_fields: ClassVar[dict[str, type]] = {"context_summary": str, "plan": str}
