"""
Manager Agent - 负责拆解任务并生成执行计划
"""
from graph.common import register_agent
from team.base import TeamAgent


@register_agent("manager", tools=None)
class ManagerAgent(TeamAgent):
    """管理者 Agent。工作流提示词见 AGENT.md 的 `## workflow:*` 小节,LLM 参数见 team_agents.json。"""
