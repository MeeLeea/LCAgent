"""
Terminator Agent - 负责汇总 Worker 执行结果并返回最终答案
"""
from graph.common import register_agent
from team.base import TeamAgent


@register_agent("terminator", tools=None)
class TerminatorAgent(TeamAgent):
    """终结者 Agent。工作流提示词见 AGENT.md 的 `## workflow:*` 小节,LLM 参数见 team_agents.json。"""
