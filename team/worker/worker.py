"""
Worker Agent - 负责执行具体子任务
"""
from typing import ClassVar

from graph.common import register_agent
from team.base import TeamAgent
from tools import all_tools


@register_agent(
    "worker",
    tools=all_tools,
    mcp_all=True,
)
class WorkerAgent(TeamAgent):
    """执行者 Agent。工作流提示词见 AGENT.md 的 `## workflow:*` 小节,LLM 参数见 team_agents.json。"""

    # 本角色产出的 workflow state 通道（供 graph 动态组合 State schema）
    state_fields: ClassVar[dict[str, type]] = {"worker_result": str}
