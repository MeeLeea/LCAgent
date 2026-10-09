"""
Architect Agent - 芯片架构工程师,负责架构方案设计、权衡分析、评审与规格文档输出
"""
from typing import ClassVar

from graph.common import register_agent
from team.base import TeamAgent
from tools.human_confirmation import request_user_confirmation


@register_agent(
    "architect",
    tools=[request_user_confirmation],
    mcp_tools=[
        "write_file",
        "edit_file",
        "list_directory",
        "read_file",
        "delete_file",
        "create_directory",
        "delete_directory"
    ],
)
class ArchiAgent(TeamAgent):
    """架构师 Agent。工作流提示词见 AGENT.md 的 `## workflow:*` 小节,LLM 参数见 team_agents.json。"""

    # 本角色产出的 workflow state 通道（供 graph 动态组合 State schema）
    state_fields: ClassVar[dict[str, type]] = {
        "arch_plan": str,
        "arch_design": str,
        "arch_analysis": str,
        "arch_review": str,
        "arch_spec": str,
    }
    