"""
Architect Agent - 芯片架构工程师,负责架构方案设计、权衡分析、评审与规格文档输出
"""
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
    