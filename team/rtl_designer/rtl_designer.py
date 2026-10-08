"""
Designer Agent - 数字芯片 RTL 设计工程师,负责规格梳理、模块拆分、Filelist 生成与可综合 RTL 编码
"""
from graph.common import register_agent
from team.base import TeamAgent
from tools.human_confirmation import request_user_confirmation


@register_agent(
    "rtl_designer",
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
class DesignerAgent(TeamAgent):
    """设计师 Agent。工作流提示词见 AGENT.md 的 `## workflow:*` 小节,LLM 参数见 team_agents.json。"""
