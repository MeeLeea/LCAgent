"""
Verification Agent - 数字芯片 RTL 验证工程师,负责验证需求梳理、验证计划、Testbench/UVM 开发与 Vivado Xsim 仿真
"""
from graph.common import register_agent
from team.base import TeamAgent
from tools.human_confirmation import request_user_confirmation


@register_agent(
    "rtl_verification",
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
class VerificationAgent(TeamAgent):
    """验证师 Agent。工作流提示词见 AGENT.md 的 `## workflow:*` 小节,LLM 参数见 team_agents.json。"""
