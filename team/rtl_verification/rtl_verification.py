"""
Verification Agent - 数字芯片 RTL 验证工程师,负责验证需求梳理、验证计划、Testbench/UVM 开发与 Vivado Xsim 仿真
"""
from typing import ClassVar

from graph.common import register_agent
from team.base import TeamAgent
from tools.human_confirmation import request_user_confirmation


@register_agent(
    "rtl_verification",
    tools=[request_user_confirmation],
    mcp_tools=["write_file","edit_file","list_directory","read_file","delete_file","create_directory","delete_directory"],
)
class VerificationAgent(TeamAgent):
    """验证师 Agent。工作流提示词见 AGENT.md 的 `## workflow:*` 小节,LLM 参数见 team_agents.json。"""

    # 验证环境始终使用 Vivado Xsim 仿真,与任务关键词无关,故作为角色级固定技能
    # 无条件注入(经 create_llm_node 读取 agent.fixed_skills 传入 injector)。
    # verification_check_node 例外:该节点经 exclude_skills=("vivado-2025.2",)
    # 显式 opt-out(其 add_files 目录通配与 Xsim sim_filelist.f 流程冲突)。
    fixed_skills: ClassVar[list[str]] = ["vivado-2025.2"]

