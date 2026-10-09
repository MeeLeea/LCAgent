"""
Worker Agent - 负责执行具体子任务
"""
from typing import ClassVar

from graph.common import register_agent
from team.base import TeamAgent
from team.node_factory import create_llm_node
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

    # ---- 工作流 LLM 节点(create_llm_node 闭包作为类属性) ----
    # NodeSpec 以类属性引用(WorkerAgent.exec_node)取未绑定函数;
    # 切勿经实例引用(描述符协议会把 self 绑到 state 参数位)。
    exec_node = create_llm_node(
        template_name="worker_exec",
        output_field="worker_result",
        template_vars_fn=lambda s: {"plan": s["plan"]},
        match_text_fn=lambda s: s["plan"],
        base_prompts="",
    )
