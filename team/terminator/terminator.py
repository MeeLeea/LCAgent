"""
Terminator Agent - 负责汇总 Worker 执行结果并返回最终答案
"""
from typing import ClassVar

from graph.common import register_agent
from team.base import TeamAgent
from team.node_factory import create_llm_node


@register_agent("terminator", tools=None)
class TerminatorAgent(TeamAgent):
    """终结者 Agent。工作流提示词见 AGENT.md 的 `## workflow:*` 小节,LLM 参数见 team_agents.json。"""

    # 本角色产出的 workflow state 通道（供 graph 动态组合 State schema）
    state_fields: ClassVar[dict[str, type]] = {"final_answer": str}

    # ---- 工作流 LLM 节点(create_llm_node 闭包作为类属性) ----
    # NodeSpec 以类属性引用(TerminatorAgent.final_node)取未绑定函数;
    # 切勿经实例引用(描述符协议会把 self 绑到 state 参数位)。
    final_node = create_llm_node(
        template_name="terminator_final",
        output_field="final_answer",
        template_vars_fn=lambda s: {"task": s["task"], "plan": s["plan"], "worker_result": s["worker_result"], "context_summary": s.get("context_summary", "")},
        match_text_fn=lambda s: s["task"],
        base_prompts="",
    )
