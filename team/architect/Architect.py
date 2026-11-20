"""
Architect Agent - 芯片架构工程师,负责架构方案设计、权衡分析、评审与规格文档输出
"""
from __future__ import annotations

from typing import Any, ClassVar

from graph.common import register_agent
from team.base import TeamAgent
from team.node_factory import create_llm_node
from tools.human_confirmation import request_user_confirmation


# ---- 上游产物拼接辅助:承载各节点原有的 parts 逻辑,供模板变量与技能匹配文本复用 ----
def _architect_design_task(s: dict[str, Any]) -> str:
    task = s["task"]
    plan = s.get("arch_plan", "")
    return f"{task}\n\n【架构执行计划】\n{plan}" if plan else task


def _architect_analyze_task(s: dict[str, Any]) -> str:
    task = s["task"]
    design = s.get("arch_design", "")
    return f"{task}\n\n【架构方案设计】\n{design}" if design else task


def _architect_review_task(s: dict[str, Any]) -> str:
    task = s["task"]
    design = s.get("arch_design", "")
    analysis = s.get("arch_analysis", "")
    parts = [task]
    if design:
        parts.append(f"【架构方案设计】\n{design}")
    if analysis:
        parts.append(f"【权衡分析】\n{analysis}")
    return "\n\n".join(parts)


def _architect_spec_task(s: dict[str, Any]) -> str:
    task = s["task"]
    design = s.get("arch_design", "")
    review = s.get("arch_review", "")
    parts = [task]
    if design:
        parts.append(f"【架构方案设计】\n{design}")
    if review:
        parts.append(f"【评审意见】\n{review}")
    return "\n\n".join(parts)


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
    """架构师 Agent。工作流提示词见 AGENT.md 的 `## workflow:*` 小节,LLM 参数见 team_agents.json。

    五阶段 LLM 节点以**类属性**形式持有(create_llm_node 闭包):
    graph/rtl_graph.py 的 NodeSpec 以**类属性引用**(``ArchiAgent.plan_node``)
    取未绑定函数,register_nodes 的经典分支补绑 ``agent=``/``injector=`` 后注册。

    警告:切勿经实例引用(``agent.plan_node``)——闭包同为函数对象,
    实例访问会触发描述符协议把 self 绑到 state 参数位,签名完全错位。
    base_prompts="" 使 prompt 与历史手写版逐字一致(不注入 load_agent_rules)。
    """

    # 本角色产出的 workflow state 通道（供 graph 动态组合 State schema）
    state_fields: ClassVar[dict[str, type]] = {
        "arch_plan": str,
        "arch_design": str,
        "arch_analysis": str,
        "arch_review": str,
        "arch_spec": str,
    }

    plan_node = create_llm_node(
        template_name="architect_plan",
        output_field="arch_plan",
        template_vars_fn=lambda s: {"task": s["task"], "context_summary": s.get("context_summary", "")},
        match_text_fn=lambda s: s["task"],
        base_prompts="",
    )

    design_node = create_llm_node(
        template_name="architect_design",
        output_field="arch_design",
        template_vars_fn=lambda s: {"task": _architect_design_task(s), "context_summary": s.get("context_summary", "")},
        match_text_fn=_architect_design_task,
        base_prompts="",
    )

    analyze_node = create_llm_node(
        template_name="architect_analyze",
        output_field="arch_analysis",
        template_vars_fn=lambda s: {"task": _architect_analyze_task(s), "context_summary": s.get("context_summary", "")},
        match_text_fn=_architect_analyze_task,
        base_prompts="",
    )

    review_node = create_llm_node(
        template_name="architect_review",
        output_field="arch_review",
        template_vars_fn=lambda s: {"task": _architect_review_task(s), "context_summary": s.get("context_summary", "")},
        match_text_fn=_architect_review_task,
        base_prompts="",
    )

    spec_node = create_llm_node(
        template_name="architect_spec",
        output_field="arch_spec",
        template_vars_fn=lambda s: {"task": _architect_spec_task(s), "context_summary": s.get("context_summary", "")},
        match_text_fn=_architect_spec_task,
        base_prompts="",
    )
