"""
Manager Agent - 负责拆解任务并生成执行计划
"""
from __future__ import annotations

from typing import Any, ClassVar, Optional

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from graph.common import register_agent
from team.base import TeamAgent
from team.interrupt import run_team_turn_with_interrupt
from team.node_factory import create_llm_node


@register_agent("manager", tools=None)
class ManagerAgent(TeamAgent):
    """管理者 Agent。工作流提示词见 AGENT.md 的 `## workflow:*` 小节,LLM 参数见 team_agents.json。"""

    # 本角色产出的 workflow state 通道（供 graph 动态组合 State schema）
    state_fields: ClassVar[dict[str, type]] = {"context_summary": str, "plan": str}

    # ---- 工作流 LLM 节点(create_llm_node 闭包作为类属性) ----
    # NodeSpec 以类属性引用(ManagerAgent.plan_node)取未绑定函数;
    # 切勿经实例引用(描述符协议会把 self 绑到 state 参数位)。
    plan_node = create_llm_node(
        template_name="manager_plan",
        output_field="plan",
        template_vars_fn=lambda s: {"task": s["task"], "context_summary": s.get("context_summary", "")},
        match_text_fn=lambda s: s["task"],
        base_prompts="",
    )

    async def summarize_context_node(
        self,
        state: dict[str, Any],
        injector=None,
        config: Optional[RunnableConfig] = None,  # noqa: UP045 - LangGraph 注解判定仅接受该字符串形态
    ) -> dict[str, Any]:
        """Manager 提炼记忆上下文,生成分发给下游节点的上下文摘要

        raw_context 为空时短路返回空串,跳过 LLM 调用。非空时把
        ``summarize_context`` 模板内容拼到 prompt 前部作为指令(经
        _astream_messages 的 system 消息语义,helper 单 prompt 通道下合并为用户消息),
        调 ``run_team_turn_with_interrupt`` 流式执行。

        config 透传(含 callbacks):使 summarize 的 LLM token 增量可流出到外层事件流。

        节点实现归属说明:本方法由 graph/simple.py 与 graph/rtl_graph.py 的
        ``summarize`` 节点共用(经 register_nodes 的 bound-method 分支注册)。
        """
        raw = state.get("raw_context", "")
        # raw 为空时短路返回空串(不调 helper)
        if not raw:
            return {"context_summary": "", "messages": [AIMessage(content="")]}
        # summarize 节点不注入技能块(不调 injector)
        prompt = f"{self.get_template('summarize_context')}\n\n{raw}"
        result = await run_team_turn_with_interrupt(self, prompt, config)
        return {"context_summary": result, "messages": [AIMessage(content=result)]}
