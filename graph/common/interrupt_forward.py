"""中断转发助手 — 将内层 TeamAgent 的 interrupt 透传给外层 LangGraph。

run_team_turn_with_interrupt 在工作流节点中调用 TeamAgent.arun_structured，
若内层被 interrupt 则调外层 ``langgraph.types.interrupt()`` 暂停外层 graph，
外层 resume 后将返回值经 ``aresume_structured`` 注入内层恢复执行。
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langgraph.types import interrupt

if TYPE_CHECKING:
    from team.base import TeamAgent

logger = logging.getLogger(__name__)


async def run_team_turn_with_interrupt(
    agent: TeamAgent,
    prompt: str,
    config: dict | None = None,
) -> str:
    """节点级助手:调 TeamAgent.arun_structured,若 interrupted 调外层
    interrupt() 暂停外层 graph,resume 后调 aresume_structured 恢复内层,
    循环处理多次 interrupt,最终返回 output str。

    用于工作流节点把内层 TeamAgent 的 interrupt 透传给外层 graph 的
    checkpointer:内层被 interrupt 时本函数主动调 ``langgraph.types.interrupt``
    暂停外层 graph,外层 resume 时把用户返回值经 ``aresume_structured``
    注入内层 agent_executor,实现"节点内嵌 interrupt"语义。

    Args:
        agent: 执行本节点任务的 TeamAgent(工具/纯文本模式皆可)
        prompt: 节点任务文本
        config: 外层 RunnableConfig,透传给 arun_structured/aresume_structured

    Returns:
        TeamAgent 正常完成/取消时的 output 字符串;cancelled 且 output 为空时
        返回 ``"{TASK_ERROR_PREFIX}: 任务被取消"``

    Raises:
        AgentTurnResult 既非 completed/interrupted/cancelled 之外的异常状态
    """
    from team.base import TASK_ERROR_PREFIX

    turn = await agent.arun_structured(prompt, config)
    while turn.is_interrupted:
        resume_value = interrupt(turn.interrupts[0].value)
        turn = await agent.aresume_structured(resume_value, config)
    if turn.is_completed:
        return turn.output or ""
    if turn.status == "cancelled":
        return turn.output or f"{TASK_ERROR_PREFIX}: 任务被取消"
    raise RuntimeError(f"AgentTurnResult 未知状态: {turn.status}")
