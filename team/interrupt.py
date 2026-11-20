"""中断转发助手 — 将内层 TeamAgent 的 interrupt 透传给外层 LangGraph。

run_team_turn_with_interrupt 在工作流节点中调用 TeamAgent.arun_structured,
若内层被 interrupt 则调外层 ``langgraph.types.interrupt()`` 暂停外层 graph,
外层 resume 后将返回值经 ``aresume_structured`` 注入内层恢复执行。

**节点重入语义(重要)**:LangGraph 恢复中断时会**从节点函数首行重新执行**该节点,
``interrupt()`` 按调用**顺序位置**匹配 resume 值(第 N 次调用消费第 N 个答案)。
因此本函数在重入时必须:

1. 依据 ``scratchpad.resume``(该 task 已消费答案的累积列表)先重放等量的
   ``interrupt(None)``,把位置计数器推进到中断点;否则第二次中断会拿到**过期答案**。
2. **跳过** ``arun_structured`` —— 它会向内层 agent_executor 追加一条新的
   ``HumanMessage``,而内层挂起的工具调用需要的是 ``Command(resume=...)``。
   走错入口会让内层 ``AIMessage.tool_calls`` 永远拿不到 ``ToolMessage``,
   LLM 请求被 OpenAI 兼容网关 400 拒绝(``must be followed by tool messages``)。
3. 仅当内层**确实存在** pending interrupts 时,才把 turn 初始化为 interrupted 
   交给下面的 while 循环;否则(非常规状态)消费掉本次 resume 值后按新任务执行。

本模块原位于 ``graph/common/interrupt_forward.py``,随节点实现下沉迁至 team/
(其 runtime 依赖本来就是 team.base,语义上属于 TeamAgent 执行桥)。
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from langgraph.config import get_config
from langgraph.types import Interrupt, interrupt

from agent.turn_types import AgentTurnResult
from team.base import TASK_ERROR_PREFIX

try:  # pragma: no cover - 私有常量,langgraph 版本升级缺失时退回字面量
    from langgraph._internal._constants import CONFIG_KEY_SCRATCHPAD
except ImportError:  # pragma: no cover
    CONFIG_KEY_SCRATCHPAD = "__pregel_scratchpad"

if TYPE_CHECKING:
    from team.base import TeamAgent

logger = logging.getLogger(__name__)

# 重放已消费轮次时使用的占位值:replay 分支会直接返回历史答案,该值不会被使用
_REPLAY_VALUE: None = None

# 全新任务开跑时清场内层陈旧挂起中断的取消载荷(兼容 human_choice / user_confirmation)
_STALE_CANCEL_PAYLOAD: dict[str, Any] = {"cancelled": True, "answers": {}}

# 陈旧中断清场最多尝试次数(每次清场后重查 pending,防止内层再次中断导致死循环)
_STALE_FLUSH_ATTEMPTS = 2


def _current_scratchpad() -> Any:
    """读取当前 task 的 PregelScratchpad

    仅在 LangGraph 运行时(节点/工具函数内)可用;独立调用(如单元测试直接
    调本模块)时 ``get_config()`` 抛 ``RuntimeError``,此时返回 None 表示
    「无中断上下文」,调用方据此回退到无位置重放的普通执行路径。
    """
    try:
        config = get_config()
    except RuntimeError:
        return None
    configurable = config.get("configurable") or {}
    return configurable.get(CONFIG_KEY_SCRATCHPAD)


def _resolved_rounds() -> int:
    """本次节点执行需要重放的已消费中断轮数

    ``scratchpad.resume`` 是该 task 已消费答案的累积列表,长度即上次中断前
    已完成的轮数;``interrupt()`` 的 re-play 分支要求第 N 次调用的位置计数器
    与列表索引对齐,故必须先用等量的占位 ``interrupt(None)`` 推进计数器。
    """
    scratchpad = _current_scratchpad()
    resumed = getattr(scratchpad, "resume", None) if scratchpad is not None else None
    if isinstance(resumed, (list, tuple)):
        return len(resumed)
    return 0


def _is_resume_run() -> bool:
    """本次执行是否由 ``Command(resume=...)`` 触发

    ``get_null_resume(consume=False)`` 只**窥视**本次运行待消费的 resume 值,
    不做消费;非 None 即说明外层图正从 interrupt 恢复(而非全新任务开跑)。
    """
    scratchpad = _current_scratchpad()
    peek = getattr(scratchpad, "get_null_resume", None) if scratchpad is not None else None
    if peek is None:
        return False
    return peek(False) is not None


async def _apending_interrupts(
    agent: TeamAgent,
    config: dict | None,
) -> list[Interrupt]:
    """读取内层 agent_executor 的 pending interrupts(容错:不支持则视为空)

    部分调用方(单元测试的 Fake Agent)只实现 ``arun_structured`` /
    ``aresume_structured``,此时视为「无挂起中断」,回退到原有执行路径。
    """
    getter = getattr(agent, "aget_pending_interrupts", None)
    if getter is None:
        return []
    return await getter(config)


async def _astart_fresh_turn(
    agent: TeamAgent,
    prompt: str,
    config: dict | None,
) -> AgentTurnResult:
    """全新任务开跑:先清场内层陈旧挂起中断,再执行本轮任务

    上一轮外层任务被用户放弃(未 resume)时,内层 checkpoint 会残留 pending
    interrupt。此时直接 ``arun_structured`` 只会让 LangGraph 重新抛出**旧中断**
    (挂起 task 被重跑,新任务文本被追加却得不到执行),用户会看到早已过期的
    确认卡片。故先用取消载荷把陈旧中断消费掉,再执行新任务。
    """
    pending = await _apending_interrupts(agent, config)
    flushed = 0
    while pending and flushed < _STALE_FLUSH_ATTEMPTS:
        await agent.aresume_structured(dict(_STALE_CANCEL_PAYLOAD), config)
        flushed += 1
        pending = await _apending_interrupts(agent, config)
    if pending:
        logger.warning(
            "[%s] 内层仍有 %d 个陈旧 pending interrupt,无法清场,继续按新任务执行",
            getattr(agent, "name", "?"),
            len(pending),
        )
    return await agent.arun_structured(prompt, config)


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

    节点被 LangGraph 重入时(即 resume 场景)不会重跑 ``arun_structured``,
    而是把内层已有的 pending interrupts 直接交给 while 循环处理(详见模块
    docstring 中的重入语义)。

    Args:
        agent: 执行本节点任务的 TeamAgent(工具/纯文本模式皆可)
        prompt: 节点任务文本
        config: 外层 RunnableConfig,透传给 arun_structured/aresume_structured

    Returns:
        TeamAgent 正常完成/取消时的 output 字符串;cancelled 且 output 为空时
        返回 ``"{TASK_ERROR_PREFIX}: 任务被取消"``

    Raises:
        RuntimeError: AgentTurnResult 出现 completed/interrupted/cancelled 之外的状态
    """
    # 1) 位置对齐:重放此前已消费的中断轮次,使 interrupt_counter 与历史答案索引一致
    resolved_rounds = _resolved_rounds()
    for _ in range(resolved_rounds):
        interrupt(_REPLAY_VALUE)

    # 2) 决定本轮起点:resume 场景直接复用内层 pending,避免 arun_structured 重放
    if _is_resume_run():
        pending = await _apending_interrupts(agent, config)
        if pending:
            turn = AgentTurnResult.interrupted(pending)
        else:
            # 非常规:外层恢复但内层无挂起中断(内层已自行完成)。此处必须消费掉
            # 本次 resume 值,否则它会以裸值写回 RESUME 通道污染后续重放位置。
            interrupt(_REPLAY_VALUE)
            turn = await agent.arun_structured(prompt, config)
    else:
        turn = await _astart_fresh_turn(agent, prompt, config)

    while turn.is_interrupted:
        resume_value = interrupt(turn.interrupts[0].value)
        turn = await agent.aresume_structured(resume_value, config)
    if turn.is_completed:
        return turn.output or ""
    if turn.status == "cancelled":
        return turn.output or f"{TASK_ERROR_PREFIX}: 任务被取消"
    raise RuntimeError(f"AgentTurnResult 未知状态: {turn.status}")
