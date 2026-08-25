"""中断处理 Mixin - AgentCore 的 interrupt 检查、恢复命令构造与拒绝处理。

从 agent_core.py 抽离，职责：
- 流结束后检查挂起 interrupt（含并行 dangerous_command 合并）
- 构造恢复命令 Command(resume=...)（多中断批量应用同一答案）
- 挂起中断模式记录 / 清除（per-session 隔离）
- 用户拒绝危险命令时的 checkpoint 修复与取消结果返回
- turn 完成后的统一中断状态管理

依赖 AgentCore 实例属性：agent_executor / _session_registry /
_session_store / max_iterations / verbose。
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from llm.message_utils import build_interrupt_event
from utils.events import AgentEvent

from .turn_types import AgentTurnResult

if TYPE_CHECKING:
    from .agent_core import AgentCore

logger = logging.getLogger(__name__)


class Interrupts:
    """中断处理 Mixin（供 AgentCore 多继承使用，自身不初始化状态）"""

    async def _ahandle_turn_completion(
        self,
        turn: AgentTurnResult,
        config: dict[str, Any],
        mode: str,
    ) -> None:
        """统一处理 turn 完成后的 interrupt 状态管理

        根据 turn 状态更新 interrupt 状态。

        Args:
            turn: Agent 执行结果
            config: LangGraph 配置对象
            mode: 执行模式 ('run', 'chat', 等)
        """
        if turn.is_interrupted:
            await self._acapture_pending_interrupt(config, mode)
        elif turn.is_completed:
            await self._aclear_pending_interrupt(self._thread_id_from_config(config))

    async def _acapture_pending_interrupt(self, config: dict[str, Any], mode: str) -> None:
        """记录挂起中断模式到 SessionStore（per-session 隔离）。"""
        thread_id = self._thread_id_from_config(config)
        if thread_id:
            await self._get_store().aset_interrupt_mode(thread_id, mode)

    async def _aclear_pending_interrupt(self, thread_id: str | None = None) -> None:
        """清除挂起中断状态（per-session 隔离）。

        Args:
            thread_id: 指定会话线程 ID。为 None 时清除当前会话的中断。
        """
        sid = self._current_sid(thread_id)
        await self._get_store().aclear_interrupt(sid)

    async def _arepair_rejected_tool_calls(
        self,
        config: dict[str, Any],
    ) -> None:
        """异步修复 checkpoint 中未完成的工具调用（补齐取消结果）。

        使用 LangGraph 的异步 state API，避免阻塞事件循环。

        Args:
            config: LangGraph 配置对象（含 configurable.thread_id）
        """
        graph = self.agent_executor
        try:
            state = await graph.aget_state(config)
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            logger.warning("读取 checkpoint 状态失败: %s", error, exc_info=True)
            return

        if state is None:
            return

        messages = list(getattr(state, "values", {}).get("messages", []))
        existing_results = [message for message in messages if isinstance(message, ToolMessage)]
        answered_ids = {message.tool_call_id for message in existing_results}
        repairs = []
        for message in messages:
            if not isinstance(message, AIMessage):
                continue
            for tool_call in message.tool_calls:
                call_id = tool_call.get("id")
                if not isinstance(call_id, str) or call_id in answered_ids:
                    continue
                repairs.append(
                    ToolMessage(
                        content="用户拒绝执行危险命令，工具调用已取消。",
                        name=tool_call.get("name"),
                        tool_call_id=call_id,
                        status="error",
                    )
                )
                answered_ids.add(call_id)
        if not repairs:
            return

        try:
            await graph.aupdate_state(
                config,
                {"messages": [*existing_results, *repairs]},
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            logger.warning("修复 checkpoint 状态失败: %s", error, exc_info=True)

    async def _ahandle_rejected_command(
        self,
        config: dict[str, Any],
    ) -> AgentTurnResult:
        """异步处理用户拒绝执行危险命令的情况

        当工具调用被用户拒绝时，异步修复 checkpoint 状态并返回取消结果。

        Args:
            config: LangGraph 配置对象

        Returns:
            状态为 'cancelled' 的 AgentTurnResult
        """
        await self._arepair_rejected_tool_calls(config)
        await self._aclear_pending_interrupt(self._thread_id_from_config(config))
        return AgentTurnResult.cancelled("用户已拒绝执行危险命令，当前任务已取消。")

    async def _acheck_interrupt_event(
        self,
        config: dict[str, Any],
        thread_id: str,
        trace_id: str,
    ) -> AgentEvent | None:
        """流结束后检查是否被中断，返回 INTERRUPT 事件或 None。

        并行工具调用（如多个危险命令）可能产生多个 pending interrupts。
        相同 kind（dangerous_command）的中断合并为单个事件，prompt 列出
        所有待确认命令，用户一次确认即可批量恢复。
        """
        graph = self.agent_executor
        try:
            state = await graph.aget_state(config)
            if state is None:
                return None

            interrupts = [
                intr
                for task in getattr(state, "tasks", []) or []
                for intr in getattr(task, "interrupts", []) or []
            ]
            if not interrupts:
                return None

            ev_dicts = [
                build_interrupt_event(getattr(intr, "value", None))
                for intr in interrupts
            ]

            # 判断是否全部为 dangerous_command（并行工具调用的典型场景）
            kinds = {
                getattr(intr, "value", {}).get("kind", "")
                if isinstance(getattr(intr, "value", None), dict)
                else ""
                for intr in interrupts
            }

            if len(interrupts) > 1 and kinds == {"dangerous_command"}:
                combined_prompt = "\n\n".join(
                    f"({i + 1}/{len(ev_dicts)}) {ev_dicts[i].get('prompt', '')}"
                    for i in range(len(ev_dicts))
                )
                return AgentEvent.interrupt(
                    prompt=combined_prompt,
                    choices=ev_dicts[0].get("choices"),
                    thread_id=thread_id,
                    trace_id=trace_id,
                )

            return AgentEvent.interrupt(
                prompt=ev_dicts[0].get("prompt", ""),
                choices=ev_dicts[0].get("choices"),
                thread_id=thread_id,
                trace_id=trace_id,
            )
        except Exception as error:
            logger.warning("检查 interrupt 失败: %s", error, exc_info=True)
        return None

    async def _abuild_resume_command(
        self,
        config: dict[str, Any],
        payload: dict[str, Any],
    ) -> Command:
        """构造恢复命令：多个 pending interrupts 时批量恢复。

        并行工具调用可能产生多个 pending interrupts（如多个危险命令确认）。
        将同一答案应用到所有 pending interrupts，避免用户逐个确认。

        Args:
            config: LangGraph 配置（用于读取当前 state 的挂起中断）
            payload: 调用方传入的恢复数据

        Returns:
            可直接传给图执行的 ``Command(resume=...)``
        """
        graph = self.agent_executor
        try:
            state = await graph.aget_state(config)
        except Exception as error:
            logger.warning("读取 pending interrupts 失败: %s", error, exc_info=True)
            return Command(resume=payload)

        interrupts = [
            intr
            for task in getattr(state, "tasks", []) or []
            for intr in getattr(task, "interrupts", []) or []
        ]
        if len(interrupts) <= 1:
            return Command(resume=payload)

        # payload 已是 {interrupt_id: value} 映射
        pending_ids = {intr.id for intr in interrupts}
        if isinstance(payload, dict) and payload and pending_ids.issuperset(payload.keys()):
            return Command(resume=payload)

        # 多中断 + 裸值：将同一答案应用到所有 pending interrupts
        return Command(resume={intr.id: payload for intr in interrupts})