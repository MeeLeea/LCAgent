"""结构化执行 Mixin - AgentCore 的 turn 级结构化入口与工具步骤记账。

从 agent_core.py 抽离，职责：
- arun_structured / achat_structured / aresume_structured（返回 AgentTurnResult）
- 工具调用步骤记账（per-session 隔离，写入 SessionStore）
- turn 结果解析（_parse_turn_result）

依赖 AgentCore 实例属性：agent_executor / _session_registry /
_session_store / metrics / verbose / llm。
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from tools.terminal_tools import UserRejectedCommandError
from utils.logging_config import TraceContext, generate_trace_id

from .terminal_retry_cap_mw import is_timeout_content
from .turn_types import AgentTurnResult

logger = logging.getLogger(__name__)


class TurnRunners:
    """结构化执行 Mixin（供 AgentCore 多继承使用，自身不初始化状态）"""

    async def arun_structured(self, task: str, thread_id: str | None = None) -> AgentTurnResult:
        """异步执行任务（结构化入口）

        使用 LangGraph 异步 invoke，避免阻塞事件循环。
        压缩由 before_model 中间件自动触发，无需手动调用。
        技能注入由 SkillInjectionMW 从 state 读取，无需手动更新。

        Args:
            task: 任务描述
            thread_id: 目标会话线程 ID（为 None 时使用当前会话）
        """
        self._ensure_not_closed()
        tid = self._current_sid(thread_id)
        with TraceContext(trace_id=generate_trace_id(), thread_id=tid):
            logger.info("arun_structured: %s", task[:100])
            config = self._invoke_config(thread_id)
            input_msg = HumanMessage(content=task)

            try:
                result = await self.agent_executor.ainvoke({"messages": [input_msg]}, config=config)
            except UserRejectedCommandError:
                return await self._ahandle_rejected_command(config)

            await self._arecord_tool_steps(result.get("messages", []), input_msg, tid)
            turn = self._parse_turn_result(result)
            await self._ahandle_turn_completion(turn, config, "run")
            self.metrics.increment_turn()

        return turn

    async def achat_structured(self, message: str, thread_id: str | None = None) -> AgentTurnResult:
        """异步对话（结构化入口）

        压缩由 before_model 中间件自动触发，无需手动调用。
        技能注入由 SkillInjectionMW 从 state 读取，无需手动更新。

        Args:
            message: 用户消息
            thread_id: 目标会话线程 ID（为 None 时使用当前会话）
        """
        self._ensure_not_closed()
        tid = self._current_sid(thread_id)
        with TraceContext(trace_id=generate_trace_id(), thread_id=tid):
            logger.info("achat_structured: %s", message[:100])
            config = self._invoke_config(thread_id)

            with self._temp_verbose(False):
                try:
                    result = await self.agent_executor.ainvoke(
                        {"messages": [HumanMessage(content=message)]},
                        config=config,
                    )
                except UserRejectedCommandError:
                    return await self._ahandle_rejected_command(config)

            turn = self._parse_turn_result(result)
            await self._ahandle_turn_completion(turn, config, "chat")
            self.metrics.increment_turn()

            return turn

    async def aresume_structured(
        self,
        payload: dict[str, Any],
        thread_id: str | None = None,
    ) -> AgentTurnResult:
        """异步恢复中断会话（结构化入口）

        Args:
            payload: 恢复数据
            thread_id: 目标会话线程 ID（为 None 时使用当前会话）
        """
        config = self._invoke_config(thread_id)
        tid = thread_id or self._thread_id_from_config(config)

        # 从 SessionStore 读取 per-session 中断模式
        mode = await self._get_store().aget_interrupt_mode(tid)
        if mode is None:
            mode = "chat"  # 默认 chat 模式（无追踪记录时尝试恢复）

        try:
            resume_command = await self._abuild_resume_command(config, payload)
            result = await self.agent_executor.ainvoke(resume_command, config=config)
        except UserRejectedCommandError:
            return await self._ahandle_rejected_command(config)

        turn = self._parse_turn_result(result)

        if turn.is_interrupted:
            await self._acapture_pending_interrupt(config, mode)
        elif turn.is_completed:
            await self._aclear_pending_interrupt(tid)

        return turn

    async def _arecord_tool_steps(
        self,
        result_messages: list[BaseMessage],
        input_msg: HumanMessage,
        session_id: str,
    ) -> None:
        """异步记录工具调用步骤到 SessionStore（per-session 隔离）。

        LangGraph 会返回当前线程的完整消息历史，按 tool_call id 去重避免重复记账。
        执行历史和去重集合均通过 SessionStore 按 session_id 隔离，
        AgentCore 实例不再持有 execution_history deque 或 _recorded_tool_call_ids set。
        """
        store = self._get_store()
        recorded_ids = await store.aget_recorded_call_ids(session_id)
        history = await store.aget_history(session_id)
        step_count = len(history)

        new_entries: list[dict[str, Any]] = []
        new_ids: set[str] = set()
        new_entries_by_call_id: dict[str, dict[str, Any]] = {}
        for msg in result_messages:
            if msg in (input_msg,):
                continue

            if isinstance(msg, AIMessage):
                tool_calls = getattr(msg, "tool_calls", None)
                if tool_calls:
                    for tc in tool_calls:
                        call_id = tc.get("id")
                        if isinstance(call_id, str) and call_id in recorded_ids:
                            continue
                        step_count += 1
                        if self.verbose:
                            logger.debug("步骤 %d | 工具: %s | 输入: %s",
                                         step_count, tc.get("name", "unknown"), tc.get("args", {}))
                        entry = {
                            "step": step_count,
                            "tool": tc.get("name"),
                            "input": tc.get("args"),
                            "observation": ""
                        }
                        new_entries.append(entry)
                        if isinstance(call_id, str):
                            new_ids.add(call_id)
                            new_entries_by_call_id[call_id] = entry

                # 记录 LLM token 用量（从 response_metadata 提取）
                # getattr 保护：测试中通过 object.__new__ 创建的实例可能没有 llm
                _llm = getattr(self, "llm", None)
                self.metrics.extract_and_record_llm_usage(
                    msg,
                    provider=getattr(_llm, "provider", ""),
                    model=getattr(_llm, "model", "") or "",
                )

            elif hasattr(msg, "content") and hasattr(msg, "tool_call_id"):
                call_id = msg.tool_call_id
                entry = new_entries_by_call_id.get(call_id)
                if entry is not None:
                    entry["observation"] = str(msg.content)[:500]
                if self.verbose and entry is not None:
                    logger.debug("结果: %s", str(msg.content)[:200])

                # 记录工具调用指标（检测超时和失败）
                if entry is not None:
                    content_str = str(msg.content)
                    timed_out = is_timeout_content(content_str)
                    success = not timed_out and getattr(msg, "status", "success") != "error"
                    self.metrics.record_tool_call(
                        name=entry.get("tool", "unknown"),
                        success=success,
                        timed_out=timed_out,
                    )

        # 批量写入 SessionStore（一次读改写，减少 Store 往返）
        if new_entries:
            await store.aextend_history(session_id, new_entries)
        if new_ids:
            await store.aadd_recorded_call_ids(session_id, new_ids)

    def _parse_turn_result(self, result: dict[str, Any]) -> AgentTurnResult:
        interrupts = result.get("__interrupt__")
        if interrupts:
            return AgentTurnResult.interrupted(list(interrupts))

        # 找到最后一条有内容的 AIMessage
        messages = result.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                return AgentTurnResult.completed(str(msg.content))
        return AgentTurnResult.completed("")