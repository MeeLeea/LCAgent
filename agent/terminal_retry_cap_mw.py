"""终端命令超时重试上限中间件 - 防止主模型无限重试超时的终端命令。

方案 B（主模型反思 + cap 中间件）：
- terminal_tools.py 超时后返回富结果（含 error_type:"timeout" + timeout_reason
  + partial_stdout），ReAct 主模型读到后自行判断超时原因并修改命令重试
  （每次重试是模型新发的 tool_call，新 id，事件干净，无 dedup 问题）。
- 本中间件只做硬性 cap：读 request.state["messages"]，统计当前会话状态里
  exec 工具（run_shell/run_python/run_cmd）的超时次数，达 MAX_TIMEOUT_RETRIES(3)
  则拦截，返回最终失败 ToolMessage(status="error")，阻止主模型再次重试。

设计要点：
- 无状态中间件（与 WorkspaceSecurityMW 一致），所有会话共享同一编译图，
  隔离由 request.state（per-thread checkpoint）保证。
- 读 state.messages 计数历史超时，不嵌入 LLM 调用（主模型本身就是 LLM，
  在 ReAct 循环里读富结果反思重试即可，无需冗余的中间件内 LLM 调用）。
- 放行时正常 await handler，让工具执行 + ToolExecutionErrorMW 正常工作。
- 拦截时返回 ToolMessage(status="error")，告知主模型已达上限停止重试。
- 与 WorkspaceSecurityMW 互补：workspace 负责 cwd/路径逃逸，本中间件负责
  超时重试上限，两者作用域一致（均限定 _EXEC_TOOLS）。
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.messages import ToolMessage
from langchain.tools.tool_node import ToolCallRequest

logger = logging.getLogger(__name__)

# 执行类终端工具名（与 workspace_mw._EXEC_TOOLS 保持一致）
_EXEC_TOOLS: frozenset[str] = frozenset({
    "run_shell",
    "run_python",
    "run_cmd",
})

# 同一会话状态内 exec 工具超时的最大允许累计次数（含已发生的）
# 达到此值后拦截后续 exec 工具调用，阻止主模型无限重试超时命令
MAX_TIMEOUT_RETRIES: int = 3


def is_timeout_content(content: Any) -> bool:
    """识别 ToolMessage.content 是否为终端命令超时结果。

    兼容三种格式：
    - terminal_tools.py 新格式：JSON/dict 含 "error_type": "timeout"
    - tool_wrapper.py 外层超时：JSON 含 "error": "tool_timeout"
    - terminal_tools.py 旧文案：含 "执行超时" / "命令超时"
    """
    if not content:
        return False
    text = content if isinstance(content, str) else str(content)
    if '"error_type": "timeout"' in text:
        return True
    if '"error": "tool_timeout"' in text:
        return True
    return "执行超时" in text or "命令超时" in text


class TerminalRetryCapMW(AgentMiddleware):
    """终端命令超时重试上限中间件。

    拦截 run_shell/run_python/run_cmd 的工具调用，在读 state 统计历史超时
    次数达上限时直接返回失败 ToolMessage，阻止主模型无限重试超时命令。
    未达上限时正常放行，让主模型在 ReAct 循环中读富结果自行反思修改命令重试
    （方案 B，不嵌入独立 LLM 调用，主模型本身就是 LLM）。
    """

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage],
    ) -> ToolMessage:
        """同步版本：超时次数达上限时拦截。"""
        tool_name = request.tool_call.get("name", "")
        if tool_name not in _EXEC_TOOLS:
            return handler(request)

        timeout_count = self._count_prior_timeouts(request.state)
        if timeout_count >= MAX_TIMEOUT_RETRIES:
            logger.warning(
                "终端命令超时重试已达上限(%d/%d)，拦截工具 %s 的执行",
                timeout_count, MAX_TIMEOUT_RETRIES, tool_name,
            )
            return self._build_cap_message(request, timeout_count)

        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ) -> Any:
        """异步版本：超时次数达上限时拦截。"""
        tool_name = request.tool_call.get("name", "")
        if tool_name not in _EXEC_TOOLS:
            return await handler(request)

        timeout_count = self._count_prior_timeouts(request.state)
        if timeout_count >= MAX_TIMEOUT_RETRIES:
            logger.warning(
                "终端命令超时重试已达上限(%d/%d)，拦截工具 %s 的执行",
                timeout_count, MAX_TIMEOUT_RETRIES, tool_name,
            )
            return self._build_cap_message(request, timeout_count)

        return await handler(request)

    @staticmethod
    def _count_prior_timeouts(state: Any) -> int:
        """统计当前会话状态 messages 里 exec 工具的超时累计次数。

        Args:
            state: request.state（dict/list/BaseModel，LangGraph 注入的当前状态快照）

        Returns:
            exec 工具超时的累计次数（不含本次尚未执行的工具调用）
        """
        if state is None:
            return 0
        # state 可能是 dict（含 messages key）、list（messages 本身）或 BaseModel
        if isinstance(state, dict):
            messages = state.get("messages", [])
        elif isinstance(state, list):
            messages = state
        else:
            messages = getattr(state, "messages", [])

        count = 0
        for msg in messages:
            # 只统计 ToolMessage（AIMessage/UserMessage 不是工具结果）
            if not isinstance(msg, ToolMessage):
                continue
            if getattr(msg, "name", "") not in _EXEC_TOOLS:
                continue
            if is_timeout_content(getattr(msg, "content", "")):
                count += 1
        return count

    @staticmethod
    def _build_cap_message(request: ToolCallRequest, timeout_count: int) -> ToolMessage:
        """构建已达超时重试上限的拦截 ToolMessage。

        返回 status="error" 的 ToolMessage，主模型读到后会停止重试该命令，
        改用其他方案或告知用户命令无法执行。
        """
        tool_call_id = request.tool_call.get("id", "")
        tool_name = request.tool_call.get("name", "未知工具")
        content = (
            f"[超时重试已达上限] {tool_name} 已连续超时 {timeout_count} 次，"
            f"停止重试。命令可能存在无法通过简单修改解决的问题"
            f"（如死循环、网络不可达、交互式等待）。"
            f"请放弃该命令，改用其他方案或告知用户命令无法执行。"
        )
        return ToolMessage(
            content=content,
            tool_call_id=tool_call_id,
            name=tool_name,
            status="error",
        )


__all__ = ["MAX_TIMEOUT_RETRIES", "TerminalRetryCapMW", "is_timeout_content"]
