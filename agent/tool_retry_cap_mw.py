"""相同参数工具失败熔断中间件 - 防止 ReAct 死循环重复调用同一失败工具。

与 TerminalRetryCapMW 互补：
- TerminalRetryCapMW 只处理终端 exec 工具的「超时」循环（按内容识别 + 工具名限定）。
- 本中间件是通用补充：任意工具只要以「完全相同参数」失败达
  MAX_IDENTICAL_FAILURES(2) 次，后续第 3 次相同调用即被拦截，返回失败
  ToolMessage(status="error")，阻止主模型继续无意义地重复同一调用
  （路径错误、MCP schema 错误、路径逃逸、空结果等均适用）。

失败识别口径（三选一即视为失败）：
1. ToolMessage.status == "error"；
2. content 文本包含已知拒绝/失败标记（_FAILURE_MARKERS）；
3. content 解析为 JSON 对象且 success is False —— 终端工具（run_python/run_cmd
   退出码非 0、run_shell 非零退出）、create_tool、search 等失败时返回 dict 而
   非抛异常，LangGraph 用 json.dumps 把它序列化成 ToolMessage content
   （如 {"success": false, "returncode": 1, ...}），前两条均无法识别。

已知误报边界：若某工具的正常结果整体就是一个含 "success": false 的 JSON 对象
（例如读取到这样的文件内容），也会被判为失败。可接受——后果仅是第 3 次完全
相同的调用被拦截，不影响其他调用。

设计要点：
- 无状态中间件（与 WorkspaceSecurityMW / TerminalRetryCapMW 一致），所有会话
  共享同一编译图，隔离由 request.state（per-thread checkpoint）保证。
- 关联需靠 tool_call_id：先用带 tool_calls 的 AIMessage 建立
  「call_id -> 指纹」映射，再按 ToolMessage.tool_call_id 回查是否为同一调用。
  无法关联（历史里没有任何 tool_calls）时 fail-open 返回 0，绝不误伤。
- 指纹 = 工具名 + 规范化 JSON 参数（sort_keys），保证确定性、与字典顺序无关。
- ask_human 永不拦截（人工确认/选择是正常控制流，不属于退化循环）。
- 放行时正常调用 handler；拦截时不调用 handler，只返回失败 ToolMessage。
- 不捕获任何异常（不触碰 GraphBubbleUp / interrupt 控制流），只决定是否调 handler。
"""
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.messages import ToolMessage
from langchain.tools.tool_node import ToolCallRequest

logger = logging.getLogger(__name__)

# 同一会话状态内，同一工具以完全相同参数失败的最大允许次数
# 达到此值后，下一次相同调用被拦截（尝试 1 失败、尝试 2 失败、尝试 3 被熔断）
MAX_IDENTICAL_FAILURES: int = 2

# 永不拦截的工具：人工确认/选择属于正常控制流，不属于退化重试
_EXCLUDED_TOOLS: frozenset[str] = frozenset({"ask_human"})

# ToolMessage.content 中表示「工具失败」的标记（用于无 status 字段时的兜底识别）
# 注：WorkspaceSecurityMW 的拒绝消息不带 status="error"，仅以「操作被拒绝：」开头，
#     必须在此登记，否则路径逃逸/越界这一最高频的失败场景不会被计数。
_FAILURE_MARKERS: tuple[str, ...] = (
    "[工具执行失败]",
    "[重复调用熔断]",
    "[超时重试已达上限]",
    "[参数冲突]",
    "操作被拒绝",
)


def _fingerprint(tool_name: str, args: Any) -> str:
    """生成工具调用的确定性指纹（工具名 + 规范化 JSON 参数）。

    Args:
        tool_name: 工具名
        args: 工具参数（可为 None / dict / 任意可序列化对象）

    Returns:
        f"{tool_name}\\x00{json.dumps(args or {}, sort_keys=True, ensure_ascii=False, default=str)}"
    """
    normalized = json.dumps(
        args or {}, sort_keys=True, ensure_ascii=False, default=str
    )
    return f"{tool_name}\x00{normalized}"


def is_failed_tool_message(msg: Any) -> bool:
    """判断 ToolMessage 是否为「工具执行失败」结果。

    非 ToolMessage 一律返回 False。判定优先级：
    1. status == "error"（主判据）；
    2. content 文本包含失败标记（兜底，兼容未设置 status 的历史消息）；
    3. content 为 JSON 对象且 success is False（覆盖终端非超时失败、
       create_tool / search 等返回 dict 而非抛异常的工具）。

    Args:
        msg: 待判定的消息对象

    Returns:
        失败返回 True，否则 False
    """
    if not isinstance(msg, ToolMessage):
        return False
    if getattr(msg, "status", None) == "error":
        return True
    content = getattr(msg, "content", "")
    text = content if isinstance(content, str) else str(content)
    if any(marker in text for marker in _FAILURE_MARKERS):
        return True
    # 结构化失败：content 是 JSON 对象且 success is False（dict 结果被 LangGraph
    # 序列化而来，如终端非超时失败 / create_tool / search）
    if not isinstance(content, str):
        return False
    if not content.lstrip().startswith("{") or '"success"' not in content:
        return False
    try:
        data = json.loads(content)
    except (ValueError, TypeError):
        return False
    return isinstance(data, dict) and data.get("success") is False


def _extract_messages(state: Any) -> list[Any] | None:
    """从 state 中提取 messages 列表，容错三种形态。

    支持：dict（含 "messages" 键）/ list（messages 本身）/ 对象（.messages 属性）。
    state 为 None、提取失败或 messages 非列表时返回 None（fail-open，不崩溃）。

    Args:
        state: request.state（LangGraph 注入的当前状态快照）

    Returns:
        messages 列表；无法提取时返回 None
    """
    if state is None:
        return None
    if isinstance(state, dict):
        messages = state.get("messages", [])
    elif isinstance(state, list):
        messages = state
    else:
        messages = getattr(state, "messages", [])
    if not isinstance(messages, list):
        return None
    return messages


def _count_identical_failures(state: Any, tool_name: str, args: Any) -> int:
    """统计历史中与目标调用「工具名 + 参数完全相同」的失败次数。

    算法：
    1. 第一遍：从带非空 tool_calls 的消息（AIMessage）建立
       「tool_call_id -> 指纹」映射（tool_calls 项兼容 dict 与对象，无 id 跳过）；
    2. 第二遍：对每个失败 ToolMessage，若其 tool_call_id 映射到的指纹等于目标
       指纹，则计数 +1。

    若历史中完全找不到任何 tool_calls（无法关联），返回 0（fail-open，
    不做仅按工具名的退化计数，避免误伤）。

    Args:
        state: request.state
        tool_name: 本次工具名
        args: 本次工具参数

    Returns:
        相同调用的历史失败次数（不含本次尚未执行的调用）
    """
    messages = _extract_messages(state)
    if messages is None:
        return 0

    # 第一遍：建立 tool_call_id -> 指纹 映射
    call_fingerprints: dict[str, str] = {}
    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            continue
        for tool_call in tool_calls:
            if isinstance(tool_call, dict):
                call_id = tool_call.get("id")
                name = tool_call.get("name")
                call_args = tool_call.get("args")
            else:
                call_id = getattr(tool_call, "id", None)
                name = getattr(tool_call, "name", None)
                call_args = getattr(tool_call, "args", None)
            if not call_id:
                continue
            call_fingerprints[str(call_id)] = _fingerprint(str(name or ""), call_args)

    # 无法关联任何工具调用，fail-open
    if not call_fingerprints:
        return 0

    # 第二遍：按 tool_call_id 回查失败 ToolMessage
    target = _fingerprint(tool_name, args)
    count = 0
    for msg in messages:
        if not is_failed_tool_message(msg):
            continue
        call_id = getattr(msg, "tool_call_id", None)
        if call_id is None:
            continue
        if call_fingerprints.get(str(call_id)) == target:
            count += 1
    return count


class ToolRetryCapMW(AgentMiddleware):
    """相同参数工具失败熔断中间件。

    拦截任意工具调用：当该工具以完全相同参数在当前会话状态中已失败达
    MAX_IDENTICAL_FAILURES 次时，直接返回失败 ToolMessage，阻止主模型继续
    重复同一失败调用；未达上限或无法关联时正常放行。
    ask_human 永不拦截。
    """

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage],
    ) -> ToolMessage:
        """同步版本：相同参数失败达上限时熔断。"""
        tool_name = request.tool_call.get("name", "")
        if tool_name in _EXCLUDED_TOOLS:
            return handler(request)

        failure_count = _count_identical_failures(
            request.state, tool_name, request.tool_call.get("args")
        )
        if failure_count >= MAX_IDENTICAL_FAILURES:
            logger.warning(
                "工具 %s 以相同参数已失败 %d 次（上限 %d），熔断拦截",
                tool_name, failure_count, MAX_IDENTICAL_FAILURES,
            )
            return self._build_cap_message(request, tool_name, failure_count)

        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ) -> Any:
        """异步版本：相同参数失败达上限时熔断。"""
        tool_name = request.tool_call.get("name", "")
        if tool_name in _EXCLUDED_TOOLS:
            return await handler(request)

        failure_count = _count_identical_failures(
            request.state, tool_name, request.tool_call.get("args")
        )
        if failure_count >= MAX_IDENTICAL_FAILURES:
            logger.warning(
                "工具 %s 以相同参数已失败 %d 次（上限 %d），熔断拦截",
                tool_name, failure_count, MAX_IDENTICAL_FAILURES,
            )
            return self._build_cap_message(request, tool_name, failure_count)

        return await handler(request)

    @staticmethod
    def _build_cap_message(
        request: ToolCallRequest,
        tool_name: str,
        failure_count: int,
    ) -> ToolMessage:
        """构建相同参数失败熔断的拦截 ToolMessage。

        返回 status="error" 的 ToolMessage，主模型读到后应更换参数、改用其他
        方案，或调用 ask_human 询问用户。
        """
        tool_call_id = request.tool_call.get("id", "")
        content = (
            f"[重复调用熔断] {tool_name} 以相同参数已失败 {failure_count} 次，"
            f"已停止继续执行。请更换参数、改用其他方案，或调用 ask_human 询问用户。"
        )
        return ToolMessage(
            content=content,
            tool_call_id=tool_call_id,
            name=tool_name,
            status="error",
        )


__all__ = ["MAX_IDENTICAL_FAILURES", "ToolRetryCapMW", "is_failed_tool_message"]
