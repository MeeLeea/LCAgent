"""消息处理与流式事件工具模块。

整合三类功能：
1. LLM 异常信息提取（extract_llm_error）
2. 消息 content 字符串化与中断事件构建（stringify_content / build_interrupt_event）
3. 流式事件生成 Mixin（StreamMixin），供 AgentCore 继承
"""
import asyncio
import logging
import re
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
)
from langgraph.types import Command

from tools.terminal_tools import UserRejectedCommandError

from .llm_client import RETRY_ATTEMPTS, RETRY_MAX_DELAY, should_retry

logger = logging.getLogger(__name__)

# ============ LLM 异常提取 ============

def _extract_status_code(raw: str) -> str:
    """从异常字符串中尽力提取 HTTP 状态码。

    Args:
        raw: 异常字符串

    Returns:
        HTTP 状态码字符串，未匹配到则返回空字符串
    """
    # 常见格式: "Error code: 429" / "status_code: 503" / "status 503"
    for pattern in (
        r"Error code:\s*(\d{3})",
        r"status[_ ]?code\s*[:=]\s*(\d{3})",
        r"status\s+(\d{3})",
    ):
        m = re.search(pattern, raw, re.IGNORECASE)
        if m:
            return m.group(1)
    return ""


def extract_llm_error(e: Exception) -> str:
    """从 LLM API 异常中提取对用户友好的错误信息，如实返回给前端。

    智谱/DeepSeek 等返回的异常字符串形如:
      Error code: 429 - {'error': {'code': '1305', 'message': '该模型当前访问量过大，请您稍后再试'}}
    某些网关过载时会返回纯文本，形如 "Service temporarily unavailable"。
    本函数把其中真正的 message 提取出来，附带 HTTP 状态码，并对常见的瞬时错误
    (429 限流 / 5xx 过载) 给出「稍后重试或切换提供商」的友好提示。

    Args:
        e: LLM API 抛出的异常对象

    Returns:
        对用户友好的错误信息字符串
    """
    raw = str(e)
    status_code = _extract_status_code(raw)
    low = raw.lower()

    # 尝试从 {'message': '...'} 结构中提取真正的错误文案
    msg_match = re.search(r"['\"]message['\"]\s*[:=]\s*['\"](.+?)['\"]\s*[,}]", raw, re.DOTALL)
    if msg_match:
        message = msg_match.group(1).strip()
        prefix = f"[HTTP {status_code}] " if status_code else ""
        return f"{prefix}{message}"

    # 429 / 限流类错误
    if status_code == "429" or "too many requests" in low or "访问量过大" in raw:
        code_hint = f"[HTTP {status_code}] " if status_code else ""
        return f"{code_hint}模型当前访问量过大或触发限流(429)，请稍后重试，或切换模型/提供商。"

    # 5xx / 服务过载类错误(含纯文本网关提示)
    transient_keywords = (
        "service temporarily unavailable",
        "service unavailable",
        "temporarily unavailable",
        "internal server error",
        "bad gateway",
        "gateway timeout",
        "server overloaded",
        "temporary failure",
    )
    is_server_error = (
        status_code.startswith("5")
        or any(k in low for k in transient_keywords)
    )
    if is_server_error:
        code_hint = f"[HTTP {status_code}] " if status_code else ""
        return f"{code_hint}模型服务暂时不可用，请稍后重试，或切换提供商。"

    # 401 / 鉴权类错误
    if status_code == "401" or "unauthorized" in low or "authentication" in low:
        return "LLM 鉴权失败(401)，请检查 API Key 是否正确配置。"

    # 兜底：返回原始信息，不再吞掉
    return f"执行出错: {raw}" if raw else f"执行出错: {type(e).__name__}"


# ============ 消息内容处理 ============

def stringify_content(content: Any) -> str:
    """把消息 content(str / list / 其他) 统一转成字符串。

    Args:
        content: LangChain 消息的 content 字段，可能是 str、list 或其他类型

    Returns:
        统一后的字符串
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", item.get("content", ""))))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def build_interrupt_event(value: Any) -> dict[str, Any]:
    """把 ask_human / 危险命令确认的 interrupt.value 转成前端可消费的事件。

    Args:
        value: interrupt 的 value 字段

    Returns:
        前端可消费的事件字典，包含 type、prompt、choices 三个字段
    """
    if isinstance(value, dict) and value.get("kind") in ("human_choice", "dangerous_command"):
        return {
            "type": "interrupt",
            "prompt": str(value.get("prompt") or "需要人工输入"),
            "choices": value.get("choices") or [],
        }
    return {"type": "interrupt", "prompt": stringify_content(value), "choices": []}


# ============ 流式事件处理器（组合模式） ============

class StreamHandler:
    """流式事件处理器，通过组合方式注入 AgentCore。

    构造时接收 agent 引用，通过 agent 访问其 agent_executor、verbose
    以及 _invoke_config 等业务方法。
    提供 astream_chat / astream_resume 两个流式接口。
    """

    def __init__(self, agent: Any):
        self.agent = agent

    async def _astream_events(
        self,
        input_or_command: Any,
        config: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        """直接消费 LangGraph 的异步事件流并映射为前端事件。

        Args:
            input_or_command: 输入消息或恢复命令
            config: LangGraph 配置对象
        """
        graph = self.agent.agent_executor
        recorded_msg_ids: set[str] = set()
        for attempt in range(RETRY_ATTEMPTS):
            emitted = False
            try:
                async for ev in graph.astream_events(
                    input_or_command,
                    config=config,
                    version="v2",
                ):
                    event_name = ev.get("event", "") if isinstance(ev, dict) else ""
                    metadata = ev.get("metadata") if isinstance(ev, dict) else None
                    data = ev.get("data") if isinstance(ev, dict) else None
                    metadata_dict = metadata if isinstance(metadata, dict) else {}
                    data_dict = data if isinstance(data, dict) else {}

                    if event_name == "on_chat_model_stream":
                        node = metadata_dict.get("langgraph_node", "")
                        chunk = data_dict.get("chunk")
                        if isinstance(chunk, AIMessageChunk) and node in ("agent", "model"):
                            text = stringify_content(chunk.content)
                            if text:
                                emitted = True
                                yield {"type": "token", "content": text}
                    elif event_name == "on_tool_start":
                        name = data_dict.get("name")
                        tool_input = data_dict.get("input")
                        emitted = True
                        yield {
                            "type": "tool_call",
                            "id": data_dict.get("tool_call_id") or name,
                            "name": name,
                            "args": tool_input,
                        }
                    elif event_name == "on_tool_end":
                        output = data_dict.get("output")
                        content = getattr(output, "content", None) if output is not None else data_dict.get("output")
                        emitted = True
                        yield {
                            "type": "tool_result",
                            "id": data_dict.get("tool_call_id") or getattr(output, "tool_call_id", None) or data_dict.get("name", ""),
                            "name": data_dict.get("name", "") or "",
                            "content": stringify_content(content),
                        }
                    elif event_name == "on_chat_model_end":
                        output = data_dict.get("output")
                        if isinstance(output, AIMessage):
                            msg_id = getattr(output, "id", None) or str(id(output))
                            if msg_id not in recorded_msg_ids:
                                recorded_msg_ids.add(msg_id)
                                llm = getattr(self.agent, "llm", None)
                                self.agent.metrics.extract_and_record_llm_usage(
                                    output,
                                    provider=getattr(llm, "provider", ""),
                                    model=getattr(llm, "model", "") or "",
                                )
                return
            except UserRejectedCommandError:
                await self.agent._arepair_rejected_tool_calls(config)
                await self.agent._aclear_pending_interrupt(self._thread_id_of(config))
                yield {"type": "cancelled", "content": "用户已拒绝执行危险命令，当前任务已取消。"}
                return
            except Exception as error:
                if emitted or not should_retry(error) or attempt == RETRY_ATTEMPTS - 1:
                    yield {"type": "error", "content": extract_llm_error(error)}
                    return
                await asyncio.sleep(min(RETRY_MAX_DELAY, 2**attempt))

    def _thread_id_of(self, config: dict[str, Any]) -> str | None:
        """从 config 提取 thread_id（用于清理对应线程的中断状态）。"""
        configurable = config.get("configurable")
        if isinstance(configurable, dict):
            tid = configurable.get("thread_id")
            if isinstance(tid, str):
                return tid
        return None

    async def _check_interrupt(
        self,
        config: dict[str, Any],
    ) -> dict[str, Any] | None:
        """流结束后异步检查是否被 ask_human 中断，返回中断事件或 None。"""
        graph = self.agent.agent_executor
        try:
            state = await graph.aget_state(config)
            if state is not None:
                for task in getattr(state, "tasks", []) or []:
                    for intr in getattr(task, "interrupts", []) or []:
                        return build_interrupt_event(getattr(intr, "value", None))
        except Exception as error:
            logger.warning("检查 interrupt 失败: %s", error, exc_info=True)
        return None

    async def astream_chat(self, message: str, thread_id: str | None = None):
        """
        流式对话：异步生成事件字典，供 SSE 推送。

        压缩由 before_model 中间件自动触发，无需手动调用。

        事件类型:
          {"type": "token", "content": str}                 # LLM 文本增量
          {"type": "tool_call", "id", "name", "args"}       # 工具调用开始
          {"type": "tool_result", "id", "name", "content"}  # 工具执行结果
          {"type": "interrupt", "prompt", "choices"}        # 被 ask_human 中断
          {"type": "cancelled", "content"}                  # 用户拒绝危险命令
          {"type": "error", "content"}                      # 执行异常
          {"type": "done"}                                  # 本轮结束

        说明：checkpoint 会自动持久化整轮对话，故无需手动 memory.add。

        Args:
            message: 用户消息
            thread_id: 目标会话线程 ID（为 None 时使用当前会话）
        """
        config = self.agent._invoke_config(thread_id)
        input_msg = HumanMessage(content=message)
        original_verbose = self.agent.verbose
        self.agent.verbose = False
        had_terminal = False
        try:
            async for ev in self._astream_events({"messages": [input_msg]}, config):
                if ev.get("type") in ("error", "cancelled"):
                    had_terminal = True
                yield ev
        finally:
            self.agent.verbose = original_verbose

        if had_terminal:
            return
        interrupt = await self._check_interrupt(config)
        if interrupt is not None:
            yield interrupt
            return
        await self.agent._aclear_pending_interrupt(self._thread_id_of(config))
        yield {"type": "done"}

    async def astream_resume(self, payload: dict[str, Any], thread_id: str | None = None):
        """流式恢复被 ask_human 中断的会话，事件格式同 astream_chat。

        Args:
            payload: 恢复数据
            thread_id: 目标会话线程 ID（为 None 时使用当前会话）
        """
        config = self.agent._invoke_config(thread_id)
        original_verbose = self.agent.verbose
        self.agent.verbose = False
        had_terminal = False
        try:
            async for ev in self._astream_events(Command(resume=payload), config):
                if ev.get("type") in ("error", "cancelled"):
                    had_terminal = True
                yield ev
        finally:
            self.agent.verbose = original_verbose

        if had_terminal:
            return
        interrupt = await self._check_interrupt(config)
        if interrupt is not None:
            yield interrupt
            return
        await self.agent._aclear_pending_interrupt(self._thread_id_of(config))
        yield {"type": "done"}


__all__ = [
    "StreamHandler",
    "build_interrupt_event",
    "extract_llm_error",
    "stringify_content",
]
