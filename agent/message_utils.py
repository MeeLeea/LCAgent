"""消息处理与流式事件工具模块。

整合三类功能：
1. LLM 异常信息提取（extract_llm_error）
2. 消息 content 字符串化与中断事件构建（stringify_content / build_interrupt_event）
3. 流式事件生成 Mixin（StreamMixin），供 AgentCore 继承
"""
import asyncio
import re
import threading
import time
from typing import Any, Dict

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    ToolMessage,
)
from langgraph.types import Command

from .llm_client import RETRY_ATTEMPTS, RETRY_MAX_DELAY, should_retry
from tools.terminal_tools import UserRejectedCommandError

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


def build_interrupt_event(value: Any) -> Dict[str, Any]:
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
    以及 _invoke_config、_create_agent_executor 等业务方法。
    提供 astream_chat / astream_resume 两个流式接口。
    """

    def __init__(self, agent: Any):
        self.agent = agent

    def _stream_events(self, input_or_command, config: Dict[str, Any]):
        """同步流式生成事件（在线程中执行，配合 SqliteSaver 的同步接口）。

        对 LLM 提供的瞬时错误(429/5xx/连接超时)做自动重试：仅当错误发生在
        任何事件输出之前才从头重试，避免工具副作用被重复执行或产出重复 token。
        """
        for attempt in range(RETRY_ATTEMPTS):
            emitted = False
            try:
                for mode, payload in self.agent.agent_executor.stream(
                    input_or_command,
                    config=config,
                    stream_mode=["messages", "updates"],
                ):
                    emitted = True
                    if mode == "messages":
                        chunk, metadata = payload
                        node = metadata.get("langgraph_node", "") if isinstance(metadata, dict) else ""
                        # 兼容不同 LangGraph 版本：token 可能从 agent 或 model 节点产出
                        if isinstance(chunk, AIMessageChunk) and node in ("agent", "model"):
                            text = stringify_content(chunk.content)
                            if text:
                                yield {"type": "token", "content": text}
                    elif mode == "updates":
                        if not isinstance(payload, dict):
                            continue
                        for _node_name, state in payload.items():
                            if not isinstance(state, dict):
                                continue
                            for m in state.get("messages", []):
                                if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                                    for tc in m.tool_calls:
                                        yield {
                                            "type": "tool_call",
                                            "id": tc.get("id"),
                                            "name": tc.get("name"),
                                            "args": tc.get("args"),
                                        }
                                elif isinstance(m, ToolMessage):
                                    yield {
                                        "type": "tool_result",
                                        "id": getattr(m, "tool_call_id", ""),
                                        "name": getattr(m, "name", "") or "",
                                        "content": stringify_content(m.content),
                                    }
                return
            except Exception as e:
                # 已产出事件(部分 token/工具调用)或非瞬时错误：不再重试，交给上层处理
                if emitted or not should_retry(e) or attempt == RETRY_ATTEMPTS - 1:
                    raise
                delay = min(RETRY_MAX_DELAY, 2**attempt)
                time.sleep(delay)

    async def _astream_from_sync(self, input_or_command, config: Dict[str, Any]):
        """把同步 .stream() 跑在后台线程，异步吐出事件。

        SqliteSaver 仅支持同步接口，无法直接用 astream；这里用线程 + 队列桥接，
        同时保持 SSE 端点的异步非阻塞特性。
        """
        loop = asyncio.get_event_loop()
        queue: "asyncio.Queue[Any]" = asyncio.Queue()
        sentinel = object()

        def _worker():
            try:
                for ev in self._stream_events(input_or_command, config):
                    asyncio.run_coroutine_threadsafe(queue.put(ev), loop)
            except UserRejectedCommandError:
                asyncio.run_coroutine_threadsafe(
                    self.agent._arepair_rejected_tool_calls(config), loop,
                )
                self.agent._clear_pending_interrupt()
                asyncio.run_coroutine_threadsafe(
                    queue.put({"type": "cancelled", "content": "用户已拒绝执行危险命令，当前任务已取消。"}),
                    loop,
                )
            except Exception as e:
                asyncio.run_coroutine_threadsafe(
                    queue.put({"type": "error", "content": extract_llm_error(e)}), loop
                )
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(sentinel), loop)

        threading.Thread(target=_worker, daemon=True).start()
        while True:
            item = await queue.get()
            if item is sentinel:
                break
            yield item

    def _check_interrupt(self, config: Dict[str, Any]) -> Dict[str, Any] | None:
        """流结束后同步检查是否被 ask_human 中断，返回中断事件或 None。"""
        try:
            state = self.agent.agent_executor.get_state(config)
            if state is not None:
                for task in getattr(state, "tasks", []) or []:
                    for intr in getattr(task, "interrupts", []) or []:
                        return build_interrupt_event(getattr(intr, "value", None))
        except Exception:
            pass
        return None

    async def astream_chat(self, message: str):
        """
        流式对话：异步生成事件字典，供 SSE 推送。

        事件类型:
          {"type": "token", "content": str}                 # LLM 文本增量
          {"type": "tool_call", "id", "name", "args"}       # 工具调用开始
          {"type": "tool_result", "id", "name", "content"}  # 工具执行结果
          {"type": "interrupt", "prompt", "choices"}        # 被 ask_human 中断
          {"type": "cancelled", "content"}                  # 用户拒绝危险命令
          {"type": "error", "content"}                      # 执行异常
          {"type": "done"}                                  # 本轮结束

        说明：checkpoint 会自动持久化整轮对话，故无需手动 memory.add。
        """
        await self.agent._acompact_if_needed()
        config = self.agent._invoke_config()
        self.agent.agent_executor = self.agent._create_agent_executor(
            self.agent._compute_skill_block(message)
        )
        input_msg = HumanMessage(content=message)
        original_verbose = self.agent.verbose
        self.agent.verbose = False
        had_terminal = False
        try:
            async for ev in self._astream_from_sync({"messages": [input_msg]}, config):
                if ev.get("type") in ("error", "cancelled"):
                    had_terminal = True
                yield ev
        finally:
            self.agent.verbose = original_verbose

        if had_terminal:
            return
        interrupt = self._check_interrupt(config)
        if interrupt is not None:
            yield interrupt
            return
        self.agent._clear_pending_interrupt()
        yield {"type": "done"}

    async def astream_resume(self, payload: Dict[str, Any]):
        """流式恢复被 ask_human 中断的会话，事件格式同 astream_chat。"""
        config = self.agent._invoke_config()
        original_verbose = self.agent.verbose
        self.agent.verbose = False
        had_terminal = False
        try:
            async for ev in self._astream_from_sync(Command(resume=payload), config):
                if ev.get("type") in ("error", "cancelled"):
                    had_terminal = True
                yield ev
        finally:
            self.agent.verbose = original_verbose

        if had_terminal:
            return
        interrupt = self._check_interrupt(config)
        if interrupt is not None:
            yield interrupt
            return
        self.agent._clear_pending_interrupt()
        yield {"type": "done"}


__all__ = [
    "extract_llm_error",
    "stringify_content",
    "build_interrupt_event",
    "StreamHandler",
]
