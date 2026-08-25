"""消息处理工具模块。

整合两类功能：
1. LLM 异常信息提取（extract_llm_error）
2. 消息 content 字符串化与中断事件构建（stringify_content / build_interrupt_event）
"""
import re
from typing import Any

from utils.events import make_interrupt_dict

from .llm_client import _RETRYABLE_KEYWORDS

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
    # 关键词清单与 llm_client.should_retry 共用 _RETRYABLE_KEYWORDS，避免两处维护漂移；
    # "too many requests" 已在上面 429 分支先行返回，此处不会受影响
    is_server_error = (
        status_code.startswith("5")
        or any(k in low for k in _RETRYABLE_KEYWORDS)
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
        return make_interrupt_dict(
            str(value.get("prompt") or "需要人工输入"),
            value.get("choices") or [],
        )
    return make_interrupt_dict(stringify_content(value), [])


__all__ = [
    "build_interrupt_event",
    "extract_llm_error",
    "stringify_content",
]
