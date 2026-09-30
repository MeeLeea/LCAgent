"""按会话解析记忆链路 LLM 的 resolver 工厂。

记忆链路（事实抽取/蒸馏/压缩）必须在会话执行结束后仍使用该会话的
provider/model，而非进程启动默认 LLM，避免跨 provider 误发请求与限流。
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from typing import Any

from llm.llm_client import LLMClient

logger = logging.getLogger(__name__)

ThreadLLMResolver = Callable[[str | None], Awaitable[Any]]


def build_memory_llm_resolver(
    session: Any,
    default_getter: Callable[[], Any],
    max_size: int = 16,
) -> ThreadLLMResolver:
    """构建 thread-aware async LLM resolver（带 LRU 缓存）。

    - thread_id 为空 / 无会话配置 / provider 为空 / 读配置或构造 LLM 异常
      → 回落 ``default_getter()``（活的 ``agent.llm``）
    - 否则按会话 provider/model/temperature/max_tokens 构造并缓存 LLMClient
    """
    if max_size < 1:
        raise ValueError("max_size 必须大于 0")
    cache: OrderedDict[tuple[Any, ...], LLMClient] = OrderedDict()

    async def resolve(thread_id: str | None) -> Any:
        if not thread_id:
            return default_getter()
        try:
            config = await session.apeek_session_config(thread_id)
        except Exception as error:  # 有意宽 catch（BLE001 语义）：记忆链路不得因取配置失败中断
            logger.debug("读取会话配置失败，回落默认 LLM [thread=%s]: %s", thread_id, error)
            return default_getter()
        if config is None or not getattr(config, "provider", None):
            return default_getter()
        key = (config.provider, config.model, config.temperature, config.max_tokens)
        cached = cache.get(key)
        if cached is not None:
            cache.move_to_end(key)
            return cached
        try:
            client = LLMClient(
                provider=config.provider,
                model=config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
        except Exception as error:  # 有意宽 catch（BLE001 语义）：构造失败回落默认 LLM
            logger.warning("按会话构造 LLM 失败，回落默认 LLM [thread=%s]: %s", thread_id, error)
            return default_getter()
        cache[key] = client
        cache.move_to_end(key)
        if len(cache) > max_size:
            cache.popitem(last=False)
        return client

    return resolve


__all__ = ["ThreadLLMResolver", "build_memory_llm_resolver"]
