"""按会话覆盖聊天模型与系统提示词。"""
from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ContextT, ModelRequest
from langchain_core.messages import SystemMessage

from llm.llm_client import LLMClient
from session.config import SessionConfig, session_config_from_runtime_context

logger = logging.getLogger(__name__)

ChatModel = Any
LLMFactory = Callable[[SessionConfig], ChatModel]


class SessionModelFactory:
    """创建并缓存会话模型，缓存容量受限于固定上限。"""

    def __init__(
        self,
        *,
        llm_factory: LLMFactory | None = None,
        max_size: int = 16,
    ) -> None:
        if max_size < 1:
            raise ValueError("max_size 必须大于 0")
        self._llm_factory = llm_factory or self._build_model
        self._max_size = max_size
        self._cache: OrderedDict[tuple[str, str | None, float | None, int | None], ChatModel] = (
            OrderedDict()
        )

    def get(self, config: SessionConfig) -> ChatModel:
        """按 provider、模型及采样参数获取缓存模型。"""
        key = (config.provider, config.model, config.temperature, config.max_tokens)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        model = self._llm_factory(config)
        self._cache[key] = model
        self._cache.move_to_end(key)
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)
        return model

    @staticmethod
    def _build_model(config: SessionConfig) -> ChatModel:
        """使用项目统一客户端构建聊天模型。"""
        return LLMClient(
            provider=config.provider,
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        ).get_chat_model()


class SessionConfigMW(AgentMiddleware):
    """从 runtime context 读取会话配置并覆盖本次模型调用。"""

    def __init__(self, *, model_factory: Callable[[SessionConfig], ChatModel], default_model: ChatModel) -> None:
        self._model_factory = model_factory
        self._default_model = default_model

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[Any]],
    ) -> Any:
        """在不改变共享 Agent 的前提下覆盖模型与系统提示词。"""
        config = session_config_from_runtime_context(request.runtime.context)
        if config is None:
            return await handler(request)
        try:
            model = self._model_factory(config)
        except (ValueError, TypeError, OSError, RuntimeError) as error:
            logger.warning("会话模型解析失败，沿用默认模型: %s", error)
            return await handler(request)

        if config.system_prompt:
            return await handler(
                request.override(model=model, system_message=SystemMessage(content=config.system_prompt))
            )
        return await handler(request.override(model=model))


def build_session_model_resolver(
    factory: SessionModelFactory,
    default_model: ChatModel,
) -> Callable[[Any], ChatModel]:
    """构建供压缩中间件使用的 runtime 模型解析器。"""
    def resolve(runtime: Any) -> ChatModel:
        config = session_config_from_runtime_context(getattr(runtime, "context", None))
        if config is None:
            return default_model
        return factory.get(config)

    return resolve


__all__ = ["SessionConfigMW", "SessionModelFactory", "build_session_model_resolver"]
