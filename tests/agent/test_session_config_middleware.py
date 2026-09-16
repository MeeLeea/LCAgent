"""会话模型中间件测试。"""
from types import SimpleNamespace

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import HumanMessage, SystemMessage

from agent.session_config_middleware import SessionConfigMW, SessionModelFactory
from session.config import SessionConfig


def _request(config: SessionConfig | None, model: object, system_message: SystemMessage | None = None) -> ModelRequest:
    context = {"configurable": {"session_config": config.to_dict()}} if config else {}
    return ModelRequest(
        model=model,
        messages=[HumanMessage(content="问题")],
        system_message=system_message,
        runtime=SimpleNamespace(context=context),
    )


@pytest.mark.anyio
async def test_session_model_and_prompt_are_overridden_without_mutating_agent() -> None:
    default = object()
    models: list[SessionConfig] = []

    def factory(config: SessionConfig) -> object:
        models.append(config)
        return config.model

    middleware = SessionConfigMW(model_factory=factory, default_model=default)
    agent = SimpleNamespace(llm=default, agent_core_prompt="旧提示")
    config_a = SessionConfig(provider="a", model="model-a", system_prompt="会话提示")
    captured: list[ModelRequest] = []

    async def handler(request: ModelRequest) -> object:
        captured.append(request)
        return request.model

    assert await middleware.awrap_model_call(_request(config_a, default), handler) == "model-a"
    assert captured[0].system_message == SystemMessage(content="会话提示")
    assert models == [config_a]
    assert agent.llm is default
    assert agent.agent_core_prompt == "旧提示"


@pytest.mark.anyio
async def test_missing_config_forwards_request_unchanged() -> None:
    default = object()
    request = _request(None, default, SystemMessage(content="原提示"))
    middleware = SessionConfigMW(model_factory=lambda _: object(), default_model=default)
    received: list[ModelRequest] = []

    async def handler(value: ModelRequest) -> object:
        received.append(value)
        return value.model

    assert await middleware.awrap_model_call(request, handler) is default
    assert received[0] is request


@pytest.mark.anyio
async def test_missing_system_prompt_preserves_incoming_message() -> None:
    incoming = SystemMessage(content="构建期提示")
    config = SessionConfig(provider="a", model="model-a")
    middleware = SessionConfigMW(model_factory=lambda _: object(), default_model=object())
    received: list[ModelRequest] = []

    async def handler(request: ModelRequest) -> object:
        received.append(request)
        return request.model

    await middleware.awrap_model_call(_request(config, object(), incoming), handler)
    assert received[0].system_message is incoming


def test_session_model_factory_caches_and_evicts() -> None:
    calls: list[SessionConfig] = []

    def build(config: SessionConfig) -> object:
        calls.append(config)
        return object()

    factory = SessionModelFactory(llm_factory=build, max_size=2)
    first = SessionConfig(provider="a", model="one")
    second = SessionConfig(provider="a", model="two")
    third = SessionConfig(provider="a", model="three")
    assert factory.get(first) is factory.get(first)
    assert factory.get(second) is not factory.get(third)
    assert len(factory._cache) == 2
    assert len(calls) == 3


@pytest.mark.anyio
async def test_model_factory_failure_keeps_turn_alive() -> None:
    default = object()
    middleware = SessionConfigMW(model_factory=lambda _: (_ for _ in ()).throw(ValueError("bad")), default_model=default)
    request = _request(SessionConfig(provider="bad"), default)
    received: list[ModelRequest] = []

    async def handler(value: ModelRequest) -> object:
        received.append(value)
        return value.model

    assert await middleware.awrap_model_call(request, handler) is default
    assert received[0] is request
