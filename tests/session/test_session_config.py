"""会话基础配置的 Store、Registry 与上下文测试。"""

import asyncio

import pytest
from langgraph.checkpoint.memory import MemorySaver

from session import (
    SessionConfig,
    SessionConfigError,
    SessionConfigPatch,
    SessionContext,
    SessionRegistry,
    SessionStore,
)


def _config(**overrides: object) -> SessionConfig:
    values: dict[str, object] = {
        "provider": "provider-a",
        "model": "model-a",
        "role": "role-a",
        "system_prompt": "prompt-a",
        "temperature": 0.4,
        "max_tokens": 1000,
        "max_iterations": 12,
        "version": 3,
    }
    values.update(overrides)
    return SessionConfig(**values)  # type: ignore[arg-type]


def test_session_config_round_trip():
    config = _config()
    assert SessionConfig.from_dict(config.to_dict()) == config


def test_apply_bumps_version_and_keeps_unset_fields():
    config = _config()
    updated = config.apply(SessionConfigPatch(model="model-b", temperature=0.8))
    assert updated.version == config.version + 1
    assert updated.model == "model-b"
    assert updated.temperature == 0.8
    assert updated.provider == config.provider
    assert updated.role == config.role


def test_apply_rejects_invalid_values():
    with pytest.raises(SessionConfigError):
        _config().apply(SessionConfigPatch(temperature=3.0))


def test_store_set_get_and_update_absent():
    store = SessionStore()

    async def run():
        config = _config()
        await store.aset_session_config("s1", config)
        stored = await store.aget_session_config("s1")
        with pytest.raises(SessionConfigError, match="尚未有配置"):
            await store.aupdate_session_config("missing", SessionConfigPatch(model="x"))
        return stored

    assert asyncio.run(run()) == _config()


def test_store_batch_read_omits_missing_and_isolates_sessions():
    store = SessionStore()

    async def run():
        first = _config(provider="provider-a")
        second = _config(provider="provider-b")
        await store.aset_session_config("a", first)
        await store.aset_session_config("b", second)
        return await store.aget_session_configs(["a", "b", "missing"])

    assert asyncio.run(run()) == {"a": _config(provider="provider-a"), "b": _config(provider="provider-b")}


def test_registry_lazy_migration_is_persisted_without_drift():
    store = SessionStore()
    registry = SessionRegistry(
        MemorySaver(), store, default_session_config=_config(provider="old")
    )

    async def run():
        first = await registry.aget_session_config("legacy")
        registry.set_default_session_config(_config(provider="new"))
        second = await registry.aget_session_config("legacy")
        return first, second

    first, second = asyncio.run(run())
    assert first == second
    assert first is not None and first.provider == "old" and first.version == 1


def test_registry_lazy_migration_without_default_returns_none():
    registry = SessionRegistry(MemorySaver(), SessionStore())
    assert asyncio.run(registry.aget_session_config("missing")) is None


def test_registry_delete_removes_session_config():
    store = SessionStore()
    registry = SessionRegistry(MemorySaver(), store)

    async def run():
        await registry.aset_session_config("s1", _config())
        await registry.adelete_session("s1")
        return await store.aget_session_config("s1")

    assert asyncio.run(run()) is None


def test_session_context_with_config_uses_session_iterations_and_snapshot():
    config = _config(max_iterations=41)
    context = SessionContext.create("s1", MemorySaver(), recursion_limit=7, session_config=config)
    assert context.config["recursion_limit"] == 41
    assert context.config["configurable"]["session_config"] == config.to_dict()
    assert context.session_config == config


def test_session_context_without_config_keeps_legacy_shape():
    context = SessionContext.create("s1", MemorySaver(), recursion_limit=7)
    assert context.config["recursion_limit"] == 7
    assert "session_config" not in context.config["configurable"]
    assert context.session_config is None
