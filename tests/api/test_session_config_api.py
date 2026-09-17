from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.server import app
from session.config import SessionConfig, SessionConfigPatch
from tests.api import test_api


@pytest.fixture
def mock_agent() -> MagicMock:
    return test_api.mock_agent.__wrapped__()


@pytest.fixture
def mock_llm() -> MagicMock:
    return test_api.mock_llm.__wrapped__()


@pytest.fixture
def client(mock_agent: MagicMock, mock_llm: MagicMock, tmp_path) -> TestClient:
    checkpoint_file = str(tmp_path / "checkpoints.sqlite")
    with patch("api.server.agent", mock_agent), patch("api.server.llm", mock_llm), patch(
        "api.server.CHECKPOINT_FILE", checkpoint_file
    ), patch("api.server.pick_default_provider", return_value="zhipu"), patch(
        "api.server.build_agent", AsyncMock(return_value=(mock_agent, mock_llm))
    ), patch("api.server.LLMClient", return_value=mock_llm), patch(
        "api.server.safety_module.set_confirm_backend"
    ), TestClient(app) as test_client:
        yield test_client



def _providers() -> dict[str, dict[str, object]]:
    return {
        "zhipu": {"name": "智谱AI", "model": "glm-4-flash", "models": ["glm-4-flash", "glm-4-plus"]},
        "deepseek": {"name": "DeepSeek", "model": "deepseek-chat", "models": ["deepseek-chat"]},
    }


def test_patch_updates_only_addressed_session(client, mock_agent) -> None:
    configs = {
        "thread-a": SessionConfig(provider="zhipu", model="glm-4-flash"),
        "thread-b": SessionConfig(provider="deepseek", model="deepseek-chat"),
    }
    mock_agent.session.aget_session_config = AsyncMock(side_effect=configs.get)

    with patch("api.server.load_providers", return_value=_providers()), patch(
        "agent.role_sw.get_available_team_roles", return_value=["manager", "worker", "default"]
    ):
        response = client.patch(
            "/api/sessions/thread-a/config", json={"model": "glm-4-plus"}
        )

    assert response.status_code == 200
    assert response.json()["session_config"]["model"] == "glm-4-plus"
    mock_agent.session.aset_session_config.assert_awaited_once()
    thread_id, updated = mock_agent.session.aset_session_config.await_args.args
    assert thread_id == "thread-a"
    assert updated.provider == "zhipu"
    assert configs["thread-b"].provider == "deepseek"
    assert configs["thread-b"].model == "deepseek-chat"


def test_patch_role_persists_resolved_system_prompt(client, mock_agent) -> None:
    role_patch = SessionConfigPatch(role="worker", system_prompt="resolved worker prompt")
    with patch("api.server.load_providers", return_value=_providers()), patch(
        "agent.role_sw.get_available_team_roles", return_value=["worker"]
    ), patch("api.server._resolve_role_patch", new_callable=AsyncMock, return_value=role_patch):
        response = client.patch(
            "/api/sessions/thread-role/config", json={"role": "worker"}
        )

    assert response.status_code == 200
    data = response.json()["session_config"]
    assert data["role"] == "worker"
    assert data["system_prompt"] == "resolved worker prompt"


@pytest.mark.parametrize(
    ("payload", "offending"),
    [
        ({"provider": "missing-provider"}, "missing-provider"),
        ({"model": "missing-model"}, "missing-model"),
        ({"role": "missing-role"}, "missing-role"),
    ],
)
def test_patch_unknown_values_return_400(client, payload: dict[str, str], offending: str) -> None:
    with patch("api.server.load_providers", return_value=_providers()), patch(
        "agent.role_sw.get_available_team_roles", return_value=["manager", "worker"]
    ):
        response = client.patch("/api/sessions/thread-invalid/config", json=payload)

    assert response.status_code == 400
    assert offending in response.json()["detail"]


def test_patch_temperature_out_of_range_returns_400(client, mock_agent) -> None:
    with patch("api.server.load_providers", return_value=_providers()), patch(
        "agent.role_sw.get_available_team_roles", return_value=[]
    ):
        response = client.patch(
            "/api/sessions/thread-temperature/config", json={"temperature": 5.0}
        )

    assert response.status_code == 400
    assert "temperature" in response.json()["detail"]


def test_patch_new_session_seeds_default_config(client, mock_agent) -> None:
    mock_agent.session.aget_session_config = AsyncMock(return_value=None)
    with patch("api.server.load_providers", return_value=_providers()), patch(
        "agent.role_sw.get_available_team_roles", return_value=[]
    ):
        response = client.patch(
            "/api/sessions/brand-new/config", json={"model": "glm-4-plus"}
        )

    assert response.status_code == 200
    assert response.json()["session_config"]["model"] == "glm-4-plus"
    assert mock_agent.session.aset_session_config.await_count == 2
    assert mock_agent.session.aset_session_config.await_args_list[0].args[0] == "brand-new"


def test_get_providers_is_session_scoped(client, mock_agent) -> None:
    mock_agent.session.aget_session_config = AsyncMock(side_effect=[
        SessionConfig(provider="zhipu", model="glm-4-flash"),
        SessionConfig(provider="deepseek", model="deepseek-chat"),
    ])
    with patch("api.server.load_providers", return_value=_providers()):
        first = client.get("/api/providers?thread_id=A")
        second = client.get("/api/providers?thread_id=B")

    assert first.json()["current_provider"] == "zhipu"
    assert second.json()["current_provider"] == "deepseek"


def test_get_providers_without_thread_keeps_legacy_shape(client) -> None:
    with patch("api.server.load_providers", return_value=_providers()):
        response = client.get("/api/providers")

    data = response.json()
    assert response.status_code == 200
    assert isinstance(data["providers"], list)
    assert data["current_provider"] == "zhipu"
    assert data["current_provider_name"] == "智谱AI"
    assert data["current_model"] == "glm-4-flash"


def test_list_threads_uses_one_batched_config_query(client, mock_agent) -> None:
    configs = {
        "thread-1": SessionConfig(provider="deepseek", model="deepseek-chat"),
        "thread-2": SessionConfig(provider="zhipu", model="glm-4-flash"),
    }
    mock_agent.session.aget_session_configs = AsyncMock(return_value=configs)
    with patch("api.server.load_providers", return_value=_providers()):
        response = client.get("/api/threads")

    assert response.status_code == 200
    entries = {entry["thread_id"]: entry for entry in response.json()["threads"]}
    assert entries["thread-1"]["session_config"]["provider"] == "deepseek"
    assert entries["thread-2"]["session_config"]["provider"] == "zhipu"
    mock_agent.session.aget_session_configs.assert_awaited_once_with(["thread-1", "thread-2"])
    mock_agent.session.aget_session_config.assert_not_awaited()


def test_legacy_provider_switch_with_thread_preserves_llm(client, mock_agent, mock_llm) -> None:
    original_llm = mock_llm
    with patch("api.server.load_providers", return_value=_providers()):
        response = client.post(
            "/api/providers/switch", json={"provider": "deepseek", "thread_id": "thread-a"}
        )

    data = response.json()
    assert data["scope"] == "session"
    assert data["session_config"]["provider"] == "deepseek"
    assert mock_llm is original_llm
    mock_agent.switch_llm.assert_not_called()


def test_legacy_model_switch_without_thread_updates_default_only(client, mock_agent) -> None:
    with patch("api.server.load_providers", return_value=_providers()):
        response = client.post("/api/models/switch", json={"model": "glm-4-plus"})

    data = response.json()
    assert data["deprecated"] is True
    assert data["scope"] == "default"
    assert data["session_config"]["model"] == "glm-4-plus"
    mock_agent.session.set_default_session_config.assert_called_once()
    mock_agent.session.aset_session_config.assert_not_awaited()


def test_legacy_role_switch_with_thread_updates_prompt_without_rebuild(client, mock_agent) -> None:
    role_patch = SessionConfigPatch(role="worker", system_prompt="worker prompt")
    with patch("api.server._resolve_role_patch", new_callable=AsyncMock, return_value=role_patch):
        response = client.post(
            "/api/roles/switch", json={"role": "worker", "thread_id": "thread-a"}
        )

    data = response.json()
    assert data["scope"] == "session"
    assert data["session_config"]["role"] == "worker"
    assert data["session_config"]["system_prompt"] == "worker prompt"
    mock_agent.switch_llm.assert_not_called()


def test_patch_provider_switch_resets_stale_model_to_provider_default(
    client, mock_agent
) -> None:
    """切 provider 且未显式给 model 时，应回落新 provider 默认模型而非 400。

    前端顶栏切换供应商只发 ``{"provider": "..."}``；若沿用旧 provider 的 model，
    该 model 不在新 provider 的 models 白名单内，validate_session_config 会判为
    「未知模型」并返回 400，前端只 console.error → 表现为「无法切换供应商」。
    """
    mock_agent.session.aget_session_config = AsyncMock(
        return_value=SessionConfig(provider="zhipu", model="glm-4-flash")
    )
    with patch("api.server.load_providers", return_value=_providers()), patch(
        "agent.role_sw.get_available_team_roles", return_value=[]
    ):
        response = client.patch("/api/sessions/thread-a/config", json={"provider": "deepseek"})

    assert response.status_code == 200
    data = response.json()["session_config"]
    assert data["provider"] == "deepseek"
    assert data["model"] == "deepseek-chat"


def test_patch_role_that_switches_provider_also_resets_model(client, mock_agent) -> None:
    """角色目录只覆盖 provider 不覆盖 model 时（worker/terminator 现状），model 需同步回落。"""
    mock_agent.session.aget_session_config = AsyncMock(
        return_value=SessionConfig(provider="zhipu", model="glm-4-flash")
    )
    role_patch = SessionConfigPatch(role="worker", provider="deepseek", system_prompt="p")
    with patch("api.server.load_providers", return_value=_providers()), patch(
        "agent.role_sw.get_available_team_roles", return_value=["worker"]
    ), patch("api.server._resolve_role_patch", new_callable=AsyncMock, return_value=role_patch):
        response = client.patch("/api/sessions/thread-role/config", json={"role": "worker"})

    assert response.status_code == 200
    data = response.json()["session_config"]
    assert data["provider"] == "deepseek"
    assert data["model"] == "deepseek-chat"
    assert data["role"] == "worker"


def test_patch_same_provider_keeps_existing_model(client, mock_agent) -> None:
    """provider 未实际变化时不得重置 model，避免覆盖用户已选的模型。"""
    mock_agent.session.aget_session_config = AsyncMock(
        return_value=SessionConfig(provider="zhipu", model="glm-4-plus")
    )
    with patch("api.server.load_providers", return_value=_providers()), patch(
        "agent.role_sw.get_available_team_roles", return_value=[]
    ):
        response = client.patch("/api/sessions/thread-a/config", json={"provider": "zhipu"})

    assert response.status_code == 200
    assert response.json()["session_config"]["model"] == "glm-4-plus"


def test_patch_provider_switch_honours_explicit_model(client, mock_agent) -> None:
    """显式给 model 时以请求为准，不被 provider 默认模型覆盖。"""
    mock_agent.session.aget_session_config = AsyncMock(
        return_value=SessionConfig(provider="deepseek", model="deepseek-chat")
    )
    with patch("api.server.load_providers", return_value=_providers()), patch(
        "agent.role_sw.get_available_team_roles", return_value=[]
    ):
        response = client.patch(
            "/api/sessions/thread-a/config",
            json={"provider": "zhipu", "model": "glm-4-plus"},
        )

    assert response.status_code == 200
    assert response.json()["session_config"]["model"] == "glm-4-plus"


def test_patch_provider_switch_falls_back_to_first_model_when_default_invalid(
    client, mock_agent
) -> None:
    """provider 默认 model 不在自身 models 白名单时（配置不一致），回落 models[0]。

    实际案例：``config/llm_config.json`` 的 ``yunlan-gpt`` 默认模型 ``qwen3.7-max``
    未列入其 ``models``；回落该默认值会再次触发「未知模型」400，使切换永久失败。
    """
    providers = {
        "zhipu": {"name": "智谱AI", "model": "glm-4-flash", "models": ["glm-4-flash"]},
        "inconsistent": {"name": "Inconsistent", "model": "not-listed", "models": ["ok-1", "ok-2"]},
    }
    mock_agent.session.aget_session_config = AsyncMock(
        return_value=SessionConfig(provider="zhipu", model="glm-4-flash")
    )
    with patch("api.server.load_providers", return_value=providers), patch(
        "agent.role_sw.get_available_team_roles", return_value=[]
    ):
        response = client.patch("/api/sessions/thread-a/config", json={"provider": "inconsistent"})

    assert response.status_code == 200
    data = response.json()["session_config"]
    assert data["provider"] == "inconsistent"
    assert data["model"] == "ok-1"
