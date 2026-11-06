"""``subagent.common.config`` 通用配置机制单元测试。

覆盖 :func:`build_config` 的字段分派/回退/环境变量覆盖/防御分支，以及
:func:`resolve_llm_settings` 的提供商校验与按 ``label`` 参数化的报错文案。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import pytest

from subagent.common.config import (
    PROJECT_ROOT,
    build_config,
    resolve_llm_settings,
)


@dataclass(frozen=True, slots=True)
class _MiniConfig:
    """与 planner 同构的最小配置（覆盖全部解析器种类）。"""

    name: str = "mini"
    provider: str | None = None
    model: str | None = None
    temperature: float = 0.5
    max_tokens: int = 100
    output_root: str = "data/mini"
    agent_prompt_file: str = "subagent/mini/AGENT.md"


_FIELD_KINDS: dict[str, str] = {
    "name": "str",
    "provider": "nullable_str",
    "model": "nullable_str",
    "temperature": "float",
    "max_tokens": "int",
    "output_root": "output_root",
    "agent_prompt_file": "path",
}


def _build(config_file: str | None, base_dir: str = PROJECT_ROOT) -> _MiniConfig:
    return build_config(
        _MiniConfig,
        config_file=config_file,
        field_kinds=_FIELD_KINDS,
        env_key="LCAGENT_MINI_ROOT",
        base_dir=base_dir,
        label="mini",
    )


# ---- build_config：文件缺失与默认值回退 ----


def test_missing_config_file_falls_back_to_defaults() -> None:
    """config.json 不存在时按 dataclass 默认值合并，相对路径基于 base_dir。"""
    config = _build(None)
    assert config.name == "mini"
    assert config.provider is None
    assert config.model is None
    assert config.temperature == 0.5
    assert config.max_tokens == 100
    assert config.output_root == os.path.normpath(
        os.path.join(PROJECT_ROOT, "data/mini")
    )
    assert config.agent_prompt_file == os.path.normpath(
        os.path.join(PROJECT_ROOT, "subagent/mini/AGENT.md")
    )


def test_valid_config_file_overrides_fields(tmp_path) -> None:
    """JSON 对象覆盖文件值：类型合法字段全部生效，相对路径基于 base_dir。"""
    config_file = tmp_path / "config.json"
    config_file.write_text(
        json.dumps(
            {
                "name": "renamed",
                "provider": "zhipu",
                "model": "glm-4",
                "temperature": 0.3,
                "max_tokens": 2048,
                "output_root": "data/renamed",
                "agent_prompt_file": "subagent/renamed/AGENT.md",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    config = _build(str(config_file), base_dir=str(tmp_path))
    assert config.name == "renamed"
    assert config.provider == "zhipu"
    assert config.model == "glm-4"
    assert config.temperature == 0.3
    assert config.max_tokens == 2048
    assert config.output_root == os.path.normpath(
        os.path.join(str(tmp_path), "data/renamed")
    )
    assert config.agent_prompt_file == os.path.normpath(
        os.path.join(str(tmp_path), "subagent/renamed/AGENT.md")
    )


def test_invalid_values_fall_back_to_defaults(tmp_path) -> None:
    """非法值逐字段回退默认：非正数温度/步数、空 provider。"""
    config_file = tmp_path / "config.json"
    config_file.write_text(
        json.dumps(
            {
                "temperature": -1.0,
                "max_tokens": 0,
                "provider": "",
                "model": 123,
                "name": 123,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    config = _build(str(config_file))
    assert config.temperature == 0.5
    assert config.max_tokens == 100
    assert config.provider is None
    assert config.model is None
    assert config.name == "123"


def test_corrupt_config_file_falls_back_with_warning(tmp_path, caplog) -> None:
    """JSON 损坏时记 warning 并完整回退默认值。"""
    config_file = tmp_path / "config.json"
    config_file.write_text("{not json", encoding="utf-8")
    with caplog.at_level("WARNING"):
        config = _build(str(config_file))
    assert config.temperature == 0.5
    assert "读取 mini 配置失败，回退默认值" in caplog.text


# ---- build_config：output_root 环境变量覆盖 ----


def test_env_var_overrides_output_root(monkeypatch, tmp_path) -> None:
    """环境变量优先于文件覆盖值与默认值。"""
    monkeypatch.setenv("LCAGENT_MINI_ROOT", str(tmp_path / "env-root"))
    config = _build(None)
    assert config.output_root == os.path.normpath(str(tmp_path / "env-root"))


def test_empty_env_var_does_not_override(monkeypatch) -> None:
    """空环境变量不覆盖（``or`` 短路回退，保持原实现语义）。"""
    monkeypatch.setenv("LCAGENT_MINI_ROOT", "")
    config = _build(None)
    assert config.output_root == os.path.normpath(
        os.path.join(PROJECT_ROOT, "data/mini")
    )


# ---- build_config：防御分支 ----


def test_unknown_kind_raises_value_error() -> None:
    """声明表含未知解析器时立即抛 ValueError（纯防御分支）。"""
    with pytest.raises(ValueError, match="未知的字段解析器"):
        build_config(
            _MiniConfig,
            config_file=None,
            field_kinds={**_FIELD_KINDS, "temperature": "bool"},
            env_key="LCAGENT_MINI_ROOT",
            base_dir=PROJECT_ROOT,
            label="mini",
        )


def test_kind_table_mismatch_raises_value_error() -> None:
    """声明表与 dataclass 字段集不一致时抛 ValueError（防字段漂移）。"""
    with pytest.raises(ValueError, match="字段声明表与 dataclass 字段不一致"):
        build_config(
            _MiniConfig,
            config_file=None,
            field_kinds={k: v for k, v in _FIELD_KINDS.items() if k != "name"},
            env_key="LCAGENT_MINI_ROOT",
            base_dir=PROJECT_ROOT,
            label="mini",
        )


# ---- resolve_llm_settings ----


def _providers() -> dict[str, dict[str, object]]:
    return {
        "zhipu": {"name": "智谱AI", "env_key": "ZHIPU_API_KEY", "model": "glm-4", "api_key": ""},
        "deepseek": {"name": "DeepSeek", "env_key": "DEEPSEEK_API_KEY", "model": "deepseek-chat", "api_key": "sk-x"},
    }


def test_resolve_rejects_when_no_providers(monkeypatch) -> None:
    from llm import llm_client

    monkeypatch.setattr(llm_client, "load_providers", dict)
    with pytest.raises(ValueError, match="未配置任何 LLM 提供商"):
        resolve_llm_settings(_MiniConfig(), label="mini")


def test_resolve_picks_provider_with_api_key(monkeypatch) -> None:
    from llm import llm_client

    # 隔离宿主机真实环境变量，保证 _pick_provider 的确定性
    monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
    monkeypatch.setattr(llm_client, "load_providers", _providers)
    provider, model = resolve_llm_settings(_MiniConfig(), label="mini")
    assert provider == "deepseek"
    assert model == "deepseek-chat"


def test_resolve_rejects_missing_api_key(monkeypatch) -> None:
    from llm import llm_client

    def _no_key() -> dict[str, dict[str, object]]:
        return {
            "zhipu": {"name": "智谱AI", "env_key": "ZHIPU_API_KEY", "model": "glm-4", "api_key": ""},
        }

    monkeypatch.setattr(llm_client, "load_providers", _no_key)
    monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
    with pytest.raises(
        ValueError, match="请提供 智谱AI 的API密钥（环境变量 ZHIPU_API_KEY"
    ):
        resolve_llm_settings(_MiniConfig(provider="zhipu"), label="mini")


def test_resolve_missing_model_message_includes_label(monkeypatch) -> None:
    """model 缺失时文案携带 label（保持各子程序原有逐字文案）。"""
    from llm import llm_client

    def _no_model() -> dict[str, dict[str, object]]:
        return {"deepseek": {"name": "DeepSeek", "env_key": "DEEPSEEK_API_KEY", "api_key": "sk-x"}}

    monkeypatch.setattr(llm_client, "load_providers", _no_model)
    with pytest.raises(
        ValueError, match="提供商 deepseek 未配置默认模型，请在 subagent/config.json 的 planner 小节指定 model"
    ):
        resolve_llm_settings(_MiniConfig(), label="planner")


def test_resolve_unknown_provider_rejected(monkeypatch) -> None:
    from llm import llm_client

    monkeypatch.setattr(llm_client, "load_providers", _providers)
    with pytest.raises(ValueError, match="不支持的提供商: nope（可用: zhipu, deepseek）"):
        resolve_llm_settings(_MiniConfig(provider="nope"), label="mini")


def test_resolve_missing_model_label_defaults_to_config_name(monkeypatch) -> None:
    """显式 label 缺省时从 ``config.name`` 推导（子程序 dataclass 均含 name 字段）。"""
    from llm import llm_client

    def _no_model() -> dict[str, dict[str, object]]:
        return {"deepseek": {"name": "DeepSeek", "env_key": "DEEPSEEK_API_KEY", "api_key": "sk-x"}}

    monkeypatch.setattr(llm_client, "load_providers", _no_model)
    with pytest.raises(
        ValueError, match="请在 subagent/config.json 的 executor 小节指定 model"
    ):
        resolve_llm_settings(_MiniConfig(provider="deepseek", name="executor"))


# ---- build_config：section 小节模式（统一配置文件 subagent/config.json）----


def test_section_reads_matching_section_only(tmp_path, monkeypatch) -> None:
    """section 模式只读取匹配小节，其他小节的同名键不得串扰。"""
    config_file = tmp_path / "config.json"
    config_file.write_text(
        json.dumps(
            {
                "other": {"temperature": 9.9, "provider": "evil"},
                "mini": {"temperature": 0.7, "provider": "mini-provider"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("LCAGENT_MINI_ROOT", raising=False)
    config = build_config(
        _MiniConfig,
        config_file=str(config_file),
        field_kinds=_FIELD_KINDS,
        env_key="LCAGENT_MINI_ROOT",
        base_dir=str(tmp_path),
        label="mini",
        section="mini",
    )
    assert config.temperature == 0.7
    assert config.provider == "mini-provider"


def test_section_missing_falls_back_with_warning(tmp_path, caplog) -> None:
    """小节缺失时整节回退默认值并记 warning。"""
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"other": {"temperature": 9.9}}), encoding="utf-8")
    config = build_config(
        _MiniConfig,
        config_file=str(config_file),
        field_kinds=_FIELD_KINDS,
        env_key="LCAGENT_MINI_ROOT",
        base_dir=str(tmp_path),
        label="mini",
        section="mini",
    )
    assert config.temperature == 0.5
    assert config.provider is None
    assert "缺少 mini 小节" in caplog.text


# ---- load_planner_config / load_executor_config：统一配置文件入口 ----


def test_load_planner_config_reads_shared_file_defaults() -> None:
    """真实 ``subagent/config.json`` 的 planner 小节与总表默认值一致。"""
    from subagent.config import (
        PLANNER_MAX_TOKENS,
        PLANNER_TEMPERATURE,
        PLANNER_TIMEOUT_SECONDS,
        load_planner_config,
    )

    config = load_planner_config()
    assert config.name == "planner"
    assert config.provider is None
    assert config.temperature == PLANNER_TEMPERATURE
    assert config.max_tokens == PLANNER_MAX_TOKENS
    assert config.timeout_seconds == PLANNER_TIMEOUT_SECONDS
    assert os.path.isabs(config.output_root)


def test_load_executor_config_reads_shared_file_defaults() -> None:
    """真实 ``subagent/config.json`` 的 executor 小节与总表默认值一致。"""
    from subagent.config import (
        EXECUTOR_MAX_STEPS,
        EXECUTOR_TEMPERATURE,
        load_executor_config,
    )

    config = load_executor_config()
    assert config.name == "executor"
    assert config.provider is None
    assert config.temperature == EXECUTOR_TEMPERATURE
    assert config.max_steps == EXECUTOR_MAX_STEPS
    assert os.path.isabs(config.output_root)


def test_load_config_file_overrides_shared_defaults(tmp_path) -> None:
    """显式传入统一配置文件路径时按传入文件合并（测试隔离用）。"""
    from subagent.config import load_planner_config

    config_file = tmp_path / "config.json"
    config_file.write_text(
        json.dumps({"planner": {"temperature": 0.9, "max_tokens": 123}}),
        encoding="utf-8",
    )
    config = load_planner_config(config_file=str(config_file), base_dir=str(tmp_path))
    assert config.temperature == 0.9
    assert config.max_tokens == 123
