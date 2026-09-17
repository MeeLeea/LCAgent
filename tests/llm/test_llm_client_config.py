"""LLMClient 采样参数默认值来源测试。

验证新配置链：agent_config.json -> llm/config.py(DEFAULTS) -> llm_client.py。
LLMClient 不再需要外部传 temperature/max_tokens，内部自动读取全局配置：
    - 显式参数 > 全局 agent_config.json > DEFAULTS 兜底
"""
import os

from llm.config import DEFAULT_AGENT_CONFIG_FILE
from llm.llm_client import LLMClient


class _FakeChatModel:
    """最小 chat model 桩，避免真实模型初始化"""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        # 同时暴露为属性，便于断言参数已真正落到构造出的模型上
        for key, value in kwargs.items():
            setattr(self, key, value)


def _monkeypatch_model_factory(monkeypatch, capture: list) -> None:
    """拦截 init_chat_model，捕获 LLMClient 传入的采样参数"""
    def fake_init_chat_model(model=None, **kwargs):
        capture.append(kwargs)
        return _FakeChatModel(**kwargs)
    monkeypatch.setattr("llm.llm_client.init_chat_model", fake_init_chat_model)


def test_internal_reads_global_config_defaults(monkeypatch, tmp_path):
    """未传采样参数时，LLMClient 从全局 agent_config.json 读取默认值"""
    captured: list[dict] = []
    _monkeypatch_model_factory(monkeypatch, captured)

    # 构造带采样参数的全局配置（覆盖 DEFAULTS）
    cfg_file = tmp_path / "agent_config.json"
    cfg_file.write_text('{"temperature": 0.5, "max_tokens": 4096}', encoding="utf-8")
    monkeypatch.setattr("llm.config.DEFAULT_AGENT_CONFIG_FILE", str(cfg_file))

    client = LLMClient(provider="zhipu", config_file="config/llm_config.json")
    assert client.temperature == 0.5
    assert client.max_tokens == 4096
    # 传入 chat model 工厂的采样参数一致
    assert captured[0]["temperature"] == 0.5
    assert captured[0]["max_tokens"] == 4096


def test_internal_falls_back_to_defaults(monkeypatch, tmp_path):
    """全局配置缺失采样参数时，回退 DEFAULTS（0.7/8192）"""
    captured: list[dict] = []
    _monkeypatch_model_factory(monkeypatch, captured)

    cfg_file = tmp_path / "agent_config.json"
    cfg_file.write_text('{"name": "x"}', encoding="utf-8")
    monkeypatch.setattr("llm.config.DEFAULT_AGENT_CONFIG_FILE", str(cfg_file))

    client = LLMClient(provider="zhipu", config_file="config/llm_config.json")
    assert client.temperature == 0.7
    assert client.max_tokens == 8192


def test_explicit_args_beat_global_config(monkeypatch, tmp_path):
    """显式传入采样参数时优先于全局配置"""
    captured: list[dict] = []
    _monkeypatch_model_factory(monkeypatch, captured)

    cfg_file = tmp_path / "agent_config.json"
    cfg_file.write_text('{"temperature": 0.5, "max_tokens": 4096}', encoding="utf-8")
    monkeypatch.setattr("llm.config.DEFAULT_AGENT_CONFIG_FILE", str(cfg_file))

    client = LLMClient(
        provider="zhipu",
        config_file="config/llm_config.json",
        temperature=0.1,
        max_tokens=512,
    )
    assert client.temperature == 0.1
    assert client.max_tokens == 512
    assert captured[0]["temperature"] == 0.1
    assert captured[0]["max_tokens"] == 512


def test_partial_explicit_args_fill_from_global(monkeypatch, tmp_path):
    """只显式传 temperature 时，max_tokens 仍从全局配置补齐"""
    captured: list[dict] = []
    _monkeypatch_model_factory(monkeypatch, captured)

    cfg_file = tmp_path / "agent_config.json"
    cfg_file.write_text('{"max_tokens": 2048}', encoding="utf-8")
    monkeypatch.setattr("llm.config.DEFAULT_AGENT_CONFIG_FILE", str(cfg_file))

    client = LLMClient(provider="zhipu", config_file="config/llm_config.json", temperature=0.9)
    assert client.temperature == 0.9
    assert client.max_tokens == 2048


def test_real_global_config_has_sampling_params():
    """真实 agent/agent_config.json 包含采样参数，且与 DEFAULTS 一致"""
    from llm.config import load_agent_config

    assert os.path.isfile(DEFAULT_AGENT_CONFIG_FILE)
    cfg = load_agent_config(DEFAULT_AGENT_CONFIG_FILE)
    assert isinstance(cfg["temperature"], float)
    assert isinstance(cfg["max_tokens"], int)


def test_stream_chunk_timeout_defaults_to_300(monkeypatch, tmp_path):
    """未配置 stream_chunk_timeout 时回退 DEFAULTS(300.0)，并透传到模型"""
    captured: list[dict] = []
    _monkeypatch_model_factory(monkeypatch, captured)

    cfg_file = tmp_path / "agent_config.json"
    cfg_file.write_text('{"name": "x"}', encoding="utf-8")
    monkeypatch.setattr("llm.config.DEFAULT_AGENT_CONFIG_FILE", str(cfg_file))

    client = LLMClient(provider="zhipu", config_file="config/llm_config.json")
    assert client.stream_chunk_timeout == 300.0
    # 已真正落到构造出的 chat model 上
    assert client.client.stream_chunk_timeout == 300.0
    assert captured[0]["stream_chunk_timeout"] == 300.0


def test_stream_chunk_timeout_reads_global_config(monkeypatch, tmp_path):
    """全局 agent_config.json 中的 stream_chunk_timeout 优先于 DEFAULTS"""
    captured: list[dict] = []
    _monkeypatch_model_factory(monkeypatch, captured)

    cfg_file = tmp_path / "agent_config.json"
    cfg_file.write_text('{"stream_chunk_timeout": 222.0}', encoding="utf-8")
    monkeypatch.setattr("llm.config.DEFAULT_AGENT_CONFIG_FILE", str(cfg_file))

    client = LLMClient(provider="zhipu", config_file="config/llm_config.json")
    assert client.stream_chunk_timeout == 222.0
    assert client.client.stream_chunk_timeout == 222.0
    assert captured[0]["stream_chunk_timeout"] == 222.0


def test_explicit_stream_chunk_timeout_beats_global_config(monkeypatch, tmp_path):
    """显式传入的 stream_chunk_timeout 优先于全局配置"""
    captured: list[dict] = []
    _monkeypatch_model_factory(monkeypatch, captured)

    cfg_file = tmp_path / "agent_config.json"
    cfg_file.write_text('{"stream_chunk_timeout": 222.0}', encoding="utf-8")
    monkeypatch.setattr("llm.config.DEFAULT_AGENT_CONFIG_FILE", str(cfg_file))

    client = LLMClient(
        provider="zhipu",
        config_file="config/llm_config.json",
        stream_chunk_timeout=45.0,
    )
    assert client.stream_chunk_timeout == 45.0
    assert client.client.stream_chunk_timeout == 45.0
    assert captured[0]["stream_chunk_timeout"] == 45.0


def test_real_global_config_has_stream_chunk_timeout():
    """真实 agent/agent_config.json 显式包含 stream_chunk_timeout=300.0"""
    from llm.config import load_agent_config

    cfg = load_agent_config(DEFAULT_AGENT_CONFIG_FILE)
    assert cfg["stream_chunk_timeout"] == 300.0