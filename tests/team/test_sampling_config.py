"""采样参数（temperature/max_tokens）分层配置测试。

分层隔离设计（team 角色与全局各自独立）：
    1. 团队场景：采样参数只来自 team/<role>/agent_config.json
       （load_agent_config 合并 DEFAULTS，角色未配置时落到 DEFAULTS 0.7/8192），
       显式 overrides 参数优先
    2. 非团队场景：LLMClient 内部从 agent/agent_config.json 读取（含 DEFAULTS 兜底）
    全局 agent/agent_config.json 的自定义采样值对团队角色不生效。
"""
import json

from team.base import TeamAgent
from team.factory import build_team_agent


class _DummyAgent:
    """捕获构造参数的假 Agent 类，避免真实 TeamAgent 初始化依赖 LLMClient/API key"""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


def _write_json(path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _make_role_tree(tmp_path, role: str, role_cfg: dict, global_cfg: dict | None = None) -> str:
    """构造 base_dir 目录树：agent/agent_config.json(全局) + team/<role>/agent_config.json"""
    base = tmp_path / "proj"
    # 全局配置（可选，缺省不创建 → 走 DEFAULTS）
    if global_cfg is not None:
        _write_json(base / "agent" / "agent_config.json", global_cfg)
    # 角色配置
    role_cfg.setdefault("name", role)
    role_cfg.setdefault("max_iterations", 10)
    role_cfg.setdefault("agent_prompt_file", f"team/{role}/AGENT.md")
    _write_json(base / "team" / role / "agent_config.json", role_cfg)
    # 角色提示词（纯文本 markdown）
    (base / "team" / role / "AGENT.md").write_text("# 测试角色\n", encoding="utf-8")
    return str(base)


def test_role_config_wins_over_global(tmp_path):
    """角色级 agent_config.json 的采样参数优先于全局配置"""
    base = _make_role_tree(
        tmp_path,
        role="my_role",
        role_cfg={"provider": "zhipu", "model": "glm-4-flash", "temperature": 0.3, "max_tokens": 4096},
        global_cfg={"temperature": 0.7, "max_tokens": 8192},
    )
    agent = build_team_agent(_DummyAgent, "team/my_role/agent_config.json", base)
    assert agent.kwargs["temperature"] == 0.3
    assert agent.kwargs["max_tokens"] == 4096


def test_role_missing_sampling_params_falls_to_defaults(tmp_path):
    """角色级未配置采样参数时，落到 DEFAULTS（0.7/8192），全局自定义值不参与"""
    base = _make_role_tree(
        tmp_path,
        role="my_role",
        role_cfg={"provider": "zhipu", "model": "glm-4-flash"},  # 未配置采样参数
        global_cfg={"temperature": 0.5, "max_tokens": 2048},  # 全局自定义值不生效
    )
    agent = build_team_agent(_DummyAgent, "team/my_role/agent_config.json", base)
    assert agent.kwargs["temperature"] == 0.7  # DEFAULTS 兜底
    assert agent.kwargs["max_tokens"] == 8192


def test_defaults_fallback_when_no_global_file(tmp_path):
    """角色级未配置且全局文件不存在时，同样落到 DEFAULTS（0.7/8192）"""
    base = _make_role_tree(
        tmp_path,
        role="my_role",
        role_cfg={"provider": "zhipu", "model": "glm-4-flash"},
        global_cfg=None,  # 全局文件不存在
    )
    agent = build_team_agent(_DummyAgent, "team/my_role/agent_config.json", base)
    assert agent.kwargs["temperature"] == 0.7
    assert agent.kwargs["max_tokens"] == 8192


def test_overrides_beat_role_config(tmp_path):
    """build_team_agent 的 **overrides 显式参数优先级最高"""
    base = _make_role_tree(
        tmp_path,
        role="my_role",
        role_cfg={"temperature": 0.3, "max_tokens": 4096},
        global_cfg={"temperature": 0.7, "max_tokens": 8192},
    )
    agent = build_team_agent(
        _DummyAgent,
        "team/my_role/agent_config.json",
        base,
        temperature=0.9,
        max_tokens=1234,
    )
    assert agent.kwargs["temperature"] == 0.9
    assert agent.kwargs["max_tokens"] == 1234


def test_team_agent_class_attr_still_applies(tmp_path, monkeypatch):
    """直接构造 TeamAgent（不经 factory）时，类属性默认值仍然生效"""
    from tests.team.test_team_base import _FakeLLM

    monkeypatch.setattr("team.base.LLMClient", _FakeLLM)
    agent = TeamAgent(name="plain", prompt_file=str(tmp_path / "no_such.md"))
    assert agent.temperature == TeamAgent.temperature  # 0.7
    assert agent.max_tokens == TeamAgent.max_tokens  # 2048
