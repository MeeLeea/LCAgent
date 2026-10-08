"""采样参数（temperature/max_tokens）分层配置测试。

分层隔离设计（team 角色与全局各自独立）：
    1. 团队场景：采样参数来自 team/team_agents.json (default + 角色覆盖)
       （load_team_agent_config 合并 default，角色未配置时落到 default），
       显式 overrides 参数优先
    2. 非团队场景：LLMClient 内部从 agent/agent_config.json 读取（含 DEFAULTS 兜底）
    全局 agent/agent_config.json 的自定义采样值对团队角色不生效。
"""
import json
from pathlib import Path

from team.base import TeamAgent
from team.factory import build_team_agent


class _DummyAgent:
    """捕获构造参数的假 Agent 类，避免真实 TeamAgent 初始化依赖 LLMClient/API key"""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


def _make_team_agents_config(tmp_path, default_cfg: dict, role_cfg: dict) -> Path:
    """构造 team/team_agents.json 统一配置文件"""
    base = tmp_path / "proj"
    team_dir = base / "team"
    team_dir.mkdir(parents=True, exist_ok=True)
    
    # 统一配置：default + 角色
    team_agents = {"default": default_cfg}
    team_agents.update(role_cfg)
    
    (team_dir / "team_agents.json").write_text(json.dumps(team_agents), encoding="utf-8")
    return base


def _make_role_prompt(tmp_path, role: str) -> None:
    """创建角色提示词文件"""
    (tmp_path / "proj" / "team" / role / "AGENT.md").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "proj" / "team" / role / "AGENT.md").write_text("# 测试角色\n", encoding="utf-8")


def test_role_config_wins_over_default(tmp_path):
    """角色级配置覆盖 default 配置"""
    base = _make_team_agents_config(
        tmp_path,
        default_cfg={"provider": "yunlan", "model": "deepseek-v4.1-flash", "temperature": 0.7, "max_tokens": 4096, "max_iterations": 10, "agent_prompt_file": "team/architect/AGENT.md"},
        role_cfg={"architect": {"name": "architect", "temperature": 0.3, "max_tokens": 2048, "agent_prompt_file": "team/architect/AGENT.md"}}
    )
    _make_role_prompt(tmp_path, "architect")
    agent = build_team_agent(_DummyAgent, "architect", str(base))
    assert agent.kwargs["temperature"] == 0.3
    assert agent.kwargs["max_tokens"] == 2048
    assert agent.kwargs["provider"] == "yunlan"


def test_role_missing_sampling_params_falls_to_default(tmp_path):
    """角色级未配置采样参数时，落到 default"""
    base = _make_team_agents_config(
        tmp_path,
        default_cfg={"provider": "yunlan", "model": "deepseek-v4.1-flash", "temperature": 0.7, "max_tokens": 4096, "max_iterations": 10, "agent_prompt_file": "team/architect/AGENT.md"},
        role_cfg={"architect": {"name": "architect", "agent_prompt_file": "team/architect/AGENT.md"}}
    )
    _make_role_prompt(tmp_path, "architect")
    agent = build_team_agent(_DummyAgent, "architect", str(base))
    assert agent.kwargs["temperature"] == 0.7  # default 兜底
    assert agent.kwargs["max_tokens"] == 4096


def test_overrides_beat_role_config(tmp_path):
    """build_team_agent 的 **overrides 显式参数优先级最高"""
    base = _make_team_agents_config(
        tmp_path,
        default_cfg={"provider": "yunlan", "model": "deepseek-v4.1-flash", "temperature": 0.7, "max_tokens": 4096, "max_iterations": 10, "agent_prompt_file": "team/architect/AGENT.md"},
        role_cfg={"architect": {"name": "architect", "temperature": 0.3, "max_tokens": 2048, "agent_prompt_file": "team/architect/AGENT.md"}}
    )
    _make_role_prompt(tmp_path, "architect")
    agent = build_team_agent(
        _DummyAgent,
        "architect",
        str(base),
        temperature=0.9,
        max_tokens=1234,
    )
    assert agent.kwargs["temperature"] == 0.9
    assert agent.kwargs["max_tokens"] == 1234


def test_role_stream_chunk_timeout_wins_over_default(tmp_path):
    """角色级配置的 stream_chunk_timeout 优先于 default"""
    base = _make_team_agents_config(
        tmp_path,
        default_cfg={"provider": "yunlan", "model": "deepseek-v4.1-flash", "temperature": 0.7, "max_tokens": 4096, "stream_chunk_timeout": 300.0, "max_iterations": 10, "agent_prompt_file": "team/architect/AGENT.md"},
        role_cfg={"architect": {"name": "architect", "stream_chunk_timeout": 45.0, "agent_prompt_file": "team/architect/AGENT.md"}}
    )
    _make_role_prompt(tmp_path, "architect")
    agent = build_team_agent(_DummyAgent, "architect", str(base))
    assert agent.kwargs["stream_chunk_timeout"] == 45.0


def test_role_missing_stream_chunk_timeout_falls_to_default(tmp_path):
    """角色级未配置 stream_chunk_timeout 时，落到 default(300.0)"""
    base = _make_team_agents_config(
        tmp_path,
        default_cfg={"provider": "yunlan", "model": "deepseek-v4.1-flash", "temperature": 0.7, "max_tokens": 4096, "stream_chunk_timeout": 300.0, "max_iterations": 10, "agent_prompt_file": "team/architect/AGENT.md"},
        role_cfg={"architect": {"name": "architect", "agent_prompt_file": "team/architect/AGENT.md"}}
    )
    _make_role_prompt(tmp_path, "architect")
    agent = build_team_agent(_DummyAgent, "architect", str(base))
    assert agent.kwargs["stream_chunk_timeout"] == 300.0


def test_stream_chunk_timeout_override_beats_role_config(tmp_path):
    """build_team_agent 的 stream_chunk_timeout 显式 override 优先级最高"""
    base = _make_team_agents_config(
        tmp_path,
        default_cfg={"provider": "yunlan", "model": "deepseek-v4.1-flash", "temperature": 0.7, "max_tokens": 4096, "stream_chunk_timeout": 300.0, "max_iterations": 10, "agent_prompt_file": "team/architect/AGENT.md"},
        role_cfg={"architect": {"name": "architect", "stream_chunk_timeout": 45.0, "agent_prompt_file": "team/architect/AGENT.md"}}
    )
    _make_role_prompt(tmp_path, "architect")
    agent = build_team_agent(
        _DummyAgent,
        "architect",
        str(base),
        stream_chunk_timeout=77.0,
    )
    assert agent.kwargs["stream_chunk_timeout"] == 77.0


def test_stream_chunk_timeout_reaches_llm_client(tmp_path, monkeypatch):
    """全链路：角色配置值经 factory → TeamAgent → LLMClient 透传(复用 _DummyAgent 捕获构造参数)"""
    base = _make_team_agents_config(
        tmp_path,
        default_cfg={"provider": "yunlan", "model": "deepseek-v4.1-flash", "temperature": 0.7, "max_tokens": 4096, "stream_chunk_timeout": 300.0, "max_iterations": 10, "agent_prompt_file": "team/architect/AGENT.md"},
        role_cfg={"architect": {"name": "architect", "stream_chunk_timeout": 45.0, "agent_prompt_file": "team/architect/AGENT.md"}}
    )
    _make_role_prompt(tmp_path, "architect")
    # 复用 _DummyAgent 作为 LLMClient 替身，捕获其构造 kwargs（不联网、不需 API key）
    monkeypatch.setattr("team.base.LLMClient", _DummyAgent)
    agent = build_team_agent(TeamAgent, "architect", str(base))
    # 角色配置值覆盖类属性默认值(300.0)，并原样传入 LLMClient
    assert agent.stream_chunk_timeout == 45.0
    assert agent.llm.kwargs["stream_chunk_timeout"] == 45.0


def test_team_agent_class_attr_still_applies(tmp_path, monkeypatch):
    """直接构造 TeamAgent（不经 factory）时，类属性默认值仍然生效"""
    from tests.team.test_team_base import _FakeLLM

    monkeypatch.setattr("team.base.LLMClient", _FakeLLM)
    agent = TeamAgent(name="plain", prompt_file=str(tmp_path / "no_such.md"))
    assert agent.temperature == TeamAgent.temperature  # 0.7
    assert agent.max_tokens == TeamAgent.max_tokens  # 2048
    assert agent.stream_chunk_timeout == TeamAgent.stream_chunk_timeout  # 300.0


def test_fallback_to_old_config_when_unified_missing(tmp_path):
    """当统一配置不存在时，回退读取旧 team/<role>/agent_config.json（兼容性）"""
    base = tmp_path / "proj"
    # 创建旧格式配置
    role_dir = base / "team" / "legacy_role"
    role_dir.mkdir(parents=True)
    (role_dir / "agent_config.json").write_text(json.dumps({
        "name": "legacy_role",
        "provider": "zhipu",
        "model": "glm-4-flash",
        "temperature": 0.5,
        "max_tokens": 2048,
        "max_iterations": 10,
        "agent_prompt_file": "team/legacy_role/AGENT.md"
    }), encoding="utf-8")
    (role_dir / "AGENT.md").write_text("# Legacy Role\n", encoding="utf-8")
    
    # 不创建 team_agents.json
    agent = build_team_agent(_DummyAgent, "legacy_role", str(base))
    assert agent.kwargs["temperature"] == 0.5
    assert agent.kwargs["max_tokens"] == 2048
    assert agent.kwargs["provider"] == "zhipu"