"""Agent 层会话配置 wiring 回归测试。"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from langgraph.checkpoint.memory import MemorySaver

from agent.agent_core import AgentCore
from agent.session_config_middleware import SessionConfigMW
from session import SessionRegistry, SessionStore
from session.config import SessionConfig


class _FakeLLM:
    def __init__(self, provider: str, model: str) -> None:
        self.provider = provider
        self.model = model

    def get_chat_model(self) -> object:
        return SimpleNamespace(provider=self.provider, model=self.model)


def _core_with_registry() -> AgentCore:
    core = object.__new__(AgentCore)
    core.llm = _FakeLLM("openai", "gpt-old")
    core.max_iterations = 25
    core._initial_thread_id = "default"
    core._session_store = SessionStore()
    core._checkpointer = MemorySaver()
    core._store = None
    core._process_type = None
    core._async_conn = None
    core._short_term_size = 10
    core._init_session_registry()
    return core


def test_create_executor_registers_session_middleware_and_dynamic_compaction(monkeypatch):
    """Given 最小 Agent 构建依赖，When 创建图，Then wiring 顺序和 resolver 完整。"""
    from agent import graph_builder

    captured: dict[str, object] = {}

    class FakeCompaction:
        def __init__(self, **kwargs: object) -> None:
            captured["compaction"] = kwargs

    class FakeSkill:
        def __init__(self, **kwargs: object) -> None:
            pass

    monkeypatch.setattr(graph_builder, "LCAgentCompactionMiddleware", FakeCompaction)
    monkeypatch.setattr(graph_builder, "SkillInjectionMW", FakeSkill)
    monkeypatch.setattr(graph_builder, "TerminalRetryCapMW", lambda: object())
    monkeypatch.setattr(graph_builder, "ToolExecutionErrorMW", lambda: object())
    monkeypatch.setattr(graph_builder, "ToolArgValidatorMW", lambda: object())
    monkeypatch.setattr(graph_builder, "WorkspaceSecurityMW", lambda: object())
    monkeypatch.setattr(
        graph_builder,
        "create_agent",
        lambda **kwargs: captured.update(kwargs) or object(),
    )

    core = object.__new__(AgentCore)
    core.llm = _FakeLLM("openai", "gpt-old")
    core.tools = []
    core.tool_timeout = None
    core.agent_core_prompt = "base"
    core.compaction_config = SimpleNamespace()
    core._metrics = SimpleNamespace(record_compaction=lambda *_args: None)
    core.skill_manager = SimpleNamespace()
    core.auto_match_skills = False
    core._extra_middleware = []
    core._checkpointer = MemorySaver()
    core._store = None

    core._create_agent_executor()

    middleware = captured["middleware"]
    assert isinstance(middleware, list)
    assert isinstance(middleware[0], SessionConfigMW)
    compaction = captured["compaction"]
    assert isinstance(compaction, dict)
    assert compaction["model"] is not None
    assert compaction["model_resolver"] is not None


def test_registry_default_session_config_comes_from_live_agent_state():
    """Given live LLM state, When 初始化 registry, Then default 配置准确反映它。"""
    core = _core_with_registry()
    default = core.session.default_session_config
    assert default == SessionConfig(provider="openai", model="gpt-old", max_iterations=25)


def test_async_config_injects_per_session_snapshot_and_iterations():
    """Given 两个 stored config，When 构建调用配置，Then 两会话互不污染。"""
    core = _core_with_registry()

    async def run() -> tuple[dict, dict]:
        await core.session.aset_session_config(
            "a", SessionConfig(provider="openai", model="gpt-a", max_iterations=7)
        )
        await core.session.aset_session_config(
            "b", SessionConfig(provider="anthropic", model="claude-b", max_iterations=11)
        )
        return await core._ainvoke_config("a"), await core._ainvoke_config("b")

    config_a, config_b = asyncio.run(run())
    assert config_a["configurable"]["session_config"]["model"] == "gpt-a"
    assert config_a["recursion_limit"] == 7
    assert config_b["configurable"]["session_config"]["provider"] == "anthropic"
    assert config_b["recursion_limit"] == 11


def test_async_config_without_default_matches_sync_config_shape():
    """Given 无 stored/default 配置，When 构建异步 config，Then 严格回退同步形状。"""
    core = object.__new__(AgentCore)
    core._initial_thread_id = "fallback"
    core.max_iterations = 25
    core._session_registry = SessionRegistry(
        checkpointer=MemorySaver(), store=SessionStore(), async_conn=None
    )

    async_config = asyncio.run(core._ainvoke_config("fallback"))
    assert async_config == core._invoke_config("fallback")
    assert "session_config" not in async_config["configurable"]


def test_aswitch_llm_updates_only_process_default():
    """Given 已有会话，When legacy 全局切换，Then 只更新 process default。"""
    core = _core_with_registry()
    asyncio.run(
        core.session.aset_session_config(
            "existing", SessionConfig(provider="openai", model="gpt-session", max_iterations=9)
        )
    )
    core._state_lock = asyncio.Lock()
    core._arebuild_agent_executor = lambda: _completed()

    async def run() -> None:
        await core.aswitch_llm(_FakeLLM("anthropic", "claude-new"))

    async def _completed() -> None:
        return None

    asyncio.run(run())
    assert core.session.default_session_config == SessionConfig(
        provider="anthropic", model="claude-new", max_iterations=25
    )
    assert asyncio.run(core.session.aget_session_config("existing")) == SessionConfig(
        provider="openai", model="gpt-session", max_iterations=9
    )
