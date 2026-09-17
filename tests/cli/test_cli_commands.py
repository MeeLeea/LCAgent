from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from cli.commands.dispatcher import dispatch_command
from cli.commands.types import CommandContext
from session.config import SessionConfig, SessionConfigPatch


@dataclass
class FakeSessionManager:
    """模拟 SessionManager 的记忆管理接口"""
    calls: list[tuple[str, Any]] = field(default_factory=list)
    _thread_id: str = "thread-1"
    agent_fact_count: int = 2
    session_config: SessionConfig = field(
        default_factory=lambda: SessionConfig(provider="zhipu", model="glm-4")
    )

    async def aget_memory_summary(self) -> dict[str, Any]:
        self.calls.append(("aget_memory_summary", None))
        return {
            "thread_id": self._thread_id,
            "checkpoint_backend": "sqlite",
            "checkpoint_file": "checkpoints.sqlite",
            "checkpoint_messages": 3,
            "long_term_count": 2,
            "agent_fact_count": self.agent_fact_count,
            "total_threads": 1,
        }

    async def acompress_memory(self) -> dict[str, Any]:
        self.calls.append(("acompress_memory", None))
        return {
            "success": True,
            "original_count": 2,
            "original_chars": 100,
            "compressed_chars": 10,
            "summary": "压缩摘要",
        }

    async def acompress_agent_memory(self) -> dict[str, Any]:
        self.calls.append(("acompress_agent_memory", None))
        return {
            "success": True,
            "original_count": 2,
            "original_chars": 100,
            "compressed_chars": 10,
            "summary": "压缩摘要",
        }


    async def aclear_long_term_memory(self, session_id: str | None = None) -> int:
        self.calls.append(("aclear_long_term_memory", session_id))
        return 2

    async def aclear_agent_memory(self) -> int:
        self.calls.append(("aclear_agent_memory", None))
        return 3

    async def arecall_agent_memory(self, limit: int | None = None) -> str:
        self.calls.append(("arecall_agent_memory", limit))
        return "【长期记忆】\n- [用户事实] 喜欢深色主题\n"

    async def aget_session_config(self, thread_id: str | None = None) -> SessionConfig:
        self.calls.append(("aget_session_config", thread_id))
        return self.session_config

    async def aupdate_session_config(
        self, patch: SessionConfigPatch, thread_id: str | None = None
    ) -> SessionConfig:
        self.calls.append(("aupdate_session_config", patch))
        self.session_config = self.session_config.apply(patch)
        return self.session_config


@dataclass
class FakeSession:
    """模拟 SessionRegistry 的会话管理接口"""
    current_session_id: str = "thread-1"
    calls: list[tuple[str, Any]] = field(default_factory=list)

    def new_session(self) -> str:
        self.calls.append(("new_session", None))
        self.current_session_id = "thread-2"
        return self.current_session_id

    def new_workflow_session(self, name: str) -> str:
        self.calls.append(("new_workflow_session", name))
        self.current_session_id = f"workflow-{name}-thread-xxx"
        return self.current_session_id


@dataclass
class FakeAgent:
    session_manager: FakeSessionManager = field(default_factory=FakeSessionManager)
    session: FakeSession = field(default_factory=FakeSession)
    calls: list[tuple[str, Any]] = field(default_factory=list)
    llm: Any = None
    local_tools: list[Any] = field(default_factory=list)
    mcp_tools: list[Any] = field(default_factory=list)
    tools: list[Any] = field(default_factory=list)
    auto_match_skills: bool = True

    def set_current_session(self, session_id: str) -> None:
        self.session.current_session_id = session_id
        self.session_manager._thread_id = session_id

    def switch_llm(self, llm: Any) -> None:
        self.calls.append(("switch_llm", llm))
        self.llm = llm

    async def aswitch_llm(self, llm: Any) -> None:
        self.switch_llm(llm)

    def reload_mcp_tools(self) -> int:
        self.calls.append(("reload_mcp_tools", None))
        return 4

    async def areload_mcp_tools(self) -> int:
        return self.reload_mcp_tools()

    def list_skills(self) -> list[dict[str, str]]:
        self.calls.append(("list_skills", None))
        return [{"name": "git-commit", "description": "commit helper"}]

    def load_skill(self, name: str) -> bool:
        self.calls.append(("load_skill", name))
        return True

    async def aclear_skills(self) -> None:
        self.calls.append(("clear_skills", None))

    def clear_skills(self) -> None:
        self.calls.append(("clear_skills", None))

    async def aload_skill(self, name: str) -> bool:
        return self.load_skill(name)

    def cot(self, task: str) -> str:
        self.calls.append(("cot", task))
        return f"cot:{task}"

    async def acot(self, task: str) -> str:
        self.calls.append(("cot", task))
        return f"cot:{task}"


@dataclass
class FakeLlm:
    provider: str = "zhipu"
    model: str = "glm-4"
    calls: list[tuple[str, Any]] = field(default_factory=list)

    def get_info(self) -> dict[str, str]:
        self.calls.append(("get_info", None))
        return {
            "provider_name": self.provider,
            "model": self.model,
            "base_url": "https://offline.invalid",
        }

    def list_models(self) -> list[str]:
        self.calls.append(("list_models", None))
        return [self.model, "glm-4-flash"]

    def switch_model(self, model: str) -> None:
        self.calls.append(("switch_model", model))
        self.model = model

    def extract_json(self, value: str) -> dict[str, str] | None:
        self.calls.append(("extract_json", value))
        return {"value": value}


@dataclass
class FakeSafetyBackend:
    config: dict[str, Any] = field(
        default_factory=lambda: {"mode": "blacklist", "confirm_dangerous": True}
    )
    calls: list[tuple[str, Any]] = field(default_factory=list)

    def load_config(self) -> dict[str, Any]:
        self.calls.append(("load_config", None))
        return dict(self.config)

    def save_config(self, config: dict[str, Any]) -> bool:
        self.calls.append(("save_config", dict(config)))
        self.config = dict(config)
        return True


@dataclass
class FakeRunners:
    calls: list[tuple[str, str]] = field(default_factory=list)

    async def structured(self, agent: FakeAgent, task: str) -> str:
        self.calls.append(("structured", task))
        return f"structured:{task}"

    async def chat(self, agent: FakeAgent, task: str) -> str:
        self.calls.append(("chat", task))
        return f"chat:{task}"


@dataclass
class Harness:
    agent: FakeAgent
    llm: FakeLlm
    runners: FakeRunners
    safety: FakeSafetyBackend
    printed: list[str]
    created: list[str]


@pytest.fixture
def harness(tmp_path: Path) -> Harness:
    agent = FakeAgent()
    llm = FakeLlm()
    runners = FakeRunners()
    safety = FakeSafetyBackend()
    printed: list[str] = []
    created: list[str] = []

    def create_llm(provider: str) -> FakeLlm:
        created.append(provider)
        return FakeLlm(provider=provider, model=f"{provider}-model")

    context = CommandContext(
        agent=agent,
        base_dir=str(tmp_path),
        config_file=str(tmp_path / "llm.json"),
        mcp_config_file=str(tmp_path / "mcp.json"),
        print_fn=printed.append,
        input_fn=lambda prompt="": "y",
        select_menu=lambda *args, **kwargs: "deepseek",
        create_llm=create_llm,
        list_providers=lambda: {
            "zhipu": {"name": "Zhipu", "model": "glm-4"},
            "deepseek": {"name": "DeepSeek", "model": "deepseek-model"},
        },
        run_structured_until_completion=runners.structured,
        chat_until_completion=runners.chat,
        safety_backend=safety,
    )
    agent.llm = llm
    return Harness(agent=context.agent, llm=agent.llm, runners=runners, safety=safety, printed=printed, created=created)


def dispatch(harness: Harness, command: str) -> Any:
    context = CommandContext(
        agent=harness.agent,
        base_dir=".",
        config_file="config/llm_config.json",
        mcp_config_file="config/mcp_servers.json",
        print_fn=harness.printed.append,
        input_fn=lambda prompt="": "y",
        select_menu=lambda *args, **kwargs: "deepseek",
        create_llm=lambda provider: FakeLlm(provider=provider, model=f"{provider}-model"),
        list_providers=lambda: {
            "zhipu": {"name": "Zhipu", "model": "glm-4"},
            "deepseek": {"name": "DeepSeek", "model": "deepseek-model"},
        },
        run_structured_until_completion=harness.runners.structured,
        chat_until_completion=harness.runners.chat,
        safety_backend=harness.safety,
    )
    return asyncio.run(dispatch_command(context, command))


@pytest.mark.parametrize("command", ["quit", "exit"])
def test_dispatch_requests_break_when_quit_command(harness: Harness, command: str) -> None:
    # Given: a live command context.
    # When: the user enters a quit alias.
    result = dispatch(harness, command)
    # Then: the dispatcher asks the caller to break the CLI loop.
    assert result.handled is True
    assert result.should_break is True


def test_dispatch_handles_info_without_running_agent(harness: Harness) -> None:
    # Given: an agent with memory summary state.
    # When: the info command is dispatched.
    result = dispatch(harness, "info")
    # Then: info is handled locally and no LLM run is started.
    assert result.handled is True
    assert ("aget_memory_summary", None) in harness.agent.session_manager.calls
    assert harness.runners.calls == []


def test_dispatch_switch_replaces_provider_llm(harness: Harness) -> None:
    # Given: the current session has a persisted configuration.
    original_llm = harness.agent.llm
    # When: a direct provider switch is dispatched.
    result = dispatch(harness, "switch:deepseek")
    # Then: only the session configuration changes.
    assert result.handled is True
    assert harness.agent.session_manager.session_config.provider == "deepseek"
    assert harness.agent.session_manager.session_config.model == "deepseek-model"
    assert harness.agent.llm is original_llm
    assert not any(call[0] == "switch_llm" for call in harness.agent.calls)


def test_dispatch_model_switch_reuses_same_llm_object(harness: Harness) -> None:
    # Given: model switching is scoped to the current session.
    original_llm = harness.agent.llm
    # When: a direct model switch is dispatched.
    result = dispatch(harness, "model:glm-4-flash")
    # Then: the session model changes and the shared LLM is untouched.
    assert result.handled is True
    assert harness.agent.session_manager.session_config.model == "glm-4-flash"
    assert harness.agent.llm is original_llm
    assert harness.llm.model == "glm-4"
    assert not any(call[0] == "switch_llm" for call in harness.agent.calls)


def test_dispatch_invalid_provider_does_not_write_session_config(harness: Harness) -> None:
    # Given: the requested provider is not in the available provider catalog.
    # When: an invalid provider switch is dispatched.
    result = dispatch(harness, "switch:invalid")
    # Then: the command is handled without changing session configuration.
    assert result.handled is True
    assert harness.agent.session_manager.session_config.provider == "zhipu"
    assert not any(call[0] == "aupdate_session_config" for call in harness.agent.session_manager.calls)


def test_dispatch_thread_new_and_clear_mutate_memory(harness: Harness) -> None:
    # Given: memory exposes thread and clear operations.
    # When: thread:new and clear all are dispatched.
    thread_result = dispatch(harness, "thread:new")
    clear_result = dispatch(harness, "clear all")
    # Then: both commands are local and call the session/memory API directly.
    assert thread_result.handled is True
    assert clear_result.handled is True
    # thread:new → session.new_session()
    assert ("new_session", None) in harness.agent.session.calls
    # clear all → session_manager.aclear_long_term_memory() + aclear_agent_memory() + session.new_session()
    assert ("aclear_long_term_memory", None) in harness.agent.session_manager.calls
    assert ("aclear_agent_memory", None) in harness.agent.session_manager.calls
    assert ("new_session", None) in harness.agent.session.calls


def test_dispatch_mcp_reload_skill_task_and_safety_mode(harness: Harness) -> None:
    # Given: local backends exist for MCP, skills, and safety.
    # When: each management command is dispatched.
    mcp_result = dispatch(harness, "mcp:reload")
    skill_result = dispatch(harness, "skill:git-commit write message")
    safety_result = dispatch(harness, "safety:mode whitelist")
    # Then: the dispatcher updates state and runs only the skill task through the agent runner.
    assert mcp_result.handled is True
    assert skill_result.handled is True
    assert safety_result.handled is True
    assert ("reload_mcp_tools", None) in harness.agent.calls
    assert ("load_skill", "git-commit") in harness.agent.calls
    assert ("structured", "write message") in harness.runners.calls
    assert harness.safety.config["mode"] == "whitelist"


@pytest.mark.parametrize(
    ("command", "expected_call"),
    [
        ("json:summarize", "structured"),
        ("react:inspect repo", "structured"),
        ("cot:reason carefully", "cot"),
        ("hello agent", "chat"),
    ],
)
def test_dispatch_execution_modes_call_expected_runner(
    harness: Harness, command: str, expected_call: str
) -> None:
    # Given: command execution is fully offline through injected fakes.
    # When: each user execution mode is dispatched.
    result = dispatch(harness, command)
    # Then: the mode routes to the expected callable without network access.
    assert result.handled is True
    calls = harness.runners.calls + harness.agent.calls
    assert any(call[0] == expected_call for call in calls)


def test_dispatch_log_shows_current_level_and_handles(harness: Harness) -> None:
    # Given: logging is configured.
    # When: log command is dispatched without arguments.
    result = dispatch(harness, "log")
    # Then: the command is handled locally and shows the current level.
    assert result.handled is True
    assert any("日志级别" in msg or "log" in msg.lower() for msg in harness.printed)


def test_dispatch_log_with_level_changes_and_confirms(harness: Harness) -> None:
    # Given: an initial log level.
    original = logging.getLogger().level
    try:
        # When: log:debug is dispatched.
        result = dispatch(harness, "log:debug")
        # Then: the level changes to DEBUG and user sees confirmation.
        assert result.handled is True
        assert logging.getLogger().level == logging.DEBUG
        assert any("DEBUG" in msg for msg in harness.printed)
    finally:
        logging.getLogger().setLevel(original)


def test_dispatch_log_with_invalid_level_shows_error(harness: Harness) -> None:
    # Given: an invalid level name.
    # When: log:invalid is dispatched.
    result = dispatch(harness, "log:invalid")
    # Then: the command is handled and an error message is shown.
    assert result.handled is True
    assert any("失败" in msg or "未知" in msg for msg in harness.printed)


def test_dispatch_clear_agent_clears_agent_level_memory(harness: Harness) -> None:
    # Given: agent-level memory is exposed through the session manager.
    # When: clear agent is dispatched.
    result = dispatch(harness, "clear agent")
    # Then: it calls aclear_agent_memory, not thread-level clear.
    assert result.handled is True
    assert ("aclear_agent_memory", None) in harness.agent.session_manager.calls
    assert ("aclear_long_term_memory", None) not in harness.agent.session_manager.calls


def test_dispatch_agent_memory_recalls_agent_level_memory(harness: Harness) -> None:
    # Given: agent-level memory recall returns formatted text.
    # When: agent memory is dispatched.
    result = dispatch(harness, "agent memory")
    # Then: the recall text is printed and no LLM run is started.
    assert result.handled is True
    assert ("arecall_agent_memory", None) in harness.agent.session_manager.calls
    assert any("喜欢深色主题" in msg for msg in harness.printed)
    assert harness.runners.calls == []


def test_dispatch_compress_agent_calls_agent_compress(harness: Harness) -> None:
    # Given: agent-level memory has facts to compress.
    # When: compress agent is dispatched.
    result = dispatch(harness, "compress agent")
    # Then: it routes to agent-level compression only.
    assert result.handled is True
    assert ("acompress_agent_memory", None) in harness.agent.session_manager.calls
    assert ("acompress_memory", None) not in harness.agent.session_manager.calls


def test_dispatch_compress_default_still_thread(harness: Harness) -> None:
    # Given: thread-level memory has facts to compress.
    # When: the default compress command is dispatched.
    result = dispatch(harness, "compress")
    # Then: it still routes to thread-level compression only.
    assert result.handled is True
    assert ("acompress_memory", None) in harness.agent.session_manager.calls
    assert ("acompress_agent_memory", None) not in harness.agent.session_manager.calls


def test_dispatch_compress_agent_no_memory_skips_llm(harness: Harness) -> None:
    # Given: agent-level memory is empty.
    harness.agent.session_manager.agent_fact_count = 0
    # When: compress agent is dispatched.
    result = dispatch(harness, "compress agent")
    # Then: the command is handled without invoking any compression.
    assert result.handled is True
    assert ("acompress_agent_memory", None) not in harness.agent.session_manager.calls
    assert ("acompress_memory", None) not in harness.agent.session_manager.calls
