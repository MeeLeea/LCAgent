from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from cli.commands.dispatcher import dispatch_command
from cli.commands.types import CommandContext


@dataclass
class FakeSessionManager:
    """模拟 SessionManager 的记忆管理接口"""
    calls: list[tuple[str, Any]] = field(default_factory=list)
    _thread_id: str = "thread-1"

    async def aget_memory_summary(self) -> dict[str, Any]:
        self.calls.append(("aget_memory_summary", None))
        return {
            "thread_id": self._thread_id,
            "checkpoint_backend": "sqlite",
            "checkpoint_file": "checkpoints.sqlite",
            "checkpoint_messages": 3,
            "long_term_count": 2,
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

    async def aclear_long_term_memory(self, session_id: str | None = None) -> int:
        self.calls.append(("aclear_long_term_memory", session_id))
        return 2


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
        list_providers=lambda: {"zhipu": {"name": "Zhipu"}, "deepseek": {"name": "DeepSeek"}},
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
        list_providers=lambda: {"zhipu": {"name": "Zhipu"}, "deepseek": {"name": "DeepSeek"}},
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
    # Given: the context can construct a replacement provider client.
    # When: a direct provider switch is dispatched.
    result = dispatch(harness, "switch:deepseek")
    # Then: the agent receives a new LLM object for that provider.
    assert result.handled is True
    switched_llm = harness.agent.calls[-1][1]
    assert harness.agent.calls[-1][0] == "switch_llm"
    assert switched_llm.provider == "deepseek"
    assert switched_llm is not harness.llm


def test_dispatch_model_switch_reuses_same_llm_object(harness: Harness) -> None:
    # Given: model switching stays within the current provider client.
    # When: a direct model switch is dispatched.
    result = dispatch(harness, "model:glm-4-flash")
    # Then: the existing LLM is mutated and passed back to the agent.
    assert result.handled is True
    assert ("switch_model", "glm-4-flash") in harness.llm.calls
    assert harness.agent.calls[-1] == ("switch_llm", harness.llm)


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
    # clear all → session_manager.aclear_long_term_memory() + session.new_session()
    assert ("aclear_long_term_memory", None) in harness.agent.session_manager.calls
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
