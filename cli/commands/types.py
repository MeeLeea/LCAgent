"""Shared command dispatcher types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, TypeAlias


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list[JsonScalar]


class LlmLike(Protocol):
    provider: str
    model: str

    def get_info(self) -> dict[str, str]: ...

    def list_models(self) -> list[str]: ...

    def switch_model(self, model: str) -> None: ...

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str: ...

    def extract_json(self, value: str) -> dict[str, JsonValue] | None: ...


class MemoryLike(Protocol):
    thread_id: str

    def list_threads(self) -> list[str]: ...
    def get_messages(self) -> list[JsonValue] | None: ...
    def switch_thread(self, thread_id: str) -> None: ...
    def new_thread(self) -> str: ...
    def delete_thread(self, thread_id: str) -> bool: ...
    def export_thread(self, thread_id: str | None = None) -> str: ...
    def clear_long_term(self) -> None: ...
    def clear_short_term(self) -> None: ...


class ToolLike(Protocol):
    name: str
    description: str


class AgentLike(Protocol):
    memory: MemoryLike
    llm: LlmLike
    local_tools: list[ToolLike]
    mcp_tools: list[ToolLike]
    tools: list[ToolLike]
    auto_match_skills: bool

    def switch_llm(self, llm: LlmLike) -> None: ...
    def get_memory_summary(self) -> dict[str, JsonValue]: ...
    def get_available_tools(self) -> list[str]: ...
    def compress_memory(self) -> dict[str, JsonValue]: ...
    def reload_mcp_tools(self) -> int: ...
    def list_skills(self) -> list[dict[str, str]]: ...
    def load_skill(self, name: str) -> bool: ...
    def clear_skills(self) -> None: ...
    def cot(self, task: str) -> str: ...


class SafetyBackend(Protocol):
    def load_config(self) -> dict[str, JsonValue]: ...
    def save_config(self, config: dict[str, JsonValue]) -> bool: ...


class McpBackend(Protocol):
    def list_configured_servers(self, config_file: str) -> list[dict[str, JsonValue]]: ...
    def add_server(
        self,
        name: str,
        transport: str,
        command: str | None = None,
        args: list[str] | None = None,
        url: str | None = None,
        config_file: str | None = None,
    ) -> None: ...
    def remove_server(self, name: str, config_file: str) -> bool: ...
    def toggle_server(self, name: str, enabled: bool, config_file: str) -> bool: ...


PrintFn: TypeAlias = Callable[[str], None]
InputFn: TypeAlias = Callable[[str], str]
SelectMenuFn: TypeAlias = Callable[..., str | tuple[str, str] | None]
CreateLlmFn: TypeAlias = Callable[[str], LlmLike]
ListProvidersFn: TypeAlias = Callable[[], dict[str, dict[str, JsonValue]]]
RunnerFn: TypeAlias = Callable[[AgentLike, str], str]


@dataclass(slots=True)
class CommandOutcome:
    """Result consumed by the interactive loop."""

    handled: bool
    should_break: bool = False


@dataclass(slots=True)
class CommandContext:
    """Dependencies and mutable session state used by command handlers."""

    agent: AgentLike
    base_dir: str
    config_file: str
    mcp_config_file: str
    print_fn: PrintFn
    input_fn: InputFn
    select_menu: SelectMenuFn
    create_llm: CreateLlmFn
    list_providers: ListProvidersFn
    run_structured_until_completion: RunnerFn
    chat_until_completion: RunnerFn
    safety_backend: SafetyBackend
    mcp_backend: McpBackend | None = None

    def print(self, value: str = "") -> None:
        self.print_fn(value)

    def input(self, prompt: str = "") -> str:
        return self.input_fn(prompt)

    def replace_llm(self, llm: LlmLike) -> None:
        # Agent 执行器持有唯一 LLM 客户端，切换后命令层直接经 agent.llm 访问。
        self.agent.switch_llm(llm)


HANDLED = CommandOutcome(handled=True)
UNHANDLED = CommandOutcome(handled=False)
BREAK = CommandOutcome(handled=True, should_break=True)
