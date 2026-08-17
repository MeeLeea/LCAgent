"""Shared command dispatcher types."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias

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


class SessionManagerLike(Protocol):
    """SessionManager 接口 — 三层架构 Session 层门面。

    封装 AgentCore + MemoryManager，所有上层流量（API / CLI / Remote）
    通过此接口访问记忆管理功能。
    """

    @property
    def memory(self) -> Any: ...

    async def aget_memory_summary(self) -> dict[str, JsonValue]: ...
    async def acompress_memory(self) -> dict[str, JsonValue]: ...
    async def aclear_long_term_memory(self, session_id: str | None = None) -> int: ...
    async def aclose(self) -> None: ...


class SessionLike(Protocol):
    """SessionRegistry 接口 - 管理 checkpointer + Store 的会话状态。

    AgentCore 无状态化后会话管理全部经此接口完成，CLI/API 不再直接操作
    AgentMemory 的 thread 级方法。
    """

    current_session_id: str

    def new_session(self, workspace_path: str | None = None) -> str: ...
    def new_workflow_session(
        self, workflow_name: str, workspace_path: str | None = None
    ) -> str: ...
    def is_workflow_session(self, session_id: str) -> bool: ...
    def workflow_name_of(self, session_id: str) -> str | None: ...
    def get_context(self, session_id: str | None = None) -> Any: ...

    async def alist_sessions(self, all_types: bool = False) -> list[str]: ...
    async def aswitch_session(self, session_id: str) -> bool: ...
    async def adelete_session(self, session_id: str) -> bool: ...
    async def aget_messages(self, session_id: str | None = None) -> list[Any]: ...
    async def aexport_session(
        self, session_id: str | None = None, fmt: str = "text"
    ) -> str: ...
    async def asummarize(self, session_id: str | None = None) -> dict[str, Any]: ...
    async def aset_workspace(
        self, workspace_path: str, session_id: str | None = None
    ) -> str: ...
    async def aclear_workspace(self, session_id: str | None = None) -> bool: ...
    async def aget_workspace(self, session_id: str | None = None) -> str | None: ...


class ToolLike(Protocol):
    name: str
    description: str


class AgentLike(Protocol):
    session: SessionLike
    session_manager: SessionManagerLike
    llm: LlmLike
    local_tools: list[ToolLike]
    mcp_tools: list[ToolLike]
    tools: list[ToolLike]
    auto_match_skills: bool

    def get_available_tools(self) -> list[str]: ...
    def list_skills(self) -> list[dict[str, str]]: ...
    def set_current_session(self, session_id: str) -> None: ...
    def cot(self, task: str) -> str: ...
    async def aswitch_llm(self, llm: LlmLike) -> None: ...
    async def manually_compact(self, force: bool = False, thread_id: str | None = None) -> dict[str, JsonValue] | None: ...
    async def arebuild_from_team_dir(self, role_name: str, task: str = "") -> None: ...
    async def areload_mcp_server(self, server_name: str) -> bool: ...
    async def areload_mcp_tools(self) -> int: ...
    async def aclear_skills(self) -> None: ...
    async def aload_skill(self, name: str) -> bool: ...


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
RunnerFn: TypeAlias = Callable[[AgentLike, str], Awaitable[str]]
# 工作流运行跟踪事件回调:接收结构化事件字典(如 {"type": "workflow_node", "node": ..., "status": ...})
WorkflowEventFn: TypeAlias = Callable[[dict[str, str]], None]


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
    # 工作流运行跟踪回调:推送节点/整体状态事件(SSE 等实时通道);None 时仅靠 print 输出
    workflow_event_cb: WorkflowEventFn | None = None
    # workflow 链路的统一 SessionManager 门面(绑定 WorkflowAdapter)。
    # 提供 arun_stream 等接口;为 None 时 workflow 命令不可用。
    workflow_sm: Any = None

    def print(self, value: str = "") -> None:
        self.print_fn(value)

    def input(self, prompt: str = "") -> str:
        return self.input_fn(prompt)

    async def replace_llm(self, llm: LlmLike) -> None:
        # Agent 执行器持有唯一 LLM 客户端，切换后命令层直接经 agent.llm 访问。
        await self.agent.aswitch_llm(llm)


HANDLED = CommandOutcome(handled=True)
UNHANDLED = CommandOutcome(handled=False)
BREAK = CommandOutcome(handled=True, should_break=True)
