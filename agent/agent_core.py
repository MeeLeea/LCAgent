"""
Agent核心调度模块 - 基于LangChain 1.x + LangGraph
使用 langchain.agents.create_agent 实现工具调用，支持ReAct式推理
支持动态加载本地工具 + MCP Server工具
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

from typing_extensions import Self

from langchain.agents import create_agent
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langgraph.types import Command, Interrupt

from tools.mcp_loader import DEFAULT_CONFIG_FILE
from tools.mcp_pool import MCPPool, ServerStatus
from tools.skills import SkillManager, default_skills_dir
from tools.terminal_tools import UserRejectedCommandError
from tools.tool_wrapper import wrap_tools_with_timeout

from .compaction import (
    CompactionConfig,
    LCAgentCompactionMiddleware,
    LCAgentState,
)
from .exceptions import AgentStateError
from .llm_client import LLMClient
from .logging_config import TraceContext, generate_trace_id
from .memory import AgentMemory
from .message_utils import StreamHandler
from .metrics import MetricsCollector
from .session import SessionRegistry, SessionStore
from .skill_middleware import SkillInjectionMiddleware

logger = logging.getLogger(__name__)

@dataclass(frozen=True, slots=True)
class AgentTurnResult:
    """Typed result for one LangGraph turn."""

    status: Literal["completed", "interrupted", "cancelled"]
    output: str | None = None
    interrupts: list[Interrupt] = field(default_factory=list)

    @classmethod
    def completed(cls, output: str) -> AgentTurnResult:
        return cls(status="completed", output=output, interrupts=[])

    @classmethod
    def interrupted(cls, interrupts: list[Interrupt]) -> AgentTurnResult:
        return cls(status="interrupted", output=None, interrupts=interrupts)

    @classmethod
    def cancelled(cls, output: str) -> AgentTurnResult:
        return cls(status="cancelled", output=output, interrupts=[])

    @property
    def is_completed(self) -> bool:
        return self.status == "completed"

    @property
    def is_interrupted(self) -> bool:
        return self.status == "interrupted"


class AgentCore:
    """基于LangChain 1.x 的Agent核心调度器"""

    def _init_common(
        self,
        llm_client: LLMClient,
        name: str,
        max_iterations: int,
        verbose: bool,
        mcp_config_file: str | None,
        enable_mcp: bool,
        skills_dir: str | None,
        auto_match_skills: bool,
        max_context_messages: int,
        context_trim_keep: int,
        process_type: str | None,
        agent_prompt_file: str | None,
        max_execution_history: int,
        tool_timeout: float,
    ) -> None:
        self.name = name
        self.llm = llm_client
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.max_context_messages = max_context_messages
        self.context_trim_keep = context_trim_keep
        self.tool_timeout = tool_timeout if tool_timeout > 0 else None
        self._closed = False

        # 压缩配置：before_model 中间件自动触发 + manually_compact 手动触发
        # summary 存入 LangGraph state（随 checkpoint 持久化），天然 per-thread 隔离
        self.compaction_config = CompactionConfig.from_kwargs(
            max_context_messages=max_context_messages,
            context_trim_keep=context_trim_keep,
        )

        # 存储核心提示词（从配置加载或使用默认值）
        from .config import _load_agent_prompt

        self.agent_core_prompt = _load_agent_prompt(agent_prompt_file)

        # 本地工具（lazy import 打破潜在循环依赖）
        from tools import all_tools as _local_tools

        self.local_tools: list[BaseTool] = list(_local_tools)
        # MCP 工具(从 MCP Server 加载)
        self.mcp_tools: list[BaseTool] = []
        # 合并后的完整工具列表
        self.tools: list[BaseTool] = list(self.local_tools)

        # MCP 配置
        self.mcp_config_file = mcp_config_file or DEFAULT_CONFIG_FILE
        self.enable_mcp = enable_mcp

        # MCP 连接池（per-server 连接管理 + 健康探测 + 自动重连）
        self._mcp_pool = MCPPool(self.mcp_config_file)

        # 异步互斥锁：保护 tools / mcp_tools / agent_executor 等共享状态
        self._state_lock = asyncio.Lock()

        # 技能阅读(本地 .agents/skills)
        if skills_dir is None:
            skills_dir = default_skills_dir()
        self.skill_manager = SkillManager(skills_dir)
        self.auto_match_skills = auto_match_skills  # 任务开始时自动匹配注入
        # 技能注入中间件引用（在 _create_agent_executor 中创建）
        self._skill_middleware: SkillInjectionMiddleware | None = None

        # 运行时指标收集器（LLM tokens / 工具耗时 / 压缩统计）
        self._metrics = MetricsCollector()

        # 创建Agent（编译一次，所有会话共享同一编译图）
        self.agent_executor = None

        # 会话注册表（管理 checkpointer + Store 的会话状态）
        # 在 __init__/acreate 中创建 AgentMemory 后初始化
        self._session_registry: SessionRegistry | None = None
        self._session_store: SessionStore | None = SessionStore(
            max_history=max_execution_history
        )

        # 执行历史 max 条数（传递给 SessionStore 做裁剪；实例不再持有 deque）
        self._max_execution_history = max_execution_history

        self._compaction_middleware = None
        self._tools_signature = frozenset()

        # 流式事件处理器（组合模式）
        self.stream = StreamHandler(self)

    def __init__(
        self,
        llm_client: LLMClient,
        name: str = "LCAgent",
        memory_size: int = 10,
        long_term_memory_file: str | None = None,
        checkpoint_file: str | None = None,
        thread_id: str | None = None,
        max_iterations: int = 25,
        verbose: bool = True,
        mcp_config_file: str | None = None,
        enable_mcp: bool = True,
        skills_dir: str | None = None,
        auto_match_skills: bool = True,
        max_context_messages: int = 0,
        context_trim_keep: int = 12,
        process_type: str | None = None,
        agent_prompt_file: str | None = None,
        max_execution_history: int = 100,
        tool_timeout: float = 60.0,
    ):
        """
        初始化Agent核心

        Args:
            llm_client: LLM客户端实例
            memory_size: 仅兼容旧 API(checkpoint 不限容量)
            long_term_memory_file: 长期记忆 JSON 文件(用于 compress)
            checkpoint_file: SQLite checkpoint 文件路径(为 None 时用内存)
            thread_id: 会话线程 ID(为 None 时自动生成)
            max_iterations: Agent最大迭代次数(langgraph recursion_limit)
            verbose: 是否打印详细执行过程
            mcp_config_file: MCP servers 配置文件路径
            enable_mcp: 是否启用 MCP 工具加载
            max_context_messages: 长上下文裁剪阈值(0=关闭);超过则自动摘要并开新会话
            context_trim_keep: 裁剪时保留的最近消息条数
            process_type: 进程类型标识(server/scheduler/feishu)，用于多进程隔离
            agent_prompt_file: Agent核心提示词文件路径(为 None 时使用配置默认值)
            max_execution_history: 执行历史最大条数(防止内存泄漏)
            tool_timeout: 工具执行默认超时秒数(0=禁用超时)
        """
        self._init_common(
            llm_client=llm_client,
            name=name,
            max_iterations=max_iterations,
            verbose=verbose,
            mcp_config_file=mcp_config_file,
            enable_mcp=enable_mcp,
            skills_dir=skills_dir,
            auto_match_skills=auto_match_skills,
            max_context_messages=max_context_messages,
            context_trim_keep=context_trim_keep,
            process_type=process_type,
            agent_prompt_file=agent_prompt_file,
            max_execution_history=max_execution_history,
            tool_timeout=tool_timeout,
        )
        self.memory = AgentMemory(
            checkpoint_file=checkpoint_file,
            long_term_file=long_term_memory_file,
            thread_id=thread_id,
            short_term_size=memory_size,
            use_sqlite=checkpoint_file is not None,
            process_type=process_type,
        )
        self._init_session_registry()

        if self.enable_mcp:
            asyncio.run(self.areload_mcp_tools())

        self.agent_executor = self._create_agent_executor()

    @classmethod
    async def acreate(
        cls,
        llm_client: LLMClient,
        name: str = "LCAgent",
        memory_size: int = 10,
        long_term_memory_file: str | None = None,
        checkpoint_file: str | None = None,
        thread_id: str | None = None,
        max_iterations: int = 25,
        verbose: bool = True,
        mcp_config_file: str | None = None,
        enable_mcp: bool = True,
        skills_dir: str | None = None,
        auto_match_skills: bool = True,
        max_context_messages: int = 0,
        context_trim_keep: int = 12,
        process_type: str | None = None,
        agent_prompt_file: str | None = None,
        max_execution_history: int = 100,
        tool_timeout: float = 60.0,
    ) -> AgentCore:
        """在运行中的事件循环内创建 AgentCore。"""
        self = cls.__new__(cls)
        self._init_common(
            llm_client=llm_client,
            name=name,
            max_iterations=max_iterations,
            verbose=verbose,
            mcp_config_file=mcp_config_file,
            enable_mcp=enable_mcp,
            skills_dir=skills_dir,
            auto_match_skills=auto_match_skills,
            max_context_messages=max_context_messages,
            context_trim_keep=context_trim_keep,
            process_type=process_type,
            agent_prompt_file=agent_prompt_file,
            max_execution_history=max_execution_history,
            tool_timeout=tool_timeout,
        )
        self.memory = await AgentMemory.acreate(
            checkpoint_file=checkpoint_file,
            long_term_file=long_term_memory_file,
            thread_id=thread_id,
            short_term_size=memory_size,
            use_sqlite=checkpoint_file is not None,
            process_type=process_type,
        )
        self._init_session_registry()

        if self.enable_mcp:
            await self.areload_mcp_tools()

        self.agent_executor = self._create_agent_executor()
        return self

    @property
    def metrics(self) -> MetricsCollector:
        """运行时指标收集器（惰性初始化，兼容 object.__new__ 创建的测试实例）"""
        mc = getattr(self, "_metrics", None)
        if mc is None:
            mc = MetricsCollector()
            self._metrics = mc
        return mc

    def _init_session_registry(self) -> None:
        """从 AgentMemory 的 checkpointer 构建 SessionRegistry。

        SessionRegistry 与 AgentMemory 共享同一 checkpointer 和 SQLite 连接，
        负责会话生命周期管理（new/switch/list/delete）+ 消息读取 + 瞬态状态隔离。
        AgentMemory 退化为仅负责长期记忆（memory.json）。
        """
        self._session_registry = SessionRegistry(
            checkpointer=self.memory.get_checkpointer(),
            store=self._session_store or SessionStore(),
            process_type=self.memory.process_type,
            recursion_limit=self.max_iterations,
            async_conn=getattr(self.memory, "_async_conn", None),
        )
        # 同步当前会话指针
        self._session_registry.current_session_id = self.memory.thread_id

    @property
    def session(self) -> SessionRegistry:
        """会话注册表（管理 checkpointer + Store 的会话状态）。

        通过此属性访问会话管理 API：
        - ``agent.session.alist_sessions()``
        - ``agent.session.adelete_session(sid)``
        - ``agent.session.aget_messages(sid)``
        - ``agent.session.current_session_id``  (getter/setter)
        """
        if self._session_registry is None:
            # 兼容测试中通过 object.__new__ 创建的实例
            self._init_session_registry()
        return self._session_registry

    def _get_store(self) -> SessionStore:
        """获取 SessionStore（惰性创建，兼容 object.__new__ 创建的测试实例）。

        execution_history / recorded_call_ids / pending_interrupts 全部
        通过此 Store 按 session_id 隔离，AgentCore 实例不再持有这些可变状态。
        """
        store = getattr(self, "_session_store", None)
        if store is None:
            store = SessionStore()
            self._session_store = store
        return store

    def set_current_session(self, session_id: str) -> None:
        """设置当前会话 ID，同步 session 注册表与 memory。

        Args:
            session_id: 目标会话 ID
        """
        self.session.current_session_id = session_id
        self.memory.thread_id = session_id

    async def areload_mcp_tools(self) -> int:
        """
        异步重新加载 MCP 工具（通过 MCPPool 全量重连）

        使用 _state_lock 保护 tools 和 agent_executor 的并发修改。
        仅当工具签名（工具名集合）变化时才重建 Graph，否则只更新系统提示词。

        对于单个 server 的重连，推荐使用 areload_mcp_server(name)。

        Returns:
            加载到的 MCP 工具数量
        """
        self._ensure_not_closed()
        async with self._state_lock:
            try:
                old_signature = getattr(self, "_tools_signature", frozenset())
                count = await self._async_load_mcp_tools()
                # 合并工具列表
                self.tools = list(self.local_tools) + list(self.mcp_tools)
                new_signature = frozenset(t.name for t in self.tools)

                if getattr(self, "agent_executor", None) is not None:
                    if new_signature != old_signature:
                        # 工具列表变化，必须重建 Graph
                        await self._arebuild_agent_executor()
                return count
            except Exception:
                logger.exception("MCP 重新加载失败")
                return 0

    async def areload_mcp_server(self, name: str) -> bool:
        """重连单个 MCP server（不影响其他 server）

        Args:
            name: server 名称

        Returns:
            True=重连成功
        """
        self._ensure_not_closed()
        async with self._state_lock:
            try:
                old_signature = getattr(self, "_tools_signature", frozenset())
                success = await self._mcp_pool.reload_server(name)
                if not success and self.verbose:
                    logger.warning("MCP %s: 重连失败或已移除", name)
                # 从池中获取最新工具列表
                self.mcp_tools = self._mcp_pool.get_all_tools()
                self.tools = list(self.local_tools) + list(self.mcp_tools)
                new_signature = frozenset(t.name for t in self.tools)

                if getattr(self, "agent_executor", None) is not None:
                    if new_signature != old_signature:
                        await self._arebuild_agent_executor()
                return success
            except Exception:
                logger.exception("MCP %s: 重连失败", name)
                return False

    async def _async_load_mcp_tools(self) -> int:
        """通过 MCPPool 初始化所有 MCP 连接"""
        tool_count = await self._mcp_pool.initialize()
        self.mcp_tools = self._mcp_pool.get_all_tools()
        if self.mcp_tools and self.verbose:
            # 按 server 分组展示
            for info in self._mcp_pool.get_server_infos():
                if info.status == ServerStatus.CONNECTED:
                    logger.info("MCP %s: %d 个工具 (%s)",
                                info.name, info.tool_count, ", ".join(info.tool_names))
                elif info.status == ServerStatus.ERROR:
                    logger.warning("MCP %s: 连接失败 - %s", info.name, info.last_error)
        elif self.verbose:
            logger.info("MCP 未加载到任何工具(可能配置为空或服务器未启用)")
        return tool_count

    def _create_agent_executor(
        self,
        skill_block: str = "",
    ):
        """创建LangGraph ReAct Agent（仅在工具列表或 LLM 变化时调用）

        集成压缩中间件（before_model 自动触发增量摘要 + 工具输出 Prune）+
        技能注入中间件（awrap_model_call 从 state 读取 active_skills 并注入提示词）。

        关键设计：system_prompt 传入静态字符串（不再使用可变 SystemMessage），
        技能隔离由 SkillInjectionMiddleware + LCAgentState.active_skills 保证
        （随 checkpoint per-thread 隔离），所有会话共享同一编译图。

        Args:
            skill_block: 构建时初始技能指引块（仅用于日志/调试，实际注入由中间件完成）
        """
        chat_model = self.llm.get_chat_model()

        # 静态系统提示词（技能注入由 SkillInjectionMiddleware 在 model 调用时完成）
        system_prompt = self._get_system_prompt()

        # 压缩中间件：消息超阈值时自动增量摘要 + Prune 工具输出
        compaction_middleware = LCAgentCompactionMiddleware(
            model=chat_model,
            config=self.compaction_config,
            on_compaction=self.metrics.record_compaction,
        )

        # 技能注入中间件：从 state.active_skills 读取技能并注入 system prompt
        skill_middleware = SkillInjectionMiddleware(
            skill_manager=self.skill_manager,
            auto_match=self.auto_match_skills,
        )
        self._skill_middleware = skill_middleware

        # create_agent 直接返回可调用的agent
        wrapped_tools = wrap_tools_with_timeout(self.tools, self.tool_timeout)
        agent = create_agent(
            model=chat_model,
            tools=wrapped_tools,
            system_prompt=system_prompt,
            checkpointer=self.memory.get_checkpointer(),
            state_schema=LCAgentState,
            middleware=[compaction_middleware, skill_middleware],
        )
        # 保存中间件引用，供手动压缩使用
        self._compaction_middleware = compaction_middleware
        # 记录工具签名，用于检测工具列表是否变化
        self._tools_signature = frozenset(t.name for t in self.tools)
        return agent

    async def _arebuild_agent_executor(self) -> None:
        """重建 Agent（仅在工具列表或 LLM 变化时使用）

        技能变化不需要重建——由 SkillInjectionMiddleware 在 model 调用时
        从 state 动态读取，无需重建 Graph。
        此方法仅在以下场景调用：
        - MCP 工具列表变化（areload_mcp_tools 检测到工具签名不同）
        - LLM 切换（aswitch_llm，model 对象变化）

        所有会话共享同一编译图，重建后对所有会话即时生效。

        注意：调用方必须已持有 _state_lock（此方法不再自行加锁，
        避免在 areload_mcp_tools 内部调用时死锁）。
        """
        self.agent_executor = self._create_agent_executor()

    async def _ahandle_turn_completion(
        self,
        turn: AgentTurnResult,
        config: dict[str, Any],
        mode: str,
        user_message: str | None = None,
        save_assistant: bool = True,
        important: bool = False
    ) -> None:
        """统一处理 turn 完成后的状态更新和记忆存储

        根据 turn 状态更新 interrupt 状态，并选择性地保存消息到记忆。

        Args:
            turn: Agent 执行结果
            config: LangGraph 配置对象
            mode: 执行模式 ('run', 'chat', 等)
            user_message: 用户消息（如需保存到记忆）
            save_assistant: 是否保存 assistant 消息到记忆
            important: assistant 消息是否标记为重要
        """
        if turn.is_interrupted:
            await self._acapture_pending_interrupt(config, mode)
        elif turn.is_completed:
            # 只清理当前线程的中断状态，避免清掉其他会话的挂起中断
            await self._aclear_pending_interrupt(self._thread_id_from_config(config))
            if user_message:
                await self.memory.aadd("user", user_message)
            if save_assistant and turn.output:
                metadata = {"important": True} if important else {}
                await self.memory.aadd("assistant", turn.output, metadata)

    def _get_system_prompt(self) -> str:
        """获取系统提示词。

        技能注入由 SkillInjectionMiddleware 在 model 调用时从 state 读取，
        历史对话摘要由压缩中间件作为 SystemMessage 放入 messages 列表头部，
        两者均与 system_prompt 分离，避免实例级共享状态污染。
        """
        return self.agent_core_prompt

    def _invoke_config(self, thread_id: str | None = None) -> dict[str, Any]:
        """构建 LangGraph 调用 config。

        优先使用 SessionRegistry（含 recursion_limit），兼容测试中无 registry 的场景。

        Args:
            thread_id: 目标会话线程 ID。为 None 时使用 session 当前会话
                       （兼容 CLI/旧调用与测试中对 get_config 的无参 mock）。

        Returns:
            config 字典（含 configurable.thread_id 与 recursion_limit）
        """
        reg = getattr(self, "_session_registry", None)
        if reg is not None:
            sid = thread_id or reg.current_session_id
            return reg.get_context(sid).config
        # Fallback：测试中通过 object.__new__ 创建的实例无 session_registry
        if thread_id is None:
            cfg = self.memory.get_config()
        else:
            cfg = self.memory.get_config(thread_id=thread_id)
        return {**cfg, "recursion_limit": self.max_iterations}

    def _thread_id_from_config(self, config: dict[str, Any]) -> str | None:
        configurable = config.get("configurable")
        if isinstance(configurable, dict):
            thread_id = configurable.get("thread_id")
            if isinstance(thread_id, str):
                return thread_id
        return None

    @contextmanager
    def _temp_verbose(self, verbose: bool):
        """临时设置 verbose 状态的上下文管理器
        
        Args:
            verbose: 临时要设置的 verbose 值
            
        Yields:
            None
            
        Example:
            with self._temp_verbose(False):
                # 这里 self.verbose 为 False
                self._do_something()
            # 这里 self.verbose 恢复原值
        """
        original = self.verbose
        self.verbose = verbose
        try:
            yield
        finally:
            self.verbose = original

    async def _acapture_pending_interrupt(self, config: dict[str, Any], mode: str) -> None:
        """记录挂起中断模式到 SessionStore（per-session 隔离）。"""
        thread_id = self._thread_id_from_config(config)
        if thread_id:
            await self._get_store().aset_interrupt_mode(thread_id, mode)

    async def _aclear_pending_interrupt(self, thread_id: str | None = None) -> None:
        """清除挂起中断状态（per-session 隔离）。

        Args:
            thread_id: 指定会话线程 ID。为 None 时清除当前会话的中断。
        """
        sid = thread_id or self.session.current_session_id
        await self._get_store().aclear_interrupt(sid)

    async def _arecord_tool_steps(
        self,
        result_messages: list[BaseMessage],
        input_msg: HumanMessage,
        session_id: str,
    ) -> None:
        """异步记录工具调用步骤到 SessionStore（per-session 隔离）。

        LangGraph 会返回当前线程的完整消息历史，按 tool_call id 去重避免重复记账。
        执行历史和去重集合均通过 SessionStore 按 session_id 隔离，
        AgentCore 实例不再持有 execution_history deque 或 _recorded_tool_call_ids set。
        """
        store = self._get_store()
        recorded_ids = await store.aget_recorded_call_ids(session_id)
        history = await store.aget_history(session_id)
        step_count = len(history)

        new_entries: list[dict[str, Any]] = []
        new_ids: set[str] = set()
        new_entries_by_call_id: dict[str, dict[str, Any]] = {}
        for msg in result_messages:
            if msg in (input_msg,):
                continue

            if isinstance(msg, AIMessage):
                tool_calls = getattr(msg, "tool_calls", None)
                if tool_calls:
                    for tc in tool_calls:
                        call_id = tc.get("id")
                        if isinstance(call_id, str) and call_id in recorded_ids:
                            continue
                        step_count += 1
                        if self.verbose:
                            logger.debug("步骤 %d | 工具: %s | 输入: %s",
                                         step_count, tc.get("name", "unknown"), tc.get("args", {}))
                        entry = {
                            "step": step_count,
                            "tool": tc.get("name"),
                            "input": tc.get("args"),
                            "observation": ""
                        }
                        new_entries.append(entry)
                        if isinstance(call_id, str):
                            new_ids.add(call_id)
                            new_entries_by_call_id[call_id] = entry

                # 记录 LLM token 用量（从 response_metadata 提取）
                # getattr 保护：测试中通过 object.__new__ 创建的实例可能没有 llm
                _llm = getattr(self, "llm", None)
                self.metrics.extract_and_record_llm_usage(
                    msg,
                    provider=getattr(_llm, "provider", ""),
                    model=getattr(_llm, "model", "") or "",
                )

            elif hasattr(msg, "content") and hasattr(msg, "tool_call_id"):
                call_id = msg.tool_call_id
                entry = new_entries_by_call_id.get(call_id)
                if entry is not None:
                    entry["observation"] = str(msg.content)[:500]
                if self.verbose and entry is not None:
                    logger.debug("结果: %s", str(msg.content)[:200])

                # 记录工具调用指标（检测超时和失败）
                if entry is not None:
                    content_str = str(msg.content)
                    timed_out = '"error": "tool_timeout"' in content_str
                    success = not timed_out and getattr(msg, "status", "success") != "error"
                    self.metrics.record_tool_call(
                        name=entry.get("tool", "unknown"),
                        success=success,
                        timed_out=timed_out,
                    )

        # 批量写入 SessionStore（一次读改写，减少 Store 往返）
        if new_entries:
            await store.aextend_history(session_id, new_entries)
        if new_ids:
            await store.aadd_recorded_call_ids(session_id, new_ids)

    def _parse_turn_result(self, result: dict[str, Any]) -> AgentTurnResult:
        interrupts = result.get("__interrupt__")
        if interrupts:
            return AgentTurnResult.interrupted(list(interrupts))

        # 找到最后一条有内容的 AIMessage
        messages = result.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                return AgentTurnResult.completed(str(msg.content))
        return AgentTurnResult.completed("")

    async def _arepair_rejected_tool_calls(
        self,
        config: dict[str, Any],
    ) -> None:
        """异步修复 checkpoint 中未完成的工具调用（补齐取消结果）。

        使用 LangGraph 的异步 state API，避免阻塞事件循环。

        Args:
            config: LangGraph 配置对象（含 configurable.thread_id）
        """
        graph = self.agent_executor
        try:
            state = await graph.aget_state(config)
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            logger.warning("读取 checkpoint 状态失败: %s", error, exc_info=True)
            return

        if state is None:
            return

        messages = list(getattr(state, "values", {}).get("messages", []))
        existing_results = [message for message in messages if isinstance(message, ToolMessage)]
        answered_ids = {message.tool_call_id for message in existing_results}
        repairs = []
        for message in messages:
            if not isinstance(message, AIMessage):
                continue
            for tool_call in message.tool_calls:
                call_id = tool_call.get("id")
                if not isinstance(call_id, str) or call_id in answered_ids:
                    continue
                repairs.append(
                    ToolMessage(
                        content="用户拒绝执行危险命令，工具调用已取消。",
                        name=tool_call.get("name"),
                        tool_call_id=call_id,
                        status="error",
                    )
                )
                answered_ids.add(call_id)
        if not repairs:
            return

        try:
            await graph.aupdate_state(
                config,
                {"messages": [*existing_results, *repairs]},
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            logger.warning("修复 checkpoint 状态失败: %s", error, exc_info=True)

    async def _ahandle_rejected_command(
        self,
        config: dict[str, Any],
    ) -> AgentTurnResult:
        """异步处理用户拒绝执行危险命令的情况

        当工具调用被用户拒绝时，异步修复 checkpoint 状态并返回取消结果。

        Args:
            config: LangGraph 配置对象

        Returns:
            状态为 'cancelled' 的 AgentTurnResult
        """
        await self._arepair_rejected_tool_calls(config)
        await self._aclear_pending_interrupt(self._thread_id_from_config(config))
        return AgentTurnResult.cancelled("用户已拒绝执行危险命令，当前任务已取消。")

    async def arun_structured(self, task: str, thread_id: str | None = None) -> AgentTurnResult:
        """异步执行任务（结构化入口）

        使用 LangGraph 异步 invoke，避免阻塞事件循环。
        压缩由 before_model 中间件自动触发，无需手动调用。
        技能注入由 SkillInjectionMiddleware 从 state 读取，无需手动更新。

        Args:
            task: 任务描述
            thread_id: 目标会话线程 ID（为 None 时使用当前会话）
        """
        self._ensure_not_closed()
        tid = thread_id or getattr(self.memory, "thread_id", "-")
        with TraceContext(trace_id=generate_trace_id(), thread_id=tid):
            logger.info("arun_structured: %s", task[:100])
            config = self._invoke_config(thread_id)
            input_msg = HumanMessage(content=task)

            try:
                result = await self.agent_executor.ainvoke({"messages": [input_msg]}, config=config)
            except UserRejectedCommandError:
                return await self._ahandle_rejected_command(config)

            await self._arecord_tool_steps(result.get("messages", []), input_msg, tid)
            turn = self._parse_turn_result(result)
            await self._ahandle_turn_completion(turn, config, "run", task, important=True)
            self.metrics.increment_turn()

        return turn

    async def achat_structured(self, message: str, thread_id: str | None = None) -> AgentTurnResult:
        """异步对话（结构化入口）

        压缩由 before_model 中间件自动触发，无需手动调用。
        技能注入由 SkillInjectionMiddleware 从 state 读取，无需手动更新。

        Args:
            message: 用户消息
            thread_id: 目标会话线程 ID（为 None 时使用当前会话）
        """
        self._ensure_not_closed()
        tid = thread_id or getattr(self.memory, "thread_id", "-")
        with TraceContext(trace_id=generate_trace_id(), thread_id=tid):
            logger.info("achat_structured: %s", message[:100])
            config = self._invoke_config(thread_id)

            with self._temp_verbose(False):
                try:
                    result = await self.agent_executor.ainvoke(
                        {"messages": [HumanMessage(content=message)]},
                        config=config,
                    )
                except UserRejectedCommandError:
                    return await self._ahandle_rejected_command(config)

            turn = self._parse_turn_result(result)
            await self._ahandle_turn_completion(turn, config, "chat", message)
            self.metrics.increment_turn()

            return turn

    async def aresume_structured(
        self,
        payload: dict[str, Any],
        thread_id: str | None = None,
    ) -> AgentTurnResult:
        """异步恢复中断会话（结构化入口）

        Args:
            payload: 恢复数据
            thread_id: 目标会话线程 ID（为 None 时使用当前会话）
        """
        config = self._invoke_config(thread_id)
        tid = thread_id or self._thread_id_from_config(config)

        # 从 SessionStore 读取 per-session 中断模式
        mode = await self._get_store().aget_interrupt_mode(tid)
        if mode is None:
            mode = "chat"  # 默认 chat 模式（无追踪记录时尝试恢复）

        try:
            result = await self.agent_executor.ainvoke(Command(resume=payload), config=config)
        except UserRejectedCommandError:
            return await self._ahandle_rejected_command(config)

        turn = self._parse_turn_result(result)

        if turn.is_interrupted:
            await self._acapture_pending_interrupt(config, mode)
        elif turn.is_completed:
            await self._aclear_pending_interrupt(tid)
            if mode == "run" and turn.output:
                await self.memory.aadd("assistant", turn.output, {"important": True})

        return turn

# ============ 流式接口（委托给 StreamHandler） ============

    async def astream_chat(self, message: str, thread_id: str | None = None):
        """流式对话，委托给 self.stream。事件格式见 StreamHandler.astream_chat。

        Args:
            message: 用户消息
            thread_id: 目标会话线程 ID（为 None 时使用当前会话）
        """
        async for ev in self.stream.astream_chat(message, thread_id=thread_id):
            yield ev

    async def astream_resume(self, payload: dict[str, Any], thread_id: str | None = None):
        """流式恢复中断会话，委托给 self.stream。

        Args:
            payload: 恢复数据
            thread_id: 目标会话线程 ID（为 None 时使用当前会话）
        """
        async for ev in self.stream.astream_resume(payload, thread_id=thread_id):
            yield ev

    async def arun(self, task: str) -> str:
        """异步执行任务（推荐使用）

        使用 arun_structured 异步执行，支持事件循环内调用。

        Args:
            task: 任务描述

        Returns:
            执行结果
        """
        logger.info("开始执行任务: %s", task)

        try:
            turn = await self.arun_structured(task)
        except Exception as e:
            error_msg = f"任务执行失败: {e!s}"
            logger.exception("%s", error_msg)
            return error_msg

        if turn.is_interrupted:
            raise RuntimeError("Agent turn interrupted; resume with aresume().")

        output = turn.output or ""
        logger.info("最终答案: %s", output)
        return output

    async def achat(self, message: str) -> str:
        """异步对话模式（推荐使用）

        Args:
            message: 用户消息

        Returns:
            助手回复
        """
        try:
            turn = await self.achat_structured(message)
        except Exception as e:
            if e.__class__.__name__ in {"GraphInterrupt", "NodeInterrupt"}:
                raise
            logger.warning("achat 降级到 fallback: %s", e, exc_info=True)
            return await self._afallback_chat(message)

        if turn.is_interrupted:
            raise RuntimeError("Agent turn interrupted; resume with aresume().")

        return turn.output or ""

    async def aresume(self, payload: dict[str, Any]) -> str:
        """异步恢复中断会话（推荐使用）

        Args:
            payload: 恢复数据

        Returns:
            助手回复
        """
        turn = await self.aresume_structured(payload)
        if turn.is_interrupted:
            raise RuntimeError("Agent turn interrupted; resume with aresume().")
        return turn.output or ""

    async def _afallback_chat(self, message: str) -> str:
        """Agent 执行失败时降级为纯 LLM 对话（异步版本）"""
        history = await self.session.aget_short_term()
        await self.memory.aadd("user", message)
        response = self.llm.chat_with_history(
            user_input=message,
            history=history,
            system_prompt="你是一个有帮助的AI助手，请用中文回答。"
        )
        await self.memory.aadd("assistant", response)
        return response

    async def acot(self, task: str) -> str:
        """
        CoT链式思考模式（不调用工具，纯推理）- 异步版本

        Args:
            task: 任务描述

        Returns:
            推理结果
        """
        logger.info("链式思考模式: %s", task)

        system_prompt = (
            "你是一个智能助手，使用链式思考(Chain of Thought)来解决问题。\n"
            "请一步步分析问题，最后给出结论。\n"
            "请用中文回答。"
        )

        # CoT 模式不调用工具,直接用 LLM + checkpoint 历史
        context = await self.session.aget_short_term() + self.memory.get_long_term(3)
        response = self.llm.chat_with_history(
            user_input=task,
            history=context,
            system_prompt=system_prompt,
            temperature=0.7
        )

        logger.info("最终答案: %s", response)

        # 存入记忆
        await self.memory.aadd("user", task)
        await self.memory.aadd("assistant", response, {"important": True})

        return response

    async def aswitch_llm(self, llm_client: LLMClient):
        """
        异步切换LLM提供商

        使用 _state_lock 保护共享状态。

        Args:
            llm_client: 新的LLM客户端实例
        """
        async with self._state_lock:
            self.llm = llm_client
            await self._arebuild_agent_executor()

    # ============ 团队角色切换(Team Role Switch) ============

    async def arebuild_from_team_dir(self, agent_name: str, *, task: str = "") -> None:
        """按 team/ 角色文件夹名重建主对话 Agent 的角色(唯一对外入口)

        具体实现委托给 agent.role_sw.arebuild_agent_from_team_dir,
        就地把当前 AgentCore 切换为目标角色的提示词/LLM。

        Args:
            agent_name: team/ 下的角色文件夹名(如 "manager"/"worker")
            task: 可选任务描述,用于切换后自动匹配注入技能

        Raises:
            KeyError: 角色文件夹不存在或缺少必需文件
            FileNotFoundError: AGENT.md 读取失败(内容为空)
        """
        from agent.role_sw import arebuild_agent_from_team_dir

        await arebuild_agent_from_team_dir(self, agent_name, task=task)

    # ============ 技能阅读(Skills) ============

    def list_skills(self) -> list[dict[str, str]]:
        """列出所有本地可用技能"""
        return self.skill_manager.list_skills()

    async def aload_skill(self, name: str, thread_id: str | None = None) -> bool:
        """异步加载技能到指定会话（写入 LCAgentState.active_skills）

        技能名存入 per-thread state（随 checkpoint 持久化），
        由 SkillInjectionMiddleware 在 model 调用时读取并注入提示词。
        无需重建 Graph，也无需维护实例级 active_skills。

        Args:
            name: 技能名(目录名或 frontmatter name)
            thread_id: 目标会话线程 ID（为 None 时使用当前会话）

        Returns:
            True=成功加载, False=技能不存在
        """
        if self.skill_manager.get_skill(name) is None:
            return False
        config = self._invoke_config(thread_id)
        state = await self.agent_executor.aget_state(config)
        current: set[str] = set(
            (state.values if state and state.values else {}).get("active_skills") or []
        )
        current.add(name)
        await self.agent_executor.aupdate_state(
            config, {"active_skills": sorted(current)}
        )
        return True

    async def aclear_skills(self, thread_id: str | None = None):
        """异步清空指定会话的技能（写入空列表到 state）

        Args:
            thread_id: 目标会话线程 ID（为 None 时使用当前会话）
        """
        config = self._invoke_config(thread_id)
        await self.agent_executor.aupdate_state(config, {"active_skills": []})

    def set_auto_match(self, enabled: bool):
        """开关任务自动匹配技能（更新中间件标志）"""
        self.auto_match_skills = enabled
        if self._skill_middleware is not None:
            self._skill_middleware.auto_match = enabled

    # ============ 长上下文裁剪 ============

    async def manually_compact(
        self,
        force: bool = False,
        thread_id: str | None = None,
    ) -> dict[str, Any] | None:
        """手动触发一次上下文压缩（CLI 命令 compact 调用）

        与 before_model 中间件使用相同的压缩逻辑（增量摘要 + 工具输出 Prune），
        但通过 update_state（asyncio.to_thread 包裹）直接写入 checkpoint，
        不依赖 LangGraph 中间件上下文。

        Args:
            force: 为 True 时跳过 max_messages 阈值检查，允许在消息数
                   未超阈值时强制压缩（仍需消息数 > keep_recent 才能安全切割）。
            thread_id: 目标会话线程 ID（为 None 时使用当前会话）

        Returns:
            {"summary": str, "messages_before": int, "messages_after": int} 或 None
        """
        msgs = await self.session.aget_messages(session_id=thread_id)
        if not msgs:
            return None

        # 读取当前 state 中的已有摘要
        config = self._invoke_config(thread_id)
        executor = self.agent_executor
        state = await executor.aget_state(config)
        existing_summary = ""
        if state and state.values:
            existing_summary = state.values.get("summary", "") or ""

        # 调用中间件的手动压缩接口
        mw = getattr(self, "_compaction_middleware", None)
        if mw is None:
            return None

        update = await mw.arun_compaction(msgs, existing_summary=existing_summary, force=force)
        if update is None:
            if self.verbose:
                logger.debug("压缩: 消息不足或无法安全切割，无需压缩")
            return None

        import time as _time
        _compact_start = _time.time()

        messages_before = len(msgs)
        await executor.aupdate_state(config, update)
        messages_after = len(update["messages"]) - 1  # 减去 RemoveMessage 标记
        summary_length = len(update.get("summary", ""))

        # 记录压缩指标
        self.metrics.record_compaction(
            trigger="manual",
            messages_before=messages_before,
            messages_after=messages_after,
            summary_length=summary_length,
            duration_ms=(_time.time() - _compact_start) * 1000,
        )

        if self.verbose:
            logger.info("压缩: %d → %d 条消息，摘要已更新", messages_before, messages_after)

        return {
            "summary": update["summary"],
            "messages_before": messages_before,
            "messages_after": messages_after,
        }

    def get_available_tools(self) -> list[str]:
        """获取可用工具名称列表"""
        return [t.name for t in self.tools]

    async def aget_execution_history(self, session_id: str | None = None) -> list[dict[str, Any]]:
        """获取执行历史（per-session 隔离）。

        Args:
            session_id: 目标会话 ID。为 None 时使用当前会话。
        """
        sid = session_id or self.session.current_session_id
        return await self._get_store().aget_history(sid)

    async def aclear_history(self, session_id: str | None = None) -> None:
        """清空执行历史和去重集合（per-session 隔离）。

        Args:
            session_id: 目标会话 ID。为 None 时使用当前会话。
        """
        sid = session_id or self.session.current_session_id
        await self._get_store().aclear_history(sid)

    async def aget_memory_summary(self) -> dict[str, Any]:
        """异步获取记忆摘要统计（当前会话消息数、长期记忆条数、checkpoint 信息）"""
        session_info = await self.session.asummarize()
        return {
            "thread_id": self.session.current_session_id,
            "checkpoint_messages": session_info["checkpoint_messages"],
            "checkpoint_backend": "sqlite" if self.memory.use_sqlite else "memory",
            "checkpoint_file": self.memory.checkpoint_file or "(内存)",
            "long_term_count": len(self.memory.long_term_memory),
            "total_threads": session_info["total_sessions"],
        }

    async def acompress_memory(self) -> dict[str, Any]:
        """异步压缩长期记忆: 委托给 memory.acompress_memory,用 LLM 生成摘要。

        LLMClient 无异步 chat 接口,阻塞的 LLM 调用放入线程池执行,避免阻塞事件循环。
        """
        async def _asummarize_text(text: str, prompt: str) -> str:
            def _sync_call() -> str:
                try:
                    return self.llm.chat([
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": text},
                    ]).strip()
                except Exception:
                    return ""

            return await asyncio.to_thread(_sync_call)

        return await self.memory.acompress_memory(_asummarize_text)

    # ============ 生命周期管理 ============

    async def aclose(self) -> None:
        """优雅关闭：释放 MCP 连接池、Checkpoint DB 等资源

        关闭后 AgentCore 不再可用，调用任何方法会抛 AgentStateError。
        可安全重复调用（幂等）。
        """
        if self._closed:
            return
        self._closed = True

        # 1. 关闭 MCP 连接池
        try:
            await self._mcp_pool.close()
        except Exception as e:
            logger.warning("MCP 连接池关闭异常: %s", e, exc_info=True)

        # 2. 关闭 Checkpoint DB 连接
        try:
            await self.memory.aclose()
        except Exception as e:
            logger.warning("Checkpoint DB 关闭异常: %s", e, exc_info=True)

        # 3. SessionStore 中的 per-session 状态（history / interrupts）随 Store GC 自动回收
        #    无需显式清理实例级可变状态（已迁移至 SessionStore）

        logger.info("AgentCore 资源已释放")

    def _ensure_not_closed(self) -> None:
        """检查 Agent 是否已关闭，已关闭则抛异常"""
        if getattr(self, "_closed", False):
            raise AgentStateError("AgentCore 已关闭，不再可用")

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.aclose()
