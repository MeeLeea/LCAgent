"""
Agent核心调度模块 - 基于LangChain 1.x + LangGraph
使用 langchain.agents.create_agent 实现工具调用，支持ReAct式推理
支持动态加载本地工具 + MCP Server工具

模块结构（按职责拆分）：
- agent_core.py      主类：构造/生命周期/共享工具方法/聚合入口
- session_mgmt.py    SessionMgmt：会话/Store/中断状态管理
- mcp_tools.py       McpTools：MCP 工具加载
- graph_builder.py   GraphBuilder：executor 构建/重建 + LLM 切换
- streaming.py       Streaming：事件流引擎（arun/aresume_events）
- interrupts.py      Interrupts：中断检查/恢复命令/拒绝处理
- turn_runners.py    TurnRunners：结构化执行入口 + 工具步骤记账
- skill_ops.py       SkillOps：技能加载/清理 + 手动压缩
- turn_types.py      AgentTurnResult（避免 mixin 反向导入 agent_core）
- skill_mw.py        技能注入中间件 SkillInjectionMW
- tool_error_mw.py   工具错误纠错中间件 ToolExecutionErrorMW
- workspace_mw.py    工作空间安全中间件 WorkspaceSecurityMW
- role_sw.py         团队角色切换
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import contextmanager
from typing import Any, Self

from langchain_core.tools import BaseTool

from llm.llm_client import LLMClient
from session import SessionStore
from tools.mcp_loader import DEFAULT_CONFIG_FILE
from tools.mcp_pool import MCPPool
from tools.skills import SkillManager, default_skills_dir
from utils.compaction import CompactionConfig
from utils.exceptions import AgentStateError
from utils.metrics import MetricsCollector

from .graph_builder import GraphBuilder
from .interrupts import Interrupts
from .mcp_tools import McpTools
from .session_mgmt import SessionMgmt
from .skill_ops import SkillOps
from .skill_mw import SkillInjectionMW
from .streaming import Streaming
from .tool_error_mw import ToolExecutionErrorMW
from .turn_runners import TurnRunners
from .turn_types import AgentTurnResult
from .workspace_mw import WorkspaceSecurityMW

logger = logging.getLogger(__name__)


class AgentCore(
    SessionMgmt,
    McpTools,
    GraphBuilder,
    Streaming,
    Interrupts,
    TurnRunners,
    SkillOps,
):
    """基于LangChain 1.x 的Agent核心调度器

    本类仅保留：
    - 构造/生命周期（_init_common / __init__ / acreate / aclose）
    - 跨 mixin 共享的私有工具方法（_invoke_config / _thread_id_from_config / _temp_verbose）
    - 独立方法（acot / 历史查询）
    """

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
        short_term_size: int,
        checkpointer: Any | None = None,
        store: Any | None = None,
        extra_middleware: list | None = None,
        initial_thread_id: str | None = None,
        async_conn: Any | None = None,
    ) -> None:
        self.name = name
        self.llm = llm_client
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.max_context_messages = max_context_messages
        self.context_trim_keep = context_trim_keep
        self.tool_timeout = tool_timeout if tool_timeout > 0 else None
        self._short_term_size = short_term_size
        self._closed = False

        # 压缩配置：before_model 中间件自动触发 + manually_compact 手动触发
        # summary 存入 LangGraph state（随 checkpoint 持久化），天然 per-thread 隔离
        self.compaction_config = CompactionConfig.from_kwargs(
            max_context_messages=max_context_messages,
            context_trim_keep=context_trim_keep,
        )

        # 存储核心提示词（从配置加载或使用默认值）
        from llm.config import _load_agent_prompt

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
        self._skill_middleware: SkillInjectionMW | None = None

        # 运行时指标收集器（LLM tokens / 工具耗时 / 压缩统计）
        self._metrics = MetricsCollector()

        # 创建Agent（编译一次，所有会话共享同一编译图）
        self.agent_executor = None

        # 会话注册表（管理 checkpointer + Store 的会话状态）
        # 在 __init__/acreate 中调用 _init_session_registry 后初始化
        self._session_registry = None
        self._session_manager = None
        self._memory_manager: Any = None
        self._session_store: SessionStore | None = SessionStore(
            max_history=max_execution_history
        )

        # LangGraph 原语（由入口程序 / MemoryContext 注入）
        self._checkpointer = checkpointer
        self._store = store
        self._extra_middleware = extra_middleware or []
        self._process_type = process_type
        self._initial_thread_id = initial_thread_id
        self._async_conn = async_conn

        # 执行历史 max 条数（传递给 SessionStore 做裁剪；实例不再持有 deque）
        self._max_execution_history = max_execution_history

        self._compaction_middleware = None
        self._tools_signature = frozenset()

    def __init__(
        self,
        llm_client: LLMClient,
        name: str = "LCAgent",
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
        short_term_size: int = 10,
        checkpointer: Any | None = None,
        store: Any | None = None,
        extra_middleware: list | None = None,
        initial_thread_id: str | None = None,
        async_conn: Any | None = None,
    ):
        """
        初始化Agent核心

        Args:
            llm_client: LLM客户端实例
            max_iterations: Agent最大迭代次数(langgraph recursion_limit)
            verbose: 是否打印详细执行过程
            mcp_config_file: MCP servers 配置文件路径
            enable_mcp: 是否启用 MCP 工具加载
            skills_dir: 技能目录路径
            auto_match_skills: 任务开始时是否自动匹配注入技能
            max_context_messages: 长上下文裁剪阈值(0=关闭);超过则自动摘要并开新会话
            context_trim_keep: 裁剪时保留的最近消息条数
            process_type: 进程类型标识(server/scheduler/feishu)，用于多进程隔离
            agent_prompt_file: Agent核心提示词文件路径(为 None 时使用配置默认值)
            max_execution_history: 执行历史最大条数(防止内存泄漏)
            tool_timeout: 工具执行默认超时秒数(0=禁用超时)
            short_term_size: 短期上下文窗口消息条数（对应配置键 latest_msg_cnt，
                             传给 SessionRegistry.aget_short_term 兜底）
            checkpointer: LangGraph checkpointer 原语（由入口程序 / MemoryContext 注入）
            store: LangGraph store 原语（长期记忆 Store）
            extra_middleware: 额外中间件列表（如记忆读写中间件）
            initial_thread_id: 初始会话线程 ID
            async_conn: 异步数据库连接（供 SessionRegistry 使用）
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
            short_term_size=short_term_size,
            checkpointer=checkpointer,
            store=store,
            extra_middleware=extra_middleware,
            initial_thread_id=initial_thread_id,
            async_conn=async_conn,
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
        short_term_size: int = 10,
        checkpointer: Any | None = None,
        store: Any | None = None,
        extra_middleware: list | None = None,
        initial_thread_id: str | None = None,
        async_conn: Any | None = None,
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
            short_term_size=short_term_size,
            checkpointer=checkpointer,
            store=store,
            extra_middleware=extra_middleware,
            initial_thread_id=initial_thread_id,
            async_conn=async_conn,
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
            sid = self._current_sid(thread_id)
            ctx = reg.get_context(sid)
            ws = ctx.config.get("configurable", {}).get("workspace_path")
            logger.info("_invoke_config: sid=%s workspace_path=%s", sid, ws)
            return ctx.config
        # Fallback：测试中通过 object.__new__ 创建的实例无 session_registry
        sid = self._current_sid(thread_id)
        return {
            "configurable": {"thread_id": sid},
            "recursion_limit": getattr(self, "max_iterations", 25),
        }

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

        short_term = await self.session.aget_short_term(
            short_term_size=self._short_term_size
        )
        response = self.llm.chat_with_history(
            user_input=task,
            history=short_term,
            system_prompt=system_prompt,
            temperature=0.7
        )

        logger.info("最终答案: %s", response)
        return response

    async def aget_execution_history(self, session_id: str | None = None) -> list[dict[str, Any]]:
        """获取执行历史（per-session 隔离）。

        Args:
            session_id: 目标会话 ID。为 None 时使用当前会话。
        """
        sid = self._current_sid(session_id)
        return await self._get_store().aget_history(sid)

    async def aclear_history(self, session_id: str | None = None) -> None:
        """清空执行历史和去重集合（per-session 隔离）。

        Args:
            session_id: 目标会话 ID。为 None 时使用当前会话。
        """
        sid = self._current_sid(session_id)
        await self._get_store().aclear_history(sid)

    # ============ 生命周期管理 ============

    async def aclose(self) -> None:
        """优雅关闭：释放 MCP 连接池等资源

        关闭后 AgentCore 不再可用，调用任何方法会抛 AgentStateError。
        可安全重复调用（幂等）。

        注意：checkpoint DB 和长期记忆资源的关闭由入口程序通过 MemoryContext.aclose() 负责。
        """
        if self._closed:
            return
        self._closed = True

        # 关闭 MCP 连接池
        try:
            await self._mcp_pool.close()
        except Exception as e:
            logger.warning("MCP 连接池关闭异常: %s", e, exc_info=True)

        logger.info("AgentCore 资源已释放")

    def _ensure_not_closed(self) -> None:
        """检查 Agent 是否已关闭，已关闭则抛异常"""
        if getattr(self, "_closed", False):
            raise AgentStateError("AgentCore 已关闭，不再可用")

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.aclose()