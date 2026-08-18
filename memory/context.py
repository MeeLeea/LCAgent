"""MemoryContext — 统一记忆上下文工厂。

封装全部 memory 组件的创建与生命周期管理，作为 AgentCore 与 SessionManager
之间记忆基础设施的唯一桥梁。

职责：
  - 创建 AgentMemory（checkpointer + Store）
  - 创建 ThreadMemoryStore（per-thread facts 存储）
  - 创建 ThreadMemoryLockPool（并发锁池）
  - 创建 MemoryManager（统一门面，内部自建读写中间件）

对 AgentCore 暴露：
  - checkpointer：LangGraph BaseCheckpointSaver
  - store：LangGraph BaseStore（create_agent store 参数）
  - read_middleware：ThreadMemoryReadMiddleware（create_agent middleware 参数）
  - thread_id / async_conn：SessionRegistry 初始化所需

对 SessionManager 暴露：
  - memory_manager：MemoryManager 实例

依赖关系：
  入口程序(main.py / api/server.py / scheduler/run.py)
    → MemoryContext.acreate(...)
    → agent = AgentCore.acreate(checkpointer=ctx.checkpointer, store=ctx.store, ...)
    → agent.session_manager  →  SessionManager(agent, memory=ctx.memory_manager)
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .agent_memory import AgentMemory
from .config import (
    MEMORY_BUFFER_DELAY_SECONDS,
    MEMORY_MAX_AGENT_FACTS,
    MEMORY_MAX_BUFFER_MESSAGES,
    MEMORY_MAX_FACTS_PER_THREAD,
    MEMORY_RECALL_LIMIT,
)
from .lock_pool import ThreadMemoryLockPool
from .manager import MemoryManager
from .middleware import ThreadMemoryReadMiddleware
from .store import ThreadMemoryStore

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.store.base import BaseStore

logger = logging.getLogger(__name__)


class MemoryContext:
    """统一记忆上下文 — 封装全部 memory 组件创建与生命周期。

    通过 ``acreate()`` 类方法异步创建，确保 AsyncSqliteSaver 绑定正确的事件循环。

    Args:
        agent_memory: AgentMemory 实例（checkpointer + Store）
        read_middleware: ThreadMemoryReadMiddleware 实例
        memory_manager: MemoryManager 实例
    """

    def __init__(
        self,
        agent_memory: AgentMemory,
        read_middleware: ThreadMemoryReadMiddleware,
        memory_manager: MemoryManager,
    ) -> None:
        self._agent_memory = agent_memory
        self._read_middleware = read_middleware
        self._memory_manager = memory_manager

    # ============ 对 AgentCore 暴露 ============

    @property
    def checkpointer(self) -> BaseCheckpointSaver:
        """LangGraph checkpointer（供 create_agent + SessionRegistry 使用）。"""
        return self._agent_memory.get_checkpointer()

    @property
    def store(self) -> BaseStore:
        """LangGraph store（供 create_agent store 参数使用）。"""
        return self._agent_memory.get_long_term_store()

    @property
    def read_middleware(self) -> ThreadMemoryReadMiddleware:
        """长期记忆读取中间件（供 create_agent middleware 参数使用）。"""
        return self._read_middleware

    @property
    def thread_id(self) -> str:
        """初始会话线程 ID。"""
        return self._agent_memory.thread_id

    @property
    def async_conn(self) -> Any:
        """异步 SQLite 连接（供 SessionRegistry 共享）。"""
        return getattr(self._agent_memory, "_async_conn", None)

    # ============ 对 SessionManager 暴露 ============

    @property
    def memory_manager(self) -> MemoryManager:
        """MemoryManager 实例（长期记忆召回/消费/压缩/清理）。"""
        return self._memory_manager

    def bind_llm(self, llm_getter: Any) -> None:
        """运行时替换记忆链路的 LLM 获取器（支持 provider 热切换）。

        入口程序在创建 AgentCore 后调用，将记忆组件（事实抽取/召回/压缩）
        的 LLM 来源动态绑定到当前 Agent 的 ``agent.llm``，
        避免切换提供商后记忆抽取仍使用启动时的旧 LLMClient。

        Args:
            llm_getter: 返回当前 LLMClient 的 callable
        """
        self._memory_manager.bind_llm(llm_getter)

    # ============ 生命周期 ============

    async def aclose(self) -> None:
        """优雅关闭：刷新记忆 buffer、关闭中间件、释放 AgentMemory 资源。"""
        try:
            await self._memory_manager.flush_all()
            await self._memory_manager.shutdown()
        except Exception as error:
            logger.warning("MemoryManager 关闭异常: %s", error, exc_info=True)

        try:
            await self._agent_memory.aclose()
        except Exception as error:
            logger.warning("AgentMemory 关闭异常: %s", error, exc_info=True)

    # ============ 工厂方法 ============

    @classmethod
    async def acreate(
        cls,
        checkpoint_file: str | None = None,
        thread_id: str | None = None,
        short_term_size: int = 10,
        use_sqlite: bool = True,
        process_type: str | None = None,
        llm_getter: Any = None,
        buffer_delay_seconds: int = MEMORY_BUFFER_DELAY_SECONDS,
        max_buffer_messages: int = MEMORY_MAX_BUFFER_MESSAGES,
        max_facts_per_thread: int = MEMORY_MAX_FACTS_PER_THREAD,
        max_agent_facts: int = MEMORY_MAX_AGENT_FACTS,
        recall_limit: int = MEMORY_RECALL_LIMIT,
    ) -> MemoryContext:
        """异步创建 MemoryContext。

        在事件循环内创建 AsyncSqliteSaver，确保绑定正确的事件循环。

        Args:
            checkpoint_file: SQLite checkpoint 文件路径
            thread_id: 会话线程 ID
            short_term_size: 兼容旧 API
            use_sqlite: True=SQLite持久化, False=内存
            process_type: 进程类型标识
            llm_getter: 返回当前 LLMClient 的 callable（支持热切换）
            buffer_delay_seconds: 防抖缓冲窗口
            max_buffer_messages: 缓冲区上限
            max_facts_per_thread: 单线程最大 fact 条数
            max_agent_facts: agent 级（跨会话共享）最大 fact 条数
            recall_limit: 召回条数上限
        """
        # 1. 创建 AgentMemory
        agent_memory = await AgentMemory.acreate(
            checkpoint_file=checkpoint_file,
            thread_id=thread_id,
            short_term_size=short_term_size,
            use_sqlite=use_sqlite,
            process_type=process_type,
        )

        # 2. 创建 ThreadMemoryStore（复用 AgentMemory 的 Store backend）
        #    process_type 用于隔离 agent 级记忆的多进程防串号；
        #    max_facts 限单 thread 容量，max_agent_facts 限 agent 级容量。
        memory_store = ThreadMemoryStore(
            backend=agent_memory.get_long_term_store(),
            max_facts=max_facts_per_thread,
            max_agent_facts=max_agent_facts,
            process_type=process_type,
        )

        # 3. 创建 per-thread 并发锁池
        lock_pool = ThreadMemoryLockPool()

        # 4. 创建读中间件（recall_limit 约束每次 LLM 调用注入的 fact 条数）
        read_middleware = ThreadMemoryReadMiddleware(
            memory_store, recall_limit=recall_limit
        )

        # 5. 创建 MemoryManager（内部自建写中间件）
        memory_manager = MemoryManager(
            memory_store=memory_store,
            lock_pool=lock_pool,
            llm_getter=llm_getter or (lambda: None),
            recall_limit=recall_limit,
            buffer_delay_seconds=buffer_delay_seconds,
            max_buffer_messages=max_buffer_messages,
        )

        return cls(
            agent_memory=agent_memory,
            read_middleware=read_middleware,
            memory_manager=memory_manager,
        )


__all__ = ["MemoryContext"]
