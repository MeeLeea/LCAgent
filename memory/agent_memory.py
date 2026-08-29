"""
记忆模块 - Checkpointer 与 Store 基础设施

会话管理（checkpointer thread 级的 new/switch/list/delete/get_messages 等）
已迁移至 session.SessionRegistry。本模块仅负责：
- 初始化 checkpointer（供 SessionRegistry 和 create_agent 共享）
- 初始化长期记忆 Store（供 ThreadMemoryStore 使用）
"""
from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import uuid
from typing import Any

import aiosqlite
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.store.sqlite.aio import AsyncSqliteStore

logger = logging.getLogger(__name__)


class AgentMemory:
    """
    基于 LangGraph Checkpoint 的记忆系统

    - checkpointer: 自动持久化 Agent 执行状态(SQLite)，供 SessionRegistry 共享
    - long_term_store: LangGraph BaseStore（供 ThreadMemoryStore 使用）
    """

    def __init__(
        self,
        checkpoint_file: str | None = None,
        thread_id: str | None = None,
        short_term_size: int = 10,  # 仅兼容旧 API
        use_sqlite: bool = True,
        process_type: str | None = None
    ):
        """
        初始化记忆系统（同步模式，主要用于测试）

        Args:
            checkpoint_file: SQLite 持久化文件路径(为 None 时用内存)
            thread_id: 会话线程 ID(为 None 时自动生成)
            short_term_size: 仅兼容旧 API(checkpoint 不限容量)
            use_sqlite: True=SQLite持久化, False=内存(调试用)
            process_type: 进程类型标识(server/scheduler/feishu)，用于多进程隔离
        """
        self.checkpoint_file = checkpoint_file
        self.process_type = process_type
        self.thread_id = thread_id or self._generate_thread_id()
        self.short_term_size = short_term_size
        self.use_sqlite = use_sqlite and checkpoint_file is not None
        self._async_mode = False
        self._async_conn: aiosqlite.Connection | None = None

        # 初始化 checkpointer
        self._checkpointer = self._init_checkpointer()

        # 长期记忆 Store（同步模式用 InMemoryStore；异步模式在 acreate 中用 AsyncSqliteStore）
        self._long_term_store: BaseStore = InMemoryStore()
        self._long_term_store_conn: aiosqlite.Connection | None = None

    @classmethod
    async def acreate(
        cls,
        checkpoint_file: str | None = None,
        thread_id: str | None = None,
        short_term_size: int = 10,
        use_sqlite: bool = True,
        process_type: str | None = None,
    ) -> AgentMemory:
        """在运行中的事件循环内创建异步记忆实例。"""
        choice = cls.__new__(cls)
        choice.checkpoint_file = checkpoint_file
        choice.process_type = process_type
        choice.thread_id = thread_id or choice._generate_thread_id()
        choice.short_term_size = short_term_size
        choice.use_sqlite = use_sqlite and checkpoint_file is not None
        choice._async_mode = True
        choice._async_conn = None

        if choice.use_sqlite:
            assert checkpoint_file is not None
            parent = os.path.dirname(os.path.abspath(checkpoint_file))
            if parent:
                await asyncio.to_thread(os.makedirs, parent, exist_ok=True)
            conn = await aiosqlite.connect(checkpoint_file)
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA busy_timeout=10000")
            choice._checkpointer = AsyncSqliteSaver(conn)
            choice._async_conn = conn

            # 长期记忆 Store：AsyncSqliteStore，复用同一 SQLite 文件，独立连接
            store_conn = await aiosqlite.connect(checkpoint_file)
            await store_conn.execute("PRAGMA journal_mode=WAL")
            await store_conn.execute("PRAGMA busy_timeout=10000")
            choice._long_term_store = AsyncSqliteStore(store_conn)
            await choice._long_term_store.setup()
            await store_conn.commit()  # 确保 setup 建表事务已提交
            choice._long_term_store_conn = store_conn
        else:
            choice._checkpointer = MemorySaver()
            choice._long_term_store = InMemoryStore()
            choice._long_term_store_conn = None

        return choice

    def _generate_thread_id(self) -> str:
        """生成新的 thread_id，自动加上 process_type 前缀（如有）"""
        suffix = uuid.uuid4().hex[:8]
        if self.process_type:
            return f"{self.process_type}-thread-{suffix}"
        return f"thread-{suffix}"

    # ============ Checkpointer 管理 ============

    def _init_checkpointer(self) -> BaseCheckpointSaver:
        """初始化 checkpointer"""
        if self.use_sqlite:
            parent = os.path.dirname(os.path.abspath(self.checkpoint_file))
            if parent:
                os.makedirs(parent, exist_ok=True)
            conn = sqlite3.connect(self.checkpoint_file, check_same_thread=False, timeout=10)
            # 启用 WAL 模式 + 忙等待：多进程(server/scheduler/remote)并发读写更友好
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=10000")
            return SqliteSaver(conn)
        else:
            return MemorySaver()

    def get_checkpointer(self) -> BaseCheckpointSaver:
        """获取 checkpointer 实例(传给 create_agent)"""
        return self._checkpointer

    def get_long_term_store(self) -> BaseStore:
        """获取长期记忆 Store 实例（供 ThreadMemoryStore 使用）。

        - 异步模式: ``AsyncSqliteStore``（持久化到 SQLite，与 checkpoint 复用同一文件）
        - 同步模式: ``InMemoryStore``（仅内存，用于测试）
        """
        return self._long_term_store

    async def aclose(self) -> None:
        """关闭异步 checkpointer 和长期记忆 Store 持有的 SQLite 连接。"""
        conn = self._async_conn
        if conn is not None:
            try:
                await conn.close()
            except (aiosqlite.Error, sqlite3.Error, AttributeError, RuntimeError, ValueError) as error:
                logger.warning("关闭异步 checkpoint 连接失败: %s", error)
            finally:
                self._async_conn = None

        # 关闭长期记忆 Store 连接（与 checkpoint 独立的连接）
        store_conn = self._long_term_store_conn
        if store_conn is not None:
            try:
                await store_conn.close()
            except (aiosqlite.Error, sqlite3.Error, AttributeError, RuntimeError, ValueError) as error:
                logger.warning("关闭长期记忆 Store 连接失败: %s", error)
            finally:
                self._long_term_store_conn = None
