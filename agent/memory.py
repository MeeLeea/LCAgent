"""
记忆模块 - 长期记忆管理（基于 JSON 持久化）

会话管理（checkpointer thread 级的 new/switch/list/delete/get_messages 等）
已迁移至 agent.session.SessionRegistry。本模块仅负责：
- 初始化 checkpointer（供 SessionRegistry 和 create_agent 共享）
- 长期记忆（memory.json）的增删查改与压缩
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import tempfile
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

from typing_extensions import Self

import aiosqlite
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.store.sqlite.aio import AsyncSqliteStore

logger = logging.getLogger(__name__)


def _naive_now() -> datetime:
    """返回本地 naive 时间(无时区)。

    记忆文件历史数据均以 naive ISO 字符串存储(如 2026-08-06T12:00:00),
    为兼容旧数据并保持格式统一,新增记录沿用 naive 格式。
    """
    return datetime.now()  # noqa: DTZ005 - 见函数 docstring,需保持 naive 格式兼容


class AgentMemory:
    """
    基于 LangGraph Checkpoint 的记忆系统

    - checkpointer: 自动持久化 Agent 执行状态(SQLite)，供 SessionRegistry 共享
    - long_term_memory: 手动标记的重要记忆(JSON,用于 compress)
    """

    def __init__(
        self,
        checkpoint_file: str | None = None,
        long_term_file: str | None = None,
        thread_id: str | None = None,
        short_term_size: int = 10,  # 仅兼容旧 API
        use_sqlite: bool = True,
        process_type: str | None = None
    ):
        """
        初始化记忆系统

        Args:
            checkpoint_file: SQLite 持久化文件路径(为 None 时用内存)
            long_term_file: 长期记忆 JSON 文件路径(用于 compress)
            thread_id: 会话线程 ID(为 None 时自动生成)
            short_term_size: 仅兼容旧 API(checkpoint 不限容量)
            use_sqlite: True=SQLite持久化, False=内存(调试用)
            process_type: 进程类型标识(server/scheduler/feishu)，用于多进程隔离
        """
        self.checkpoint_file = checkpoint_file
        self.long_term_file = long_term_file
        self.process_type = process_type
        self.thread_id = thread_id or self._generate_thread_id()
        self.short_term_size = short_term_size
        self.use_sqlite = use_sqlite and checkpoint_file is not None
        self._async_mode = False
        self._async_conn: aiosqlite.Connection | None = None

        # 长期记忆(独立于 checkpoint,用于 compress)
        self.long_term_memory: list[dict[str, Any]] = []

        # 初始化 checkpointer
        self._checkpointer = self._init_checkpointer()

        # 长期记忆 Store（同步模式用 InMemoryStore；异步模式在 acreate 中用 AsyncSqliteStore）
        self._long_term_store: BaseStore = InMemoryStore()
        self._long_term_store_conn: aiosqlite.Connection | None = None

        # 加载长期记忆
        if long_term_file and os.path.exists(long_term_file):
            self._load_long_term_memory()

    @classmethod
    async def acreate(
        cls,
        checkpoint_file: str | None = None,
        long_term_file: str | None = None,
        thread_id: str | None = None,
        short_term_size: int = 10,
        use_sqlite: bool = True,
        process_type: str | None = None,
    ) -> AgentMemory:
        """在运行中的事件循环内创建异步记忆实例。"""
        choice = cls.__new__(cls)
        choice.checkpoint_file = checkpoint_file
        choice.long_term_file = long_term_file
        choice.process_type = process_type
        choice.thread_id = thread_id or choice._generate_thread_id()
        choice.short_term_size = short_term_size
        choice.use_sqlite = use_sqlite and checkpoint_file is not None
        choice._async_mode = True
        choice._async_conn = None
        choice.long_term_memory = []

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

        if long_term_file and await asyncio.to_thread(os.path.exists, long_term_file):
            await choice._a_load_long_term_memory()

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

    def close(self) -> None:
        """关闭 checkpointer 持有的 SQLite 连接。

        sqlite3 的 connection 上下文管理器只提交事务、不关闭连接,
        因此长驻连接必须显式关闭,否则 WAL 无法回收、Windows 上文件也删不掉。
        """
        self._check_not_async("close")
        conn = getattr(self._checkpointer, "conn", None)
        if conn is None:
            return
        try:
            conn.close()
        except sqlite3.Error as error:
            logger.warning("关闭 checkpoint 连接失败: %s", error)

    def __enter__(self) -> Self:
        self._check_not_async("__enter__")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.aclose()

    def get_config(self, thread_id: str | None = None) -> dict[str, Any]:
        """获取 invoke 需要传入的 config

        Args:
            thread_id: 显式指定会话线程 ID（并发多会话时用于隔离）。
                       为 None 时使用当前会话（self.thread_id），兼容旧调用。

        Returns:
            LangGraph config，形如 {"configurable": {"thread_id": "..."}}
        """
        target = thread_id or self.thread_id
        return {"configurable": {"thread_id": target}}

    def _check_not_async(self, method: str) -> None:
        """在异步模式下调用同步方法直接报错，杜绝双模式混跑。

        Args:
            method: 被调用的同步方法名(用于错误提示)
        """
        if self._async_mode:
            raise RuntimeError(
                f"AgentMemory.{method}() 只能在同步模式(直接 __init__)调用；"
                f"异步模式(acreate)请使用对应异步版本"
            )

    # ============ 长期记忆(用于 compress) ============

    def add(self, role: str, content: str, metadata: dict[str, Any] | None = None):
        """
        添加记忆

        注意:checkpoint 会在 agent invoke 时自动保存对话,
        此方法只处理 important=True 的重要记忆(写入 memory.json 供 compress)。
        普通对话不需要调用此方法(由 checkpoint 自动处理)。
        """
        self._check_not_async("add")
        if metadata and metadata.get("important", False):
            item = {
                "role": role,
                "content": content,
                "timestamp": _naive_now().isoformat(),
                "metadata": metadata
            }
            self.long_term_memory.append(item)
            self._save_long_term_memory()

    async def aadd(self, role: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        """异步添加记忆(语义同 :meth:`add`,仅 important=True 时写入 memory.json)"""
        if metadata and metadata.get("important", False):
            item = {
                "role": role,
                "content": content,
                "timestamp": _naive_now().isoformat(),
                "metadata": metadata
            }
            self.long_term_memory.append(item)
            await self._a_save_long_term_memory()

    def get_long_term(self, limit: int = 5) -> list[dict[str, str]]:
        """获取长期记忆"""
        recent = self.long_term_memory[-limit:] if limit else self.long_term_memory
        return [
            {"role": item["role"], "content": item["content"]}
            for item in recent
        ]

    def clear_long_term(self):
        """清空长期记忆"""
        self._check_not_async("clear_long_term")
        self.long_term_memory.clear()
        if self.long_term_file and os.path.exists(self.long_term_file):
            os.remove(self.long_term_file)

    async def aclear_long_term(self) -> None:
        """异步清空长期记忆"""
        self.long_term_memory.clear()
        if self.long_term_file and await asyncio.to_thread(os.path.exists, self.long_term_file):
            await asyncio.to_thread(os.remove, self.long_term_file)

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

    def _save_long_term_memory(self):
        """保存长期记忆到文件(原子写,崩溃时不产生坏 JSON)"""
        if not self.long_term_file:
            return
        parent = os.path.dirname(os.path.abspath(self.long_term_file))
        if parent:
            os.makedirs(parent, exist_ok=True)
        # 先写临时文件,成功后原子替换;中途崩溃不会损坏现有文件
        fd, tmp_path = tempfile.mkstemp(dir=parent, prefix=".memory_", suffix=".json.tmp")
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(self.long_term_memory, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self.long_term_file)
        except Exception:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise

    def _load_long_term_memory(self):
        """从文件加载长期记忆"""
        if not self.long_term_file or not os.path.exists(self.long_term_file):
            return
        try:
            with open(self.long_term_file, 'r', encoding='utf-8') as f:
                self.long_term_memory = json.load(f)
        except (OSError, json.JSONDecodeError):
            self.long_term_memory = []

    async def _a_save_long_term_memory(self) -> None:
        """异步保存长期记忆到文件(原子写,崩溃时不产生坏 JSON)"""
        if not self.long_term_file:
            return
        parent = os.path.dirname(os.path.abspath(self.long_term_file))
        if parent:
            await asyncio.to_thread(os.makedirs, parent, exist_ok=True)
        fd, tmp_path = await asyncio.to_thread(
            tempfile.mkstemp, dir=parent, prefix=".memory_", suffix=".json.tmp"
        )

        def _atomic_write() -> None:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(self.long_term_memory, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self.long_term_file)

        try:
            await asyncio.to_thread(_atomic_write)
        except Exception:
            try:
                await asyncio.to_thread(os.remove, tmp_path)
            except OSError:
                pass
            raise

    async def _a_load_long_term_memory(self) -> None:
        """异步从文件加载长期记忆"""
        if not self.long_term_file:
            return
        if not await asyncio.to_thread(os.path.exists, self.long_term_file):
            return

        def _read() -> list[dict[str, Any]]:
            with open(self.long_term_file, 'r', encoding='utf-8') as f:
                return json.load(f)

        try:
            self.long_term_memory = await asyncio.to_thread(_read)
        except (OSError, json.JSONDecodeError):
            self.long_term_memory = []

    # ============ 长上下文裁剪与压缩 ============

    def compress_memory(self, summarize_callback: Callable[[str, str], str]) -> dict[str, Any]:
        """
        压缩长期记忆

        将 memory.json 中所有长期记忆发送给 LLM，生成摘要后替换原内容。
        这样可以在保留关键信息的同时大幅减少 token 占用。

        Args:
            summarize_callback: 生成摘要的回调函数(接收文本与提示,返回摘要字符串)

        Returns:
            {
                "success": bool,
                "original_count": int,      # 原记忆条数
                "original_chars": int,      # 原字符数
                "compressed_chars": int,    # 压缩后字符数
                "summary": str,             # 摘要内容
                "error": str (失败时)
            }
        """
        self._check_not_async("compress_memory")
        if not self.long_term_memory:
            return {
                "success": False,
                "error": "没有长期记忆可压缩"
            }

        # 1. 拼接所有长期记忆为文本
        history_lines = []
        original_chars = 0
        for idx, item in enumerate(self.long_term_memory, 1):
            role = item.get("role", "unknown")
            content = item.get("content", "")
            ts = item.get("timestamp", "")
            history_lines.append(f"[{idx}] ({ts}) {role}: {content}")
            original_chars += len(content)

        history_text = "\n\n".join(history_lines)

        # 2. 调用 LLM 生成摘要
        system_prompt = (
            "你是一个记忆压缩助手。请将以下历史对话记录压缩成一份简洁的摘要，要求：\n"
            "1. 保留所有关键信息、用户意图、重要决策和事实\n"
            "2. 去除重复和冗余内容\n"
            "3. 按主题分条目组织，使用 '- ' 开头\n"
            "4. 保持事实准确，不要添加推测内容\n"
            "5. 用中文输出"
        )
        summary = summarize_callback(
            f"以下是历史对话记录，请压缩成摘要:\n\n{history_text}",
            system_prompt,
        )
        if not summary:
            return {
                "success": False,
                "error": "LLM 调用失败或返回空摘要"
            }

        # 4. 用摘要替换原长期记忆
        original_count = len(self.long_term_memory)
        compressed_chars = len(summary)

        self.long_term_memory = [{
            "role": "system",
                "content": f"[历史记忆摘要 {_naive_now().isoformat()}]\n{summary}",
            "timestamp": _naive_now().isoformat(),
            "metadata": {
                "important": True,
                "type": "summary",
                "original_count": original_count,
                "original_chars": original_chars,
                "compressed_chars": compressed_chars
            }
        }]

        # 5. 保存回 memory.json
        self._save_long_term_memory()

        return {
            "success": True,
            "original_count": original_count,
            "original_chars": original_chars,
            "compressed_chars": compressed_chars,
            "summary": summary
        }

    async def acompress_memory(
        self,
        summarize_callback: Callable[[str, str], Awaitable[str]],
    ) -> dict[str, Any]:
        """异步压缩长期记忆(语义同 :meth:`compress_memory`,回调为异步版本)

        Args:
            summarize_callback: 生成摘要的异步回调(接收文本与提示,返回摘要字符串)
        """
        if not self.long_term_memory:
            return {
                "success": False,
                "error": "没有长期记忆可压缩"
            }

        # 1. 拼接所有长期记忆为文本
        history_lines = []
        original_chars = 0
        for idx, item in enumerate(self.long_term_memory, 1):
            role = item.get("role", "unknown")
            content = item.get("content", "")
            ts = item.get("timestamp", "")
            history_lines.append(f"[{idx}] ({ts}) {role}: {content}")
            original_chars += len(content)

        history_text = "\n\n".join(history_lines)

        # 2. 调用 LLM 生成摘要
        system_prompt = (
            "你是一个记忆压缩助手。请将以下历史对话记录压缩成一份简洁的摘要，要求：\n"
            "1. 保留所有关键信息、用户意图、重要决策和事实\n"
            "2. 去除重复和冗余内容\n"
            "3. 按主题分条目组织，使用 '- ' 开头\n"
            "4. 保持事实准确，不要添加推测内容\n"
            "5. 用中文输出"
        )
        summary = await summarize_callback(
            f"以下是历史对话记录，请压缩成摘要:\n\n{history_text}",
            system_prompt,
        )
        if not summary:
            return {
                "success": False,
                "error": "LLM 调用失败或返回空摘要"
            }

        # 3. 用摘要替换原长期记忆
        original_count = len(self.long_term_memory)
        compressed_chars = len(summary)

        self.long_term_memory = [{
            "role": "system",
                "content": f"[历史记忆摘要 {_naive_now().isoformat()}]\n{summary}",
            "timestamp": _naive_now().isoformat(),
            "metadata": {
                "important": True,
                "type": "summary",
                "original_count": original_count,
                "original_chars": original_chars,
                "compressed_chars": compressed_chars
            }
        }]

        # 4. 保存回 memory.json
        await self._a_save_long_term_memory()

        return {
            "success": True,
            "original_count": original_count,
            "original_chars": original_chars,
            "compressed_chars": compressed_chars,
            "summary": summary
        }
