"""Checkpointer 工厂 - 集中管理 LangGraph checkpointer 的创建与连接生命周期。

从 AgentMemory 中提取 checkpointer 初始化逻辑，使会话级基础设施由 session
模块统一管理。AgentMemory 退化为仅负责长期记忆（memory.json）。

当前仍为单一连接（与原 AgentMemory 行为一致）。未来可在本模块引入连接池，
无需改动 AgentCore / AgentMemory。
"""
from __future__ import annotations

import asyncio
import logging
import os

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)


def create_checkpointer(
    checkpoint_file: str | None = None,
    use_sqlite: bool = True,
) -> BaseCheckpointSaver:
    """创建同步 checkpointer（供同步路径/测试使用）。

    Args:
        checkpoint_file: SQLite 文件路径；为 None 或 use_sqlite=False 时用内存。
        use_sqlite: True=SQLite 持久化，False=内存（调试用）。

    Returns:
        SqliteSaver 或 MemorySaver。
    """
    if not use_sqlite or checkpoint_file is None:
        return MemorySaver()

    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    parent = os.path.dirname(os.path.abspath(checkpoint_file))
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(checkpoint_file, check_same_thread=False, timeout=10)
    # WAL 模式 + 忙等待：多进程(server/scheduler/remote)并发读写更友好
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    return SqliteSaver(conn)


async def create_async_checkpointer(
    checkpoint_file: str | None = None,
    use_sqlite: bool = True,
) -> tuple[BaseCheckpointSaver, object | None]:
    """创建异步 checkpointer（供 LangGraph 异步热路径使用）。

    Args:
        checkpoint_file: SQLite 文件路径；为 None 或 use_sqlite=False 时用内存。
        use_sqlite: True=SQLite 持久化，False=内存。

    Returns:
        ``(checkpointer, conn)`` 元组。conn 为 None 时表示内存模式；
        调用方负责在 aclose 时关闭 conn。
    """
    if not use_sqlite or checkpoint_file is None:
        return MemorySaver(), None

    import aiosqlite

    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    parent = os.path.dirname(os.path.abspath(checkpoint_file))
    if parent:
        await asyncio.to_thread(os.makedirs, parent, exist_ok=True)
    conn = await aiosqlite.connect(checkpoint_file)
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("PRAGMA busy_timeout=10000")
    return AsyncSqliteSaver(conn), conn


__all__ = ["create_checkpointer", "create_async_checkpointer"]
