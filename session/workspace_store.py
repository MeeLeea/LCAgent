"""工作空间存储 - session_id ↔ workspace_path 持久映射 + 内存缓存。

复用 SessionRegistry 已持有的 aiosqlite async_conn，与 checkpoints/writes 表
共存于同一 checkpoints_async.sqlite 文件，保证断点续跑可恢复工作空间绑定。

同步/异步边界设计：
- new_session / get_context 是同步方法，无法 await DB
- 解决：内部维护内存缓存 dict（同步 O(1) 读），SQLite 做持久化
- new_session(workspace_path) 同步写缓存立即可用 → 后台异步持久化
- get_context(sid) 同步读缓存
- aswitch_session / aresume_events 等异步入口 await aget() warm 缓存
- 进程重启后缓存空，首个异步入口从 DB 回填缓存

设计要点：
- workspace_path 是 Session 固有持久属性，会话创建时写入，仅会话删除时清除
- 内存缓存与 DB 双写，缓存为权威读源，DB 为持久化源
- async_conn 为 None 时纯内存模式（测试用）
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# 幂等建表 SQL（与 checkpoints/writes 表同库共存）
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS session_workspaces (
    session_id    TEXT PRIMARY KEY,
    workspace_path TEXT NOT NULL,
    created_at    TEXT NOT NULL
)
"""


class WorkspaceStore:
    """session_id ↔ workspace_path 持久映射 + 内存缓存。

    Args:
        async_conn: aiosqlite 连接（与 SessionRegistry 共享）。None 时纯内存模式。
    """

    def __init__(self, async_conn: Any | None = None) -> None:
        self._conn = async_conn
        # 内存缓存：同步读权威源，DB 为持久化源
        self._cache: dict[str, str] = {}
        self._initialized = False

    async def _ensure_table(self) -> None:
        """幂等建表（仅 SQLite 模式，首次访问时执行）。"""
        if self._initialized or self._conn is None:
            self._initialized = True
            return
        try:
            await self._conn.execute(_CREATE_TABLE_SQL)
            await self._conn.commit()
        except Exception as error:
            logger.warning("session_workspaces 建表失败（降级内存模式）: %s", error)
        finally:
            self._initialized = True

    # ============ 同步读（缓存） ============

    def get_cached(self, session_id: str) -> str | None:
        """同步读缓存（O(1)），供同步方法 get_context 使用。

        未命中返回 None。异步入口（aswitch/aresume）应先 await aget() warm 缓存。

        Returns:
            工作空间绝对路径，缓存未命中或无记录时返回 None
        """
        return self._cache.get(session_id)

    def set_cached(self, session_id: str, workspace_path: str) -> None:
        """同步写缓存（立即对同步读可见），供同步方法 new_session 使用。

        仅写缓存，不持久化。持久化由 aset() 完成（new_session 内部 fire-and-forget）。

        Args:
            session_id: 会话 ID
            workspace_path: 工作空间绝对路径
        """
        self._cache[session_id] = workspace_path

    # ============ 异步读写（缓存 + DB 双写） ============

    async def aset(self, session_id: str, workspace_path: str) -> None:
        """写入缓存 + DB（双写）。供异步入口持久化使用。

        new_session 同步路径应先 set_cached()，再 fire-and-forget aset() 持久化。

        Args:
            session_id: 会话 ID
            workspace_path: 工作空间绝对路径
        """
        self._cache[session_id] = workspace_path
        if self._conn is None:
            return
        await self._ensure_table()
        try:
            await self._conn.execute(
                "INSERT INTO session_workspaces (session_id, workspace_path, created_at) "
                "VALUES (?, ?, datetime('now')) "
                "ON CONFLICT(session_id) DO UPDATE SET "
                "workspace_path=excluded.workspace_path",
                (session_id, workspace_path),
            )
            await self._conn.commit()
        except Exception as error:
            logger.warning("workspace 持久化失败（缓存已写）: %s", error)

    async def aget(self, session_id: str) -> str | None:
        """异步读：缓存命中直接返回，未命中查 DB 并回填缓存。

        供 aswitch_session / aresume_events 等异步入口 warm 缓存使用。
        warm 后 get_cached() 同步读即可命中。

        Returns:
            工作空间绝对路径，无记录时返回 None
        """
        cached = self._cache.get(session_id)
        if cached is not None:
            return cached
        if self._conn is None:
            return None
        await self._ensure_table()
        try:
            cursor = await self._conn.execute(
                "SELECT workspace_path FROM session_workspaces WHERE session_id = ?",
                (session_id,),
            )
            try:
                row = await cursor.fetchone()
            finally:
                await cursor.close()
        except Exception as error:
            logger.warning("workspace 读取失败: %s", error)
            return None
        if row:
            self._cache[session_id] = row[0]
            return row[0]
        return None

    async def adelete(self, session_id: str) -> bool:
        """删除缓存 + DB 记录。

        Returns:
            True=缓存或 DB 中有记录被删除，False=无记录
        """
        existed = self._cache.pop(session_id, None) is not None
        if self._conn is None:
            return existed
        await self._ensure_table()
        try:
            before = self._conn.total_changes
            await self._conn.execute(
                "DELETE FROM session_workspaces WHERE session_id = ?",
                (session_id,),
            )
            await self._conn.commit()
            db_deleted = self._conn.total_changes > before
        except Exception as error:
            logger.warning("workspace 删除失败（缓存已清）: %s", error)
            return existed
        return existed or db_deleted

    async def awarm_from_db(self, session_id: str) -> None:
        """从 DB 加载指定 session 的 workspace 到缓存（若 DB 有记录）。

        供断点续跑/进程重启后的异步入口使用：warm 后同步读即可命中。
        等价于 aget 但不返回值，语义更明确。
        """
        await self.aget(session_id)


__all__ = ["WorkspaceStore"]
