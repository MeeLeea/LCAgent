"""会话注册表 - 管理会话生命周期，桥接 checkpointer（thread 维度）与 Store（session 维度）。

替代原 AgentMemory 中的 new_thread/switch_thread/list_threads/delete_thread/
get_messages 等会话管理职责。AgentCore 不再持有 memory 实例，而是通过
SessionRegistry 获取 SessionContext，通过 SessionStore 读写瞬态状态。

并发说明：
- ``current_session_id`` 仅用于 CLI 单会话语义（兼容旧调用）。多会话并发场景
  必须向 ``get_context`` / 执行方法传入显式 ``session_id``，不依赖该共享指针。
- checkpointer 与 Store 实例本身线程安全，可在多会话间共享。
"""
from __future__ import annotations

import logging
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.base import BaseCheckpointSaver

from .context import SessionContext
from .store import SessionStore

logger = logging.getLogger(__name__)

# 工具输出在降级对话历史中的前缀
TOOL_RESULT_PREFIX = "[工具结果] "


def _naive_now() -> datetime:
    """返回本地 naive 时间（无时区），兼容历史数据格式。"""
    return datetime.now()  # noqa: DTZ005


def _message_role(msg: BaseMessage) -> str:
    """把 LangChain 消息对象映射为 OpenAI 风格的 role 字符串。"""
    if isinstance(msg, (ToolMessage, AIMessage)):
        return "assistant"
    if isinstance(msg, SystemMessage):
        return "system"
    return "user"


def _message_text(msg: BaseMessage) -> str:
    """取消息文本内容。多模态 content 为 list 时归一化为字符串。"""
    content = getattr(msg, "content", "")
    if isinstance(content, list):
        return " ".join(str(x) for x in content)
    return str(content)


class SessionRegistry:
    """会话生命周期管理。

    Args:
        checkpointer: 共享 checkpointer（所有会话复用）。
        store: 共享 SessionStore（execution_history / pending_interrupts）。
        process_type: 进程类型标识（server/scheduler/feishu），用于多进程隔离。
        recursion_limit: LangGraph 递归上限（= max_iterations）。
        async_conn: aiosqlite 连接（用于会话列表查询与关闭）；内存模式为 None。
    """

    def __init__(
        self,
        checkpointer: BaseCheckpointSaver,
        store: SessionStore,
        process_type: str | None = None,
        recursion_limit: int = 25,
        async_conn: Any | None = None,
    ) -> None:
        self._checkpointer = checkpointer
        self._store = store
        self._process_type = process_type
        self._recursion_limit = recursion_limit
        self._async_conn = async_conn
        # CLI 单会话当前指针（并发场景必须传显式 session_id，不依赖此值）
        self.current_session_id: str = self.generate_session_id()

    # ============ Session ID 生成 ============

    def generate_session_id(self, workflow_name: str | None = None) -> str:
        """生成新会话 ID（process_type 前缀 + 可选工作流名 + 随机后缀）。"""
        suffix = uuid.uuid4().hex[:8]
        parts: list[str] = []
        if self._process_type:
            parts.append(self._process_type)
        if workflow_name:
            safe = workflow_name.replace("-", "_").replace(" ", "_")
            parts.append(f"workflow-{safe}")
        parts.append(f"thread-{suffix}")
        return "-".join(parts)

    def new_workflow_session(self, workflow_name: str) -> str:
        """开启新的专属工作流会话（原会话保留在数据库）。"""
        self.current_session_id = self.generate_session_id(workflow_name)
        return self.current_session_id

    def new_session(self) -> str:
        """开启新会话（原会话保留在数据库），更新 current_session_id。"""
        self.current_session_id = self.generate_session_id()
        return self.current_session_id

    def is_workflow_session(self, session_id: str) -> bool:
        """判断 session_id 是否为专属工作流会话。"""
        return session_id.startswith("workflow-") or "-workflow-" in session_id

    def workflow_name_of(self, session_id: str) -> str | None:
        """从工作流会话的 session_id 反解绑定的工作流名称。

        session_id 格式: ``[process_type-]workflow-{name}-thread-{suffix}``
        name 中的 ``-`` 已在生成时替换为 ``_``，因此用 ``-thread-`` 分割安全。
        """
        if session_id.startswith("workflow-"):
            body = session_id[len("workflow-"):]
        elif "-workflow-" in session_id:
            body = session_id.split("-workflow-", 1)[1]
        else:
            return None
        name, _, _ = body.rpartition("-thread-")
        return name if name else body

    def _matches_process_type(self, session_id: str) -> bool:
        """判断 session_id 是否属于当前 process_type。"""
        if not self._process_type:
            return True
        return session_id.startswith(f"{self._process_type}-")

    # ============ 上下文获取 ============

    def get_context(self, session_id: str | None = None) -> SessionContext:
        """获取指定会话的 SessionContext。

        Args:
            session_id: 目标会话 ID。为 None 时使用 ``current_session_id``
                       （仅 CLI 单会话语义；并发场景必须传显式值）。
        """
        sid = session_id or self.current_session_id
        return SessionContext.create(
            session_id=sid,
            checkpointer=self._checkpointer,
            recursion_limit=self._recursion_limit,
        )

    # ============ 会话查询（从 checkpointer） ============

    async def astored_session_ids(self, all_types: bool = False) -> list[str]:
        """异步读取 checkpointer 中真实存在的 session_id（thread_id）。"""
        if self._async_conn is None:
            return []
        try:
            cursor = await self._async_conn.execute(
                "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
            )
            try:
                rows = await cursor.fetchall()
            finally:
                await cursor.close()
        except sqlite3.OperationalError as error:
            if "no such table" in str(error).lower():
                return []
            raise

        stored = [row[0] for row in rows]
        if all_types:
            return stored
        return [sid for sid in stored if self._matches_process_type(sid)]

    async def alist_sessions(self, all_types: bool = False) -> list[str]:
        """列出所有可见 session_id（数据库存量 ∪ 当前会话）。"""
        stored = await self.astored_session_ids(all_types=all_types)
        if self.current_session_id in stored:
            return stored
        return sorted(stored + [self.current_session_id])

    async def aswitch_session(self, session_id: str) -> bool:
        """切换到指定会话（恢复历史）。

        Returns:
            True=该会话有 checkpoint 记录，False=新会话（无历史）。
        """
        existed = session_id in await self.astored_session_ids()
        self.current_session_id = session_id
        return existed

    async def adelete_session(self, session_id: str) -> bool:
        """删除会话：清除 checkpoint + Store 中的会话状态。

        Returns:
            True=删除成功，False=失败或不存在。
        """
        deleted = False
        delete_fn = getattr(self._checkpointer, "adelete_thread", None)
        if callable(delete_fn):
            before = (
                self._async_conn.total_changes if self._async_conn is not None else 0
            )
            try:
                await delete_fn(session_id)
            except (sqlite3.Error, AttributeError, TypeError) as error:
                logger.error("删除会话失败(通过 Saver 契约): %s", error)
            else:
                if self._async_conn is not None:
                    deleted = self._async_conn.total_changes > before
                else:
                    deleted = True

        if not deleted and self._async_conn is not None:
            try:
                before = self._async_conn.total_changes
                await self._async_conn.execute(
                    "DELETE FROM checkpoints WHERE thread_id = ?", (session_id,)
                )
                try:
                    await self._async_conn.execute(
                        "DELETE FROM writes WHERE thread_id = ?", (session_id,)
                    )
                except sqlite3.OperationalError as error:
                    if "no such table" not in str(error).lower():
                        raise
                await self._async_conn.commit()
                deleted = self._async_conn.total_changes > before
            except sqlite3.OperationalError as error:
                if "no such table" in str(error).lower():
                    return False
                logger.error("删除会话失败(异步兜底路径): %s", error)
                return False

        # 清理 Store 中的瞬态状态
        await self._store.adelete_session(session_id)

        if deleted and session_id == self.current_session_id:
            remaining = await self.astored_session_ids()
            self.current_session_id = (
                remaining[0] if remaining else self.generate_session_id()
            )
        return deleted

    # ============ 消息获取（操作 checkpointer） ============

    async def aget_messages(self, session_id: str | None = None) -> list[BaseMessage]:
        """从 checkpoint 获取指定会话的所有消息。"""
        sid = session_id or self.current_session_id
        config = {"configurable": {"thread_id": sid}}
        try:
            getter = getattr(self._checkpointer, "aget_tuple", None)
            if callable(getter):
                tup = await getter(config)
            else:
                sync_getter = getattr(self._checkpointer, "get_tuple", None)
                if not callable(sync_getter):
                    return []
                tup = sync_getter(config)
            if tup and tup.checkpoint:
                channel_values = tup.checkpoint.get("channel_values", {})
                if "messages" in channel_values:
                    return list(channel_values["messages"])
        except Exception as error:
            logger.warning(
                "读取 checkpoint 消息失败 [%s]: %s", sid, error, exc_info=True
            )
        return []

    async def aget_short_term(
        self, session_id: str | None = None, limit: int | None = None,
        short_term_size: int = 10,
    ) -> list[dict[str, str]]:
        """从 checkpoint 取消息转为 dict 格式（兼容旧 API）。"""
        msgs = await self.aget_messages(session_id)
        if limit:
            msgs = msgs[-limit:]
        elif short_term_size:
            msgs = msgs[-short_term_size:]
        result: list[dict[str, str]] = []
        for m in msgs:
            role = _message_role(m)
            content = _message_text(m)
            if isinstance(m, ToolMessage):
                content = TOOL_RESULT_PREFIX + content
            result.append({"role": role, "content": content})
        return result

    async def aexport_session(
        self, session_id: str | None = None, fmt: str = "text"
    ) -> str:
        """将指定会话的对话导出为可读文本。"""
        sid = session_id or self.current_session_id
        msgs = await self.aget_messages(sid) or []

        blocks = []
        for m in msgs:
            if isinstance(m, HumanMessage):
                role = "用户"
            elif isinstance(m, AIMessage):
                role = "助手"
            elif isinstance(m, SystemMessage):
                role = "系统"
            else:
                role = "工具"
            text = _message_text(m).strip()
            if not text:
                continue
            if fmt == "markdown":
                blocks.append(f"**{role}**:\n\n{text}")
            else:
                blocks.append(f"【{role}】\n{text}")

        sep = "\n\n---\n\n" if fmt == "markdown" else "\n\n"
        header = (
            f"# 对话导出 - {sid}\n\n"
            if fmt == "markdown"
            else f"对话导出 - {sid}\n{'=' * 40}\n"
        )
        return header + sep.join(blocks)

    async def asummarize(self, session_id: str | None = None) -> dict[str, Any]:
        """获取会话摘要统计。"""
        sid = session_id or self.current_session_id
        msgs = await self.aget_messages(sid)
        sessions = await self.alist_sessions()
        return {
            "session_id": sid,
            "checkpoint_messages": len(msgs),
            "total_sessions": len(sessions),
        }

    # ============ 生命周期 ============

    async def aclose(self) -> None:
        """关闭 checkpointer 持有的 SQLite 连接。"""
        conn = self._async_conn
        if conn is None:
            return
        try:
            await conn.close()
        except Exception as error:
            logger.warning("关闭 checkpoint 连接失败: %s", error)
        finally:
            self._async_conn = None


__all__ = ["SessionRegistry"]
