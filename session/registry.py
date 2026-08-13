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
from .workspace_store import WorkspaceStore

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
        # 工作空间持久存储（复用 async_conn，内存模式降级为纯缓存）
        self._workspace_store = WorkspaceStore(async_conn)
        # CLI 单会话当前指针（并发场景必须传显式 session_id，不依赖此值）
        self.current_session_id: str = self.generate_session_id()

    # ============ 属性暴露 ============

    @property
    def checkpointer(self) -> BaseCheckpointSaver:
        """共享 checkpointer 实例（供工作流构建器注入持久化）。"""
        return self._checkpointer

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

    @staticmethod
    def _validate_workspace_path(path: str) -> str:
        """校验工作空间路径合法性，返回规范化绝对路径。

        校验规则（文档第三节、第七节）：
        - 必须是绝对路径
        - 必须是已存在的目录
        - 禁止宿主根目录、用户主目录、系统目录
        - 规范化为 realpath（解析符号链接）

        Args:
            path: 用户指定的工作空间路径

        Returns:
            规范化后的绝对路径

        Raises:
            ValueError: 路径不合法时抛出，附带中文原因
        """
        import os

        if not path or not path.strip():
            raise ValueError("工作空间路径不能为空")

        # 规范化为绝对路径（不要求路径存在，下面单独检查）
        abs_path = os.path.abspath(path)
        real_path = os.path.realpath(abs_path)

        if not os.path.isdir(real_path):
            raise ValueError(f"工作空间路径不存在或不是目录: {real_path}")

        # 禁止宿主根目录、系统目录、用户主目录
        import sys

        forbidden = [
            os.path.realpath(os.path.sep),  # 根目录 /
            os.path.realpath(os.path.expanduser("~")),  # 用户主目录
        ]
        # Windows 系统目录
        if sys.platform == "win32":
            sys_root = os.environ.get("SystemRoot", r"C:\Windows")
            forbidden.append(os.path.realpath(sys_root))
            forbidden.append(os.path.realpath(os.path.join(sys_root, "System32")))

        for forbidden_path in forbidden:
            try:
                if (
                    os.path.commonpath([forbidden_path, real_path]) == forbidden_path
                    and real_path == forbidden_path
                ):
                    raise ValueError(
                        f"禁止使用系统关键目录作为工作空间: {real_path}"
                    )
            except ValueError:
                # commonpath 在不同盘符时抛 ValueError，跳过
                continue

        return real_path

    def new_workflow_session(
        self, workflow_name: str, workspace_path: str | None = None
    ) -> str:
        """开启新的专属工作流会话（原会话保留在数据库）。

        Args:
            workflow_name: 工作流名称
            workspace_path: 工作空间绝对路径。为 None 时不绑定工作空间（兼容旧调用）。

        Returns:
            新会话 ID
        """
        self.current_session_id = self.generate_session_id(workflow_name)
        if workspace_path is not None:
            real_path = self._validate_workspace_path(workspace_path)
            # 同步写缓存，立即可被 get_context 同步读
            self._workspace_store.set_cached(self.current_session_id, real_path)
            # 后台异步持久化（fire-and-forget，由调用方在异步上下文触发）
            self._schedule_workspace_persist(self.current_session_id, real_path)
        return self.current_session_id

    def new_session(self, workspace_path: str | None = None) -> str:
        """开启新会话（原会话保留在数据库），更新 current_session_id。

        Args:
            workspace_path: 工作空间绝对路径。为 None 时不绑定工作空间（兼容旧调用）。

        Returns:
            新会话 ID
        """
        self.current_session_id = self.generate_session_id()
        if workspace_path is not None:
            real_path = self._validate_workspace_path(workspace_path)
            self._workspace_store.set_cached(self.current_session_id, real_path)
            self._schedule_workspace_persist(self.current_session_id, real_path)
        return self.current_session_id

    def _schedule_workspace_persist(self, session_id: str, workspace_path: str) -> None:
        """调度异步持久化 workspace 绑定（fire-and-forget）。

        尝试复用现有事件循环提交 aset() 任务。若无运行中的事件循环（纯同步上下文），
        跳过异步持久化——下次 aset/aget 调用时会补写缓存，DB 在后续异步入口补写。
        """
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._workspace_store.aset(session_id, workspace_path))
        except RuntimeError:
            # 无运行中的事件循环（如同步测试），缓存已写，DB 跳过
            logger.debug("无事件循环，workspace 持久化跳过（缓存已写）: %s", session_id)

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

        workspace_path 从内存缓存同步读取（new_session 时已写入）。
        若会话通过 aswitch_session 切换而来，aswitch 已 warm 缓存。
        若缓存未命中（如断点续跑首次同步访问），workspace_path 不注入——
        调用方应在异步入口 await awarm_workspace() 后再调用本方法。

        Args:
            session_id: 目标会话 ID。为 None 时使用 ``current_session_id``
                       （仅 CLI 单会话语义；并发场景必须传显式值）。
        """
        sid = session_id or self.current_session_id
        workspace_path = self._workspace_store.get_cached(sid)
        return SessionContext.create(
            session_id=sid,
            checkpointer=self._checkpointer,
            recursion_limit=self._recursion_limit,
            workspace_path=workspace_path,
        )

    async def awarm_workspace(self, session_id: str) -> str | None:
        """从 DB 加载指定 session 的 workspace 到缓存。

        断点续跑 / 进程重启后，缓存为空。在异步入口（aswitch_session /
        aresume_events / achat_stream 等）调用本方法 warm 缓存后，
        get_context() 同步读即可命中 workspace_path。

        Args:
            session_id: 目标会话 ID

        Returns:
            工作空间路径，无记录返回 None
        """
        return await self._workspace_store.aget(session_id)

    async def aset_workspace(
        self, workspace_path: str, session_id: str | None = None
    ) -> str:
        """异步设置/修改指定会话的工作空间绑定（缓存 + DB 双写）。

        用于运行时修改已存在会话的 workspace（如 CLI ``workspace <path>`` 命令）。
        路径校验同 new_session：必须是已存在的目录、非系统关键目录。

        Args:
            workspace_path: 工作空间路径（绝对或相对，将规范化为 realpath）
            session_id: 目标会话 ID。为 None 时使用 current_session_id。

        Returns:
            规范化后的绝对路径

        Raises:
            ValueError: 路径不合法（不存在、非目录、系统关键目录等）
        """
        sid = session_id or self.current_session_id
        real_path = self._validate_workspace_path(workspace_path)
        await self._workspace_store.aset(sid, real_path)
        return real_path

    async def aclear_workspace(self, session_id: str | None = None) -> bool:
        """清除指定会话的工作空间绑定（缓存 + DB）。

        Args:
            session_id: 目标会话 ID。为 None 时使用 current_session_id。

        Returns:
            True=原绑定存在已清除，False=原本无绑定
        """
        sid = session_id or self.current_session_id
        return await self._workspace_store.adelete(sid)

    async def aget_workspace(self, session_id: str | None = None) -> str | None:
        """获取指定会话的工作空间路径（缓存优先，未命中查 DB）。

        Args:
            session_id: 目标会话 ID。为 None 时使用 current_session_id。

        Returns:
            工作空间绝对路径，无绑定返回 None
        """
        sid = session_id or self.current_session_id
        return await self._workspace_store.aget(sid)

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

        切换时 warm workspace 缓存：从 DB 读取该会话绑定的 workspace_path
        到内存，使后续 get_context() 同步读可命中。

        Returns:
            True=该会话有 checkpoint 记录，False=新会话（无历史）。
        """
        existed = session_id in await self.astored_session_ids()
        self.current_session_id = session_id
        # warm workspace 缓存，使 get_context 同步读可命中
        await self.awarm_workspace(session_id)
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

        # 清理 workspace 绑定记录（缓存 + DB）
        await self._workspace_store.adelete(session_id)

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
