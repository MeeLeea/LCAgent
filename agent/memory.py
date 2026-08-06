"""
记忆模块 - 基于 LangGraph Checkpoint

兼容说明:
- 旧同步路径: 继续使用 sqlite3 + SqliteSaver, 供现有调用方与测试使用
- 新异步路径: 通过 `acreate()` 构建 aiosqlite + AsyncSqliteSaver, 供 LangGraph 异步热路径使用

替代原 deque + memory.json 方案:
- 短期记忆 → checkpoint 自动管理(SQLite 持久化,自动恢复)
- 长期记忆 → memory.json 手动标记(用于 compress 摘要)

特性:
1. 自动持久化: Agent 每步执行后自动保存完整状态
2. 跨会话恢复: 通过 thread_id 恢复历史对话
3. 中断恢复: 程序崩溃可从最近 checkpoint 续跑
4. 多会话隔离: 不同 thread_id 独立
5. 工具调用中间状态: 完整保存(tool_calls、tool_outputs)
"""
import asyncio
import json
import logging
import os
import sqlite3
import tempfile
import uuid
from collections.abc import Awaitable, Callable
from contextlib import closing
from datetime import datetime
from typing import Any, Self

import aiosqlite
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

logger = logging.getLogger(__name__)

# 工具输出在降级对话历史中的前缀:标明来源,避免以用户口吻混入上下文
TOOL_RESULT_PREFIX = "[工具结果] "


def _naive_now() -> datetime:
    """返回本地 naive 时间(无时区)。

    记忆文件历史数据均以 naive ISO 字符串存储(如 2026-08-06T12:00:00),
    为兼容旧数据并保持格式统一,新增记录沿用 naive 格式。
    """
    return datetime.now()  # noqa: DTZ005 - 见函数 docstring,需保持 naive 格式兼容


def _message_role(msg: BaseMessage) -> str:
    """把 LangChain 消息对象映射为 OpenAI 风格的 role 字符串。

    ToolMessage 与 AIMessage 无继承关系,但仍显式前置判断以防未来变动。
    未知类型统一归为 'user',与历史行为保持一致。
    """
    if isinstance(msg, ToolMessage):
        return "assistant"
    if isinstance(msg, AIMessage):
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


class AgentMemory:
    """
    基于 LangGraph Checkpoint 的记忆系统

    - checkpointer: 自动持久化 Agent 执行状态(SQLite)
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
        else:
            choice._checkpointer = MemorySaver()

        if long_term_file and await asyncio.to_thread(os.path.exists, long_term_file):
            await choice._a_load_long_term_memory()

        return choice

    def _generate_thread_id(self) -> str:
        """生成新的 thread_id，自动加上 process_type 前缀（如有）"""
        suffix = uuid.uuid4().hex[:8]
        if self.process_type:
            return f"{self.process_type}-thread-{suffix}"
        return f"thread-{suffix}"

    def _generate_workflow_thread_id(self, workflow_name: str) -> str:
        """生成专属工作流会话的 thread_id（编码工作流名，便于恢复时自动包装命令）。

        Args:
            workflow_name: 工作流名称（注册表 WORKFLOWS 的键）

        Returns:
            形如 `{process_type}-workflow-{name}-{suffix}`（无 process_type 时去掉前缀）。
            名称中的 `-`/空格被替换为 `_`，保证从 thread_id 反解名称时不会歧义。
        """
        suffix = uuid.uuid4().hex[:8]
        safe = workflow_name.replace("-", "_").replace(" ", "_")
        if self.process_type:
            return f"{self.process_type}-workflow-{safe}-{suffix}"
        return f"workflow-{safe}-{suffix}"

    def new_workflow_thread(self, workflow_name: str) -> str:
        """开启新的专属工作流会话（绑定指定工作流名称，原会话保留在数据库）。

        Args:
            workflow_name: 工作流名称

        Returns:
            新生成的 thread_id
        """
        self.thread_id = self._generate_workflow_thread_id(workflow_name)
        return self.thread_id

    def is_workflow_thread(self, thread_id: str) -> bool:
        """判断 thread_id 是否为专属工作流会话。

        Args:
            thread_id: 会话 ID

        Returns:
            True=工作流会话，False=普通会话
        """
        return thread_id.startswith("workflow-") or "-workflow-" in thread_id

    def workflow_name_of(self, thread_id: str) -> str | None:
        """从工作流会话的 thread_id 反解绑定的工作流名称。

        Args:
            thread_id: 工作流会话的 thread_id

        Returns:
            工作流名称；非工作流会话返回 None
        """
        if thread_id.startswith("workflow-"):
            body = thread_id[len("workflow-"):]
        elif "-workflow-" in thread_id:
            body = thread_id.split("-workflow-", 1)[1]
        else:
            return None
        name, _, _ = body.rpartition("-")
        return name if name else body

    def _matches_process_type(self, thread_id: str) -> bool:
        """判断 thread_id 是否属于当前 process_type
        
        - 如果未设置 process_type，匹配所有 thread
        - 如果设置了 process_type，只匹配 "{type}-" 前缀的 thread
        """
        if not self.process_type:
            return True
        return thread_id.startswith(f"{self.process_type}-")

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

    def _connect(self) -> sqlite3.Connection:
        """为一次性查询打开连接。调用方必须负责关闭(用 closing 包裹)。

        与 _init_checkpointer 保持相同的 timeout/busy_timeout 设置,
        避免并发写入时短连接立刻抛 database is locked。
        """
        conn = sqlite3.connect(self.checkpoint_file, check_same_thread=False, timeout=10)
        conn.execute("PRAGMA busy_timeout=10000")
        return conn

    async def _async_stored_thread_ids(self, all_types: bool = False) -> list[str]:
        """异步读取数据库中真实存在的 thread_id。"""
        if not self.use_sqlite or self._async_conn is None:
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
        return [thread_id for thread_id in stored if self._matches_process_type(thread_id)]

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

    def get_config(self) -> dict[str, Any]:
        """获取 invoke 需要传入的 config"""
        return {"configurable": {"thread_id": self.thread_id}}

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

    # ============ 会话(Thread)管理 ============

    def new_thread(self) -> str:
        """开启新会话(原会话保留在数据库)"""
        self.thread_id = self._generate_thread_id()
        return self.thread_id

    async def aswitch_thread(self, thread_id: str) -> bool:
        """
        切换到指定会话(恢复历史) - 异步版本

        Returns:
            True=该 thread 有 checkpoint 记录, False=新会话(无历史)
        """
        # 必须先查存量再赋值:否则当前 thread_id 总会出现在列表里,返回值恒为 True
        existed = thread_id in await self._async_stored_thread_ids()
        self.thread_id = thread_id
        return existed

    def switch_thread(self, thread_id: str) -> bool:
        """
        切换到指定会话(恢复历史)

        .. deprecated::
            请迁移到异步版本 :meth:`aswitch_thread`，该同步方法将在未来版本中移除。

        Returns:
            True=该 thread 有 checkpoint 记录, False=新会话(无历史)
        """
        self._check_not_async("switch_thread")
        # 必须先查存量再赋值:否则当前 thread_id 总会出现在列表里,返回值恒为 True
        # 同步模式：直接查询数据库
        if not self.use_sqlite:
            existed = False
        else:
            try:
                with closing(self._connect()) as conn:
                    cursor = conn.execute(
                        "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
                    )
                    rows = [row[0] for row in cursor.fetchall()]
                existed = thread_id in rows
            except sqlite3.OperationalError as error:
                if "no such table" in str(error).lower():
                    existed = False
                else:
                    raise
        self.thread_id = thread_id
        return existed



    def list_threads(self, all_types: bool = False) -> list[str]:
        """列出所有可见 thread_id(数据库存量 ∪ 当前会话)。

        .. deprecated::
            请迁移到异步版本 :meth:`alist_threads`，该同步方法将在未来版本中移除。

        Args:
            all_types: True=列出所有进程类型的 thread(跨 type 查询),
                       False=只列出当前 process_type 的 thread(默认)

        当前会话可能还没写入 checkpoint,但对用户而言它是存在的,
        因此并入结果,保持 CLI/API 的显示语义不变。
        """
        self._check_not_async("list_threads")
        # 同步模式：直接查询数据库
        if not self.use_sqlite:
            stored = []
        else:
            try:
                with closing(self._connect()) as conn:
                    cursor = conn.execute(
                        "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
                    )
                    rows = [row[0] for row in cursor.fetchall()]
            except sqlite3.OperationalError as error:
                if "no such table" in str(error).lower():
                    rows = []
                else:
                    raise
            if all_types:
                stored = rows
            else:
                stored = [tid for tid in rows if self._matches_process_type(tid)]
        # 当前 thread 始终可见，即使它不匹配 all_types 的查询范围
        if self.thread_id in stored:
            return stored
        return sorted(stored + [self.thread_id])

    async def alist_threads(self, all_types: bool = False) -> list[str]:
        """异步列出所有可见 thread_id(数据库存量 ∪ 当前会话)。"""
        stored = await self._async_stored_thread_ids(all_types=all_types)
        if self.thread_id in stored:
            return stored
        return sorted(stored + [self.thread_id])

    def delete_thread(self, thread_id: str) -> bool:
        """
        删除指定会话(从 SQLite checkpoint 表移除)

        Args:
            thread_id: 要删除的会话 ID

        Returns:
            True=删除成功, False=失败或不存在
        """
        self._check_not_async("delete_thread")
        if not self.use_sqlite:
            # 内存模式:无法删除单个 thread,只能清空
            return False

        # 优先走 SqliteSaver 自带的 delete_thread 契约,用其长驻连接和锁
        delete_fn = getattr(self._checkpointer, "delete_thread", None)
        if callable(delete_fn):
            conn = getattr(self._checkpointer, "conn", None)
            if conn is None:
                return False
            try:
                before = conn.total_changes
                delete_fn(thread_id)
                deleted = conn.total_changes > before
            except sqlite3.Error as error:
                logger.error("删除会话失败(通过 Saver 契约): %s", error)
                return False
        else:
            # 兜底:直接用 SQL 删除(开新连接,单事务)
            try:
                with closing(self._connect()) as conn:
                    cursor = conn.execute(
                        "DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,)
                    )
                    deleted = cursor.rowcount > 0
                    # writes 表可能不存在(老版本 LangGraph 或未初始化的库)
                    try:
                        conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
                    except sqlite3.OperationalError:
                        pass
                    conn.commit()
            except sqlite3.Error as error:
                logger.error("删除会话失败(兜底路径): %s", error)
                return False

        if not deleted:
            return False

        # 如果删的是当前会话,自动切到一个剩余的会话(或新建)
        if thread_id == self.thread_id:
            # 同步模式：直接查询数据库
            try:
                with closing(self._connect()) as conn:
                    cursor = conn.execute(
                        "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
                    )
                    remaining = [row[0] for row in cursor.fetchall()]
            except sqlite3.OperationalError as error:
                if "no such table" in str(error).lower():
                    remaining = []
                else:
                    raise
            self.thread_id = remaining[0] if remaining else self.new_thread()
        return True

    async def adelete_thread(self, thread_id: str) -> bool:
        """异步删除指定会话(从 SQLite checkpoint 表移除)。"""
        if not self.use_sqlite:
            return False

        deleted = False
        delete_fn = getattr(self._checkpointer, "adelete_thread", None)
        if callable(delete_fn):
            before_changes = self._async_conn.total_changes if self._async_conn is not None else 0
            try:
                await delete_fn(thread_id)
            except (aiosqlite.Error, sqlite3.Error, AttributeError, TypeError) as error:
                logger.error("删除会话失败(通过异步 Saver 契约): %s", error)
            else:
                if self._async_conn is not None:
                    deleted = self._async_conn.total_changes > before_changes
                else:
                    deleted = True

        if not deleted and self._async_conn is not None:
            try:
                before_changes = self._async_conn.total_changes
                await self._async_conn.execute(
                    "DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,)
                )
                try:
                    await self._async_conn.execute(
                        "DELETE FROM writes WHERE thread_id = ?", (thread_id,)
                    )
                except sqlite3.OperationalError as error:
                    if "no such table" not in str(error).lower():
                        raise
                await self._async_conn.commit()
                deleted = self._async_conn.total_changes > before_changes
            except sqlite3.OperationalError as error:
                if "no such table" in str(error).lower():
                    return False
                logger.error("删除会话失败(异步兜底路径): %s", error)
                return False
            except (aiosqlite.Error, AttributeError, TypeError) as error:
                logger.error("删除会话失败(异步兜底路径): %s", error)
                return False

        if not deleted:
            return False

        if thread_id == self.thread_id:
            remaining = await self._async_stored_thread_ids()
            self.thread_id = remaining[0] if remaining else self.new_thread()
        return True

    # ============ 消息获取 ============

    def get_messages(self, thread_id: str | None = None) -> list[BaseMessage]:
        """从 checkpoint 获取指定会话(默认当前会话)的所有消息

        .. deprecated::
            请迁移到异步版本 :meth:`aget_messages`，该同步方法将在未来版本中移除。

        Args:
            thread_id: 要读取的会话 ID。为 None 时读取当前会话。

        Returns:
            消息列表。无 checkpoint 或会话不存在时返回空列表。
        """
        self._check_not_async("get_messages")
        target = thread_id if thread_id is not None else self.thread_id
        config = {"configurable": {"thread_id": target}}
        try:
            # langgraph 1.x: 用 get_tuple 获取 CheckpointTuple
            tup = self._checkpointer.get_tuple(config)
            if tup and tup.checkpoint:
                channel_values = tup.checkpoint.get("channel_values", {})
                if "messages" in channel_values:
                    return list(channel_values["messages"])
        except Exception as error:
            logger.warning("读取 checkpoint 消息失败 [%s]: %s", target, error, exc_info=True)
        return []

    async def aget_messages(self, thread_id: str | None = None) -> list[BaseMessage]:
        """从异步 checkpoint 获取指定会话(默认当前会话)的所有消息。"""
        target = thread_id if thread_id is not None else self.thread_id
        config = {"configurable": {"thread_id": target}}
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
        except (aiosqlite.Error, sqlite3.Error, AttributeError, TypeError) as error:
            logger.warning("读取异步 checkpoint 消息失败 [%s]: %s", target, error, exc_info=True)
        return []

    async def aexport_thread(self, thread_id: str | None = None, fmt: str = "text") -> str:
        """
        将指定会话(默认当前会话)的对话导出为可读文本 - 异步版本

        Args:
            thread_id: 要导出的会话 ID(为 None 时导出当前会话)
            fmt: 导出格式,目前支持 'text'(默认) 与 'markdown'

        Returns:
            格式化后的对话文本
        """
        target = thread_id if thread_id is not None else self.thread_id
        msgs = await self.aget_messages(thread_id=target) or []

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
        header = f"# 对话导出 - {target}\n\n" if fmt == "markdown" else f"对话导出 - {target}\n{'='*40}\n"
        return header + sep.join(blocks)

    def export_thread(self, thread_id: str | None = None, fmt: str = "text") -> str:
        """
        将指定会话(默认当前会话)的对话导出为可读文本 - 同步版本（已废弃）
        
        @deprecated: 请使用 aexport_thread() 异步版本
        """
        self._check_not_async("export_thread")
        # 同步模式：直接实现
        target = thread_id if thread_id is not None else self.thread_id
        msgs = self.get_messages(thread_id=target) or []

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
        header = f"# 对话导出 - {target}\n\n" if fmt == "markdown" else f"对话导出 - {target}\n{'='*40}\n"
        return header + sep.join(blocks)

    def get_short_term(self, limit: int | None = None) -> list[dict[str, str]]:
        """兼容旧 API:从 checkpoint 取消息转为 dict 格式

        .. deprecated::
            请迁移到异步版本 :meth:`aget_short_term`，该同步方法将在未来版本中移除。

        工具输出映射为 assistant 角色并加前缀,避免在降级对话中冒充用户。
        """
        self._check_not_async("get_short_term")
        # 同步模式：直接读取 checkpoint
        msgs = self.get_messages()
        
        if limit:
            msgs = msgs[-limit:]
        elif self.short_term_size:
            msgs = msgs[-self.short_term_size:]
        result = []
        for m in msgs:
            role = _message_role(m)
            content = _message_text(m)
            # ToolMessage 输出标注来源,避免以用户口吻混入 LLM 上下文
            if isinstance(m, ToolMessage):
                content = TOOL_RESULT_PREFIX + content
            result.append({"role": role, "content": content})
        return result

    async def aget_short_term(self, limit: int | None = None) -> list[dict[str, str]]:
        """异步兼容旧 API:从 checkpoint 取消息转为 dict 格式。"""
        msgs = await self.aget_messages()
        if limit:
            msgs = msgs[-limit:]
        elif self.short_term_size:
            msgs = msgs[-self.short_term_size:]
        result: list[dict[str, str]] = []
        for m in msgs:
            role = _message_role(m)
            content = _message_text(m)
            if isinstance(m, ToolMessage):
                content = TOOL_RESULT_PREFIX + content
            result.append({"role": role, "content": content})
        return result

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

    def clear_short_term(self):
        """清空当前 thread(开启新会话替代删除)"""
        self.new_thread()

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
        """关闭异步 checkpointer 持有的 SQLite 连接。"""
        conn = self._async_conn
        if conn is None:
            return
        try:
            await conn.close()
        except (aiosqlite.Error, sqlite3.Error, AttributeError, RuntimeError, ValueError) as error:
            logger.warning("关闭异步 checkpoint 连接失败: %s", error)
        finally:
            self._async_conn = None

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

    async def asummarize(self) -> dict[str, Any]:
        """获取记忆摘要统计 - 异步版本"""
        msgs = await self.aget_messages()
        threads = await self.alist_threads()
        
        return {
            "thread_id": self.thread_id,
            "checkpoint_messages": len(msgs),
            "checkpoint_backend": "sqlite" if self.use_sqlite else "memory",
            "checkpoint_file": self.checkpoint_file or "(内存)",
            "long_term_count": len(self.long_term_memory),
            "total_threads": len(threads)
        }

    def summarize(self) -> dict[str, Any]:
        """获取记忆摘要统计 - 同步版本（已废弃）
        
        @deprecated: 请使用 asummarize() 异步版本
        """
        self._check_not_async("summarize")
        # 同步模式：直接实现
        msgs = self.get_messages()
        threads = self.list_threads()
        
        return {
            "thread_id": self.thread_id,
            "checkpoint_messages": len(msgs),
            "checkpoint_backend": "sqlite" if self.use_sqlite else "memory",
            "checkpoint_file": self.checkpoint_file or "(内存)",
            "long_term_count": len(self.long_term_memory),
            "total_threads": len(threads)
        }

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
