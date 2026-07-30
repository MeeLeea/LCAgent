"""
记忆模块 - 基于 LangGraph Checkpoint

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
import os
import json
import sqlite3
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver


class AgentMemory:
    """
    基于 LangGraph Checkpoint 的记忆系统

    - checkpointer: 自动持久化 Agent 执行状态(SQLite)
    - long_term_memory: 手动标记的重要记忆(JSON,用于 compress)
    """

    def __init__(
        self,
        checkpoint_file: Optional[str] = None,
        long_term_file: Optional[str] = None,
        thread_id: Optional[str] = None,
        short_term_size: int = 10,  # 仅兼容旧 API
        use_sqlite: bool = True
    ):
        """
        初始化记忆系统

        Args:
            checkpoint_file: SQLite 持久化文件路径(为 None 时用内存)
            long_term_file: 长期记忆 JSON 文件路径(用于 compress)
            thread_id: 会话线程 ID(为 None 时自动生成)
            short_term_size: 仅兼容旧 API(checkpoint 不限容量)
            use_sqlite: True=SQLite持久化, False=内存(调试用)
        """
        self.checkpoint_file = checkpoint_file
        self.long_term_file = long_term_file
        self.thread_id = thread_id or f"thread-{uuid.uuid4().hex[:8]}"
        self.short_term_size = short_term_size
        self.use_sqlite = use_sqlite and checkpoint_file is not None

        # 长期记忆(独立于 checkpoint,用于 compress)
        self.long_term_memory: List[Dict[str, Any]] = []

        # 初始化 checkpointer
        self._checkpointer = self._init_checkpointer()

        # 加载长期记忆
        if long_term_file and os.path.exists(long_term_file):
            self._load_long_term_memory()

    # ============ Checkpointer 管理 ============

    def _init_checkpointer(self):
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

    def get_checkpointer(self):
        """获取 checkpointer 实例(传给 create_agent)"""
        return self._checkpointer

    def get_config(self) -> Dict[str, Any]:
        """获取 invoke 需要传入的 config"""
        return {"configurable": {"thread_id": self.thread_id}}

    # ============ 会话(Thread)管理 ============

    def new_thread(self) -> str:
        """开启新会话(原会话保留在数据库)"""
        self.thread_id = f"thread-{uuid.uuid4().hex[:8]}"
        return self.thread_id

    def switch_thread(self, thread_id: str) -> bool:
        """
        切换到指定会话(恢复历史)

        Returns:
            True=该 thread 有 checkpoint 记录, False=新会话(无历史)
        """
        self.thread_id = thread_id
        # 如果数据库里有记录,返回 True;否则是新建会话
        return thread_id in self.list_threads()

    def list_threads(self) -> List[str]:
        """列出数据库中所有 thread_id"""
        if not self.use_sqlite:
            return [self.thread_id]
        try:
            with sqlite3.connect(self.checkpoint_file, check_same_thread=False) as conn:
                cursor = conn.execute(
                    "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
                )
                return [row[0] for row in cursor.fetchall()]
        except Exception:
            return [self.thread_id]

    def delete_thread(self, thread_id: str) -> bool:
        """
        删除指定会话(从 SQLite checkpoint 表移除)

        Args:
            thread_id: 要删除的会话 ID

        Returns:
            True=删除成功, False=失败或不存在
        """
        if not self.use_sqlite:
            # 内存模式:无法删除单个 thread,只能清空
            return False
        try:
            with sqlite3.connect(self.checkpoint_file, check_same_thread=False) as conn:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                    (thread_id,)
                )
                checkpoint_rows = cursor.fetchone()[0]
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'writes'"
                )
                has_writes = cursor.fetchone() is not None
                write_rows = 0
                if has_writes:
                    cursor = conn.execute(
                        "SELECT COUNT(*) FROM writes WHERE thread_id = ?",
                        (thread_id,)
                    )
                    write_rows = cursor.fetchone()[0]
            deleted_rows = checkpoint_rows + write_rows
            if deleted_rows == 0:
                return False

            # 优先使用 Saver 的删除契约，它会同步清理 checkpoints 与 writes。
            delete_thread = getattr(self._checkpointer, "delete_thread", None)
            if callable(delete_thread):
                delete_thread(thread_id)
            else:
                with sqlite3.connect(self.checkpoint_file, check_same_thread=False) as conn:
                    conn.execute(
                        "DELETE FROM checkpoints WHERE thread_id = ?",
                        (thread_id,)
                    )
                    if has_writes:
                        conn.execute(
                            "DELETE FROM writes WHERE thread_id = ?",
                            (thread_id,)
                        )
            # 如果删的是当前会话,自动切到一个剩余的会话(或新建)
            if thread_id == self.thread_id:
                remaining = self.list_threads()
                self.thread_id = remaining[0] if remaining else self.new_thread()
            return True
        except Exception as e:
            print(f"[删除会话失败] {e}")
            return False

    # ============ 消息获取 ============

    def get_messages(self) -> List[BaseMessage]:
        """从 checkpoint 获取当前 thread 的所有消息"""
        config = self.get_config()
        try:
            # langgraph 1.x: 用 get_tuple 获取 CheckpointTuple
            tup = self._checkpointer.get_tuple(config)
            if tup and tup.checkpoint:
                channel_values = tup.checkpoint.get("channel_values", {})
                if "messages" in channel_values:
                    return list(channel_values["messages"])
        except Exception:
            pass
        return []

    def export_thread(self, thread_id: Optional[str] = None, fmt: str = "text") -> str:
        """
        将指定会话(默认当前会话)的对话导出为可读文本

        Args:
            thread_id: 要导出的会话 ID(为 None 时导出当前会话)
            fmt: 导出格式,目前支持 'text'(默认) 与 'markdown'

        Returns:
            格式化后的对话文本
        """
        target = thread_id or self.thread_id
        saved = self.thread_id
        self.thread_id = target
        try:
            msgs = self.get_messages() or []
        finally:
            self.thread_id = saved

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
            content = getattr(m, "content", "")
            if isinstance(content, list):
                content = " ".join(str(x) for x in content)
            text = str(content).strip()
            if not text:
                continue
            if fmt == "markdown":
                blocks.append(f"**{role}**:\n\n{text}")
            else:
                blocks.append(f"【{role}】\n{text}")

        sep = "\n\n---\n\n" if fmt == "markdown" else "\n\n"
        header = f"# 对话导出 - {target}\n\n" if fmt == "markdown" else f"对话导出 - {target}\n{'='*40}\n"
        return header + sep.join(blocks)

    def get_short_term(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """兼容旧 API:从 checkpoint 取消息转为 dict 格式"""
        msgs = self.get_messages()
        if limit:
            msgs = msgs[-limit:]
        elif self.short_term_size:
            msgs = msgs[-self.short_term_size:]
        result = []
        for m in msgs:
            role = "user"
            if isinstance(m, AIMessage):
                role = "assistant"
            elif isinstance(m, SystemMessage):
                role = "system"
            result.append({"role": role, "content": m.content})
        return result

    # ============ 长期记忆(用于 compress) ============

    def add(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """
        添加记忆

        注意:checkpoint 会在 agent invoke 时自动保存对话,
        此方法只处理 important=True 的重要记忆(写入 memory.json 供 compress)。
        普通对话不需要调用此方法(由 checkpoint 自动处理)。
        """
        if metadata and metadata.get("important", False):
            item = {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata
            }
            self.long_term_memory.append(item)
            self._save_long_term_memory()

    def get_long_term(self, limit: int = 5) -> List[Dict[str, str]]:
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
        self.long_term_memory.clear()
        if self.long_term_file and os.path.exists(self.long_term_file):
            os.remove(self.long_term_file)

    def _save_long_term_memory(self):
        """保存长期记忆到文件"""
        if not self.long_term_file:
            return
        parent = os.path.dirname(os.path.abspath(self.long_term_file))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.long_term_file, 'w', encoding='utf-8') as f:
            json.dump(self.long_term_memory, f, ensure_ascii=False, indent=2)

    def _load_long_term_memory(self):
        """从文件加载长期记忆"""
        if not self.long_term_file or not os.path.exists(self.long_term_file):
            return
        try:
            with open(self.long_term_file, 'r', encoding='utf-8') as f:
                self.long_term_memory = json.load(f)
        except (json.JSONDecodeError, IOError):
            self.long_term_memory = []

    def summarize(self) -> Dict[str, Any]:
        """获取记忆摘要统计"""
        return {
            "thread_id": self.thread_id,
            "checkpoint_messages": len(self.get_messages()),
            "checkpoint_backend": "sqlite" if self.use_sqlite else "memory",
            "checkpoint_file": self.checkpoint_file or "(内存)",
            "long_term_count": len(self.long_term_memory),
            "total_threads": len(self.list_threads())
        }

    # ============ 长上下文裁剪与压缩 ============

    def maybe_compact(
        self,
        max_context_messages: int,
        context_trim_keep: int,
        summarize_callback,
        recreate_agent_callback,
        verbose: bool = False
    ):
        """
        长上下文裁剪: 当当前会话消息数超过 max_context_messages 时,
        将较早的消息用 LLM 摘要,开启新会话并把摘要注入后续 system prompt,
        从而避免撞上 LLM 上下文窗口。

        阈值 max_context_messages <= 0 时关闭(默认)。

        Args:
            max_context_messages: 最大消息数阈值
            context_trim_keep: 裁剪时保留最近 N 条消息
            summarize_callback: 生成摘要的回调函数(接收消息列表,返回摘要字符串)
            recreate_agent_callback: 重建 Agent 的回调函数(接收摘要,返回新 executor)
            verbose: 是否打印裁剪信息

        Returns:
            摘要字符串(未触发裁剪时返回 None)
        """
        if not max_context_messages or max_context_messages <= 0:
            return None

        msgs = self.get_messages()
        if len(msgs) <= max_context_messages:
            return None

        keep = min(context_trim_keep, max(len(msgs) - 1, 0))
        old = msgs[:-keep] if keep > 0 else msgs
        retained = msgs[-keep:] if keep > 0 else []
        summary = summarize_callback(old)
        if not summary:
            return None

        # 开启新会话,把历史摘要注入 system prompt(保留上下文精华)
        old_tid = self.thread_id
        self.new_thread()

        # 通过回调重建 Agent
        new_executor = recreate_agent_callback(summary)
        if retained:
            # 摘要只覆盖旧消息；最近消息原样写入新线程，避免裁剪后丢失上下文。
            new_executor.update_state(
                self.get_config(),
                {"messages": retained},
            )

        if verbose:
            print(
                f"\n[上下文裁剪] 会话过长({len(msgs)} 条),已自动摘要历史并开启新会话: "
                f"{self.thread_id} (原 {old_tid})"
            )

        return summary

    def compress_memory(self, summarize_callback) -> Dict[str, Any]:
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
            "content": f"[历史记忆摘要 {datetime.now().isoformat()}]\n{summary}",
            "timestamp": datetime.now().isoformat(),
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
