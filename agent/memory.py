"""
记忆模块 - 基于LangChain Memory，支持短期记忆和长期记忆
"""
from typing import List, Dict, Any, Optional
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from collections import deque
import json
import os
from datetime import datetime


class AgentMemory:
    """记忆管理类，结合LangChain消息和持久化长期记忆"""

    def __init__(
        self,
        short_term_size: int = 10,
        long_term_file: Optional[str] = None
    ):
        """
        初始化记忆系统

        Args:
            short_term_size: 短期记忆容量
            long_term_file: 长期记忆存储文件路径
        """
        self.short_term_size = short_term_size
        self.short_term_memory = deque(maxlen=short_term_size)
        self.long_term_file = long_term_file
        self.long_term_memory: List[Dict[str, Any]] = []

        # LangChain 消息历史（用于与LangChain组件交互）
        self.chat_history = InMemoryChatMessageHistory()

        if long_term_file and os.path.exists(long_term_file):
            self._load_long_term_memory()

    def add(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """
        添加一条记忆

        Args:
            role: 角色 (user/assistant/system)
            content: 内容
            metadata: 额外元数据，设置 {"important": True} 会存入长期记忆
        """
        memory_item = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }

        # 添加到短期记忆
        self.short_term_memory.append(memory_item)

        # 添加到LangChain消息历史
        if role == "user":
            self.chat_history.add_message(HumanMessage(content=content))
        elif role == "assistant":
            self.chat_history.add_message(AIMessage(content=content))
        elif role == "system":
            self.chat_history.add_message(SystemMessage(content=content))

        # 根据重要性决定是否存入长期记忆
        if metadata and metadata.get("important", False):
            self.long_term_memory.append(memory_item)
            self._save_long_term_memory()

    def get_short_term(self) -> List[Dict[str, str]]:
        """获取短期记忆（字典格式，用于对话上下文）"""
        return [
            {"role": item["role"], "content": item["content"]}
            for item in self.short_term_memory
        ]

    def get_langchain_messages(self) -> List:
        """获取LangChain消息列表（用于LangChain组件）"""
        return self.chat_history.messages

    def get_long_term(self, limit: int = 5) -> List[Dict[str, str]]:
        """获取长期记忆"""
        recent = self.long_term_memory[-limit:] if limit else self.long_term_memory
        return [
            {"role": item["role"], "content": item["content"]}
            for item in recent
        ]

    def get_all_context(self, long_term_limit: int = 3) -> List[Dict[str, str]]:
        """获取完整的上下文（长期记忆 + 短期记忆）"""
        context = []
        context.extend(self.get_long_term(long_term_limit))
        context.extend(self.get_short_term())
        return context

    def clear_short_term(self):
        """清空短期记忆"""
        self.short_term_memory.clear()
        self.chat_history = InMemoryChatMessageHistory()

    def clear_long_term(self):
        """清空长期记忆"""
        self.long_term_memory.clear()
        if self.long_term_file and os.path.exists(self.long_term_file):
            os.remove(self.long_term_file)

    def _save_long_term_memory(self):
        """保存长期记忆到文件"""
        if not self.long_term_file:
            return
        # 自动创建父目录
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
            "short_term_count": len(self.short_term_memory),
            "short_term_capacity": self.short_term_size,
            "long_term_count": len(self.long_term_memory)
        }
