"""会话上下文 - 封装单个会话的运行时引用（无状态数据载体）。

AgentCore 的所有执行方法接收 SessionContext，从中获取 config 与 session_id，
不再依赖实例上的可变会话状态。AgentCore 实例本身只持有不可变配置 +
共享的编译图/Store/checkpointer，可安全在多会话间复用。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver


@dataclass(slots=True)
class SessionContext:
    """单个会话的运行时上下文。

    Attributes:
        session_id: 逻辑会话标识（= LangGraph thread_id，用于 checkpoint 隔离）。
        config: LangGraph 调用 config，形如
            ``{"configurable": {"thread_id": session_id}, "recursion_limit": N}``。
        checkpointer: 共享的 checkpointer 引用（所有会话复用同一实例）。
    """

    session_id: str
    config: dict[str, Any]
    checkpointer: BaseCheckpointSaver

    @classmethod
    def create(
        cls,
        session_id: str,
        checkpointer: BaseCheckpointSaver,
        recursion_limit: int = 25,
    ) -> SessionContext:
        """构建会话上下文。

        Args:
            session_id: 会话线程 ID。
            checkpointer: 共享 checkpointer。
            recursion_limit: LangGraph 递归上限（= max_iterations）。

        Returns:
            SessionContext 实例。
        """
        config: dict[str, Any] = {
            "configurable": {"thread_id": session_id},
            "recursion_limit": recursion_limit,
        }
        return cls(session_id=session_id, config=config, checkpointer=checkpointer)


__all__ = ["SessionContext"]
