"""Session 模块 - 集中管理 checkpointer + Store 的会话状态，支撑 AgentCore 无状态化。

核心组件：
- SessionContext: 单个会话的运行时上下文（session_id + config + checkpointer）
- SessionStore: 基于 LangGraph Store 的 per-session 瞬态状态封装
                 （execution_history / pending_interrupts）
- SessionRegistry: 会话生命周期管理（生成/查询/删除/消息读取），桥接 checkpointer 与 Store
- create_checkpointer / create_async_checkpointer: checkpointer 工厂

设计要点：
- AgentCore 实例只持有不可变配置 + 共享的编译图/Store/checkpointer，可安全在多会话间复用
- active_skills 放入 LCAgentState（随 checkpoint per-thread 隔离），不在此处
- 所有会话级可变状态通过 session_id 显式隔离，消除实例级隐式状态
"""
from .checkpointer import create_async_checkpointer, create_checkpointer
from .context import SessionContext
from .registry import SessionRegistry
from .store import SessionStore

__all__ = [
    "SessionContext",
    "SessionRegistry",
    "SessionStore",
    "create_async_checkpointer",
    "create_checkpointer",
]
