"""会话/Store 管理 Mixin - AgentCore 的会话生命周期与会话状态访问。

从 agent_core.py 抽离，职责：
- SessionRegistry / SessionManager 懒初始化与访问
- MemoryManager 注入（必须在 session_manager 首次访问前）
- SessionStore 访问 + 当前会话 ID 解析
- checkpoint 后端元信息

依赖 AgentCore 实例属性：_checkpointer / _session_store / _process_type /
max_iterations / _async_conn / _short_term_size / _initial_thread_id /
_memory_manager / _session_registry。
"""
from __future__ import annotations

from typing import Any

from session import SessionManager, SessionRegistry, SessionStore

logger = None  # 占位，实际 logger 由 agent_core 模块提供


class SessionMgmt:
    """会话管理 Mixin（供 AgentCore 多继承使用，自身不初始化状态）"""

    def _init_session_registry(self) -> None:
        """从传入的 LangGraph 原语构建 SessionRegistry。

        SessionRegistry 负责会话生命周期管理（new/switch/list/delete）+
        消息读取 + 瞬态状态隔离。
        checkpointer / store / async_conn 由入口程序 / MemoryContext 注入。
        """
        self._session_registry = SessionRegistry(
            checkpointer=self._checkpointer,
            store=self._session_store or SessionStore(),
            process_type=self._process_type,
            recursion_limit=self.max_iterations,
            async_conn=self._async_conn,
            short_term_size=self._short_term_size,
        )
        # 同步当前会话指针
        self._session_registry.current_session_id = self._initial_thread_id or self._session_registry.current_session_id

    @property
    def session(self) -> SessionRegistry:
        """会话注册表（管理 checkpointer + Store 的会话状态）。

        通过此属性访问会话管理 API：
        - ``agent.session.alist_sessions()``
        - ``agent.session.adelete_session(sid)``
        - ``agent.session.aget_messages(sid)``
        - ``agent.session.current_session_id``  (getter/setter)
        """
        if getattr(self, "_session_registry", None) is None:
            # 兼容测试中通过 object.__new__ 创建的实例
            self._init_session_registry()
        return self._session_registry  # type: ignore[return-value]

    @property
    def session_manager(self) -> SessionManager:
        """SessionManager 实例（三层架构 Session 层门面）。

        懒初始化：首次访问时构建。如果 ``_memory_manager`` 已由入口程序
        通过 ``set_memory_manager()`` 注入，则传递给 SessionManager。
        所有上层流量（流式/非流式/会话管理/记忆管理）应通过此入口。

        通过此属性访问对外接口：
        - ``await agent.session_manager.achat_stream(message)``
        - ``await agent.session_manager.achat(message)``
        - ``await agent.session_manager.aresume_stream(payload)``
        - ``agent.session_manager.new_session()``
        """
        if getattr(self, "_session_manager", None) is None:
            mm = getattr(self, "_memory_manager", None)
            self._session_manager = SessionManager(self, memory=mm)
        return self._session_manager

    def set_memory_manager(self, memory_manager: Any) -> None:
        """注入 MemoryManager（由入口程序在创建 MemoryContext 后调用）。

        必须在首次访问 ``session_manager`` 属性之前调用，
        否则 SessionManager 将在没有记忆功能的情况下初始化。

        Args:
            memory_manager: MemoryManager 实例
        """
        if getattr(self, "_session_manager", None) is not None:
            raise RuntimeError(
                "SessionManager 已初始化，无法再注入 MemoryManager。"
                "请在访问 session_manager 之前调用 set_memory_manager()。"
            )
        self._memory_manager = memory_manager

    def _get_store(self) -> SessionStore:
        """获取 SessionStore（惰性创建，兼容 object.__new__ 创建的测试实例）。

        execution_history / recorded_call_ids / pending_interrupts 全部
        通过此 Store 按 session_id 隔离，AgentCore 实例不再持有这些可变状态。
        """
        store = getattr(self, "_session_store", None)
        if store is None:
            store = SessionStore()
            self._session_store = store
        return store

    def _current_sid(self, thread_id: str | None = None) -> str:
        """获取当前会话 ID（兼容测试中通过 object.__new__ 创建的无 _session_registry 实例）。

        优先使用显式 thread_id；否则从 SessionRegistry 读取；
        若 _session_registry 不存在或无 current_session_id 属性则回退到 _initial_thread_id，
        最终回退到 "default"。
        """
        if thread_id is not None:
            return thread_id
        reg = getattr(self, "_session_registry", None)
        if reg is not None:
            sid = getattr(reg, "current_session_id", None)
            if sid is not None:
                return sid
        return getattr(self, "_initial_thread_id", None) or "default"

    @property
    def checkpoint_info(self) -> dict[str, Any]:
        """checkpoint 后端元信息（backend 类型 + 文件路径）。"""
        if self._checkpointer is not None:
            cls_name = type(self._checkpointer).__name__
            return {
                "checkpoint_backend": "sqlite" if "Sqlite" in cls_name else "memory",
                "checkpoint_file": getattr(self._checkpointer, "checkpoint_file", "(内存)") if hasattr(self._checkpointer, "checkpoint_file") else "(内存)",
            }
        return {"checkpoint_backend": "memory", "checkpoint_file": "(内存)"}

    def set_current_session(self, session_id: str) -> None:
        """设置当前会话 ID。

        Args:
            session_id: 目标会话 ID
        """
        self.session.current_session_id = session_id