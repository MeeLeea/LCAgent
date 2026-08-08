"""会话状态存储 - 基于 LangGraph Store 的 per-session 隔离封装。

按 session_id（= thread_id）隔离两类瞬态会话状态：
- execution_history: 工具调用执行历史（有界队列，仅 CLI 展示用）
- pending_interrupts: 挂起的中断模式（run/chat，记录是哪种执行被中断）

注意：
- ``active_skills`` 不在此处，而是放入 ``LCAgentState``（随 checkpoint per-thread
  持久化），由 ``SkillInjectionMiddleware`` 在 model 调用时从 state 直接读取。
- Store 实例本身不持有任何会话级可变状态，可安全在多会话间共享。
- 使用异步 API（aput/aget/adelete）保持与持久化 backend（如 PostgresStore）兼容；
  InMemoryStore 的异步方法内部即内存操作，不阻塞事件循环。
"""
from __future__ import annotations

import logging
from typing import Any

from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

logger = logging.getLogger(__name__)

# namespace 前缀常量
_NS_PREFIX = "lcagent"
_NS_SESSIONS = "sessions"


def _ns(session_id: str, kind: str) -> tuple[str, ...]:
    """构建 4 层 namespace tuple，按 session_id + 状态类别隔离。"""
    return (_NS_PREFIX, _NS_SESSIONS, session_id, kind)


class SessionStore:
    """LangGraph Store 的会话状态封装层。

    Args:
        backend: 底层 BaseStore 实例。为 None 时用 InMemoryStore。
        max_history: 单会话 execution_history 最大条数（超出裁剪旧条目）。
    """

    def __init__(self, backend: BaseStore | None = None, max_history: int = 100):
        self._store: BaseStore = backend or InMemoryStore()
        self._max_history = max_history

    @property
    def backend(self) -> BaseStore:
        """暴露底层 Store，供 ``create_agent(store=...)`` 使用。"""
        return self._store

    @property
    def max_history(self) -> int:
        return self._max_history

    # ============ execution_history ============

    async def aget_history(self, session_id: str) -> list[dict[str, Any]]:
        """读取会话的执行历史（完整列表）。无记录时返回空列表。"""
        item = await self._store.aget(_ns(session_id, "history"), key="entries")
        if item is None:
            return []
        return list(item.value.get("entries", []))

    async def aappend_history(self, session_id: str, entry: dict[str, Any]) -> None:
        """追加单条执行历史，自动裁剪到 ``max_history``。

        读改写语义：同一会话内 ainvoke 串行调用，不同会话 namespace 隔离，
        故无需额外加锁。
        """
        history = await self.aget_history(session_id)
        history.append(entry)
        if len(history) > self._max_history:
            history = history[-self._max_history:]
        await self._store.aput(
            _ns(session_id, "history"),
            key="entries",
            value={"entries": history},
        )

    async def aextend_history(self, session_id: str, entries: list[dict[str, Any]]) -> None:
        """批量追加多条执行历史，自动裁剪到 ``max_history``。

        比 ``aappend_history`` 更高效：一次读改写即可追加多条，
        适用于 ``_arecord_tool_steps`` 一次 turn 内记录多个工具调用。
        """
        if not entries:
            return
        history = await self.aget_history(session_id)
        history.extend(entries)
        if len(history) > self._max_history:
            history = history[-self._max_history:]
        await self._store.aput(
            _ns(session_id, "history"),
            key="entries",
            value={"entries": history},
        )

    async def aget_recorded_call_ids(self, session_id: str) -> set[str]:
        """读取已记录的 tool_call_id 集合（用于跨 turn 去重）。"""
        item = await self._store.aget(_ns(session_id, "history"), key="call_ids")
        if item is None:
            return set()
        return set(item.value.get("ids", []))

    async def aadd_recorded_call_ids(self, session_id: str, new_ids: set[str]) -> None:
        """批量追加多个 tool_call_id 到去重集合。

        比 ``aappend_history`` 更高效：一次读改写即可追加多个 ID，
        适用于 ``_arecord_tool_steps`` 一次 turn 内记录多个工具调用。
        """
        if not new_ids:
            return
        ids = await self.aget_recorded_call_ids(session_id)
        ids.update(new_ids)
        await self._store.aput(
            _ns(session_id, "history"),
            key="call_ids",
            value={"ids": sorted(ids)},
        )

    async def aclear_history(self, session_id: str) -> None:
        """清空执行历史 + tool_call_id 去重集合。"""
        await self._safe_delete(_ns(session_id, "history"), "entries")
        await self._safe_delete(_ns(session_id, "history"), "call_ids")

    # ============ pending_interrupts ============

    async def aget_interrupt_mode(self, session_id: str) -> str | None:
        """读取会话的挂起中断模式（None=无中断）。"""
        item = await self._store.aget(_ns(session_id, "interrupts"), key="pending")
        if item is None:
            return None
        return item.value.get("mode")

    async def aset_interrupt_mode(self, session_id: str, mode: str) -> None:
        """记录挂起中断模式（'run' / 'chat'）。"""
        await self._store.aput(
            _ns(session_id, "interrupts"),
            key="pending",
            value={"mode": mode},
        )

    async def aclear_interrupt(self, session_id: str) -> None:
        """清除挂起中断状态。"""
        await self._safe_delete(_ns(session_id, "interrupts"), "pending")

    # ============ 会话级清理 ============

    async def adelete_session(self, session_id: str) -> None:
        """删除指定会话在 Store 中的全部状态（history + interrupts）。"""
        await self.aclear_history(session_id)
        await self.aclear_interrupt(session_id)

    # ============ 内部辅助 ============

    async def _safe_delete(self, namespace: tuple[str, ...], key: str) -> None:
        """幂等删除：key 不存在时不报错。"""
        try:
            await self._store.adelete(namespace, key=key)
        except Exception as error:  # noqa: BLE001 - Store backend 差异需兜底
            logger.debug("Store 删除失败 [%s/%s]: %s", namespace, key, error)


__all__ = ["SessionStore"]
