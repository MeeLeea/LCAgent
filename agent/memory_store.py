"""长期记忆 Store 业务读写封装 — 基于 LangGraph BaseStore 的 per-thread 隔离。

与 :class:`session.store.SessionStore` 平行设计，但负责长期记忆（facts），
而非瞬态会话状态（execution_history / interrupts）。

隔离方式：
- namespace = ``(thread_id, "thread_facts")``，按 thread_id 天然隔离
- 不同 thread 完全并行，无需加锁
- 同一 thread 的写入由上层 :class:`ThreadMemoryLockPool` 保护

存储后端：
- 异步模式: ``AsyncSqliteStore``（可复用 checkpoint 的 SQLite 文件）
- 同步/测试模式: ``InMemoryStore``
"""
from __future__ import annotations

import logging
from typing import Any

from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from .memory_models import MemoryCategory, ThreadFactItem

logger = logging.getLogger(__name__)

# namespace 常量
_NS_FACTS = "thread_facts"

# asearch 默认上限（LangGraph Store asearch 默认 limit=10，此处提高以确保取全）
_SEARCH_LIMIT = 200


def _facts_namespace(thread_id: str) -> tuple[str, ...]:
    """构建 thread facts 的 namespace tuple。

    格式: ``(thread_id, "thread_facts")``
    """
    return (thread_id, _NS_FACTS)


class ThreadMemoryStore:
    """LangGraph Store 的长期记忆业务封装。

    薄封装层：Store 原始 API 直接使用，只做序列化/反序列化和业务逻辑。
    锁、防抖、LLM 抽取全部在上层中间件实现。

    Args:
        backend: 底层 BaseStore 实例。为 None 时用 InMemoryStore。
        max_facts: 单 thread 最大 fact 条数（超出时 LRU 淘汰）。
    """

    def __init__(self, backend: BaseStore | None = None, max_facts: int = 50):
        self._store: BaseStore = backend or InMemoryStore()
        self._max_facts = max_facts

    @property
    def backend(self) -> BaseStore:
        """暴露底层 Store，供 ``create_agent(store=...)`` 使用。"""
        return self._store

    @property
    def max_facts(self) -> int:
        return self._max_facts

    # ============ 读操作（不加锁，允许并发读） ============

    async def query_facts(self, thread_id: str) -> list[ThreadFactItem]:
        """读取 thread 全部长期记忆 facts。

        读操作不加锁，允许并发读。按 ``create_time`` 升序排列。

        Args:
            thread_id: 会话线程 ID

        Returns:
            该 thread 的全部 facts 列表（可能为空）
        """
        ns = _facts_namespace(thread_id)
        items = await self._store.asearch(ns, limit=_SEARCH_LIMIT)
        facts = [ThreadFactItem.from_dict(item.value) for item in items]
        facts.sort(key=lambda f: f.create_time)
        return facts

    async def query_facts_by_category(
        self, thread_id: str, category: MemoryCategory
    ) -> list[ThreadFactItem]:
        """按分类筛选 facts。

        Args:
            thread_id: 会话线程 ID
            category: 记忆分类

        Returns:
            匹配分类的 facts 列表
        """
        facts = await self.query_facts(thread_id)
        return [f for f in facts if f.category == category.value]

    async def count_facts(self, thread_id: str) -> int:
        """统计 thread 的 fact 数量。"""
        return len(await self.query_facts(thread_id))

    # ============ 写操作（上层应加锁） ============

    async def save_fact(self, thread_id: str, item: ThreadFactItem) -> None:
        """写入单条 fact。

        使用 ``fact_id`` 作为 Store key，namespace + key 唯一定位。
        同一 ``fact_id`` 重复写入会覆盖（幂等）。

        Args:
            thread_id: 会话线程 ID
            item: 记忆条目（``item.thread_id`` 会被自动设为 ``thread_id``）
        """
        item.thread_id = thread_id
        await self._store.aput(
            _facts_namespace(thread_id),
            key=item.fact_id,
            value=item.to_dict(),
        )

    async def save_facts_batch(
        self, thread_id: str, items: list[ThreadFactItem]
    ) -> None:
        """批量写入 facts。

        比逐条 ``save_fact`` 更高效：减少 Store 交互次数。

        Args:
            thread_id: 会话线程 ID
            items: 记忆条目列表
        """
        ns = _facts_namespace(thread_id)
        for item in items:
            item.thread_id = thread_id
            await self._store.aput(ns, key=item.fact_id, value=item.to_dict())

    async def delete_fact(self, thread_id: str, fact_id: str) -> None:
        """删除单条 fact。

        Args:
            thread_id: 会话线程 ID
            fact_id: 记忆条目 ID
        """
        await self._store.adelete(_facts_namespace(thread_id), key=fact_id)

    # ============ LRU 淘汰 ============

    async def prune_facts(self, thread_id: str) -> int:
        """按 ``last_used_at`` 淘汰最久未使用 fact，防止单 thread 记忆膨胀。

        当 fact 数量超过 ``max_facts`` 时，淘汰最旧的条目。

        Args:
            thread_id: 会话线程 ID

        Returns:
            被淘汰的 fact 数量
        """
        facts = await self.query_facts(thread_id)
        if len(facts) <= self._max_facts:
            return 0

        # 按 last_used_at 升序，淘汰最久未使用
        facts.sort(key=lambda f: f.last_used_at)
        to_remove = facts[: len(facts) - self._max_facts]
        ns = _facts_namespace(thread_id)
        for fact in to_remove:
            await self._store.adelete(ns, key=fact.fact_id)

        logger.info(
            "thread %s 记忆淘汰: %d 条 (剩余 %d)",
            thread_id,
            len(to_remove),
            len(facts) - len(to_remove),
        )
        return len(to_remove)

    async def touch_fact(self, thread_id: str, fact_id: str) -> None:
        """更新 fact 的 ``last_used_at``（用于 LRU 淘汰）。

        读取 facts 时调用，标记最近使用。非阻塞语义：失败仅记日志。

        Args:
            thread_id: 会话线程 ID
            fact_id: 记忆条目 ID
        """
        from .memory_models import _naive_now

        item = await self._store.aget(_facts_namespace(thread_id), key=fact_id)
        if item is None:
            return
        try:
            fact = ThreadFactItem.from_dict(item.value)
            fact.last_used_at = _naive_now().isoformat()
            await self._store.aput(
                _facts_namespace(thread_id),
                key=fact.fact_id,
                value=fact.to_dict(),
            )
        except Exception as error:  # noqa: BLE001
            logger.debug("touch_fact 失败 [%s/%s]: %s", thread_id, fact_id, error)

    # ============ 会话级清理 ============

    async def clear_thread_memory(self, thread_id: str) -> int:
        """清空指定 thread 的全部长期记忆。

        会话删除时调用，一键清空该 thread 全部 facts。

        Args:
            thread_id: 会话线程 ID

        Returns:
            被清除的 fact 数量
        """
        facts = await self.query_facts(thread_id)
        ns = _facts_namespace(thread_id)
        for fact in facts:
            await self._store.adelete(ns, key=fact.fact_id)

        if facts:
            logger.info("清理 thread %s 长期记忆: %d 条", thread_id, len(facts))
        return len(facts)

    # ============ 压缩摘要 ============

    async def replace_with_summary(
        self, thread_id: str, summary: str
    ) -> dict[str, Any]:
        """用摘要替换全部 facts（替代旧 ``acompress_memory``）。

        将所有旧 facts 替换为单条摘要条目，大幅减少 token 占用。

        Args:
            thread_id: 会话线程 ID
            summary: LLM 生成的摘要文本

        Returns:
            ``{"success": bool, "original_count": int, "summary": str}``
        """
        facts = await self.query_facts(thread_id)
        original_count = len(facts)

        # 清空旧 facts
        ns = _facts_namespace(thread_id)
        for fact in facts:
            await self._store.adelete(ns, key=fact.fact_id)

        # 写入摘要条目
        summary_item = ThreadFactItem(
            thread_id=thread_id,
            content=f"[历史记忆摘要]\n{summary}",
            category=MemoryCategory.IMPORTANT_CONVERSATION.value,
            confidence=1.0,
        )
        await self._store.aput(ns, key=summary_item.fact_id, value=summary_item.to_dict())

        logger.info(
            "thread %s 记忆压缩: %d 条 → 1 条摘要",
            thread_id,
            original_count,
        )
        return {
            "success": True,
            "original_count": original_count,
            "summary": summary,
        }


__all__ = ["ThreadMemoryStore"]
