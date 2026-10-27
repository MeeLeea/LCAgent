"""长期记忆 Store 业务读写封装 — 基于 LangGraph BaseStore 的分层隔离。

与 :class:`session.store.SessionStore` 平行设计，但负责长期记忆（facts），
而非瞬态会话状态（execution_history / interrupts）。

隔离方式（两级 namespace）：
- thread 级 namespace = ``(thread_id, "thread_facts")``，按 thread_id 天然隔离
  （conv / business 类记忆，会话内有效）
- agent 级 namespace = ``(agent_key, "global_facts")``，agent 级 namespace 跨进程共享
  （user_fact / lesson 类记忆；agent_key 默认 ``"global"``，可显式覆盖以隔离）
- 不同 thread 完全并行，无需加锁
- 同一 namespace 的写入由上层 :class:`ThreadMemoryLockPool` 保护

存储后端：
- 异步模式: ``AsyncSqliteStore``（可复用 checkpoint 的 SQLite 文件）
- 同步/测试模式: ``InMemoryStore``
"""
from __future__ import annotations

import logging
from typing import Any

from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from .models import (
    FACT_STATUS_ACTIVE,
    FACT_STATUS_STALE,
    FACT_STATUS_SUPERSEDED,
    MemoryCategory,
    ThreadFactItem,
    _naive_now,
    memory_score,
)

logger = logging.getLogger(__name__)

# namespace 常量
_NS_FACTS = "thread_facts"
_NS_AGENT = "global_facts"

# 淘汰状态优先级（显式分区，数值小者先淘汰）：
# - superseded / stale：已被取代或过时，审计价值最低 → 优先物理删除
# - active：按 memory_score 升序淘汰最低分
# 说明：Phase 2 将新增 forgotten（软删除，跳过不删且不占容量）作为第三分区，
# 届时只需在此映射补一个更大的优先级值并在 _prune_ns 中按映射分区，无需重写函数。
_PRUNE_STATUS_PRIORITY: dict[str, int] = {
    FACT_STATUS_SUPERSEDED: 0,
    FACT_STATUS_STALE: 0,
    FACT_STATUS_ACTIVE: 1,
}
"""状态 → 淘汰优先级映射（数值小者先淘汰）。未登记状态回退 _PRUNE_UNKNOWN_PRIORITY。"""

_PRUNE_UNKNOWN_PRIORITY = 0
"""未登记状态的默认淘汰优先级（保守：视为最高淘汰优先级，与旧 `0 if != active` 语义一致）。"""

# asearch 默认上限（LangGraph Store asearch 默认 limit=10，此处按容量上限动态取值，
# 确保 agent 级配置 max_agent_facts > 200 时不会被静默截断，避免漏读/漏淘汰）


class ThreadMemoryStore:
    """LangGraph Store 的长期记忆业务封装。

    薄封装层：Store 原始 API 直接使用，只做序列化/反序列化和业务逻辑。
    锁、防抖、LLM 抽取全部在上层中间件实现。

    Args:
        backend: 底层 BaseStore 实例。为 None 时用 InMemoryStore。
        max_facts: 单 thread 最大 fact 条数（超出时 LRU 淘汰）。
        max_agent_facts: agent 级（跨会话共享）最大 fact 条数（超出时 LRU 淘汰）。
        agent_key: agent 级 namespace 标识，agent 级 namespace 跨进程共享，
            默认 ``"global"``，可显式覆盖以隔离不同的 agent 级记忆。
    """

    def __init__(
        self,
        backend: BaseStore | None = None,
        max_facts: int = 50,
        max_agent_facts: int = 200,
        agent_key: str = "global",
    ):
        self._store: BaseStore = backend or InMemoryStore()
        self._max_facts = max_facts
        self._max_agent_facts = max_agent_facts
        self._agent_key = agent_key

    @property
    def backend(self) -> BaseStore:
        """暴露底层 Store，供 ``create_agent(store=...)`` 使用。"""
        return self._store

    @property
    def max_facts(self) -> int:
        return self._max_facts

    @property
    def max_agent_facts(self) -> int:
        """agent 级记忆容量上限。"""
        return self._max_agent_facts

    @property
    def agent_key(self) -> str:
        """agent 级 namespace 标识（跨进程共享，默认 ``"global"``）。"""
        return self._agent_key

    # ============ namespace 构建 ============

    def _facts_namespace(
        self, thread_id: str, scope: str = "thread"
    ) -> tuple[str, ...]:
        """构建 facts 的 namespace tuple。

        - ``scope == "agent"``：返回 ``(self._agent_key, "global_facts")``，
          agent 级 namespace 跨进程共享（agent_key 默认 ``"global"``，可显式覆盖以隔离）
        - 其它（默认 ``"thread"``）：返回 ``(thread_id, "thread_facts")``，
          per-thread 隔离的会话级记忆

        Args:
            thread_id: 会话线程 ID（agent 作用域下仅作占位，实际用 ``self._agent_key``）
            scope: 作用域，``"thread"`` 或 ``"agent"``

        Returns:
            namespace tuple
        """
        if scope == "agent":
            return (self._agent_key, _NS_AGENT)
        return (thread_id, _NS_FACTS)

    # ============ 内部公共方法（thread / agent 共用，消除重复） ============

    async def _query_facts_ns(
        self, ns: tuple[str, ...], active_only: bool = False
    ) -> list[ThreadFactItem]:
        """读取指定 namespace 全部 facts，按 ``create_time`` 升序排列。

        Args:
            ns: namespace tuple
            active_only: 为 True 时仅返回 ``status == "active"`` 的条目
                （superseded / stale 被过滤，供 prompt 注入与召回使用）；
                默认 False 返回全部（供审计 / 压缩 / 清理使用）
        """
        # 查询上限取 thread 级与 agent 级容量上限的较大值，确保 asearch 不会
        # 在 max_agent_facts 配到 >200 时静默截断（原硬编码 200 与默认值撞车）
        limit = max(self._max_facts, self._max_agent_facts, 1)
        items = await self._store.asearch(ns, limit=limit)
        facts = [ThreadFactItem.from_dict(item.value) for item in items]
        if active_only:
            facts = [f for f in facts if f.status == FACT_STATUS_ACTIVE]
        facts.sort(key=lambda f: f.create_time)
        return facts

    async def _save_facts_batch_ns(
        self, ns: tuple[str, ...], items: list[ThreadFactItem]
    ) -> None:
        """批量写入 facts 到指定 namespace（按 ``fact_id`` 幂等覆盖）。"""
        for item in items:
            await self._store.aput(ns, key=item.fact_id, value=item.to_dict())

    async def _prune_ns(self, ns: tuple[str, ...], max_facts: int) -> int:
        """容量淘汰：状态分区 + active 组内按 ``memory_score`` 升序淘汰。

        淘汰优先级（分区映射 ``_PRUNE_STATUS_PRIORITY``，数值小者先淘汰）：
        1. ``superseded`` / ``stale``（已被取代或过时，审计价值最低）优先物理删除
        2. ``active`` 条目按 ``memory_score`` 升序（分数最低者先淘汰；
           score = confidence^0.5 × 时间衰减，低置信/久未访问者分数低）

        score 为读取时计算的派生值，**不落库**（避免存储 stale score）。

        时序说明：本方法内以 ``_naive_now()`` 为基准时间逐条计算 score，
        同一轮淘汰内所有条目的基准时间一致（同一次 prune 调用），排序稳定。

        Returns:
            被淘汰的 fact 数量
        """
        facts = await self._query_facts_ns(ns)
        if len(facts) <= max_facts:
            return 0

        now = _naive_now()

        def _sort_key(fact: ThreadFactItem) -> tuple[int, float]:
            """淘汰排序键：(状态优先级升序, active 组内 score 升序)。"""
            priority = _PRUNE_STATUS_PRIORITY.get(
                fact.status, _PRUNE_UNKNOWN_PRIORITY
            )
            score = memory_score(
                fact.confidence,
                fact.last_used_at or None,
                fact.create_time or None,
                now=now,
            )
            return (priority, score)

        facts.sort(key=_sort_key)
        to_remove = facts[: len(facts) - max_facts]
        for fact in to_remove:
            await self._store.adelete(ns, key=fact.fact_id)
        return len(to_remove)

    async def _supersede_facts_ns(
        self, ns: tuple[str, ...], fact_ids: list[str]
    ) -> int:
        """将指定 facts 标记为 ``superseded``（被同主题新条目取代，保留审计）。

        仅对 ``status == "active"`` 的条目生效：已是非 active 的条目跳过，
        保留其原始 ``superseded_at`` 时间戳。

        Args:
            ns: namespace tuple
            fact_ids: 待标记的 fact_id 列表

        Returns:
            实际标记的 fact 数量（不存在或已非 active 的不计入）
        """
        now = _naive_now().isoformat()
        count = 0
        for fact_id in fact_ids:
            item = await self._store.aget(ns, key=fact_id)
            if item is None:
                continue
            fact = ThreadFactItem.from_dict(item.value)
            if fact.status != FACT_STATUS_ACTIVE:
                continue
            fact.status = FACT_STATUS_SUPERSEDED
            fact.superseded_at = now
            await self._store.aput(ns, key=fact.fact_id, value=fact.to_dict())
            count += 1
        return count

    async def _clear_ns(self, ns: tuple[str, ...]) -> int:
        """清空指定 namespace 全部 facts。"""
        facts = await self._query_facts_ns(ns)
        for fact in facts:
            await self._store.adelete(ns, key=fact.fact_id)
        return len(facts)

    # ============ 读操作（thread 级，不加锁，允许并发读） ============

    async def query_facts(
        self, thread_id: str, active_only: bool = False
    ) -> list[ThreadFactItem]:
        """读取 thread 全部长期记忆 facts。

        读操作不加锁，允许并发读。按 ``create_time`` 升序排列。

        Args:
            thread_id: 会话线程 ID
            active_only: 为 True 时仅返回 ``status == "active"`` 条目
                （superseded / stale 过滤，供召回与 prompt 注入使用）

        Returns:
            该 thread 的 facts 列表（可能为空）
        """
        return await self._query_facts_ns(
            self._facts_namespace(thread_id), active_only=active_only
        )

    async def count_facts(self, thread_id: str) -> int:
        """统计 thread 的 fact 数量。"""
        return len(await self.query_facts(thread_id))

    # ============ 写操作（thread 级，上层应加锁） ============

    async def save_fact(self, thread_id: str, item: ThreadFactItem) -> None:
        """写入单条 fact。

        使用 ``fact_id`` 作为 Store key，namespace + key 唯一定位。
        同一 ``fact_id`` 重复写入会覆盖（幂等）。

        注意：生产写入链路统一走 :meth:`save_facts_batch`，本方法仅保留供测试
        与少量单条写入场景使用，勿在生产循环中逐条调用。

        Args:
            thread_id: 会话线程 ID
            item: 记忆条目（``item.thread_id`` 会被自动设为 ``thread_id``）
        """
        item.thread_id = thread_id
        await self._store.aput(
            self._facts_namespace(thread_id),
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
        ns = self._facts_namespace(thread_id)
        for item in items:
            item.thread_id = thread_id
        await self._save_facts_batch_ns(ns, items)

    # ============ LRU 淘汰 ============

    async def prune_facts(self, thread_id: str) -> int:
        """容量淘汰：状态分区 + score 衰减，防止单 thread 记忆膨胀。

        当 fact 数量超过 ``max_facts`` 时，先淘汰 ``superseded`` / ``stale``
        条目，仍超限再按 ``memory_score`` 升序（分数最低者先淘汰）淘汰 active 条目。
        score = confidence^0.5 × 时间衰减（低置信/久未访问者分数低）。

        Args:
            thread_id: 会话线程 ID

        Returns:
            被淘汰的 fact 数量
        """
        pruned = await self._prune_ns(
            self._facts_namespace(thread_id), self._max_facts
        )
        if pruned:
            logger.info(
                "thread %s 记忆淘汰: %d 条 (剩余 %d)",
                thread_id,
                pruned,
                self._max_facts,
            )
        return pruned

    async def supersede_facts(self, thread_id: str, fact_ids: list[str]) -> int:
        """将 thread 级指定 facts 标记为 ``superseded``（同主题新条目取代）。

        写入合并路径调用：新记忆与既有 active 记忆 ``topic_key`` 相同时，
        旧条目标记取代（保留审计痕迹），由新条目接管该主题。

        Args:
            thread_id: 会话线程 ID
            fact_ids: 待标记的 fact_id 列表

        Returns:
            实际标记的条目数
        """
        return await self._supersede_facts_ns(
            self._facts_namespace(thread_id), fact_ids
        )

    async def touch_fact(self, thread_id: str, fact_id: str) -> None:
        """更新 fact 的 ``last_used_at``（用于 LRU 淘汰）。

        读取 facts 时调用，标记最近使用。非阻塞语义：失败仅记日志。

        Args:
            thread_id: 会话线程 ID
            fact_id: 记忆条目 ID
        """
        from .models import _naive_now

        ns = self._facts_namespace(thread_id)
        item = await self._store.aget(ns, key=fact_id)
        if item is None:
            return
        try:
            fact = ThreadFactItem.from_dict(item.value)
            fact.last_used_at = _naive_now().isoformat()
            await self._store.aput(
                ns,
                key=fact.fact_id,
                value=fact.to_dict(),
            )
        except Exception as error:
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
        cleared = await self._clear_ns(self._facts_namespace(thread_id))
        if cleared:
            logger.info("清理 thread %s 长期记忆: %d 条", thread_id, cleared)
        return cleared

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
        ns = self._facts_namespace(thread_id)
        facts = await self._query_facts_ns(ns)
        original_count = len(facts)

        # 清空旧 facts
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

    async def replace_agent_facts_with_summary(self, summary: str) -> dict[str, Any]:
        """用摘要替换全部 agent 级 facts（:meth:`replace_with_summary` 的 agent 作用域孪生）。

        删除 agent namespace（``(agent_key, "global_facts")``）下的全部旧 facts，
        替换为单条摘要条目，供 agent 级记忆压缩使用；不影响 thread 级记忆。

        Args:
            summary: LLM 生成的摘要文本

        Returns:
            ``{"success": bool, "original_count": int, "summary": str}``
        """
        ns = self._facts_namespace("", scope="agent")
        facts = await self._query_facts_ns(ns)
        original_count = len(facts)

        # 清空旧 facts
        for fact in facts:
            await self._store.adelete(ns, key=fact.fact_id)

        # 写入摘要条目（dataclass 默认 scope="thread"，此处必须显式指定 "agent"）
        summary_item = ThreadFactItem(
            thread_id=self._agent_key,
            content=f"[历史记忆摘要]\n{summary}",
            category=MemoryCategory.IMPORTANT_CONVERSATION.value,
            confidence=1.0,
            scope="agent",
        )
        await self._store.aput(ns, key=summary_item.fact_id, value=summary_item.to_dict())

        logger.info("agent %s 记忆压缩: %d 条 → 1 条摘要", self._agent_key, original_count)
        return {
            "success": True,
            "original_count": original_count,
            "summary": summary,
        }

    # ============ Agent 级长期记忆（跨会话共享） ============

    async def query_agent_facts(
        self, active_only: bool = False
    ) -> list[ThreadFactItem]:
        """读取 agent 级全部长期记忆 facts。

        agent 级记忆跨会话共享（user_fact / lesson 类），按 ``create_time`` 升序排列。

        Args:
            active_only: 为 True 时仅返回 ``status == "active"`` 条目
                （superseded / stale 过滤，供召回与 prompt 注入使用）

        Returns:
            agent 级 facts 列表（可能为空）
        """
        return await self._query_facts_ns(
            self._facts_namespace("", scope="agent"), active_only=active_only
        )

    async def save_agent_fact(self, item: ThreadFactItem) -> None:
        """写入单条 agent 级 fact。

        ``item.thread_id`` 会被强制设为 ``self._agent_key``，便于溯源
        （与 thread 级 :meth:`save_fact` 强制 thread_id 的做法对称）。
        同一 ``fact_id`` 重复写入会覆盖（幂等）。

        注意：生产写入链路统一走 :meth:`save_agent_facts_batch`，本方法仅保留
        供测试与少量单条写入场景使用，勿在生产循环中逐条调用。

        Args:
            item: 记忆条目
        """
        item.thread_id = self._agent_key
        await self._store.aput(
            self._facts_namespace("", scope="agent"),
            key=item.fact_id,
            value=item.to_dict(),
        )

    async def save_agent_facts_batch(
        self, items: list[ThreadFactItem]
    ) -> None:
        """批量写入 agent 级 facts。

        Args:
            items: 记忆条目列表（``item.thread_id`` 会被统一设为 ``self._agent_key``）
        """
        for item in items:
            item.thread_id = self._agent_key
        await self._save_facts_batch_ns(
            self._facts_namespace("", scope="agent"), items
        )

    async def count_agent_facts(self) -> int:
        """统计 agent 级 fact 数量。"""
        return len(await self.query_agent_facts())

    async def supersede_agent_facts(self, fact_ids: list[str]) -> int:
        """将 agent 级指定 facts 标记为 ``superseded``（同主题新条目取代）。

        agent namespace 跨进程共享，调用方（写中间件）必须已持有 agent 锁，
        本方法不再自行加锁（与 :meth:`save_agent_facts_batch` 一致）。

        Args:
            fact_ids: 待标记的 fact_id 列表

        Returns:
            实际标记的条目数
        """
        return await self._supersede_facts_ns(
            self._facts_namespace("", scope="agent"), fact_ids
        )

    async def clear_agent_facts(self) -> int:
        """清空 agent 级全部长期记忆。

        Returns:
            被清除的 fact 数量
        """
        cleared = await self._clear_ns(self._facts_namespace("", scope="agent"))
        if cleared:
            logger.info("清理 agent %s 长期记忆: %d 条", self._agent_key, cleared)
        return cleared

    async def prune_agent_facts(self) -> int:
        """按 ``memory_score`` 淘汰最低分 fact，控制 agent 级记忆在上限内。

        当 fact 数量超过 ``max_agent_facts`` 时，先淘汰非 active 条目，
        其余 active 条目按 score（置信度软化 × 时间衰减）升序淘汰。

        Returns:
            被淘汰的 fact 数量
        """
        pruned = await self._prune_ns(
            self._facts_namespace("", scope="agent"), self._max_agent_facts
        )
        if pruned:
            logger.info(
                "agent %s 记忆淘汰: %d 条 (剩余 %d)",
                self._agent_key,
                pruned,
                self._max_agent_facts,
            )
        return pruned

    async def touch_agent_fact(self, fact_id: str) -> None:
        """更新 agent 级 fact 的 ``last_used_at``（用于 LRU 淘汰）。

        agent 级 facts 在读中间件注入 prompt 时被使用，调用本方法标记最近使用，
        与 thread 级 :meth:`touch_fact` 对称。非阻塞语义：失败仅记日志，
        不影响主流程。

        Args:
            fact_id: 记忆条目 ID
        """
        from .models import _naive_now

        ns = self._facts_namespace("", scope="agent")
        item = await self._store.aget(ns, key=fact_id)
        if item is None:
            return
        try:
            fact = ThreadFactItem.from_dict(item.value)
            fact.last_used_at = _naive_now().isoformat()
            await self._store.aput(
                ns,
                key=fact.fact_id,
                value=fact.to_dict(),
            )
        except Exception as error:
            logger.debug("touch_agent_fact 失败 [%s]: %s", fact_id, error)


__all__ = ["ThreadMemoryStore"]
