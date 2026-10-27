"""BASELINE 表征测试 — 固定改造前（纯 LRU）的 prune 淘汰行为。

用途：在 score 时间衰减改造**之前**先锁定既有行为，确保改造只改变
"active 组内排序依据"，不改变"superseded 优先淘汰"这一层语义。

时序口径：所有时间字段由固定基准时间派生，测试不依赖真实墙钟。
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

from memory.models import ThreadFactItem
from memory.store import ThreadMemoryStore

# 固定基准时间（表征测试仅比较相对新旧，不注入 now 到生产代码）
_BASE = datetime(2026, 6, 1, 12, 0, 0)  # noqa: DTZ001 — 项目时间字段统一 naive


def _iso(days_ago: float) -> str:
    """生成相对 _BASE 过去 days_ago 天的 naive ISO 字符串。"""
    return (_BASE - timedelta(days=days_ago)).isoformat()


def test_baseline_superseded_evicted_before_active():
    """BASELINE：超限时 superseded / stale 无条件先淘汰，active 全部保留。"""

    async def run():
        store = ThreadMemoryStore(max_facts=3)
        # 3 条 active：均"久未访问"（last_used_at 都早），仍应全部存活
        for i in range(3):
            await store.save_fact(
                "t1",
                ThreadFactItem(
                    content=f"active-{i}",
                    last_used_at=_iso(100 - i),
                    create_time=_iso(100 - i),
                ),
            )
        # 2 条 superseded（内容更新但状态非 active）
        for i in range(2):
            item = ThreadFactItem(content=f"superseded-{i}")
            await store.save_fact("t1", item)
            await store.supersede_facts("t1", [item.fact_id])

        removed = await store.prune_facts("t1")

        assert removed == 2
        remaining = await store.query_facts("t1")
        assert len(remaining) == 3
        assert all(f.status == "active" for f in remaining)
        assert {f.content for f in remaining} == {
            "active-0",
            "active-1",
            "active-2",
        }

    asyncio.run(run())


def test_baseline_active_oldest_last_used_evicted_first():
    """BASELINE：active 组内按 last_used_at 升序淘汰（最久未用先出，纯 LRU）。

    该用例固定的是"同 confidence 下旧条目先淘汰"的**可观测结果**——
    改造为 score 排序后，同 confidence 的 score 仍随 Δt 单调递减，
    故淘汰对象保持不变（守恒断言，改造后继续通过）。
    """

    async def run():
        store = ThreadMemoryStore(max_facts=2)
        for i in range(3):
            await store.save_fact(
                "t1",
                ThreadFactItem(
                    content=f"active-{i}",
                    confidence=0.8,  # 同档同值 → 排序只由 Δt 决定
                    last_used_at=_iso(30 - i),  # active-0 最旧
                    create_time=_iso(30 - i),
                ),
            )

        removed = await store.prune_facts("t1")

        assert removed == 1
        remaining = await store.query_facts("t1")
        contents = {f.content for f in remaining}
        assert len(remaining) == 2
        assert "active-0" not in contents  # 最旧者被淘汰
        assert contents == {"active-1", "active-2"}

    asyncio.run(run())


def test_baseline_prune_under_limit_returns_zero():
    """BASELINE：未超上限时 prune 为 no-op（返回 0，不改动数据）。"""

    async def run():
        store = ThreadMemoryStore(max_facts=5)
        for i in range(3):
            await store.save_fact("t1", ThreadFactItem(content=f"f-{i}"))

        assert await store.prune_facts("t1") == 0
        assert len(await store.query_facts("t1")) == 3

    asyncio.run(run())


def test_baseline_eviction_count_exact():
    """BASELINE：淘汰数量恰为超限条数（5 条 → max 3 → 淘汰 2）。"""

    async def run():
        store = ThreadMemoryStore(max_facts=3)
        for i in range(5):
            await store.save_fact(
                "t1",
                ThreadFactItem(
                    content=f"f-{i}",
                    last_used_at=_iso(50 - i),
                    create_time=_iso(50 - i),
                ),
            )

        assert await store.prune_facts("t1") == 2
        assert len(await store.query_facts("t1")) == 3

    asyncio.run(run())