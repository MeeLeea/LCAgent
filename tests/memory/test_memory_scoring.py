"""记忆淘汰 score 时间衰减测试 — memory_score 纯函数 + _prune_ns score 排序。

覆盖：
- memory_score 边界：Δt=0、三档 λ（高/中/低 confidence）、档位边界值 0.7 / 0.9、
  last_used_at 缺失回退 create_time、both 缺失回退 timeless、confidence 钳制、
  now 注入（测试不依赖真实墙钟）
- prune 行为（store 级 + 真实 AsyncSqliteStore 集成）：
  superseded 仍先于 active 被淘汰（回归）；
  active 组内按 memory_score 升序淘汰（低分先淘汰）；
  淘汰数量精确等于超限条数（max_facts / max_agent_facts）

时序口径：所有涉及时间的断言显式传入 now，禁用真实墙钟。
"""
from __future__ import annotations

import asyncio
import math
from datetime import UTC, datetime, timedelta

from langgraph.store.memory import InMemoryStore

from memory.agent_memory import AgentMemory
from memory.models import (
    FACT_STATUS_SUPERSEDED,
    ThreadFactItem,
    memory_score,
)
from memory.store import ThreadMemoryStore

# ── 测试基准时间（显式注入，杜绝墙钟依赖） ──
_NOW = datetime(2026, 6, 1, 12, 0, 0)  # noqa: DTZ001 — 项目时间字段统一 naive，与生产口径一致


def _iso(days_ago: float | None) -> str:
    """生成相对 _NOW 过去 days_ago 天的 naive ISO 字符串；None → 空串。"""
    if days_ago is None:
        return ""
    return (_NOW - timedelta(days=days_ago)).isoformat()


def _make_fact(content: str = "test fact", **kwargs) -> ThreadFactItem:
    return ThreadFactItem(content=content, **kwargs)


# ════════════════════════════════════════════════════════════════════════
#  memory_score — 基础公式与分档
# ════════════════════════════════════════════════════════════════════════


class TestMemoryScoreFormula:
    def test_delta_t_zero_returns_softened_confidence(self):
        """Δt=0：时间衰减因子为 1，score = confidence ** 0.5。"""
        conf = 0.81
        score = memory_score(conf, _NOW.isoformat(), None, now=_NOW)
        assert math.isclose(score, conf ** 0.5, rel_tol=1e-12)

    def test_high_band_half_life_180_days(self):
        """confidence=0.9（高档）：180 天后衰减为一半（λ=ln2/180）。"""
        start = _NOW - timedelta(days=180)
        score = memory_score(0.9, start.isoformat(), None, now=_NOW)
        assert math.isclose(score, (0.9 ** 0.5) * 0.5, rel_tol=1e-9)

    def test_mid_band_half_life_90_days(self):
        """confidence=0.8（中档）：90 天后衰减为一半（λ=ln2/90）。"""
        start = _NOW - timedelta(days=90)
        score = memory_score(0.8, start.isoformat(), None, now=_NOW)
        assert math.isclose(score, (0.8 ** 0.5) * 0.5, rel_tol=1e-9)

    def test_low_band_half_life_30_days(self):
        """confidence=0.5（低档）：30 天后衰减为一半（λ=ln2/30）。"""
        start = _NOW - timedelta(days=30)
        score = memory_score(0.5, start.isoformat(), None, now=_NOW)
        assert math.isclose(score, (0.5 ** 0.5) * 0.5, rel_tol=1e-9)

    def test_band_boundary_0_7_is_mid(self):
        """confidence=0.7 恰落在中档（0.7 ≤ c < 0.9）→ 半衰期 90 天。

        与 0.69（低档，半衰期 30 天）在同样 Δt 下衰减速度不同，可据此区分档位：
        取 Δt=45 天后，中档 score 应 > 低档 score（中档衰变更慢）。
        """
        start = _NOW - timedelta(days=45)
        mid = memory_score(0.7, start.isoformat(), None, now=_NOW)
        low = memory_score(0.69, start.isoformat(), None, now=_NOW)
        assert math.isclose(mid, (0.7 ** 0.5) * math.exp(-math.log(2) / 90 * 45), rel_tol=1e-9)
        # 中档 45 天 = 半衰 90 天的一半 → e^(-ln2/2)=1/sqrt(2)≈0.707
        assert mid > low

    def test_band_boundary_0_9_is_high(self):
        """confidence=0.9 恰落在高档（c ≥ 0.9）→ 半衰期 180 天。

        与 0.89（中档，半衰期 90 天）在 Δt=90 天后比较：高档衰减更慢。
        Δt=90 天对高档 = 半个半衰期 → e^(-ln2/2) = 1/√2。
        """
        start = _NOW - timedelta(days=90)
        high = memory_score(0.9, start.isoformat(), None, now=_NOW)
        assert math.isclose(high, (0.9 ** 0.5) * (1 / math.sqrt(2)), rel_tol=1e-9)
        # 90 天对高档=半个半衰期 → 1/√2；对中档=整个半衰期 → 1/2
        mid = memory_score(0.89, start.isoformat(), None, now=_NOW)
        assert high > mid  # 同 Δt 下，高档衰减因子(0.707) > 中档(0.5)，且 0.9**0.5 > 0.89**0.5


class TestMemoryScoreFallback:
    def test_missing_last_used_falls_back_to_create_time(self):
        """last_used_at 为 None/空 → 回退 create_time 计算 Δt。"""
        create = _NOW - timedelta(days=30)
        score_none = memory_score(0.5, None, create.isoformat(), now=_NOW)
        score_empty = memory_score(0.5, "", create.isoformat(), now=_NOW)
        expected = (0.5 ** 0.5) * 0.5  # 30 天=低档半衰期
        assert math.isclose(score_none, expected, rel_tol=1e-9)
        assert math.isclose(score_empty, expected, rel_tol=1e-9)

    def test_both_missing_returns_timeless_score(self):
        """last_used_at 与 create_time 均缺失 → 无时间衰减，score = confidence ** 0.5。"""
        score = memory_score(0.64, None, None, now=_NOW)
        assert math.isclose(score, 0.64 ** 0.5, rel_tol=1e-12)

    def test_legacy_empty_strings_treated_as_missing(self):
        """legacy 数据空串时间字段 → 不抛异常，返回 timeless score。"""
        score = memory_score(0.5, "", "", now=_NOW)
        assert math.isclose(score, 0.5 ** 0.5, rel_tol=1e-12)


class TestMemoryScoreClampingAndMalformed:
    def test_confidence_above_one_clamped(self):
        """confidence=1.5 防御性钳制到 1.0 → score 与 1.0 一致。"""
        assert math.isclose(
            memory_score(1.5, _NOW.isoformat(), None, now=_NOW),
            memory_score(1.0, _NOW.isoformat(), None, now=_NOW),
            rel_tol=1e-12,
        )

    def test_confidence_below_zero_clamped(self):
        """confidence=-0.5 钳制到 0.0（且档位取低档），不抛异常。"""
        score = memory_score(-0.5, _NOW.isoformat(), None, now=_NOW)
        assert math.isclose(score, 0.0, abs_tol=1e-12)

    def test_confidence_nan_does_not_raise(self):
        """confidence=NaN 不抛异常（安全回退为 0 分基线）。"""
        score = memory_score(float("nan"), _NOW.isoformat(), None, now=_NOW)
        assert isinstance(score, float)
        assert not math.isnan(score)

    def test_confidence_none_does_not_raise(self):
        """confidence=None 不抛异常（按 0.0 处理）。"""
        score = memory_score(None, _NOW.isoformat(), None, now=_NOW)  # type: ignore[arg-type]
        assert math.isclose(score, 0.0, abs_tol=1e-12)

    def test_now_injection_never_uses_wall_clock(self):
        """传入 now 时结果只取决于 now，与真实墙钟无关（同一入参多次调用稳定）。"""
        s1 = memory_score(0.8, _iso(10), None, now=_NOW)
        s2 = memory_score(0.8, _iso(10), None, now=_NOW)
        assert math.isclose(s1, s2, rel_tol=1e-12)

    def test_future_last_used_clamped_to_no_decay(self):
        """last_used_at 晚于 now（时钟漂移）→ Δt 负值被钳制到 0，不产生 >1 放大。"""
        future = (_NOW + timedelta(days=5)).isoformat()
        score = memory_score(0.64, future, None, now=_NOW)
        assert math.isclose(score, 0.64 ** 0.5, rel_tol=1e-12)


class TestMemoryScoreTimezoneNormalization:
    """tz-aware datetime 入参归一化 — 不得因 naive/aware 相减抛 TypeError。"""

    def test_tz_aware_datetime_anchor_matches_naive(self):
        """tz-aware datetime 锚点与同墙钟值的 naive 锚点得分一致，且不抛异常。"""
        naive_anchor = _NOW - timedelta(days=30)
        aware_anchor = naive_anchor.replace(tzinfo=UTC)
        score_naive = memory_score(0.8, naive_anchor, None, now=_NOW)
        score_aware = memory_score(0.8, aware_anchor, None, now=_NOW)
        assert math.isclose(score_aware, score_naive, rel_tol=1e-12)

    def test_tz_aware_now_matches_naive_now(self):
        """tz-aware now 与同墙钟值的 naive now 得分一致，且不抛异常。"""
        aware_now = _NOW.replace(tzinfo=UTC)
        score_naive = memory_score(0.8, _iso(30), None, now=_NOW)
        score_aware = memory_score(0.8, _iso(30), None, now=aware_now)
        assert math.isclose(score_aware, score_naive, rel_tol=1e-12)


# ════════════════════════════════════════════════════════════════════════
#  BASELINE 表征测试已移至 test_memory_prune_baseline.py（改造前先锁行为）
# ════════════════════════════════════════════════════════════════════════
#  NEW 行为 — active 组内按 memory_score 升序淘汰（低分先淘汰）
# ════════════════════════════════════════════════════════════════════════


class TestPruneScoreOrdering:
    def test_low_confidence_recent_evicted_before_high_confidence_old(self):
        """核心新行为：低置信新近条目先于高置信久远条目被淘汰。

        计算基准（now 显式注入 _NOW）：
          A 低置信新近：confidence=0.3, Δt=1 天
            λ_low  = ln2/30  → score = 0.3**0.5 * e^(-ln2/30 * 1) ≈ 0.535
          B 高置信久远：confidence=0.95, Δt=100 天
            λ_high = ln2/180 → score = 0.95**0.5 * e^(-ln2/180 * 100) ≈ 0.663
        score_A < score_B → 超限 1 条时淘汰 A（低分先出）。
        旧纯 LRU 语义下会淘汰 B（last_used_at 更旧），故该用例在改造前必然 FAIL。
        """
        async def run():
            store = ThreadMemoryStore(max_facts=1)
            low_recent = ThreadFactItem(
                content="low-conf-recent",
                confidence=0.3,
                last_used_at=_iso(1),
                create_time=_iso(1),
            )
            high_old = ThreadFactItem(
                content="high-conf-old",
                confidence=0.95,
                last_used_at=_iso(100),
                create_time=_iso(100),
            )
            # 先验证公式确实给出期望序（防测试自身假设漂移）
            assert memory_score(0.3, low_recent.last_used_at, None, now=_NOW) < memory_score(
                0.95, high_old.last_used_at, None, now=_NOW
            )
            await store.save_fact("t1", high_old)
            await store.save_fact("t1", low_recent)

            removed = await store.prune_facts("t1")
            assert removed == 1
            remaining = await store.query_facts("t1")
            assert len(remaining) == 1
            # 低分（低置信新近）被淘汰，高置信久远存活
            assert remaining[0].content == "high-conf-old"

        asyncio.run(run())

    def test_agent_ns_score_ordering(self):
        """agent 级 prune 同样按 score 升序淘汰最低分。"""
        async def run():
            store = ThreadMemoryStore(max_agent_facts=1)
            low = ThreadFactItem(
                content="agent-low", category="user_fact", confidence=0.3,
                last_used_at=_iso(1), create_time=_iso(1),
            )
            high = ThreadFactItem(
                content="agent-high", category="user_fact", confidence=0.95,
                last_used_at=_iso(100), create_time=_iso(100),
            )
            await store.save_agent_fact(high)
            await store.save_agent_fact(low)

            removed = await store.prune_agent_facts()
            assert removed == 1
            remaining = await store.query_agent_facts()
            assert [f.content for f in remaining] == ["agent-high"]

        asyncio.run(run())

    def test_eviction_count_exact_with_mixed_scores(self):
        """淘汰数量精确等于超限条数（4 条 → 上限 2 → 恰好淘汰最低分 2 条）。"""
        async def run():
            store = ThreadMemoryStore(max_facts=2)
            # 分数从低到高：c=0.3/dt=1 < c=0.5/dt=10 < c=0.8/dt=5 < c=0.95/dt=1
            facts = [
                ThreadFactItem(content="s-low", confidence=0.3, last_used_at=_iso(1), create_time=_iso(1)),
                ThreadFactItem(content="s-midlow", confidence=0.5, last_used_at=_iso(10), create_time=_iso(10)),
                ThreadFactItem(content="s-midhigh", confidence=0.8, last_used_at=_iso(5), create_time=_iso(5)),
                ThreadFactItem(content="s-high", confidence=0.95, last_used_at=_iso(1), create_time=_iso(1)),
            ]
            for f in facts:
                await store.save_fact("t1", f)

            removed = await store.prune_facts("t1")
            assert removed == 2
            remaining = {f.content for f in await store.query_facts("t1")}
            assert remaining == {"s-midhigh", "s-high"}

        asyncio.run(run())

    def test_superseded_first_then_active_by_score(self):
        """混合场景：1 条 superseded + 3 条 active 超 max=3 → 先淘汰 superseded（回归），active 全保留。"""
        async def run():
            store = ThreadMemoryStore(max_facts=3)
            for i in range(3):
                await store.save_fact(
                    "t1",
                    ThreadFactItem(
                        content=f"active-{i}",
                        confidence=0.9,
                        last_used_at=_iso(1),
                        create_time=_iso(1),
                    ),
                )
            dead = ThreadFactItem(content="dead", confidence=0.1)
            await store.save_fact("t1", dead)
            await store.supersede_facts("t1", [dead.fact_id])

            removed = await store.prune_facts("t1")
            assert removed == 1
            remaining = {f.content for f in await store.query_facts("t1")}
            assert remaining == {"active-0", "active-1", "active-2"}
            assert dead.content not in remaining

        asyncio.run(run())

    def test_legacy_facts_missing_time_fields_no_exception(self):
        """legacy 数据缺 last_used_at/create_time → 不抛异常，按 timeless score 排序。"""
        async def run():
            store = ThreadMemoryStore(max_facts=1)
            # 两条均无有效时间：高置信 timeless 应存活
            low = ThreadFactItem(content="legacy-low", confidence=0.2)
            high = ThreadFactItem(content="legacy-high", confidence=0.9)
            low.last_used_at = ""
            low.create_time = ""
            high.last_used_at = ""
            high.create_time = ""
            await store.save_fact("t1", low)
            await store.save_fact("t1", high)

            removed = await store.prune_facts("t1")
            assert removed == 1
            remaining = await store.query_facts("t1")
            assert [f.content for f in remaining] == ["legacy-high"]

        asyncio.run(run())


# ════════════════════════════════════════════════════════════════════════
#  集成 — 真实 AsyncSqliteStore（经 AgentMemory 临时 sqlite 文件）
# ════════════════════════════════════════════════════════════════════════


def test_integration_prune_score_on_real_async_sqlite_store(tmp_path):
    """真实 AsyncSqliteStore 上验证 score 升序淘汰（DB 层真正被驱动）。

    复用 AgentMemory.acreate 构造 AsyncSqliteStore（与生产同款），
    经 ThreadMemoryStore(backend=...) 直接行使 _prune_ns。
    """
    db_path = str(tmp_path / "scoring.sqlite")

    async def scenario():
        mem = await AgentMemory.acreate(checkpoint_file=db_path, use_sqlite=True)
        try:
            store = ThreadMemoryStore(backend=mem.get_long_term_store(), max_facts=1)
            low = ThreadFactItem(
                content="sqlite-low", confidence=0.3,
                last_used_at=_iso(1), create_time=_iso(1),
            )
            high = ThreadFactItem(
                content="sqlite-high", confidence=0.95,
                last_used_at=_iso(100), create_time=_iso(100),
            )
            await store.save_fact("t1", high)
            await store.save_fact("t1", low)

            removed = await store.prune_facts("t1")
            assert removed == 1
            remaining = await store.query_facts("t1")
            assert [f.content for f in remaining] == ["sqlite-high"]
        finally:
            await mem.aclose()

    asyncio.run(scenario())


def test_integration_agent_ns_prune_on_real_async_sqlite_store(tmp_path):
    """真实 AsyncSqliteStore 上验证 agent 级 score 淘汰。"""
    db_path = str(tmp_path / "scoring_agent.sqlite")

    async def scenario():
        mem = await AgentMemory.acreate(checkpoint_file=db_path, use_sqlite=True)
        try:
            store = ThreadMemoryStore(backend=mem.get_long_term_store(), max_agent_facts=1)
            low = ThreadFactItem(
                content="sqlite-agent-low", category="user_fact", confidence=0.2,
                last_used_at=_iso(1), create_time=_iso(1),
            )
            high = ThreadFactItem(
                content="sqlite-agent-high", category="user_fact", confidence=0.95,
                last_used_at=_iso(90), create_time=_iso(90),
            )
            await store.save_agent_fact(high)
            await store.save_agent_fact(low)

            removed = await store.prune_agent_facts()
            assert removed == 1
            remaining = await store.query_agent_facts()
            assert [f.content for f in remaining] == ["sqlite-agent-high"]
        finally:
            await mem.aclose()

    asyncio.run(scenario())


# ════════════════════════════════════════════════════════════════════════
#  状态分区可扩展性 — 供 Phase 2 加 forgotten 时无需重写
# ════════════════════════════════════════════════════════════════════════


def test_prune_status_partition_helper_exists():
    """_prune_ns 使用显式状态优先级映射（Phase 2 forgotten 可加档而不重写）。"""
    from memory import store as store_module

    assert hasattr(store_module, "_PRUNE_STATUS_PRIORITY")
    assert "active" in store_module._PRUNE_STATUS_PRIORITY


def test_status_priority_ordering_pure_function():
    """状态分区映射：superseded/stale 优先级 < active（数值小者先淘汰）。"""
    from memory.store import _PRUNE_STATUS_PRIORITY

    assert _PRUNE_STATUS_PRIORITY[FACT_STATUS_SUPERSEDED] < _PRUNE_STATUS_PRIORITY["active"]
    assert _PRUNE_STATUS_PRIORITY["stale"] < _PRUNE_STATUS_PRIORITY["active"]


def test_inmemory_backend_type_smoke():
    """默认 backend 为 InMemoryStore（store 单测惯例）。"""
    assert isinstance(ThreadMemoryStore().backend, InMemoryStore)