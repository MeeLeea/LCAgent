"""Phase 2a — 显式遗忘（forgotten 软删除）核心层测试。

覆盖：
- models：FACT_STATUS_FORGOTTEN / forgotten_at 字段 / from_dict 兼容 / to_dict 含字段
- store.forget_facts / forget_agent_facts：匹配优先级（fact_id > topic_key > keyword）、
  ValueError（全空条款）、计数正确、仅标记命中项
- 读取侧：active_only 排除 forgotten；全量查询仍含 forgotten（供 purge/审计）
- 容量：forgotten 不占容量基数、prune 永不删除 forgotten、superseded-first 回归
- purge_forgotten：只删 forgotten、older_than_days 过滤、幂等、不误伤 active/superseded
- middleware 去重基准剔除 forgotten → 遗忘后同 content 可重新学习
- MemoryManager.forget：缓冲清理 + 锁串行化，遗忘后 flush 不复活
- 集成：真实 AsyncSqliteStore 上 forget 后 DB 层 status == 'forgotten'

时序口径：forgotten_at 只断言"非空 / 可解析"，不比对墙钟精确值。
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

from memory.lock_pool import ThreadMemoryLockPool
from memory.manager import MemoryManager
from memory.middleware import (
    ThreadMemoryWriteMiddleware,
    default_thread_llm_resolver,
)
from memory.models import (
    FACT_STATUS_ACTIVE,
    FACT_STATUS_FORGOTTEN,
    FACT_STATUS_SUPERSEDED,
    ThreadFactItem,
)
from memory.store import ThreadMemoryStore


def _make_fact(content: str = "test fact", **kwargs) -> ThreadFactItem:
    return ThreadFactItem(content=content, **kwargs)


def _iso(days_ago: float) -> str:
    """相对"当前"生成 naive ISO（仅用于老化 forgotten_at 的 purge 测试）。

    断言只比较"早于/晚于阈值"的相对关系，不比对墙钟精确值，语义与时区无关。
    """
    base = datetime.now()  # noqa: DTZ005 — 项目时间字段统一 naive，与生产 _naive_now 同口径
    return (base - timedelta(days=days_ago)).isoformat()


class _FakeLLM:
    def __init__(self, response: str = "[]") -> None:
        self._response = response

    def chat(self, messages: list[dict[str, str]]) -> str:
        return self._response


def _make_write_middleware(llm_response: str = "[]") -> tuple[ThreadMemoryWriteMiddleware, ThreadMemoryStore]:
    store = ThreadMemoryStore()
    llm = _FakeLLM(response=llm_response)
    mw = ThreadMemoryWriteMiddleware(
        memory_store=store,
        lock_pool=ThreadMemoryLockPool(),
        llm_getter=default_thread_llm_resolver(lambda: llm),
        buffer_delay_seconds=999,
    )
    return mw, store


# ════════════════════════════════════════════════════════════════════════
#  models — 状态常量 & 字段兼容
# ════════════════════════════════════════════════════════════════════════


class TestForgottenModel:
    def test_forgotten_status_constant(self):
        assert FACT_STATUS_FORGOTTEN == "forgotten"

    def test_default_forgotten_at_is_none(self):
        assert ThreadFactItem().forgotten_at is None

    def test_to_dict_includes_forgotten_at(self):
        assert "forgotten_at" in ThreadFactItem().to_dict()

    def test_from_dict_legacy_without_forgotten_at(self):
        """legacy 数据缺少 forgotten_at 字段 → 默认 None，不抛异常。"""
        item = ThreadFactItem.from_dict({"content": "old", "status": "active"})
        assert item.forgotten_at is None


# ════════════════════════════════════════════════════════════════════════
#  store.forget_facts — 匹配路径 & 优先级
# ════════════════════════════════════════════════════════════════════════


class TestForgetThreadFacts:
    def test_forget_by_fact_id(self):
        """fact_id 精确匹配 → 仅命中项标记 forgotten（断言具体 fact_id）。"""
        async def run():
            store = ThreadMemoryStore()
            a = _make_fact("alpha")
            b = _make_fact("beta")
            await store.save_facts_batch("t1", [a, b])

            count = await store.forget_facts("t1", fact_id=a.fact_id)
            assert count == 1
            facts = await store.query_facts("t1")
            by_id = {f.fact_id: f for f in facts}
            assert by_id[a.fact_id].status == FACT_STATUS_FORGOTTEN
            assert by_id[a.fact_id].forgotten_at is not None
            assert by_id[b.fact_id].status == FACT_STATUS_ACTIVE

        asyncio.run(run())

    def test_forget_by_keyword_case_insensitive(self):
        """keyword 对 content 不区分大小写子串匹配。"""
        async def run():
            store = ThreadMemoryStore()
            a = _make_fact("User prefers Dark Mode")
            b = _make_fact("unrelated note")
            await store.save_facts_batch("t1", [a, b])

            count = await store.forget_facts("t1", keyword="dark mode")
            assert count == 1
            by_id = {f.fact_id: f for f in await store.query_facts("t1")}
            assert by_id[a.fact_id].status == FACT_STATUS_FORGOTTEN
            assert by_id[b.fact_id].status == FACT_STATUS_ACTIVE

        asyncio.run(run())

    def test_forget_by_topic_key_normalized(self):
        """topic_key 输入先归一化再匹配（Dim-Y → dim_y）。"""
        async def run():
            store = ThreadMemoryStore()
            a = _make_fact("v1", topic_key="dim_y_formula")
            b = _make_fact("other", topic_key="other_topic")
            await store.save_facts_batch("t1", [a, b])

            count = await store.forget_facts("t1", topic_key="Dim-Y Formula")
            assert count == 1
            by_id = {f.fact_id: f for f in await store.query_facts("t1")}
            assert by_id[a.fact_id].status == FACT_STATUS_FORGOTTEN
            assert by_id[b.fact_id].status == FACT_STATUS_ACTIVE

        asyncio.run(run())

    def test_priority_fact_id_over_keyword(self):
        """同时给 fact_id 与 keyword → fact_id 优先，仅按 id 命中。"""
        async def run():
            store = ThreadMemoryStore()
            a = _make_fact("match me")
            b = _make_fact("match me too")
            await store.save_facts_batch("t1", [a, b])

            count = await store.forget_facts("t1", fact_id=a.fact_id, keyword="match")
            assert count == 1
            by_id = {f.fact_id: f for f in await store.query_facts("t1")}
            assert by_id[a.fact_id].status == FACT_STATUS_FORGOTTEN
            assert by_id[b.fact_id].status == FACT_STATUS_ACTIVE

        asyncio.run(run())

    def test_priority_topic_key_over_keyword(self):
        """同时给 topic_key 与 keyword（topic 命中）→ topic_key 优先。"""
        async def run():
            store = ThreadMemoryStore()
            a = _make_fact("content with keyword", topic_key="target_topic")
            b = _make_fact("another keyword hit", topic_key="other")
            await store.save_facts_batch("t1", [a, b])

            count = await store.forget_facts("t1", topic_key="target_topic", keyword="keyword")
            assert count == 1
            by_id = {f.fact_id: f for f in await store.query_facts("t1")}
            assert by_id[a.fact_id].status == FACT_STATUS_FORGOTTEN
            assert by_id[b.fact_id].status == FACT_STATUS_ACTIVE

        asyncio.run(run())

    def test_all_criteria_absent_raises(self):
        async def run():
            store = ThreadMemoryStore()
            await store.save_fact("t1", _make_fact("x"))
            for kwargs in ({}, {"keyword": None}, {"keyword": ""}, {"keyword": "  "}):
                try:
                    await store.forget_facts("t1", **kwargs)
                except ValueError:
                    continue
                raise AssertionError(f"应对全空条件抛 ValueError: {kwargs}")

        asyncio.run(run())

    def test_forget_nonexistent_returns_zero(self):
        async def run():
            store = ThreadMemoryStore()
            assert await store.forget_facts("t1", keyword="nothing") == 0

        asyncio.run(run())


class TestForgetAgentFacts:
    def test_forget_agent_by_keyword(self):
        async def run():
            store = ThreadMemoryStore()
            a = _make_fact("Prefers Chinese", category="user_fact", scope="agent")
            b = _make_fact("Uses Vim", category="user_fact", scope="agent")
            await store.save_agent_facts_batch([a, b])

            count = await store.forget_agent_facts(keyword="chinese")
            assert count == 1
            by_id = {f.fact_id: f for f in await store.query_agent_facts()}
            assert by_id[a.fact_id].status == FACT_STATUS_FORGOTTEN
            assert by_id[b.fact_id].status == FACT_STATUS_ACTIVE

        asyncio.run(run())


# ════════════════════════════════════════════════════════════════════════
#  读取侧 — forgotten 不注入但全量可查
# ════════════════════════════════════════════════════════════════════════


class TestReadFiltering:
    def test_active_only_excludes_forgotten(self):
        async def run():
            store = ThreadMemoryStore()
            a = _make_fact("forget-me")
            b = _make_fact("keep-me")
            await store.save_facts_batch("t1", [a, b])
            await store.forget_facts("t1", fact_id=a.fact_id)

            active = await store.query_facts("t1", active_only=True)
            assert {f.content for f in active} == {"keep-me"}

            full = await store.query_facts("t1")
            assert {f.content for f in full} == {"forget-me", "keep-me"}
            assert any(f.status == FACT_STATUS_FORGOTTEN for f in full)

        asyncio.run(run())

    def test_count_facts_excludes_forgotten(self):
        """用户可见计数排除 forgotten。"""
        async def run():
            store = ThreadMemoryStore()
            a = _make_fact("a")
            b = _make_fact("b")
            await store.save_facts_batch("t1", [a, b])
            await store.forget_facts("t1", fact_id=a.fact_id)
            assert await store.count_facts("t1") == 1

        asyncio.run(run())

    def test_manager_recall_excludes_forgotten(self):
        async def run():
            store = ThreadMemoryStore()
            a = _make_fact("forgotten-item")
            b = _make_fact("kept-item")
            await store.save_facts_batch("t1", [a, b])
            await store.forget_facts("t1", fact_id=a.fact_id)

            mgr = MemoryManager(
                memory_store=store,
                lock_pool=ThreadMemoryLockPool(),
                llm_getter=default_thread_llm_resolver(lambda: None),
            )
            facts = await mgr.recall("t1")
            assert [f.content for f in facts] == ["kept-item"]

        asyncio.run(run())


# ════════════════════════════════════════════════════════════════════════
#  容量 — forgotten 不占基数 / 不被 prune 删除
# ════════════════════════════════════════════════════════════════════════


class TestPruneWithForgotten:
    def test_forgotten_not_counted_in_capacity(self):
        """max_facts=2，2 forgotten + 2 active → 无 active 被淘汰。"""
        async def run():
            store = ThreadMemoryStore(max_facts=2)
            active = [_make_fact(f"active-{i}", confidence=0.8) for i in range(2)]
            forgotten = [_make_fact(f"forgot-{i}") for i in range(2)]
            await store.save_facts_batch("t1", [*active, *forgotten])
            for f in forgotten:
                await store.forget_facts("t1", fact_id=f.fact_id)

            removed = await store.prune_facts("t1")
            assert removed == 0
            full = await store.query_facts("t1")
            assert len(full) == 4  # 全部保留（forgotten 跳过 + active 未超基数）
            active_contents = {f.content for f in full if f.status == FACT_STATUS_ACTIVE}
            assert active_contents == {"active-0", "active-1"}

        asyncio.run(run())

    def test_prune_never_deletes_forgotten(self):
        """2 active + 2 forgotten 超 max=2 → 只淘汰 active 超限部分，forgotten 全留。"""
        async def run():
            store = ThreadMemoryStore(max_facts=2)
            active = [
                _make_fact(f"active-{i}", confidence=0.8, last_used_at=_iso(10 - i), create_time=_iso(10 - i))
                for i in range(2)
            ]
            forgotten = [_make_fact(f"forgot-{i}") for i in range(2)]
            await store.save_facts_batch("t1", [*active, *forgotten])
            for f in forgotten:
                await store.forget_facts("t1", fact_id=f.fact_id)

            # 再加 1 条 active → active 计数 3 > max 2，需淘汰 1 条 active
            extra = _make_fact("active-new", confidence=0.8, last_used_at=_iso(0), create_time=_iso(0))
            await store.save_fact("t1", extra)

            removed = await store.prune_facts("t1")
            assert removed == 1
            full = await store.query_facts("t1")
            # forgotten 2 条全部保留
            forgotten_ids = {f.fact_id for f in forgotten}
            assert forgotten_ids <= {f.fact_id for f in full}
            # active 剩 2 条（淘汰 1 条最低分）
            assert len([f for f in full if f.status == FACT_STATUS_ACTIVE]) == 2

        asyncio.run(run())

    def test_superseded_first_still_holds_with_forgotten(self):
        """回归：有 forgotten 时 superseded 仍先于 active 被淘汰。"""
        async def run():
            store = ThreadMemoryStore(max_facts=2)
            await store.save_fact("t1", _make_fact("active-0", confidence=0.8))
            await store.save_fact("t1", _make_fact("active-1", confidence=0.8))
            dead = _make_fact("dead")
            await store.save_fact("t1", dead)
            await store.supersede_facts("t1", [dead.fact_id])
            fg = _make_fact("forgot-x")
            await store.save_fact("t1", fg)
            await store.forget_facts("t1", fact_id=fg.fact_id)

            # active=2（未超基数），superseded=1 → prune 删除 superseded
            removed = await store.prune_facts("t1")
            assert removed == 1
            full = await store.query_facts("t1")
            assert dead.fact_id not in {f.fact_id for f in full}
            assert fg.fact_id in {f.fact_id for f in full}

        asyncio.run(run())


# ════════════════════════════════════════════════════════════════════════
#  purge_forgotten — 物理清理
# ════════════════════════════════════════════════════════════════════════


class TestPurgeForgotten:
    def test_purge_thread_forgotten_only(self):
        async def run():
            store = ThreadMemoryStore()
            a = _make_fact("active")
            fg = _make_fact("forgotten")
            await store.save_facts_batch("t1", [a, fg])
            await store.forget_facts("t1", fact_id=fg.fact_id)

            count = await store.purge_forgotten("t1", older_than_days=None)
            assert count == 1
            remaining = await store.query_facts("t1")
            assert {f.fact_id for f in remaining} == {a.fact_id}

        asyncio.run(run())

    def test_purge_idempotent(self):
        async def run():
            store = ThreadMemoryStore()
            fg = _make_fact("forgotten")
            await store.save_fact("t1", fg)
            await store.forget_facts("t1", fact_id=fg.fact_id)

            assert await store.purge_forgotten("t1", older_than_days=None) == 1
            assert await store.purge_forgotten("t1", older_than_days=None) == 0

        asyncio.run(run())

    def test_purge_respects_older_than_days(self):
        """只有 forgotten_at 早于阈值者被清理；近期遗忘者保留。"""
        async def run():
            store = ThreadMemoryStore()
            old = _make_fact("old-forgotten")
            recent = _make_fact("recent-forgotten")
            await store.save_facts_batch("t1", [old, recent])
            await store.forget_facts("t1", fact_id=old.fact_id)
            await store.forget_facts("t1", fact_id=recent.fact_id)
            # 手动回填 old 的 forgotten_at 为 100 天前
            facts = await store.query_facts("t1")
            for f in facts:
                if f.fact_id == old.fact_id:
                    f.forgotten_at = _iso(100)
                    await store._store.aput(
                        store._facts_namespace("t1"), key=f.fact_id, value=f.to_dict()
                    )

            count = await store.purge_forgotten("t1", older_than_days=30)
            assert count == 1
            remaining = {f.fact_id for f in await store.query_facts("t1")}
            assert old.fact_id not in remaining
            assert recent.fact_id in remaining

        asyncio.run(run())

    def test_purge_does_not_touch_superseded(self):
        async def run():
            store = ThreadMemoryStore()
            dead = _make_fact("superseded-x")
            await store.save_fact("t1", dead)
            await store.supersede_facts("t1", [dead.fact_id])
            fg = _make_fact("forgotten-y")
            await store.save_fact("t1", fg)
            await store.forget_facts("t1", fact_id=fg.fact_id)

            assert await store.purge_forgotten("t1", older_than_days=None) == 1
            remaining = {f.fact_id for f in await store.query_facts("t1")}
            assert dead.fact_id in remaining
            assert fg.fact_id not in remaining

        asyncio.run(run())

    def test_purge_legacy_forgotten_at_none_with_threshold_skips(self):
        """legacy forgotten 条目 forgotten_at=None + 设阈值 → 跳过（不清理）。"""
        async def run():
            store = ThreadMemoryStore()
            fg = _make_fact("legacy-forgotten")
            await store.save_fact("t1", fg)
            await store.forget_facts("t1", fact_id=fg.fact_id)
            # 模拟 legacy：清空 forgotten_at
            facts = await store.query_facts("t1")
            f = facts[0]
            f.forgotten_at = None
            await store._store.aput(store._facts_namespace("t1"), key=f.fact_id, value=f.to_dict())

            count = await store.purge_forgotten("t1", older_than_days=30)
            assert count == 0
            assert len(await store.query_facts("t1")) == 1

        asyncio.run(run())

    def test_purge_agent_forgotten(self):
        async def run():
            store = ThreadMemoryStore()
            a = _make_fact("agent-forgotten", category="user_fact", scope="agent")
            await store.save_agent_fact(a)
            await store.forget_agent_facts(fact_id=a.fact_id)

            assert await store.purge_forgotten(thread_id=None, older_than_days=None) == 1
            assert await store.query_agent_facts() == []

        asyncio.run(run())


# ════════════════════════════════════════════════════════════════════════
#  middleware 去重基准剔除 forgotten → 重新学习
# ════════════════════════════════════════════════════════════════════════


class TestRelearnAfterForget:
    def test_same_content_relearned_after_forget(self):
        """遗忘后同一 content 经写流水线再次写入 → 作为新的 active fact 入库。"""
        async def run():
            import json

            facts = json.dumps([{"content": "用户偏好深色主题", "category": "conv", "confidence": 0.9}])
            mw, store = _make_write_middleware(llm_response=facts)
            # 预置并遗忘同 content 的旧 active
            old = _make_fact("用户偏好深色主题", category="conv")
            await store.save_fact("t1", old)
            await store.forget_facts("t1", fact_id=old.fact_id)
            assert await store.query_facts("t1", active_only=True) == []

            await mw.submit_event("t1", "user", "记住我喜欢深色主题")
            await mw._aflush_thread("t1")

            active = await store.query_facts("t1", active_only=True)
            assert len(active) == 1
            assert active[0].content == "用户偏好深色主题"
            assert active[0].status == FACT_STATUS_ACTIVE
            # 旧 forgotten 条目仍在（软删除保留）
            full = await store.query_facts("t1")
            assert any(f.status == FACT_STATUS_FORGOTTEN for f in full)

        asyncio.run(run())

    def test_superseded_still_dedups(self):
        """回归：superseded 仍参与去重基准（不因 Phase 2 改动而放开）。"""
        async def run():
            import json

            facts = json.dumps([{"content": "已被取代的内容", "category": "conv", "confidence": 0.9}])
            mw, store = _make_write_middleware(llm_response=facts)
            old = _make_fact("已被取代的内容", category="conv")
            await store.save_fact("t1", old)
            await store.supersede_facts("t1", [old.fact_id])

            await mw.submit_event("t1", "user", "再说一遍")
            await mw._aflush_thread("t1")

            # superseded 内容仍在基准中 → 不新增
            full = await store.query_facts("t1")
            assert len(full) == 1
            assert full[0].status == FACT_STATUS_SUPERSEDED

        asyncio.run(run())


# ════════════════════════════════════════════════════════════════════════
#  MemoryManager.forget — 缓冲清理 + 锁串行化（不复活）
# ════════════════════════════════════════════════════════════════════════


class TestManagerForgetSerialization:
    def test_forget_then_flush_does_not_revive(self):
        """forget（持锁 + 清缓冲）后立即 flush 同 content → 不复活 forgotten。"""
        async def run():
            import json

            facts = json.dumps([{"content": "要遗忘的内容", "category": "conv", "confidence": 0.9}])
            store = ThreadMemoryStore()
            mgr = MemoryManager(
                memory_store=store,
                lock_pool=ThreadMemoryLockPool(),
                llm_getter=default_thread_llm_resolver(lambda: _FakeLLM(response=facts)),
                buffer_delay_seconds=999,
            )
            # 预置 active 条目
            old = _make_fact("要遗忘的内容", category="conv")
            await store.save_fact("t1", old)
            # 投递一条同内容事件到 buffer（未 flush）
            await mgr.submit_user_message("t1", "再说一次要遗忘的内容")

            # 先遗忘（内部弃缓冲 + 持锁）
            count = await mgr.forget("t1", fact_id=old.fact_id)
            assert count == 1
            # 再 flush 全部 buffer（应已被丢弃）
            await mgr.flush_all()

            full = await store.query_facts("t1")
            assert len(full) == 1
            assert full[0].status == FACT_STATUS_FORGOTTEN

        asyncio.run(run())

    def test_manager_forget_agent(self):
        async def run():
            store = ThreadMemoryStore()
            a = _make_fact("agent-pref", category="user_fact", scope="agent")
            await store.save_agent_fact(a)
            mgr = MemoryManager(
                memory_store=store,
                lock_pool=ThreadMemoryLockPool(),
                llm_getter=default_thread_llm_resolver(lambda: None),
            )
            count = await mgr.forget_agent(keyword="agent-pref")
            assert count == 1
            assert await store.query_agent_facts(active_only=True) == []

        asyncio.run(run())


# ════════════════════════════════════════════════════════════════════════
#  集成 — 真实 AsyncSqliteStore 上 status == 'forgotten' 落库
# ════════════════════════════════════════════════════════════════════════


def test_integration_forget_persists_forgotten_status(tmp_path):
    """真实 AsyncSqliteStore：forget 后从 backend 直接读回 status == 'forgotten'。"""
    from memory.agent_memory import AgentMemory

    db_path = str(tmp_path / "forget.sqlite")

    async def scenario():
        mem = await AgentMemory.acreate(checkpoint_file=db_path, use_sqlite=True)
        try:
            store = ThreadMemoryStore(backend=mem.get_long_term_store())
            item = _make_fact("persist-forgotten")
            await store.save_fact("t1", item)

            count = await store.forget_facts("t1", fact_id=item.fact_id)
            assert count == 1

            # 从 DB 原始层读回（非业务查询），确认落库状态
            raw = await store.backend.aget(store._facts_namespace("t1"), key=item.fact_id)
            assert raw is not None
            assert raw.value["status"] == "forgotten"
            assert raw.value["forgotten_at"]
        finally:
            await mem.aclose()

    asyncio.run(scenario())