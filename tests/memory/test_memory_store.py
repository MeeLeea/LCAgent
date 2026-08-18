"""ThreadMemoryStore 单元测试

覆盖：
- save_fact / query_facts 基本读写
- save_facts_batch 批量写入与幂等性
- count_facts 计数
- prune_facts LRU 淘汰
- touch_fact 更新 last_used_at
- clear_thread_memory 全量清理
- replace_with_summary 压缩摘要替换
- per-thread 隔离
"""
from __future__ import annotations

import asyncio

from langgraph.store.memory import InMemoryStore

from memory.models import MemoryCategory, ThreadFactItem
from memory.store import ThreadMemoryStore


def _make_fact(content: str = "test fact", category: str = "conv") -> ThreadFactItem:
    return ThreadFactItem(content=content, category=category)


# ════════════════════════════════════════════════════════════════════════
#  基本读写
# ════════════════════════════════════════════════════════════════════════


class TestSaveAndQuery:
    def test_save_and_query_single_fact(self):
        async def run():
            store = ThreadMemoryStore()
            item = _make_fact("用户喜欢深色主题", "user_fact")
            await store.save_fact("thread-1", item)

            facts = await store.query_facts("thread-1")
            assert len(facts) == 1
            assert facts[0].content == "用户喜欢深色主题"
            assert facts[0].category == "user_fact"
            assert facts[0].thread_id == "thread-1"

        asyncio.run(run())

    def test_query_empty_thread_returns_empty_list(self):
        async def run():
            store = ThreadMemoryStore()
            facts = await store.query_facts("nonexistent")
            assert facts == []

        asyncio.run(run())

    def test_save_fact_overwrites_same_id(self):
        async def run():
            store = ThreadMemoryStore()
            item = _make_fact("original")
            await store.save_fact("thread-1", item)

            item.content = "updated"
            await store.save_fact("thread-1", item)

            facts = await store.query_facts("thread-1")
            assert len(facts) == 1
            assert facts[0].content == "updated"

        asyncio.run(run())

    def test_query_facts_sorted_by_create_time(self):
        async def run():
            store = ThreadMemoryStore()
            for i in range(3):
                item = _make_fact(f"fact-{i}")
                item.create_time = f"2026-01-0{i+1}T00:00:00"
                await store.save_fact("thread-1", item)

            facts = await store.query_facts("thread-1")
            assert len(facts) == 3
            assert facts[0].content == "fact-0"
            assert facts[2].content == "fact-2"

        asyncio.run(run())


# ════════════════════════════════════════════════════════════════════════
#  批量写入
# ════════════════════════════════════════════════════════════════════════


class TestBatchOperations:
    def test_save_facts_batch(self):
        async def run():
            store = ThreadMemoryStore()
            items = [_make_fact(f"batch-{i}") for i in range(5)]
            await store.save_facts_batch("thread-1", items)

            facts = await store.query_facts("thread-1")
            assert len(facts) == 5
            contents = {f.content for f in facts}
            assert contents == {f"batch-{i}" for i in range(5)}

        asyncio.run(run())

    def test_save_facts_batch_sets_thread_id(self):
        async def run():
            store = ThreadMemoryStore()
            items = [_make_fact("a"), _make_fact("b")]
            await store.save_facts_batch("thread-42", items)

            facts = await store.query_facts("thread-42")
            assert all(f.thread_id == "thread-42" for f in facts)

        asyncio.run(run())

    def test_save_facts_batch_empty_list(self):
        async def run():
            store = ThreadMemoryStore()
            await store.save_facts_batch("thread-1", [])
            facts = await store.query_facts("thread-1")
            assert facts == []

        asyncio.run(run())


# ════════════════════════════════════════════════════════════════════════
#  分类筛选 & 计数
# ════════════════════════════════════════════════════════════════════════


class TestCategoryAndCount:
    def test_count_facts(self):
        async def run():
            store = ThreadMemoryStore()
            await store.save_fact("t1", _make_fact("a"))
            await store.save_fact("t1", _make_fact("b"))
            await store.save_fact("t1", _make_fact("c"))

            assert await store.count_facts("t1") == 3
            assert await store.count_facts("t2") == 0

        asyncio.run(run())


# ════════════════════════════════════════════════════════════════════════
#  删除 & 清理
# ════════════════════════════════════════════════════════════════════════


class TestDeleteAndClear:
    def test_clear_thread_memory(self):
        async def run():
            store = ThreadMemoryStore()
            for i in range(3):
                await store.save_fact("t1", _make_fact(f"f{i}"))
            await store.save_fact("t2", _make_fact("other"))

            cleared = await store.clear_thread_memory("t1")
            assert cleared == 3
            assert await store.count_facts("t1") == 0
            assert await store.count_facts("t2") == 1

        asyncio.run(run())

    def test_clear_empty_thread_returns_zero(self):
        async def run():
            store = ThreadMemoryStore()
            cleared = await store.clear_thread_memory("empty")
            assert cleared == 0

        asyncio.run(run())


# ════════════════════════════════════════════════════════════════════════
#  LRU 淘汰
# ════════════════════════════════════════════════════════════════════════


class TestPruneAndTouch:
    def test_prune_facts_removes_oldest(self):
        async def run():
            store = ThreadMemoryStore(max_facts=3)
            for i in range(5):
                item = _make_fact(f"fact-{i}")
                item.last_used_at = f"2026-01-0{i+1}T00:00:00"
                item.create_time = f"2026-01-0{i+1}T00:00:00"
                await store.save_fact("t1", item)

            pruned = await store.prune_facts("t1")
            assert pruned == 2
            facts = await store.query_facts("t1")
            assert len(facts) == 3
            contents = {f.content for f in facts}
            assert "fact-4" in contents
            assert "fact-0" not in contents

        asyncio.run(run())

    def test_prune_facts_under_limit_returns_zero(self):
        async def run():
            store = ThreadMemoryStore(max_facts=10)
            await store.save_fact("t1", _make_fact("only-one"))

            pruned = await store.prune_facts("t1")
            assert pruned == 0

        asyncio.run(run())

    def test_touch_fact_updates_last_used_at(self):
        async def run():
            store = ThreadMemoryStore()
            item = _make_fact("touch-me")
            old_time = "2026-01-01T00:00:00"
            item.last_used_at = old_time
            await store.save_fact("t1", item)

            await asyncio.sleep(0.01)
            await store.touch_fact("t1", item.fact_id)

            facts = await store.query_facts("t1")
            assert len(facts) == 1
            assert facts[0].last_used_at != old_time

        asyncio.run(run())

    def test_touch_nonexistent_fact_no_error(self):
        async def run():
            store = ThreadMemoryStore()
            await store.touch_fact("t1", "nonexistent")

        asyncio.run(run())


# ════════════════════════════════════════════════════════════════════════
#  压缩摘要替换
# ════════════════════════════════════════════════════════════════════════


class TestReplaceWithSummary:
    def test_replace_with_summary(self):
        async def run():
            store = ThreadMemoryStore()
            for i in range(3):
                await store.save_fact("t1", _make_fact(f"fact-{i}"))

            result = await store.replace_with_summary("t1", "这是压缩摘要")
            assert result["success"] is True
            assert result["original_count"] == 3
            assert result["summary"] == "这是压缩摘要"

            facts = await store.query_facts("t1")
            assert len(facts) == 1
            assert "历史记忆摘要" in facts[0].content
            assert "这是压缩摘要" in facts[0].content
            assert facts[0].category == MemoryCategory.IMPORTANT_CONVERSATION.value

        asyncio.run(run())

    def test_replace_with_summary_empty_thread(self):
        async def run():
            store = ThreadMemoryStore()
            result = await store.replace_with_summary("empty", "summary")
            assert result["success"] is True
            assert result["original_count"] == 0

            facts = await store.query_facts("empty")
            assert len(facts) == 1

        asyncio.run(run())


# ════════════════════════════════════════════════════════════════════════
#  Per-thread 隔离
# ════════════════════════════════════════════════════════════════════════


class TestThreadIsolation:
    def test_facts_isolated_between_threads(self):
        async def run():
            store = ThreadMemoryStore()
            await store.save_fact("t1", _make_fact("thread-1-fact"))
            await store.save_fact("t2", _make_fact("thread-2-fact"))

            assert await store.count_facts("t1") == 1
            assert await store.count_facts("t2") == 1
            assert (await store.query_facts("t1"))[0].content == "thread-1-fact"
            assert (await store.query_facts("t2"))[0].content == "thread-2-fact"

        asyncio.run(run())

    def test_clear_one_thread_does_not_affect_other(self):
        async def run():
            store = ThreadMemoryStore()
            await store.save_fact("t1", _make_fact("a"))
            await store.save_fact("t2", _make_fact("b"))

            await store.clear_thread_memory("t1")
            assert await store.count_facts("t1") == 0
            assert await store.count_facts("t2") == 1

        asyncio.run(run())


# ════════════════════════════════════════════════════════════════════════
#  属性暴露
# ════════════════════════════════════════════════════════════════════════


class TestProperties:
    def test_backend_property(self):
        store = ThreadMemoryStore()
        assert isinstance(store.backend, InMemoryStore)

    def test_max_facts_property(self):
        store = ThreadMemoryStore(max_facts=42)
        assert store.max_facts == 42

    def test_default_max_facts(self):
        store = ThreadMemoryStore()
        assert store.max_facts == 50
