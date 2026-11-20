"""Phase 3 — 后台 purge 协程测试。

覆盖：
- 任务生命周期：在当前 loop 创建、stop/aclose 取消、重复 start 幂等
- tick 行为：forgotten 超龄删除、未超龄保留、superseded >90d 删除、异常被吞不杀循环
- AgentMemoryLock 在 tick 期间被持有（跨进程串行化的单元级证明）
- 间隔：注入极小间隔（0.01s）+ 假 store，无真实墙钟长等待
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

from memory.lock_pool import AgentMemoryLock, ThreadMemoryLockPool
from memory.manager import MemoryManager
from memory.middleware import default_thread_llm_resolver
from memory.models import ThreadFactItem
from memory.store import ThreadMemoryStore


def _iso(days_ago: float) -> str:
    """相对当前生成 naive ISO（仅用于老化 forgotten_at / superseded_at）。"""
    base = datetime.now()  # noqa: DTZ005 — 项目时间字段统一 naive
    return (base - timedelta(days=days_ago)).isoformat()


def _make_manager(store: ThreadMemoryStore, **kwargs) -> MemoryManager:
    return MemoryManager(
        memory_store=store,
        lock_pool=ThreadMemoryLockPool(),
        llm_getter=default_thread_llm_resolver(lambda: None),
        **kwargs,
    )


class _RecordingLock(AgentMemoryLock):
    """记录 acquire 次数的进程内锁（path=None 退化为 asyncio.Lock）。"""

    def __init__(self) -> None:
        super().__init__(path=None)
        self.entries = 0

    async def __aenter__(self):  # type: ignore[override]
        self.entries += 1
        return await super().__aenter__()


class _ExplodingStore(ThreadMemoryStore):
    """purge 时抛异常的 store（验证循环吞异常且不退出）。"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = 0

    async def purge_forgotten(self, thread_id=None, older_than_days=None) -> int:  # type: ignore[override]
        self.calls += 1
        raise RuntimeError("模拟 purge 失败")


# ════════════════════════════════════════════════════════════════════════
#  tick 行为 — 真实 store 级别
# ════════════════════════════════════════════════════════════════════════


class TestPurgeOnce:
    def test_purge_once_deletes_old_forgotten_keeps_new(self):
        """purge_once：forgotten 超龄删除、未超龄保留（agent namespace）。"""

        async def run() -> None:
            store = ThreadMemoryStore()
            old = ThreadFactItem(content="old-forgotten", category="user_fact", scope="agent")
            fresh = ThreadFactItem(content="fresh-forgotten", category="user_fact", scope="agent")
            await store.save_agent_facts_batch([old, fresh])
            await store.forget_agent_facts(fact_id=old.fact_id)
            await store.forget_agent_facts(fact_id=fresh.fact_id)
            # 回填 old 的 forgotten_at 为 30 天前
            await _set_forgotten_at(store, None, old.fact_id, _iso(30))

            mgr = _make_manager(store)
            result = await mgr.purge_once(forgotten_retention_days=7)

            assert result["forgotten"] == 1
            remaining = {f.fact_id for f in await store.query_agent_facts()}
            assert old.fact_id not in remaining
            assert fresh.fact_id in remaining

        asyncio.run(run())

    def test_purge_once_deletes_old_superseded(self):
        """purge_once：superseded 超 90 天删除，未超龄保留。"""

        async def run() -> None:
            store = ThreadMemoryStore()
            old = ThreadFactItem(content="old-superseded", category="user_fact", scope="agent")
            recent = ThreadFactItem(content="recent-superseded", category="user_fact", scope="agent")
            await store.save_agent_facts_batch([old, recent])
            await store.supersede_agent_facts([old.fact_id, recent.fact_id])
            await _set_superseded_at(store, None, old.fact_id, _iso(120))

            mgr = _make_manager(store)
            result = await mgr.purge_once(superseded_retention_days=90)

            assert result["superseded"] == 1
            remaining = {f.fact_id for f in await store.query_agent_facts()}
            assert old.fact_id not in remaining
            assert recent.fact_id in remaining

        asyncio.run(run())

    def test_purge_once_legacy_forgotten_without_timestamp_skipped(self):
        """legacy forgotten（无 forgotten_at）+ 阈值 → 跳过（不误删）。"""

        async def run() -> None:
            store = ThreadMemoryStore()
            fg = ThreadFactItem(content="legacy", category="user_fact", scope="agent")
            await store.save_agent_fact(fg)
            await store.forget_agent_facts(fact_id=fg.fact_id)
            await _set_forgotten_at(store, None, fg.fact_id, None)

            mgr = _make_manager(store)
            result = await mgr.purge_once(forgotten_retention_days=7)

            assert result["forgotten"] == 0
            assert len(await store.query_agent_facts()) == 1

        asyncio.run(run())

    def test_purge_once_idempotent(self):
        """重复 purge_once 无副作用（第二次删除 0 条）。"""

        async def run() -> None:
            store = ThreadMemoryStore()
            fg = ThreadFactItem(content="fg", category="user_fact", scope="agent")
            await store.save_agent_fact(fg)
            await store.forget_agent_facts(fact_id=fg.fact_id)

            mgr = _make_manager(store)
            first = await mgr.purge_once(forgotten_retention_days=0)
            second = await mgr.purge_once(forgotten_retention_days=0)
            assert first["forgotten"] == 1
            assert second["forgotten"] == 0

        asyncio.run(run())


# ════════════════════════════════════════════════════════════════════════
#  任务生命周期 — 当前 loop 创建 / 取消 / 幂等
# ════════════════════════════════════════════════════════════════════════


class TestPurgeTaskLifecycle:
    def test_start_creates_task_on_current_loop(self):
        async def run() -> None:
            store = ThreadMemoryStore()
            mgr = _make_manager(store)
            task = mgr.start_purge_task(interval_seconds=0.01)
            try:
                assert isinstance(task, asyncio.Task)
                # 任务运行于当前事件循环
                assert task.get_loop() is asyncio.get_running_loop()
            finally:
                await mgr.stop_purge_task()

        asyncio.run(run())

    def test_double_start_is_safe(self):
        async def run() -> None:
            store = ThreadMemoryStore()
            mgr = _make_manager(store)
            t1 = mgr.start_purge_task(interval_seconds=0.01)
            t2 = mgr.start_purge_task(interval_seconds=0.01)
            try:
                # 二次 start 返回同一任务（不重复创建协程）
                assert t1 is t2
            finally:
                await mgr.stop_purge_task()

        asyncio.run(run())

    def test_stop_cancels_task(self):
        async def run() -> None:
            store = ThreadMemoryStore()
            mgr = _make_manager(store)
            task = mgr.start_purge_task(interval_seconds=0.01)
            await mgr.stop_purge_task()
            # stop 内已 await 任务结束；此处断言已完结/取消
            assert task.done()
            assert mgr.purge_task_running is False

        asyncio.run(run())

    def test_stop_without_start_is_noop(self):
        async def run() -> None:
            mgr = _make_manager(ThreadMemoryStore())
            await mgr.stop_purge_task()  # 不抛异常
            await mgr.stop_purge_task()

        asyncio.run(run())

    def test_tick_actually_runs_and_purges(self):
        """极小间隔下循环触发一次 tick，真实删除超龄 forgotten。"""

        async def run() -> None:
            store = ThreadMemoryStore()
            fg = ThreadFactItem(content="fg", category="user_fact", scope="agent")
            await store.save_agent_fact(fg)
            await store.forget_agent_facts(fact_id=fg.fact_id)
            await _set_forgotten_at(store, None, fg.fact_id, _iso(30))

            mgr = _make_manager(store)
            mgr.start_purge_task(interval_seconds=0.01, forgotten_retention_days=7)
            try:
                # 有界等待：确认 tick 已删除该条目
                for _ in range(200):
                    if not await store.query_agent_facts():
                        break
                    await asyncio.sleep(0.01)
                assert await store.query_agent_facts() == []
            finally:
                await mgr.stop_purge_task()

        asyncio.run(run())


# ════════════════════════════════════════════════════════════════════════
#  lock 持有 — 跨进程串行化
# ════════════════════════════════════════════════════════════════════════


class TestPurgeLock:
    def test_tick_holds_agent_lock(self):
        """每个 tick 期间持有 AgentMemoryLock（acquire 次数随 tick 增加）。"""

        async def run() -> None:
            store = ThreadMemoryStore()
            lock = _RecordingLock()
            mgr = _make_manager(store, agent_lock=lock)
            mgr.start_purge_task(interval_seconds=0.01)
            try:
                for _ in range(200):
                    if lock.entries >= 1:
                        break
                    await asyncio.sleep(0.01)
                assert lock.entries >= 1
            finally:
                await mgr.stop_purge_task()

        asyncio.run(run())


# ════════════════════════════════════════════════════════════════════════
#  错误韧性 — store 抛异常不杀循环
# ════════════════════════════════════════════════════════════════════════


class TestPurgeErrorResilience:
    def test_store_error_does_not_kill_loop(self):
        """store.purge 抛异常 → 循环继续（后续 tick 仍被调用）。"""

        async def run() -> None:
            store = _ExplodingStore()
            mgr = _make_manager(store)
            mgr.start_purge_task(interval_seconds=0.01)
            try:
                for _ in range(200):
                    if store.calls >= 2:
                        break
                    await asyncio.sleep(0.01)
                assert store.calls >= 2  # 第一次异常后仍继续 tick
                assert mgr.purge_task_running is True
            finally:
                await mgr.stop_purge_task()

        asyncio.run(run())


# ════════════════════════════════════════════════════════════════════════
#  MemoryContext 生命周期 — enable 标志 + aclose 取消（真实 loop）
# ════════════════════════════════════════════════════════════════════════


class TestContextPurgeLifecycle:
    def test_context_acreate_starts_and_aclose_cancels(self):
        """enable_purge_task=True → acreate 在当前 loop 启动；aclose 取消它。"""
        from memory.context import MemoryContext

        async def run() -> None:
            ctx = await MemoryContext.acreate(
                checkpoint_file=None,
                use_sqlite=False,
                enable_purge_task=True,
                purge_interval_seconds=0.01,
            )
            try:
                assert ctx.memory_manager.purge_task_running is True
            finally:
                await ctx.aclose()
            # aclose 后任务已停止
            assert ctx.memory_manager.purge_task_running is False
            # aclose 可重复调用（幂等）
            await ctx.aclose()

        asyncio.run(run())

    def test_context_acreate_without_flag_does_not_start(self):
        """默认（enable_purge_task=False）不启动 purge 任务（CLI 场景）。"""
        from memory.context import MemoryContext

        async def run() -> None:
            ctx = await MemoryContext.acreate(checkpoint_file=None, use_sqlite=False)
            try:
                assert ctx.memory_manager.purge_task_running is False
            finally:
                await ctx.aclose()

        asyncio.run(run())


# ════════════════════════════════════════════════════════════════════════
#  内部辅助 — 直接回填时间戳（模拟老化数据）
# ════════════════════════════════════════════════════════════════════════


async def _set_forgotten_at(
    store: ThreadMemoryStore, thread_id: str | None, fact_id: str, value: str | None
) -> None:
    ns = (
        store._facts_namespace("", scope="agent")
        if thread_id is None
        else store._facts_namespace(thread_id)
    )
    item = await store.backend.aget(ns, key=fact_id)
    assert item is not None
    fact = ThreadFactItem.from_dict(item.value)
    fact.forgotten_at = value
    await store.backend.aput(ns, key=fact_id, value=fact.to_dict())


async def _set_superseded_at(
    store: ThreadMemoryStore, thread_id: str | None, fact_id: str, value: str | None
) -> None:
    ns = (
        store._facts_namespace("", scope="agent")
        if thread_id is None
        else store._facts_namespace(thread_id)
    )
    item = await store.backend.aget(ns, key=fact_id)
    assert item is not None
    fact = ThreadFactItem.from_dict(item.value)
    fact.superseded_at = value
    await store.backend.aput(ns, key=fact_id, value=fact.to_dict())