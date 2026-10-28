"""Phase 3 修复 — scheduler purge host（守护线程 + 专属事件循环）测试。

验证：scheduler 的 purge 周期任务运行在 run.py 的**专属守护线程事件循环**上，
而非 executor 的 per-task 临时 asyncio.run loop。

覆盖：
- host 在真实临时 sqlite 上启动，极小间隔下真实物理删除超龄 forgotten
- 重复调用/停止幂等
- 无墙钟相等断言（仅相对阈值）
"""
from __future__ import annotations

import asyncio
import time

from memory.agent_memory import AgentMemory
from memory.models import ThreadFactItem
from memory.store import ThreadMemoryStore


def _forget_old_agent_fact(db_path: str, days_ago: float) -> str:
    """在临时 sqlite 上写入并遗忘一条 agent fact，回填 forgotten_at 为 days_ago 前。"""

    async def _setup() -> str:
        mem = await AgentMemory.acreate(checkpoint_file=db_path, use_sqlite=True)
        try:
            store = ThreadMemoryStore(backend=mem.get_long_term_store())
            fact = ThreadFactItem(content="purge-host-target", category="user_fact", scope="agent")
            await store.save_agent_fact(fact)
            await store.forget_agent_facts(fact_id=fact.fact_id)
            # 回填 forgotten_at 为超龄（超过默认 7 天），使 host 的 purge 命中
            ns = store._facts_namespace("", scope="agent")
            item = await store.backend.aget(ns, key=fact.fact_id)
            assert item is not None
            updated = ThreadFactItem.from_dict(item.value)
            updated.forgotten_at = _iso_days_ago(days_ago)
            await store.backend.aput(ns, key=fact.fact_id, value=updated.to_dict())
            return fact.fact_id
        finally:
            await mem.aclose()

    return asyncio.run(_setup())


def _iso_days_ago(days_ago: float) -> str:
    from datetime import datetime, timedelta

    base = datetime.now()  # noqa: DTZ005 — 项目时间字段统一 naive
    return (base - timedelta(days=days_ago)).isoformat()


def _count_agent_facts(db_path: str) -> int:
    async def _count() -> int:
        mem = await AgentMemory.acreate(checkpoint_file=db_path, use_sqlite=True)
        try:
            store = ThreadMemoryStore(backend=mem.get_long_term_store())
            return len(await store.query_agent_facts())
        finally:
            await mem.aclose()

    return asyncio.run(_count())


def test_purge_host_on_daemon_thread_deletes_old_forgotten(tmp_path):
    """host 在专属线程 loop 上周期性物理删除超龄 forgotten（有界轮询）。"""
    from scheduler.run import _start_purge_host

    db_path = str(tmp_path / "purge_host.sqlite")
    _forget_old_agent_fact(db_path, days_ago=30)
    assert _count_agent_facts(db_path) == 1

    host = _start_purge_host(
        db_path, interval=0.05, forgotten_retention_days=7, superseded_retention_days=90
    )
    try:
        deadline = time.monotonic() + 4.0
        while time.monotonic() < deadline:
            if _count_agent_facts(db_path) == 0:
                break
            time.sleep(0.05)
        assert _count_agent_facts(db_path) == 0
    finally:
        host.stop()


def test_purge_host_stop_is_idempotent(tmp_path):
    """停止 host 可重复调用（幂等），且线程在限时内退出。"""
    from scheduler.run import _start_purge_host

    db_path = str(tmp_path / "purge_host_stop.sqlite")
    host = _start_purge_host(db_path, interval=0.05)
    host.stop()
    host.stop()
    assert not host.thread.is_alive()


def test_purge_host_exposes_thread_and_loop():
    """host 暴露守护线程（daemon=True）。"""
    import tempfile

    from scheduler.run import _start_purge_host

    with tempfile.TemporaryDirectory() as d:
        import os

        db_path = os.path.join(d, "purge_host_meta.sqlite")
        host = _start_purge_host(db_path, interval=10.0)
        try:
            assert host.thread.daemon is True
            assert host.thread.is_alive()
        finally:
            host.stop()