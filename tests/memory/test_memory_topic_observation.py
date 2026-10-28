"""Phase 4 — topic_key 漂移观测计数测试（仅观测，不改行为）。

覆盖：
- 批内 topic_key 命中既有主题（supersede 命中）→ 日志计数为命中 1
- 批内全为新增 topic_key → 新增数 = 全部
- 批内无 topic_key → 不记日志

断言日志记录的具体数字（非仅"发生了一次日志"）。
"""
from __future__ import annotations

import asyncio
import json
import logging

from memory.lock_pool import ThreadMemoryLockPool
from memory.middleware import (
    ThreadMemoryWriteMiddleware,
    default_thread_llm_resolver,
)
from memory.models import ThreadFactItem
from memory.store import ThreadMemoryStore


class _FakeLLM:
    def __init__(self, response: str = "[]") -> None:
        self._response = response

    def chat(self, messages: list[dict[str, str]]) -> str:
        return self._response


def _make_middleware(llm_response: str) -> tuple[ThreadMemoryWriteMiddleware, ThreadMemoryStore]:
    store = ThreadMemoryStore()
    llm = _FakeLLM(response=llm_response)
    mw = ThreadMemoryWriteMiddleware(
        memory_store=store,
        lock_pool=ThreadMemoryLockPool(),
        llm_getter=default_thread_llm_resolver(lambda: llm),
        buffer_delay_seconds=999,
    )
    return mw, store


def _observation_records(caplog) -> list[str]:
    return [
        rec.getMessage()
        for rec in caplog.records
        if "topic_key 观测" in rec.getMessage()
    ]


def test_supersede_hit_counted(caplog):
    """批内 topic_key 命中既有 active 同主题 → supersede 命中 1、新增 0。"""

    async def run() -> None:
        facts = json.dumps([
            {"content": "dim_y = 2*out", "category": "lesson",
             "confidence": 0.9, "persistence": "global", "topic_key": "dim_y_formula"},
        ])
        mw, store = _make_middleware(facts)
        # 预置同 topic_key 的 active 旧条目
        await store.save_agent_fact(
            ThreadFactItem(
                content="dim_y = 2*out + 2.5", category="lesson", scope="agent",
                topic_key="dim_y_formula", create_time="2026-01-01T00:00:00",
            )
        )
        with caplog.at_level(logging.INFO, logger="memory.middleware"):
            await mw.submit_event("t1", "user", "修正 dim_y")
            await mw._aflush_thread("t1")

    asyncio.run(run())
    records = _observation_records(caplog)
    assert len(records) == 1
    assert "supersede命中 1/1" in records[0]
    assert "新增topic_key 0/1" in records[0]


def test_all_new_topic_keys_counted(caplog):
    """批内全为新增 topic_key → 新增数 = 全部。"""

    async def run() -> None:
        facts = json.dumps([
            {"content": "fact-a", "category": "conv", "topic_key": "brand_new_a"},
            {"content": "fact-b", "category": "conv", "topic_key": "brand_new_b"},
        ])
        mw, _store = _make_middleware(facts)
        with caplog.at_level(logging.INFO, logger="memory.middleware"):
            await mw.submit_event("t1", "user", "chat")
            await mw._aflush_thread("t1")

    asyncio.run(run())
    records = _observation_records(caplog)
    assert len(records) == 1
    assert "supersede命中 0/2" in records[0]
    assert "新增topic_key 2/2" in records[0]


def test_no_topic_keys_no_log(caplog):
    """批内无 topic_key → 不记观测日志。"""

    async def run() -> None:
        facts = json.dumps([{"content": "plain", "category": "conv"}])
        mw, _store = _make_middleware(facts)
        with caplog.at_level(logging.INFO, logger="memory.middleware"):
            await mw.submit_event("t1", "user", "chat")
            await mw._aflush_thread("t1")

    asyncio.run(run())
    assert _observation_records(caplog) == []


def test_partial_new_vs_existing(caplog):
    """既有 key 与新增 key 混合 → 新增率按非空 key 总数计算。"""

    async def run() -> None:
        facts = json.dumps([
            {"content": "reuse", "category": "lesson",
             "confidence": 0.9, "persistence": "global", "topic_key": "known_topic"},
            {"content": "fresh", "category": "conv", "topic_key": "novel_topic"},
        ])
        mw, store = _make_middleware(facts)
        await store.save_agent_fact(
            ThreadFactItem(
                content="old", category="lesson", scope="agent",
                topic_key="known_topic", create_time="2026-01-01T00:00:00",
            )
        )
        with caplog.at_level(logging.INFO, logger="memory.middleware"):
            await mw.submit_event("t1", "user", "chat")
            await mw._aflush_thread("t1")

    asyncio.run(run())
    records = _observation_records(caplog)
    assert len(records) == 1
    # known_topic 命中 supersede 1；novel_topic 为新增 → 新增 1/2
    assert "supersede命中 1/2" in records[0]
    assert "新增topic_key 1/2" in records[0]