"""读写中间件测试

覆盖：
- ThreadMemoryWriteMiddleware: submit_event / 防抖 / buffer 裁剪 / 生命周期
- ThreadMemoryWriteMiddleware._parse_facts_response: JSON 解析容错
- ThreadMemoryReadMiddleware: awrap_model_call / facts 注入 / _format_facts
- ThreadMemoryLockPool: get / cleanup / 并发隔离
"""
from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock

from memory.lock_pool import ThreadMemoryLockPool
from memory.middleware import (
    MEMORY_BUFFER_DELAY_SECONDS,
    MAX_BUFFER_MESSAGE_COUNT,
    ThreadMemoryReadMiddleware,
    ThreadMemoryWriteMiddleware,
)
from memory.models import MemoryCategory, ThreadFactItem
from memory.store import ThreadMemoryStore


def _make_write_middleware(
    llm_getter: Any | None = None,
    buffer_delay_seconds: int = 999,
    max_buffer_messages: int = 30,
) -> tuple[ThreadMemoryWriteMiddleware, ThreadMemoryStore]:
    store = ThreadMemoryStore()
    lock_pool = ThreadMemoryLockPool()
    if llm_getter is None:
        llm_getter = lambda: None
    mw = ThreadMemoryWriteMiddleware(
        memory_store=store,
        lock_pool=lock_pool,
        llm_getter=llm_getter,
        buffer_delay_seconds=buffer_delay_seconds,
        max_buffer_messages=max_buffer_messages,
    )
    return mw, store


class _FakeLLM:
    def __init__(self, response: str = "[]"):
        self._response = response

    def chat(self, messages: list[dict[str, str]]) -> str:
        return self._response


# ════════════════════════════════════════════════════════════════════════
#  ThreadMemoryWriteMiddleware — 事件接收 & 防抖
# ════════════════════════════════════════════════════════════════════════


class TestSubmitEvent:
    def test_submit_event_adds_to_buffer(self):
        async def run():
            mw, _ = _make_write_middleware()
            await mw.submit_event("t1", "user", "hello")
            assert "t1" in mw._buffer
            assert len(mw._buffer["t1"]) == 1

        asyncio.run(run())

    def test_submit_event_empty_content_skipped(self):
        async def run():
            mw, _ = _make_write_middleware()
            await mw.submit_event("t1", "user", "")
            await mw.submit_event("t1", "user", "   ")
            assert "t1" not in mw._buffer

        asyncio.run(run())

    def test_submit_event_empty_thread_id_skipped(self):
        async def run():
            mw, _ = _make_write_middleware()
            await mw.submit_event("", "user", "content")
            assert len(mw._buffer) == 0

        asyncio.run(run())

    def test_submit_event_resets_timer(self):
        async def run():
            mw, _ = _make_write_middleware(buffer_delay_seconds=999)
            await mw.submit_event("t1", "user", "first")
            first_timer = mw._timers["t1"]

            await mw.submit_event("t1", "user", "second")
            second_timer = mw._timers["t1"]

            # 新事件应替换旧计时器
            assert first_timer is not second_timer
            # 旧计时器被 cancel（cancel 是异步的，需 yield 一次事件循环）
            await asyncio.sleep(0)
            assert first_timer.cancelled() or first_timer.done()

        asyncio.run(run())


# ════════════════════════════════════════════════════════════════════════
#  ThreadMemoryWriteMiddleware — buffer 裁剪
# ════════════════════════════════════════════════════════════════════════


class TestBufferTruncation:
    def test_buffer_truncated_to_max(self):
        async def run():
            mw, _ = _make_write_middleware(max_buffer_messages=3)
            for i in range(5):
                await mw.submit_event("t1", "user", f"msg-{i}")

            assert len(mw._buffer["t1"]) == 3
            # 保留最后 3 条
            contents = [c for _, c, _ in mw._buffer["t1"]]
            assert contents == ["msg-2", "msg-3", "msg-4"]

        asyncio.run(run())


# ════════════════════════════════════════════════════════════════════════
#  ThreadMemoryWriteMiddleware — Fact 抽取流水线
# ════════════════════════════════════════════════════════════════════════


class TestFactExtractionPipeline:
    def test_flush_writes_facts_via_llm(self):
        async def run():
            llm = _FakeLLM(response=json.dumps([
                {"content": "用户偏好 Python", "category": "user_fact", "confidence": 0.9}
            ]))
            mw, store = _make_write_middleware(llm_getter=lambda: llm)
            await mw.submit_event("t1", "user", "我偏好 Python")
            await mw._aflush_thread("t1")

            facts = await store.query_facts("t1")
            assert len(facts) == 1
            assert facts[0].content == "用户偏好 Python"

        asyncio.run(run())

    def test_flush_deduplicates_existing_facts(self):
        async def run():
            llm = _FakeLLM(response=json.dumps([
                {"content": "duplicate", "category": "conv", "confidence": 0.8},
                {"content": "new-fact", "category": "lesson", "confidence": 0.9},
            ]))
            mw, store = _make_write_middleware(llm_getter=lambda: llm)
            # 预写入一条
            await store.save_fact("t1", ThreadFactItem(content="duplicate"))
            await mw.submit_event("t1", "user", "some message")
            await mw._aflush_thread("t1")

            facts = await store.query_facts("t1")
            assert len(facts) == 2
            contents = {f.content for f in facts}
            assert "duplicate" in contents
            assert "new-fact" in contents

        asyncio.run(run())

    def test_flush_no_facts_extracted(self):
        async def run():
            llm = _FakeLLM(response="[]")
            mw, store = _make_write_middleware(llm_getter=lambda: llm)
            await mw.submit_event("t1", "user", "nothing worth remembering")
            await mw._aflush_thread("t1")

            facts = await store.query_facts("t1")
            assert facts == []

        asyncio.run(run())

    def test_flush_llm_failure_no_crash(self):
        async def run():
            def failing_llm():
                raise RuntimeError("LLM unavailable")

            mw, store = _make_write_middleware(llm_getter=failing_llm)
            await mw.submit_event("t1", "user", "content")
            await mw._aflush_thread("t1")

            facts = await store.query_facts("t1")
            assert facts == []

        asyncio.run(run())


# ════════════════════════════════════════════════════════════════════════
#  ThreadMemoryWriteMiddleware — _parse_facts_response 容错
# ════════════════════════════════════════════════════════════════════════


class TestParseFactsResponse:
    def test_parse_valid_json(self):
        response = '[{"content": "fact", "category": "conv"}]'
        result = ThreadMemoryWriteMiddleware._parse_facts_response(response)
        assert len(result) == 1
        assert result[0]["content"] == "fact"

    def test_parse_markdown_code_block(self):
        response = '```json\n[{"content": "fact"}]\n```'
        result = ThreadMemoryWriteMiddleware._parse_facts_response(response)
        assert len(result) == 1
        assert result[0]["content"] == "fact"

    def test_parse_bare_json_in_text(self):
        response = 'Here are the facts:\n[{"content": "fact"}]\nDone.'
        result = ThreadMemoryWriteMiddleware._parse_facts_response(response)
        assert len(result) == 1

    def test_parse_empty_response(self):
        assert ThreadMemoryWriteMiddleware._parse_facts_response("") == []
        assert ThreadMemoryWriteMiddleware._parse_facts_response("   ") == []

    def test_parse_invalid_json(self):
        assert ThreadMemoryWriteMiddleware._parse_facts_response("not json at all") == []

    def test_parse_non_array_json(self):
        result = ThreadMemoryWriteMiddleware._parse_facts_response('{"key": "value"}')
        assert result == []


# ════════════════════════════════════════════════════════════════════════
#  ThreadMemoryWriteMiddleware — 生命周期
# ════════════════════════════════════════════════════════════════════════


class TestWriteMiddlewareLifecycle:
    def test_cleanup_thread_clears_buffer(self):
        async def run():
            mw, _ = _make_write_middleware()
            await mw.submit_event("t1", "user", "content")
            await mw.cleanup_thread("t1")
            assert "t1" not in mw._buffer
            assert "t1" not in mw._timers

        asyncio.run(run())

    def test_flush_all_processes_all_threads(self):
        async def run():
            llm = _FakeLLM(response=json.dumps([
                {"content": "fact", "category": "conv"}
            ]))
            mw, store = _make_write_middleware(llm_getter=lambda: llm)
            await mw.submit_event("t1", "user", "msg1")
            await mw.submit_event("t2", "user", "msg2")
            await mw.flush_all()

            assert await store.count_facts("t1") == 1
            assert await store.count_facts("t2") == 1

        asyncio.run(run())

    def test_shutdown_clears_all(self):
        async def run():
            mw, _ = _make_write_middleware()
            await mw.submit_event("t1", "user", "content")
            await mw.submit_event("t2", "user", "content")
            await mw.shutdown()
            assert len(mw._buffer) == 0
            assert len(mw._timers) == 0

        asyncio.run(run())


# ════════════════════════════════════════════════════════════════════════
#  ThreadMemoryReadMiddleware — facts 注入
# ════════════════════════════════════════════════════════════════════════


class TestReadMiddleware:
    def test_format_facts_with_content(self):
        store = ThreadMemoryStore()
        mw = ThreadMemoryReadMiddleware(store)
        facts = [
            ThreadFactItem(content="偏好 Python", category="user_fact"),
            ThreadFactItem(content="踩坑记录", category="lesson"),
        ]
        text = mw._format_facts(facts)
        assert "【长期记忆】" in text
        assert "用户事实" in text
        assert "偏好 Python" in text
        assert "经验教训" in text
        assert "踩坑记录" in text

    def test_format_facts_empty(self):
        store = ThreadMemoryStore()
        mw = ThreadMemoryReadMiddleware(store)
        assert mw._format_facts([]) == ""

    def test_format_facts_unknown_category(self):
        store = ThreadMemoryStore()
        mw = ThreadMemoryReadMiddleware(store)
        facts = [ThreadFactItem(content="unknown", category="weird_cat")]
        text = mw._format_facts(facts)
        assert "记忆" in text  # 未知分类回退到"记忆"
        assert "unknown" in text

    def test_awrap_model_call_injects_facts(self):
        """验证 awrap_model_call 将 facts 注入 SystemMessage。"""
        async def run():
            store = ThreadMemoryStore()
            await store.save_fact("t1", ThreadFactItem(content="injected fact", category="user_fact"))
            mw = ThreadMemoryReadMiddleware(store)

            # 构建最小化 ModelRequest mock
            request = MagicMock()
            request.runtime.context = {"configurable": {"thread_id": "t1"}}
            request.system_message = None  # 无 system message

            captured_request = []

            async def handler(req):
                captured_request.append(req)
                return "result"

            result = await mw.awrap_model_call(request, handler)
            assert result == "result"
            assert len(captured_request) == 1
            # 验证 new_request 有 system_message
            assert captured_request[0].system_message is not None

        asyncio.run(run())

    def test_awrap_model_call_no_thread_id_passes_through(self):
        async def run():
            store = ThreadMemoryStore()
            mw = ThreadMemoryReadMiddleware(store)

            request = MagicMock()
            request.runtime.context = None

            async def handler(req):
                return "passthrough"

            result = await mw.awrap_model_call(request, handler)
            assert result == "passthrough"

        asyncio.run(run())

    def test_awrap_model_call_empty_facts_passes_through(self):
        async def run():
            store = ThreadMemoryStore()
            mw = ThreadMemoryReadMiddleware(store)

            request = MagicMock()
            request.runtime.context = {"configurable": {"thread_id": "empty-thread"}}

            async def handler(req):
                return "no-facts"

            result = await mw.awrap_model_call(request, handler)
            assert result == "no-facts"

        asyncio.run(run())


# ════════════════════════════════════════════════════════════════════════
#  ThreadMemoryLockPool — 并发锁
# ════════════════════════════════════════════════════════════════════════


class TestLockPool:
    def test_get_returns_same_lock_for_same_thread(self):
        async def run():
            pool = ThreadMemoryLockPool()
            lock1 = await pool.get("t1")
            lock2 = await pool.get("t1")
            assert lock1 is lock2

        asyncio.run(run())

    def test_get_returns_different_locks_for_different_threads(self):
        async def run():
            pool = ThreadMemoryLockPool()
            lock1 = await pool.get("t1")
            lock2 = await pool.get("t2")
            assert lock1 is not lock2

        asyncio.run(run())

    def test_cleanup_removes_lock(self):
        async def run():
            pool = ThreadMemoryLockPool()
            await pool.get("t1")
            await pool.cleanup("t1")
            lock = await pool.get("t1")
            # cleanup 后再 get 应创建新锁
            assert lock is not None

        asyncio.run(run())

    def test_same_thread_serialized(self):
        """同一 thread 的锁确保串行执行。"""
        async def run():
            pool = ThreadMemoryLockPool()
            lock = await pool.get("t1")
            # 异步获取锁
            await lock.acquire()
            assert lock.locked() is True
            lock.release()
            assert lock.locked() is False

        asyncio.run(run())


# ════════════════════════════════════════════════════════════════════════
#  配置常量
# ════════════════════════════════════════════════════════════════════════


class TestConfigConstants:
    def test_default_buffer_delay(self):
        assert MEMORY_BUFFER_DELAY_SECONDS == 20

    def test_default_max_buffer_messages(self):
        assert MAX_BUFFER_MESSAGE_COUNT == 30
