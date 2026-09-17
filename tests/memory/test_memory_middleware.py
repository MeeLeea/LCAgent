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
from typing import Any, Self
from unittest.mock import MagicMock, patch

from memory.lock_pool import ThreadMemoryLockPool
from memory.middleware import (
    ThreadMemoryReadMiddleware,
    ThreadMemoryWriteMiddleware,
)
from memory.models import ThreadFactItem
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
            contents = [c for _, c, _, _, _ in mw._buffer["t1"]]
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

            # user_fact 走 agent 级（跨会话共享），thread 级应为空
            thread_facts = await store.query_facts("t1")
            assert thread_facts == []
            agent_facts = await store.query_agent_facts()
            assert len(agent_facts) == 1
            assert agent_facts[0].content == "用户偏好 Python"
            assert agent_facts[0].scope == "agent"

        asyncio.run(run())

    def test_flush_deduplicates_existing_facts(self):
        async def run():
            llm = _FakeLLM(response=json.dumps([
                {"content": "duplicate", "category": "conv", "confidence": 0.8},
                {"content": "new-fact", "category": "lesson", "confidence": 0.9},
            ]))
            mw, store = _make_write_middleware(llm_getter=lambda: llm)
            # 预写入一条 thread 级 fact（conv 类）
            await store.save_fact("t1", ThreadFactItem(content="duplicate"))
            await mw.submit_event("t1", "user", "some message")
            await mw._aflush_thread("t1")

            # conv 类走 thread 级：duplicate 已存在应被去重，保留预写入的那条
            thread_facts = await store.query_facts("t1")
            assert len(thread_facts) == 1
            assert thread_facts[0].content == "duplicate"
            # lesson 类走 agent 级：new-fact 应写入 agent namespace
            agent_facts = await store.query_agent_facts()
            assert len(agent_facts) == 1
            assert agent_facts[0].content == "new-fact"
            assert agent_facts[0].scope == "agent"

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


class _FakeRuntime:
    """带 ``context`` 属性的 runtime 替身（对应 ModelRequest.runtime.context）。"""

    def __init__(self, context):
        self.context = context


class _FakeModelRequest:
    """最小化 ModelRequest 替身：override() 返回携带真实 SystemMessage 的新实例。"""

    def __init__(self, context, system_message=None):
        self.runtime = _FakeRuntime(context)
        self.system_message = system_message

    def override(self, system_message=None):
        return _FakeModelRequest(self.runtime.context, system_message=system_message)


def _sys_content_text(system_message) -> str:
    """从 SystemMessage 提取纯文本（content 可能为 text block 列表）。"""
    content = system_message.content
    if isinstance(content, list):
        return "".join(
            str(block.get("text", "")) if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


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

    def test_awrap_model_call_respects_recall_limit(self):
        """recall_limit 约束注入的 fact 条数（取最近 N 条）。"""

        async def run():
            store = ThreadMemoryStore()
            # 显式递增 create_time，保证 query_facts 升序确定
            for i in range(3):
                await store.save_fact(
                    "t1",
                    ThreadFactItem(
                        content=f"fact-{i}",
                        category="user_fact",
                        create_time=f"2026-01-01T00:00:00.{i:03d}",
                    ),
                )
            mw = ThreadMemoryReadMiddleware(store, recall_limit=2)

            request = _FakeModelRequest(context={"configurable": {"thread_id": "t1"}})
            captured = []

            async def handler(req):
                captured.append(req)
                return "ok"

            result = await mw.awrap_model_call(request, handler)
            assert result == "ok"
            assert len(captured) == 1
            text = _sys_content_text(captured[0].system_message)
            assert "fact-1" in text and "fact-2" in text
            assert "fact-0" not in text

        asyncio.run(run())

    def test_awrap_model_call_without_limit_injects_all(self):
        """未配置 recall_limit 时注入全部 facts（保持原行为）。"""

        async def run():
            store = ThreadMemoryStore()
            for i in range(3):
                await store.save_fact(
                    "t1",
                    ThreadFactItem(
                        content=f"fact-{i}",
                        category="user_fact",
                        create_time=f"2026-01-01T00:00:00.{i:03d}",
                    ),
                )
            mw = ThreadMemoryReadMiddleware(store)  # recall_limit=None

            request = _FakeModelRequest(context={"configurable": {"thread_id": "t1"}})
            captured = []

            async def handler(req):
                captured.append(req)
                return "ok"

            await mw.awrap_model_call(request, handler)
            text = _sys_content_text(captured[0].system_message)
            assert "fact-0" in text and "fact-1" in text and "fact-2" in text

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
    """中间件默认值来自 memory/config.py（单一来源，避免分叉常量）。"""

    def test_default_values_come_from_config(self):
        from memory.config import (
            MEMORY_BUFFER_DELAY_SECONDS,
            MEMORY_MAX_BUFFER_MESSAGES,
        )

        store = ThreadMemoryStore()
        lock_pool = ThreadMemoryLockPool()
        mw = ThreadMemoryWriteMiddleware(
            memory_store=store,
            lock_pool=lock_pool,
            llm_getter=lambda: None,
        )
        # 未显式传参时，中间件默认值应取自 config.py 的唯一来源
        assert mw._buffer_delay_seconds == MEMORY_BUFFER_DELAY_SECONDS
        assert mw._max_buffer_messages == MEMORY_MAX_BUFFER_MESSAGES


# ════════════════════════════════════════════════════════════════════════
#  Agent 级 / Thread 级 分层路由与读聚合
# ════════════════════════════════════════════════════════════════════════


class TestAgentScopeRouting:
    """LLM 返回不同 category 的 fact 时按 scope 路由写入对应 namespace。"""

    def test_user_fact_routed_to_agent_namespace(self):
        """user_fact 类 fact → 写入 agent namespace，thread 级为空。"""
        async def run():
            llm = _FakeLLM(response=json.dumps([
                {"content": "用户偏好深色主题", "category": "user_fact", "confidence": 0.9}
            ]))
            mw, store = _make_write_middleware(llm_getter=lambda: llm)
            await mw.submit_event("t1", "user", "我喜欢深色主题")
            await mw._aflush_thread("t1")

            thread_facts = await store.query_facts("t1")
            assert thread_facts == []
            agent_facts = await store.query_agent_facts()
            assert len(agent_facts) == 1
            assert agent_facts[0].content == "用户偏好深色主题"
            assert agent_facts[0].scope == "agent"

        asyncio.run(run())

    def test_lesson_routed_to_agent_namespace(self):
        """lesson 类 fact → 写入 agent namespace。"""
        async def run():
            llm = _FakeLLM(response=json.dumps([
                {"content": "踩坑：API 限流", "category": "lesson", "confidence": 0.9}
            ]))
            mw, store = _make_write_middleware(llm_getter=lambda: llm)
            await mw.submit_event("t1", "user", "API 限流了")
            await mw._aflush_thread("t1")

            assert await store.query_facts("t1") == []
            agent_facts = await store.query_agent_facts()
            assert len(agent_facts) == 1
            assert agent_facts[0].category == "lesson"

        asyncio.run(run())

    def test_conv_routed_to_thread_namespace(self):
        """conv 类 fact → 写入 thread namespace，agent 级为空。"""
        async def run():
            llm = _FakeLLM(response=json.dumps([
                {"content": "本次对话要点", "category": "conv", "confidence": 0.8}
            ]))
            mw, store = _make_write_middleware(llm_getter=lambda: llm)
            await mw.submit_event("t1", "user", "讨论要点")
            await mw._aflush_thread("t1")

            agent_facts = await store.query_agent_facts()
            assert agent_facts == []
            thread_facts = await store.query_facts("t1")
            assert len(thread_facts) == 1
            assert thread_facts[0].content == "本次对话要点"
            assert thread_facts[0].scope == "thread"

        asyncio.run(run())

    def test_business_routed_to_thread_namespace(self):
        """business 类 fact → 写入 thread namespace。"""
        async def run():
            llm = _FakeLLM(response=json.dumps([
                {"content": "项目配置 X", "category": "business", "confidence": 0.9}
            ]))
            mw, store = _make_write_middleware(llm_getter=lambda: llm)
            await mw.submit_event("t1", "user", "配置 X")
            await mw._aflush_thread("t1")

            assert await store.query_agent_facts() == []
            thread_facts = await store.query_facts("t1")
            assert len(thread_facts) == 1
            assert thread_facts[0].category == "business"

        asyncio.run(run())

    def test_read_aggregates_agent_and_thread_facts(self):
        """agent namespace 预置 1 条 + thread 预置 1 条 → SystemMessage 同时含两者。"""
        async def run():
            store = ThreadMemoryStore()
            await store.save_agent_fact(
                ThreadFactItem(content="agent-fact-shared", category="user_fact")
            )
            await store.save_fact(
                "t1", ThreadFactItem(content="thread-fact-local", category="conv")
            )
            mw = ThreadMemoryReadMiddleware(store)

            request = _FakeModelRequest(context={"configurable": {"thread_id": "t1"}})
            captured = []

            async def handler(req):
                captured.append(req)
                return "ok"

            await mw.awrap_model_call(request, handler)
            text = _sys_content_text(captured[0].system_message)
            assert "agent-fact-shared" in text
            assert "thread-fact-local" in text

        asyncio.run(run())

    def test_read_merge_agent_priority_on_duplicate_content(self):
        """content 重复时 agent 版本优先（保留 agent 级版本）。"""
        async def run():
            store = ThreadMemoryStore()
            same_content = "重复内容"
            # 两级 namespace 预置相同 content
            await store.save_agent_fact(
                ThreadFactItem(content=same_content, category="user_fact", scope="agent")
            )
            await store.save_fact(
                "t1",
                ThreadFactItem(content=same_content, category="conv", scope="thread"),
            )
            mw = ThreadMemoryReadMiddleware(store)

            request = _FakeModelRequest(context={"configurable": {"thread_id": "t1"}})
            captured = []

            async def handler(req):
                captured.append(req)
                return "ok"

            await mw.awrap_model_call(request, handler)
            text = _sys_content_text(captured[0].system_message)
            # 注入文本只应出现一次（去重后保留 agent 版本）
            assert text.count(same_content) == 1
            # agent 版本 category 为 user_fact → "用户事实" 标签
            assert "用户事实" in text

        asyncio.run(run())

    def test_read_recall_limit_applies_to_merged_list(self):
        """recall_limit 截取对合并后的列表生效（agent 3 + thread 5, limit=5 → 5 条）。"""
        async def run():
            store = ThreadMemoryStore()
            # agent 级 3 条（create_time 较早）
            for i in range(3):
                await store.save_agent_fact(
                    ThreadFactItem(
                        content=f"agent-{i}",
                        category="user_fact",
                        create_time=f"2026-01-01T00:00:0{i}",
                    )
                )
            # thread 级 5 条（create_time 较晚）
            for i in range(5):
                await store.save_fact(
                    "t1",
                    ThreadFactItem(
                        content=f"thread-{i}",
                        category="conv",
                        create_time=f"2026-01-02T00:00:0{i}",
                    ),
                )
            mw = ThreadMemoryReadMiddleware(store, recall_limit=5)

            request = _FakeModelRequest(context={"configurable": {"thread_id": "t1"}})
            captured = []

            async def handler(req):
                captured.append(req)
                return "ok"

            await mw.awrap_model_call(request, handler)
            text = _sys_content_text(captured[0].system_message)
            # 合并 8 条按 create_time 升序，取最近 5 条 → 应为 thread-0..thread-4
            # （create_time 较晚的 5 条 thread facts）
            for i in range(5):
                assert f"thread-{i}" in text
            # agent-* 较早，应被截掉
            for i in range(3):
                assert f"agent-{i}" not in text

        asyncio.run(run())

    def test_read_touch_dispatches_by_scope(self):
        """awrap_model_call 触发 touch 时按 scope 分发：
        agent 级调 touch_agent_fact，thread 级调 touch_fact。"""
        async def run():
            store = ThreadMemoryStore()
            agent_item = ThreadFactItem(
                content="agent-touch-target", category="user_fact", scope="agent"
            )
            thread_item = ThreadFactItem(
                content="thread-touch-target", category="conv", scope="thread"
            )
            agent_item.last_used_at = "2026-01-01T00:00:00"
            thread_item.last_used_at = "2026-01-01T00:00:00"
            await store.save_agent_fact(agent_item)
            await store.save_fact("t1", thread_item)

            mw = ThreadMemoryReadMiddleware(store)
            request = _FakeModelRequest(context={"configurable": {"thread_id": "t1"}})

            async def handler(req):
                return "ok"

            await mw.awrap_model_call(request, handler)
            # 等待 asyncio.create_task 调度的 touch 完成
            await asyncio.sleep(0.05)

            agent_facts = await store.query_agent_facts()
            thread_facts = await store.query_facts("t1")
            assert agent_facts[0].last_used_at != "2026-01-01T00:00:00"
            assert thread_facts[0].last_used_at != "2026-01-01T00:00:00"

        asyncio.run(run())


# ════════════════════════════════════════════════════════════════════════
#  确定性预判定（judge_long_term_memory 精简后的行为）
# ════════════════════════════════════════════════════════════════════════


class _RecordingLLM:
    """记录 chat 调用消息的 mock，用于验证 important 标注透传。"""

    def __init__(self, response: str = "[]"):
        self._response = response
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        return self._response


class TestDeterministicJudgment:
    """失败次数驱动的确定性判定：SKIP 生效 + 失败≥2 绕过 LLM 直接记 lesson。"""

    def test_single_failed_tool_result_skipped(self):
        """单次失败的工具结果应被确定性丢弃，不触发 LLM 抽取也不落库。"""
        async def run():
            llm = _RecordingLLM(response=json.dumps([]))
            mw, store = _make_write_middleware(llm_getter=lambda: llm)
            await mw.submit_event(
                "t1",
                "assistant",
                "[工具执行失败] run_shell 抛出了 TimeoutError",
                event_type="tool_result",
                tool_name="run_shell",
            )
            await mw._aflush_thread("t1")

            # 单次失败 → SKIP：既不调 LLM，也不落库
            assert llm.calls == []
            assert await store.query_facts("t1") == []
            assert await store.query_agent_facts() == []

        asyncio.run(run())

    def test_two_failed_tool_results_become_lesson(self):
        """同类失败 ≥2 次应确定性记为 lesson（agent 级），绕过 LLM。"""
        async def run():
            llm = _RecordingLLM(response=json.dumps([]))
            mw, store = _make_write_middleware(llm_getter=lambda: llm)
            for _ in range(2):
                await mw.submit_event(
                    "t1",
                    "assistant",
                    "命令超时 run_shell",
                    event_type="tool_result",
                    tool_name="run_shell",
                )
            await mw._aflush_thread("t1")

            # 两次失败 → 确定性 lesson，直接写入 agent 级，无需 LLM
            assert llm.calls == []
            agent_facts = await store.query_agent_facts()
            assert len(agent_facts) == 1
            assert agent_facts[0].category == "lesson"
            assert agent_facts[0].scope == "agent"

        asyncio.run(run())

    def test_important_message_marked_in_llm_prompt(self):
        """important=True 的消息应带 [用户明确要求记住] 标注传入 LLM。"""
        async def run():
            llm = _RecordingLLM(response=json.dumps([]))
            mw, _ = _make_write_middleware(llm_getter=lambda: llm)
            await mw.submit_event("t1", "user", "记住我喜欢蓝色", important=True)
            await mw._aflush_thread("t1")

            # important 消息进入 LLM 抽取，且对话文本带标注
            assert llm.calls
            conversation = llm.calls[0][-1]["content"]
            assert "用户明确要求记住" in conversation
            assert "我喜欢蓝色" in conversation

        asyncio.run(run())


# ════════════════════════════════════════════════════════════════════════
#  Agent 级写入/淘汰：条件化 prune + agent 锁（FIX 1 / FIX 2d）
# ════════════════════════════════════════════════════════════════════════


class _FakeAgentLock:
    """agent 锁测试替身：仅实现 async 上下文协议并记录持锁状态。"""

    def __init__(self) -> None:
        self.held = False

    async def __aenter__(self) -> Self:
        self.held = True
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.held = False


class TestAgentPruneConditional:
    """仅当本批次确实写入 agent facts 时才触发 prune_agent_facts。"""

    def test_prune_agent_facts_not_called_when_no_agent_items(self):
        """仅抽出 thread 级 category（conv）时不得调用 prune_agent_facts。"""

        async def run():
            llm = _FakeLLM(response=json.dumps([
                {"content": "本次对话要点", "category": "conv", "confidence": 0.8}
            ]))
            mw, store = _make_write_middleware(llm_getter=lambda: llm)
            calls: list[str] = []
            original_prune = store.prune_agent_facts

            async def spy_prune() -> int:
                calls.append("prune_agent_facts")
                return await original_prune()

            with patch.object(store, "prune_agent_facts", spy_prune):
                await mw.submit_event("t1", "user", "讨论要点")
                await mw._aflush_thread("t1")

            assert calls == []
            assert await store.query_agent_facts() == []
            assert len(await store.query_facts("t1")) == 1

        asyncio.run(run())

    def test_prune_agent_facts_called_when_agent_items_written(self):
        """抽出 user_fact（agent 级）时应调用 prune_agent_facts。"""

        async def run():
            llm = _FakeLLM(response=json.dumps([
                {"content": "用户偏好深色主题", "category": "user_fact", "confidence": 0.9}
            ]))
            mw, store = _make_write_middleware(llm_getter=lambda: llm)
            calls: list[str] = []
            original_prune = store.prune_agent_facts

            async def spy_prune() -> int:
                calls.append("prune_agent_facts")
                return await original_prune()

            with patch.object(store, "prune_agent_facts", spy_prune):
                await mw.submit_event("t1", "user", "我喜欢深色主题")
                await mw._aflush_thread("t1")

            assert calls == ["prune_agent_facts"]
            assert await store.count_agent_facts() == 1

        asyncio.run(run())

    def test_agent_write_and_prune_hold_agent_lock(self):
        """agent 级批量写入与 prune 必须在 agent 锁持有期间执行。"""

        async def run():
            llm = _FakeLLM(response=json.dumps([
                {"content": "用户偏好深色主题", "category": "user_fact", "confidence": 0.9}
            ]))
            store = ThreadMemoryStore()
            fake_lock = _FakeAgentLock()
            mw = ThreadMemoryWriteMiddleware(
                memory_store=store,
                lock_pool=ThreadMemoryLockPool(),
                llm_getter=lambda: llm,
                buffer_delay_seconds=999,
                max_buffer_messages=30,
                agent_lock=fake_lock,
            )
            observed: list[bool] = []
            original_save = store.save_agent_facts_batch
            original_prune = store.prune_agent_facts

            async def spy_save(items: list[ThreadFactItem]) -> None:
                observed.append(fake_lock.held)
                await original_save(items)

            async def spy_prune() -> int:
                observed.append(fake_lock.held)
                return await original_prune()

            with patch.object(store, "save_agent_facts_batch", spy_save), patch.object(
                store, "prune_agent_facts", spy_prune
            ):
                await mw.submit_event("t1", "user", "我喜欢深色主题")
                await mw._aflush_thread("t1")

            assert observed == [True, True]
            assert fake_lock.held is False

        asyncio.run(run())
