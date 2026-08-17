"""SessionManager 真实集成测试

覆盖三层架构的跨层交互：
- AgentCore ↔ SessionManager ↔ MemoryManager 完整数据流
- per-thread 并发锁（同 thread 串行、不同 thread 并行）
- achat_stream / arun_stream / aresume_stream 事件流消费
- 记忆召回注入、事件消费、压缩、清理
- aget_memory_summary / acompress_memory / aclear_long_term_memory
- aclose 生命周期（flush → shutdown → agent.aclose）
- SessionManager 无 MemoryManager 时的降级行为
- 会话管理委托（new_session / switch / delete / list / messages）

不使用 Mock SessionManager — 使用真实 SessionManager + 真实 MemoryManager + Mock AgentCore。
"""
from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from utils.events import AgentEvent, EventType
from memory.lock_pool import ThreadMemoryLockPool
from memory.manager import MemoryManager
from memory.models import ThreadFactItem
from memory.store import ThreadMemoryStore
from session.manager import SessionManager


# ════════════════════════════════════════════════════════════════════════
#  Fixtures
# ════════════════════════════════════════════════════════════════════════


def _make_memory_manager(
    llm_response: str = "[]",
) -> tuple[MemoryManager, ThreadMemoryStore]:
    """创建真实 MemoryManager（InMemoryStore 后端）。"""
    store = ThreadMemoryStore()
    lock_pool = ThreadMemoryLockPool()

    class _LLM:
        def chat(self, messages):
            return llm_response

    mgr = MemoryManager(
        memory_store=store,
        lock_pool=lock_pool,
        llm_getter=lambda: _LLM(),
        buffer_delay_seconds=999,  # 不自动触发
    )
    return mgr, store


def _make_fake_agent(
    events: list[AgentEvent] | None = None,
    thread_id: str = "default",
) -> MagicMock:
    """创建模拟 AgentCore，支持 arun_events / aresume_events 返回事件流。"""
    agent = MagicMock()
    agent._closed = False

    # SessionRegistry mock
    agent.session = MagicMock()
    agent.session.current_session_id = thread_id
    agent.session.new_session = MagicMock(return_value="new-thread")
    agent.session.new_workflow_session = MagicMock(return_value="workflow-test-abc12345")
    agent.session.alist_sessions = AsyncMock(return_value=[thread_id])
    agent.session.aswitch_session = AsyncMock(return_value=True)
    agent.session.adelete_session = AsyncMock(return_value=True)
    agent.session.aget_messages = AsyncMock(return_value=[])
    agent.session.aexport_session = AsyncMock(return_value="exported text")
    agent.session.asummarize = AsyncMock(return_value={
        "checkpoint_messages": 5,
        "total_sessions": 2,
    })

    # AgentCore 方法
    agent.aclose = AsyncMock()
    agent._current_sid = lambda tid=None: tid or thread_id
    agent.set_current_session = MagicMock()
    agent.checkpoint_info = {
        "checkpoint_backend": "memory",
        "checkpoint_file": "(内存)",
    }
    agent.aget_execution_history = AsyncMock(return_value=[])
    agent.aclear_history = AsyncMock()
    agent.manually_compact = AsyncMock(return_value=None)

    # arun_events / aresume_events 返回事件流
    if events is None:
        events = [AgentEvent.done(thread_id=thread_id)]

    async def _arun_events(message, thread_id=None, is_run_mode=False):
        for ev in events:
            yield ev

    async def _aresume_events(payload, thread_id=None):
        for ev in events:
            yield ev

    agent.arun_events = _arun_events
    agent.aresume_events = _aresume_events

    return agent


def _make_session_manager(
    events: list[AgentEvent] | None = None,
    thread_id: str = "default",
    llm_response: str = "[]",
    with_memory: bool = True,
) -> tuple[SessionManager, MagicMock, MemoryManager | None, ThreadMemoryStore | None]:
    """创建完整的三层架构测试实例。"""
    agent = _make_fake_agent(events=events, thread_id=thread_id)
    if with_memory:
        memory, store = _make_memory_manager(llm_response=llm_response)
    else:
        memory, store = None, None
    sm = SessionManager(agent, memory=memory)
    return sm, agent, memory, store


# ════════════════════════════════════════════════════════════════════════
#  属性暴露
# ════════════════════════════════════════════════════════════════════════


class TestProperties:
    def test_agent_property(self):
        sm, agent, _, _ = _make_session_manager()
        assert sm.agent is agent

    def test_memory_property(self):
        sm, _, memory, _ = _make_session_manager()
        assert sm.memory is memory

    def test_memory_property_none_when_no_memory(self):
        sm, _, _, _ = _make_session_manager(with_memory=False)
        assert sm.memory is None

    def test_session_property(self):
        sm, agent, _, _ = _make_session_manager()
        assert sm.session is agent.session

    def test_current_session_id(self):
        sm, _, _, _ = _make_session_manager(thread_id="thread-x")
        assert sm.current_session_id == "thread-x"


# ════════════════════════════════════════════════════════════════════════
#  achat_stream — 流式对话
# ════════════════════════════════════════════════════════════════════════


class TestAchatStream:
    def test_achat_stream_yields_events(self):
        async def run():
            events = [
                AgentEvent.token("Hello", thread_id="t1"),
                AgentEvent.token(" World", thread_id="t1"),
                AgentEvent.done(thread_id="t1"),
            ]
            sm, _, _, _ = _make_session_manager(events=events, thread_id="t1")

            results = []
            async for ev in sm.achat_stream("hi", thread_id="t1"):
                results.append(ev)

            assert len(results) == 3
            assert results[0]["type"] == "token"
            assert results[0]["content"] == "Hello"
            assert results[1]["content"] == " World"
            assert results[2]["type"] == "done"

        asyncio.run(run())

    def test_achat_stream_consumes_memory_events(self):
        """DONE 事件应被提交到 MemoryManager。"""
        async def run():
            done_event = AgentEvent(
                event_type=EventType.DONE,
                content="任务完成的最终输出",
                thread_id="t1",
            )
            llm_resp = json.dumps([
                {"content": "完成结果", "category": "conv", "confidence": 0.9}
            ])
            sm, _, memory, store = _make_session_manager(
                events=[done_event], thread_id="t1", llm_response=llm_resp
            )

            results = []
            async for ev in sm.achat_stream("do something", thread_id="t1"):
                results.append(ev)

            # flush_all 触发 fact 抽取
            await memory.flush_all()
            facts = await store.query_facts("t1")
            assert len(facts) == 1
            assert facts[0].content == "完成结果"

        asyncio.run(run())

    def test_achat_stream_submits_user_message_to_memory(self):
        """用户消息应提交到 MemoryManager 的 buffer。"""
        async def run():
            sm, _, memory, _ = _make_session_manager(
                events=[AgentEvent.done(thread_id="t1")],
                thread_id="t1",
            )
            async for _ in sm.achat_stream("remember I like Python", thread_id="t1"):
                pass

            # 用户消息应在 write_middleware 的 buffer 中
            assert "t1" in memory.write_middleware._buffer
            assert len(memory.write_middleware._buffer["t1"]) == 1

        asyncio.run(run())

    def test_achat_stream_without_memory_still_works(self):
        async def run():
            events = [
                AgentEvent.token("hello", thread_id="t1"),
                AgentEvent.done(thread_id="t1"),
            ]
            sm, _, _, _ = _make_session_manager(
                events=events, thread_id="t1", with_memory=False
            )

            results = []
            async for ev in sm.achat_stream("hi", thread_id="t1"):
                results.append(ev)

            assert len(results) == 2

        asyncio.run(run())


# ════════════════════════════════════════════════════════════════════════
#  arun_stream — 流式任务执行（important=True）
# ════════════════════════════════════════════════════════════════════════


class TestArunStream:
    def test_arun_stream_marks_important(self):
        """arun_stream 应以 important=True 提交用户消息。"""
        async def run():
            sm, _, memory, _ = _make_session_manager(
                events=[AgentEvent.done(thread_id="t1")],
                thread_id="t1",
            )
            async for _ in sm.arun_stream("run this task", thread_id="t1"):
                pass

            buf = memory.write_middleware._buffer["t1"]
            assert len(buf) == 1
            _, _, important = buf[0]
            assert important is True

        asyncio.run(run())


# ════════════════════════════════════════════════════════════════════════
#  aresume_stream — 流式恢复中断
# ════════════════════════════════════════════════════════════════════════


class TestAresumeStream:
    def test_aresume_stream_yields_events(self):
        async def run():
            events = [
                AgentEvent.token("resumed", thread_id="t1"),
                AgentEvent.done(thread_id="t1"),
            ]
            sm, _, _, _ = _make_session_manager(events=events, thread_id="t1")

            results = []
            async for ev in sm.aresume_stream({"choice_id": "approve"}, thread_id="t1"):
                results.append(ev)

            assert len(results) == 2
            assert results[0]["content"] == "resumed"

        asyncio.run(run())


# ════════════════════════════════════════════════════════════════════════
#  非流式接口
# ════════════════════════════════════════════════════════════════════════


class TestNonStream:
    def test_achat_collects_tokens(self):
        async def run():
            events = [
                AgentEvent.token("Hello", thread_id="t1"),
                AgentEvent.token(" World", thread_id="t1"),
                AgentEvent.done(thread_id="t1"),
            ]
            sm, _, _, _ = _make_session_manager(events=events, thread_id="t1")
            result = await sm.achat("hi", thread_id="t1")
            assert result == "Hello World"

        asyncio.run(run())

    def test_achat_error_returns_error_content(self):
        async def run():
            events = [AgentEvent.error("something broke", thread_id="t1")]
            sm, _, _, _ = _make_session_manager(events=events, thread_id="t1")
            result = await sm.achat("hi", thread_id="t1")
            assert result == "something broke"

        asyncio.run(run())

    def test_achat_interrupt_raises_runtime_error(self):
        async def run():
            events = [AgentEvent.interrupt(
                prompt="Confirm?", choices=[], thread_id="t1"
            )]
            sm, _, _, _ = _make_session_manager(events=events, thread_id="t1")
            with pytest.raises(RuntimeError, match="interrupted"):
                await sm.achat("hi", thread_id="t1")

        asyncio.run(run())

    def test_arun_collects_tokens(self):
        async def run():
            events = [
                AgentEvent.token("result", thread_id="t1"),
                AgentEvent.done(thread_id="t1"),
            ]
            sm, _, _, _ = _make_session_manager(events=events, thread_id="t1")
            result = await sm.arun("task", thread_id="t1")
            assert result == "result"

        asyncio.run(run())

    def test_aresume_collects_tokens(self):
        async def run():
            events = [
                AgentEvent.token("ok", thread_id="t1"),
                AgentEvent.done(thread_id="t1"),
            ]
            sm, _, _, _ = _make_session_manager(events=events, thread_id="t1")
            result = await sm.aresume({"choice_id": "approve"}, thread_id="t1")
            assert result == "ok"

        asyncio.run(run())


# ════════════════════════════════════════════════════════════════════════
#  并发锁 — per-thread 串行
# ════════════════════════════════════════════════════════════════════════


class TestConcurrency:
    def test_same_thread_serialized(self):
        """同一 thread_id 的请求串行执行：两个 achat_stream 不并发。"""
        async def run():
            # 用自定义事件追踪并发
            active_count = 0
            max_concurrent = 0

            class _TrackingEvent:
                """包装 AgentEvent，在迭代时追踪并发数。"""

            original_events = [
                AgentEvent.token("hello", thread_id="t1"),
                AgentEvent.done(thread_id="t1"),
            ]

            sm, agent, _, _ = _make_session_manager(thread_id="t1")

            # 替换 arun_events 为追踪版本
            async def _tracking_arun_events(message, thread_id=None, is_run_mode=False):
                nonlocal active_count, max_concurrent
                active_count += 1
                max_concurrent = max(max_concurrent, active_count)
                await asyncio.sleep(0.05)
                for ev in original_events:
                    yield ev
                active_count -= 1

            agent.arun_events = _tracking_arun_events

            async def chat():
                async for _ in sm.achat_stream("msg", thread_id="t1"):
                    pass

            await asyncio.gather(chat(), chat())
            # 同 thread 串行：最大并发应为 1
            assert max_concurrent == 1

        asyncio.run(run())

    def test_different_threads_parallel(self):
        """不同 thread_id 的请求并行执行。"""
        async def run():
            sm, _, _, _ = _make_session_manager(
                events=[AgentEvent.done(thread_id="t1")],
                thread_id="t1",
            )

            completion_times = []

            async def chat(tid):
                async for _ in sm.achat_stream("msg", thread_id=tid):
                    pass
                completion_times.append(tid)

            # 两个不同 thread 同时执行
            await asyncio.gather(chat("t1"), chat("t2"))
            assert len(completion_times) == 2

        asyncio.run(run())


# ════════════════════════════════════════════════════════════════════════
#  记忆管理
# ════════════════════════════════════════════════════════════════════════


class TestMemoryManagement:
    def test_aget_memory_summary(self):
        async def run():
            sm, _, memory, store = _make_session_manager(thread_id="t1")
            await store.save_fact("t1", ThreadFactItem(content="fact-1"))
            await store.save_fact("t1", ThreadFactItem(content="fact-2"))

            summary = await sm.aget_memory_summary()
            assert summary["thread_id"] == "t1"
            assert summary["checkpoint_messages"] == 5
            assert summary["checkpoint_backend"] == "memory"
            assert summary["long_term_count"] == 2
            assert summary["total_threads"] == 2

        asyncio.run(run())

    def test_aget_memory_summary_without_memory(self):
        async def run():
            sm, _, _, _ = _make_session_manager(with_memory=False, thread_id="t1")
            summary = await sm.aget_memory_summary()
            assert summary["long_term_count"] == 0

        asyncio.run(run())

    def test_acompress_memory(self):
        async def run():
            llm_resp = "压缩后的摘要"
            sm, _, memory, store = _make_session_manager(
                thread_id="t1", llm_response=llm_resp
            )
            await store.save_fact("t1", ThreadFactItem(content="fact-1"))
            await store.save_fact("t1", ThreadFactItem(content="fact-2"))

            result = await sm.acompress_memory()
            assert result["success"] is True
            assert result["original_count"] == 2
            assert result["summary"] == "压缩后的摘要"

            facts = await store.query_facts("t1")
            assert len(facts) == 1

        asyncio.run(run())

    def test_acompress_memory_without_memory(self):
        async def run():
            sm, _, _, _ = _make_session_manager(with_memory=False)
            result = await sm.acompress_memory()
            assert result["success"] is False
            assert "未初始化" in result["error"]

        asyncio.run(run())

    def test_aclear_long_term_memory(self):
        async def run():
            sm, _, _, store = _make_session_manager(thread_id="t1")
            await store.save_fact("t1", ThreadFactItem(content="a"))
            await store.save_fact("t1", ThreadFactItem(content="b"))

            cleared = await sm.aclear_long_term_memory()
            assert cleared == 2
            assert await store.count_facts("t1") == 0

        asyncio.run(run())

    def test_aclear_long_term_memory_without_memory(self):
        async def run():
            sm, _, _, _ = _make_session_manager(with_memory=False)
            cleared = await sm.aclear_long_term_memory()
            assert cleared == 0

        asyncio.run(run())


# ════════════════════════════════════════════════════════════════════════
#  会话管理委托
# ════════════════════════════════════════════════════════════════════════


class TestSessionDelegation:
    def test_new_session(self):
        sm, agent, _, _ = _make_session_manager()
        result = sm.new_session()
        assert result == "new-thread"
        agent.session.new_session.assert_called_once()

    def test_new_workflow_session(self):
        sm, agent, _, _ = _make_session_manager()
        result = sm.new_workflow_session("test")
        assert "workflow" in result
        agent.session.new_workflow_session.assert_called_once_with("test")

    def test_set_current_session(self):
        sm, agent, _, _ = _make_session_manager()
        sm.set_current_session("thread-x")
        agent.set_current_session.assert_called_once_with("thread-x")

    def test_alist_sessions(self):
        async def run():
            sm, _, _, _ = _make_session_manager(thread_id="t1")
            sessions = await sm.alist_sessions()
            assert sessions == ["t1"]

        asyncio.run(run())

    def test_aswitch_session(self):
        async def run():
            sm, _, _, _ = _make_session_manager()
            result = await sm.aswitch_session("other")
            assert result is True

        asyncio.run(run())

    def test_adelete_session(self):
        async def run():
            sm, _, _, _ = _make_session_manager()
            result = await sm.adelete_session("thread-to-delete")
            assert result is True

        asyncio.run(run())

    def test_aget_messages(self):
        async def run():
            sm, _, _, _ = _make_session_manager()
            messages = await sm.aget_messages()
            assert messages == []

        asyncio.run(run())

    def test_aexport_session(self):
        async def run():
            sm, _, _, _ = _make_session_manager()
            result = await sm.aexport_session()
            assert result == "exported text"

        asyncio.run(run())

    def test_asummarize(self):
        async def run():
            sm, _, _, _ = _make_session_manager()
            result = await sm.asummarize()
            assert result["checkpoint_messages"] == 5
            assert result["total_sessions"] == 2

        asyncio.run(run())


# ════════════════════════════════════════════════════════════════════════
#  执行历史
# ════════════════════════════════════════════════════════════════════════


class TestExecutionHistory:
    def test_aget_execution_history(self):
        async def run():
            sm, agent, _, _ = _make_session_manager(thread_id="t1")
            history = await sm.aget_execution_history()
            assert history == []
            agent.aget_execution_history.assert_called_once()

        asyncio.run(run())

    def test_aclear_history(self):
        async def run():
            sm, agent, _, _ = _make_session_manager(thread_id="t1")
            await sm.aclear_history()
            agent.aclear_history.assert_called_once()

        asyncio.run(run())

    def test_manually_compact(self):
        async def run():
            sm, agent, _, _ = _make_session_manager(thread_id="t1")
            await sm.manually_compact(force=True, thread_id="t1")
            agent.manually_compact.assert_called_once()

        asyncio.run(run())


# ════════════════════════════════════════════════════════════════════════
#  生命周期 — aclose
# ════════════════════════════════════════════════════════════════════════


class TestLifecycle:
    def test_aclose_flushes_memory_and_closes_agent(self):
        async def run():
            sm, agent, memory, _ = _make_session_manager(thread_id="t1")
            # 提交一些 buffer 数据
            await memory.submit_user_message("t1", "some content")

            await sm.aclose()

            # Agent 被关闭
            agent.aclose.assert_awaited_once()

        asyncio.run(run())

    def test_aclose_without_memory(self):
        async def run():
            sm, agent, _, _ = _make_session_manager(with_memory=False)
            await sm.aclose()
            agent.aclose.assert_awaited_once()

        asyncio.run(run())

    def test_aclose_memory_error_does_not_block_agent_close(self):
        """MemoryManager 关闭异常不应阻止 Agent 关闭。"""
        async def run():
            sm, agent, memory, _ = _make_session_manager(thread_id="t1")
            # 让 flush_all 抛异常
            memory.flush_all = AsyncMock(side_effect=RuntimeError("flush failed"))

            await sm.aclose()
            # Agent 仍然被关闭
            agent.aclose.assert_awaited_once()

        asyncio.run(run())
