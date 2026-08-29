"""MemoryManager 集成测试

覆盖：
- recall / recall_text 召回与格式化
- submit_user_message 用户消息提交
- consume_event 事件消费与过滤
- compress LLM 压缩摘要
- clear / count_facts 清理与计数
- flush_all / shutdown 生命周期管理
- MemoryManager ↔ ThreadMemoryStore ↔ WriteMiddleware 跨组件集成
"""
from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock

from memory.lock_pool import ThreadMemoryLockPool
from memory.manager import MemoryManager
from memory.models import MemoryCategory, ThreadFactItem
from memory.store import ThreadMemoryStore


def _make_manager(
    llm_getter: Any | None = None,
    buffer_delay_seconds: int = 0,
    max_buffer_messages: int = 30,
    recall_limit: int = 10,
) -> tuple[MemoryManager, ThreadMemoryStore]:
    """创建真实组件的 MemoryManager（InMemoryStore 后端）。"""
    store = ThreadMemoryStore()
    lock_pool = ThreadMemoryLockPool()
    if llm_getter is None:
        llm_getter = lambda: None
    mgr = MemoryManager(
        memory_store=store,
        lock_pool=lock_pool,
        llm_getter=llm_getter,
        recall_limit=recall_limit,
        buffer_delay_seconds=buffer_delay_seconds,
        max_buffer_messages=max_buffer_messages,
    )
    return mgr, store


class _FakeLLM:
    """最小化 LLM mock，支持 chat(messages) → str。"""

    def __init__(self, response: str = "[]"):
        self._response = response
        self.calls: list[list[dict]] = []

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        return self._response


# ════════════════════════════════════════════════════════════════════════
#  召回
# ════════════════════════════════════════════════════════════════════════


class TestRecall:
    def test_recall_returns_facts(self):
        async def run():
            mgr, store = _make_manager()
            await store.save_fact("t1", ThreadFactItem(content="fact-a", category="user_fact"))
            await store.save_fact("t1", ThreadFactItem(content="fact-b", category="lesson"))

            facts = await mgr.recall("t1")
            assert len(facts) == 2
            assert {f.content for f in facts} == {"fact-a", "fact-b"}

        asyncio.run(run())

    def test_recall_empty_thread_returns_empty(self):
        async def run():
            mgr, _ = _make_manager()
            facts = await mgr.recall("empty")
            assert facts == []

        asyncio.run(run())

    def test_recall_empty_thread_id_returns_empty(self):
        async def run():
            mgr, _ = _make_manager()
            facts = await mgr.recall("")
            assert facts == []

        asyncio.run(run())

    def test_recall_respects_limit(self):
        async def run():
            mgr, store = _make_manager(recall_limit=2)
            for i in range(5):
                item = ThreadFactItem(content=f"fact-{i}")
                item.create_time = f"2026-01-0{i+1}T00:00:00"
                await store.save_fact("t1", item)

            facts = await mgr.recall("t1")
            assert len(facts) == 2
            # 返回最近创建的
            assert facts[-1].content == "fact-4"

        asyncio.run(run())

    def test_recall_text_formats_with_header(self):
        async def run():
            mgr, store = _make_manager()
            await store.save_fact("t1", ThreadFactItem(content="喜欢 Python", category="user_fact"))

            text = await mgr.recall_text("t1")
            assert "【长期记忆】" in text
            assert "用户事实" in text
            assert "喜欢 Python" in text

        asyncio.run(run())

    def test_recall_text_empty_returns_empty_string(self):
        async def run():
            mgr, _ = _make_manager()
            text = await mgr.recall_text("empty")
            assert text == ""

        asyncio.run(run())


# ════════════════════════════════════════════════════════════════════════
#  用户消息提交 & 事件消费
# ════════════════════════════════════════════════════════════════════════


class TestSubmitAndConsume:
    def test_submit_user_message_does_not_crash(self):
        async def run():
            mgr, _ = _make_manager()
            await mgr.submit_user_message("t1", "记住我喜欢深色主题")

        asyncio.run(run())

    def test_submit_user_message_empty_content_skipped(self):
        async def run():
            mgr, _ = _make_manager()
            await mgr.submit_user_message("t1", "")
            await mgr.submit_user_message("t1", "   ")

        asyncio.run(run())

    def test_submit_user_message_empty_thread_id_skipped(self):
        async def run():
            mgr, _ = _make_manager()
            await mgr.submit_user_message("", "content")

        asyncio.run(run())

    def test_consume_event_skips_non_memory_worthy(self):
        """TOKEN 事件不提交记忆。"""
        async def run():
            from utils.events import AgentEvent, EventType

            mgr, _ = _make_manager()
            event = AgentEvent.token("hello", thread_id="t1")
            # TOKEN 不 is_memory_worthy，consume_event 应跳过
            await mgr.consume_event(event)

        asyncio.run(run())

    def test_consume_event_done_with_content(self):
        """DONE 事件（有内容）提交记忆。"""
        async def run():
            from utils.events import AgentEvent, EventType

            mgr, _ = _make_manager()
            event = AgentEvent(
                event_type=EventType.DONE,
                content="任务完成",
                thread_id="t1",
            )
            assert event.is_memory_worthy
            await mgr.consume_event(event)

        asyncio.run(run())

    def test_consume_event_skips_empty_thread_id(self):
        async def run():
            from utils.events import AgentEvent, EventType

            mgr, _ = _make_manager()
            event = AgentEvent(
                event_type=EventType.DONE,
                content="content",
                thread_id="",
            )
            await mgr.consume_event(event)

        asyncio.run(run())



# ════════════════════════════════════════════════════════════════════════
#  压缩 & 清理
# ════════════════════════════════════════════════════════════════════════


class TestCompressAndClear:
    def test_compress_with_llm(self):
        async def run():
            llm = _FakeLLM(response="压缩后的摘要内容")
            mgr, store = _make_manager(llm_getter=lambda: llm)
            await store.save_fact("t1", ThreadFactItem(content="fact-1"))
            await store.save_fact("t1", ThreadFactItem(content="fact-2"))

            result = await mgr.compress("t1")
            assert result["success"] is True
            assert result["original_count"] == 2
            assert result["summary"] == "压缩后的摘要内容"
            assert result["compressed_chars"] > 0

            # 压缩后只剩 1 条摘要
            facts = await store.query_facts("t1")
            assert len(facts) == 1

        asyncio.run(run())

    def test_compress_empty_thread_returns_error(self):
        async def run():
            mgr, _ = _make_manager()
            result = await mgr.compress("empty")
            assert result["success"] is False
            assert "没有长期记忆" in result["error"]

        asyncio.run(run())

    def test_compress_llm_failure_returns_error(self):
        async def run():
            llm = _FakeLLM(response="")
            mgr, store = _make_manager(llm_getter=lambda: llm)
            await store.save_fact("t1", ThreadFactItem(content="fact"))

            result = await mgr.compress("t1")
            assert result["success"] is False
            assert "LLM" in result["error"] or "空摘要" in result["error"]

        asyncio.run(run())

    def test_clear_removes_all_facts(self):
        async def run():
            mgr, store = _make_manager()
            await store.save_fact("t1", ThreadFactItem(content="a"))
            await store.save_fact("t1", ThreadFactItem(content="b"))

            cleared = await mgr.clear("t1")
            assert cleared == 2
            assert await mgr.count_facts("t1") == 0

        asyncio.run(run())

    def test_clear_does_not_remove_agent_facts(self):
        """clear 仅清 thread 级记忆，agent 级跨会话共享记忆不受影响。"""
        async def run():
            mgr, store = _make_manager()
            await store.save_fact("t1", ThreadFactItem(content="thread-a"))
            await store.save_agent_fact(
                ThreadFactItem(content="global-a", category="user_fact")
            )

            cleared = await mgr.clear("t1")
            assert cleared == 1
            assert await mgr.count_facts("t1") == 0
            assert await mgr.count_agent_facts() == 1

        asyncio.run(run())

    def test_clear_drops_unflushed_buffer(self):
        """clear 应丢弃未 flush 的缓冲事件，防止清完被回写“复活”。"""
        async def run():
            # 大延迟确保防抖定时器在测试期间不会触发 flush
            mgr, store = _make_manager(buffer_delay_seconds=1000)
            await mgr.write_middleware.submit_event("t1", "user", "待沉淀事件", False)
            assert "t1" in mgr.write_middleware._buffer

            cleared = await mgr.clear("t1")
            assert cleared == 0  # buffer 被丢弃，未落库
            assert "t1" not in mgr.write_middleware._buffer
            assert await mgr.count_facts("t1") == 0

        asyncio.run(run())

    def test_clear_release_lock_frees_pool(self):
        """release_lock=True 时删除完成后释放锁池缓存。"""
        async def run():
            mgr, store = _make_manager()
            await store.save_fact("t1", ThreadFactItem(content="a"))

            await mgr.clear("t1", release_lock=True)
            assert mgr._lock_pool._locks.get("t1") is None

        asyncio.run(run())

    def test_clear_keeps_lock_without_release(self):
        """release_lock 默认 False（显式清记忆命令路径）时保留锁池缓存。"""
        async def run():
            mgr, store = _make_manager()
            await store.save_fact("t1", ThreadFactItem(content="a"))

            await mgr.clear("t1")
            assert mgr._lock_pool._locks.get("t1") is not None

        asyncio.run(run())

    def test_count_facts(self):
        async def run():
            mgr, store = _make_manager()
            await store.save_fact("t1", ThreadFactItem(content="x"))
            await store.save_fact("t1", ThreadFactItem(content="y"))

            assert await mgr.count_facts("t1") == 2
            assert await mgr.count_facts("t2") == 0

        asyncio.run(run())


# ════════════════════════════════════════════════════════════════════════
#  生命周期
# ════════════════════════════════════════════════════════════════════════


class TestLifecycle:
    def test_flush_all_processes_buffer(self):
        """flush_all 应处理 buffer 中待处理事件（需要 LLM 抽取）。"""
        async def run():
            llm = _FakeLLM(response=json.dumps([
                {"content": "用户喜欢 Rust", "category": "user_fact", "confidence": 0.9}
            ]))
            mgr, store = _make_manager(llm_getter=lambda: llm, buffer_delay_seconds=999)
            # 提交用户消息到 buffer（防抖窗口 999s，不会自动触发）
            await mgr.submit_user_message("t1", "我喜欢 Rust 语言")
            # flush_all 立即处理
            await mgr.flush_all()
            # user_fact 走 agent 级（跨会话共享），thread 级应为空
            thread_facts = await store.query_facts("t1")
            assert thread_facts == []
            agent_facts = await store.query_agent_facts()
            assert len(agent_facts) == 1
            assert agent_facts[0].content == "用户喜欢 Rust"

        asyncio.run(run())

    def test_shutdown_cleans_up(self):
        async def run():
            mgr, _ = _make_manager(buffer_delay_seconds=999)
            await mgr.submit_user_message("t1", "content")
            await mgr.shutdown()

        asyncio.run(run())


# ════════════════════════════════════════════════════════════════════════
#  属性暴露
# ════════════════════════════════════════════════════════════════════════


class TestProperties:
    def test_store_property(self):
        mgr, store = _make_manager()
        assert mgr.store is store

    def test_read_middleware_property(self):
        from memory.middleware import ThreadMemoryReadMiddleware

        mgr, _ = _make_manager()
        assert isinstance(mgr.read_middleware, ThreadMemoryReadMiddleware)

    def test_write_middleware_property(self):
        from memory.middleware import ThreadMemoryWriteMiddleware

        mgr, _ = _make_manager()
        assert isinstance(mgr.write_middleware, ThreadMemoryWriteMiddleware)


# ════════════════════════════════════════════════════════════════════════
#  LLM 热切换绑定（bind_llm）
# ════════════════════════════════════════════════════════════════════════


class TestBindLlm:
    """绑定/替换记忆 LLM 获取器：provider 热切换后记忆抽取应跟随新 LLM。"""

    def test_bind_llm_replaces_manager_and_write_middleware_getter(self):
        """bind_llm 后 MemoryManager 与写中间件的 llm_getter 应返回新 LLM。"""
        old_llm = _FakeLLM()
        new_llm = _FakeLLM()
        mgr, _ = _make_manager(llm_getter=lambda: old_llm)

        assert mgr._llm_getter() is old_llm
        assert mgr.write_middleware._llm_getter() is old_llm

        mgr.bind_llm(lambda: new_llm)

        assert mgr._llm_getter() is new_llm
        assert mgr.write_middleware._llm_getter() is new_llm

    def test_fact_extraction_uses_bound_llm_after_provider_switch(self):
        """模拟切换 provider：bind 新 LLM 后 flush 抽取应使用新 LLM 请求。"""
        old_llm = _FakeLLM(response="[]")  # 旧 LLM 返回空（不抽取）
        new_llm = _FakeLLM(response=json.dumps([
            {"content": "切换后抽取的事实", "category": "user_fact", "confidence": 0.9}
        ]))
        mgr, store = _make_manager(llm_getter=lambda: old_llm, buffer_delay_seconds=999)

        async def run():
            # 切换 provider：替换为读取 agent.llm 的 getter（模拟入口 bind 调用）
            mgr.bind_llm(lambda: new_llm)
            await mgr.submit_user_message("t1", "用户偏好信息")
            await mgr.flush_all()

        asyncio.run(run())

        # 旧 LLM 未被调用，新 LLM 完成抽取并落库
        assert old_llm.calls == []
        assert new_llm.calls != []
        # user_fact 走 agent 级（跨会话共享），thread 级应为空
        thread_facts = asyncio.run(store.query_facts("t1"))
        assert thread_facts == []
        agent_facts = asyncio.run(store.query_agent_facts())
        assert len(agent_facts) == 1
        assert agent_facts[0].content == "切换后抽取的事实"

    def test_memory_context_bind_llm_forwards_to_manager(self):
        """MemoryContext.bind_llm 应透传到内部 MemoryManager。"""
        from memory.context import MemoryContext

        old_llm = _FakeLLM()
        new_llm = _FakeLLM()
        mgr, _ = _make_manager(llm_getter=lambda: old_llm)
        ctx = MemoryContext(
            agent_memory=MagicMock(),
            read_middleware=MagicMock(),
            memory_manager=mgr,
        )

        assert ctx.memory_manager._llm_getter() is old_llm

        ctx.bind_llm(lambda: new_llm)

        assert ctx.memory_manager._llm_getter() is new_llm
        assert ctx.memory_manager.write_middleware._llm_getter() is new_llm


# ════════════════════════════════════════════════════════════════════════
#  Agent 级长期记忆接口（recall_agent / recall_agent_text / count / clear）
# ════════════════════════════════════════════════════════════════════════


class TestAgentInterfaces:
    """MemoryManager 暴露的 agent 级长期记忆接口测试。"""

    def test_recall_agent_returns_facts(self):
        """预置 agent facts → recall_agent 返回。"""
        async def run():
            mgr, store = _make_manager()
            await store.save_agent_fact(
                ThreadFactItem(content="agent-fact-1", category="user_fact")
            )
            await store.save_agent_fact(
                ThreadFactItem(content="agent-fact-2", category="lesson")
            )

            facts = await mgr.recall_agent()
            assert len(facts) == 2
            assert {f.content for f in facts} == {"agent-fact-1", "agent-fact-2"}

        asyncio.run(run())

    def test_recall_agent_empty(self):
        """无 agent facts → 返回 []。"""
        async def run():
            mgr, _ = _make_manager()
            facts = await mgr.recall_agent()
            assert facts == []

        asyncio.run(run())

    def test_recall_agent_text_format(self):
        """非空 → 含"【长期记忆】"前缀和类别标签；空 → ""。"""
        async def run():
            mgr, store = _make_manager()
            await store.save_agent_fact(
                ThreadFactItem(content="跨会话偏好", category="user_fact")
            )
            await store.save_agent_fact(
                ThreadFactItem(content="踩坑教训", category="lesson")
            )

            text = await mgr.recall_agent_text()
            assert "【长期记忆】" in text
            assert "用户事实" in text
            assert "跨会话偏好" in text
            assert "经验教训" in text
            assert "踩坑教训" in text

            # 空场景
            mgr_empty, _ = _make_manager()
            assert await mgr_empty.recall_agent_text() == ""

        asyncio.run(run())

    def test_recall_agent_respects_limit(self):
        """recall_agent 受 limit 约束（取最近 N 条）。"""
        async def run():
            mgr, store = _make_manager()
            for i in range(5):
                item = ThreadFactItem(content=f"agent-{i}", category="user_fact")
                item.create_time = f"2026-01-0{i+1}T00:00:00"
                await store.save_agent_fact(item)

            facts = await mgr.recall_agent(limit=2)
            assert len(facts) == 2
            # 返回最近创建的
            assert facts[-1].content == "agent-4"

        asyncio.run(run())

    def test_count_and_clear_agent_facts(self):
        """count_agent_facts 统计，clear_agent_facts 清空返回计数。"""
        async def run():
            mgr, store = _make_manager()
            await store.save_agent_fact(
                ThreadFactItem(content="a", category="user_fact")
            )
            await store.save_agent_fact(
                ThreadFactItem(content="b", category="user_fact")
            )

            assert await mgr.count_agent_facts() == 2

            cleared = await mgr.clear_agent_facts()
            assert cleared == 2
            assert await mgr.count_agent_facts() == 0

        asyncio.run(run())
