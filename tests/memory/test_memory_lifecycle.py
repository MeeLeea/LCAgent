"""记忆生命周期机制测试 — 主题合并取代 / persistence 路由 / 状态过滤 / 证据标注。

覆盖：
- normalize_topic_key: 主题键归一化规则
- ThreadFactItem.from_dict: 未知字段过滤 + 旧数据缺字段兜底
- 写入合并：同 topic_key 新条目取代旧条目（superseded），非并存
- 批内同 topic_key 合并（保留置信度最高）
- persistence 路由：session→thread / global→agent / 缺失回退 category /
  低置信 global 降级 thread
- 读取：superseded 过滤、同 topic 收敛最新、推断/低置信标注
- prune 优先级：superseded 先于 active LRU 淘汰
- MemoryManager.recall 过滤非 active
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from memory.lock_pool import ThreadMemoryLockPool
from memory.manager import MemoryManager
from memory.middleware import (
    ThreadMemoryReadMiddleware,
    ThreadMemoryWriteMiddleware,
    default_thread_llm_resolver,
)
from memory.models import (
    FACT_STATUS_SUPERSEDED,
    ThreadFactItem,
    normalize_topic_key,
)
from memory.store import ThreadMemoryStore


def _make_write_middleware(
    llm_response: str = "[]",
    confidence_floor: float | None = None,
) -> tuple[ThreadMemoryWriteMiddleware, ThreadMemoryStore]:
    store = ThreadMemoryStore()
    llm = _FakeLLM(response=llm_response)
    mw = ThreadMemoryWriteMiddleware(
        memory_store=store,
        lock_pool=ThreadMemoryLockPool(),
        llm_getter=default_thread_llm_resolver(lambda: llm),
        buffer_delay_seconds=999,
        confidence_floor=confidence_floor,
    )
    return mw, store


class _FakeLLM:
    def __init__(self, response: str = "[]") -> None:
        self._response = response

    def chat(self, messages: list[dict[str, str]]) -> str:
        return self._response


# ════════════════════════════════════════════════════════════════════════
#  normalize_topic_key — 归一化
# ════════════════════════════════════════════════════════════════════════


class TestNormalizeTopicKey:
    def test_basic_normalization(self):
        assert normalize_topic_key("Dim-Y Formula") == "dim_y_formula"
        assert normalize_topic_key("  cli.usage  ") == "cli_usage"

    def test_chinese_and_symbols_fall_to_empty(self):
        # 纯中文/符号归一化后为空 → 视为无主题（退化 append）
        assert normalize_topic_key("维度公式") == ""
        assert normalize_topic_key("---") == ""

    def test_consecutive_underscores_collapsed(self):
        assert normalize_topic_key("a__b---c") == "a_b_c"

    def test_truncated_to_max_chars(self):
        long_key = "a" * 80
        assert len(normalize_topic_key(long_key)) == 50

    def test_empty_input(self):
        assert normalize_topic_key("") == ""


# ════════════════════════════════════════════════════════════════════════
#  ThreadFactItem.from_dict — schema 兼容
# ════════════════════════════════════════════════════════════════════════


class TestFromDictCompat:
    def test_unknown_fields_filtered(self):
        # 未来字段（或回滚场景多余字段）不导致 TypeError
        item = ThreadFactItem.from_dict({
            "content": "x",
            "future_field": "y",
        })
        assert item.content == "x"
        assert item.status == "active"
        assert item.topic_key == ""
        assert item.evidence == "stated"

    def test_legacy_row_defaults(self):
        # 旧数据缺新字段 → dataclass 默认值兜底
        item = ThreadFactItem.from_dict({"content": "old", "category": "lesson"})
        assert item.status == "active"
        assert item.superseded_at == ""


# ════════════════════════════════════════════════════════════════════════
#  写入合并 — 同 topic_key 取代而非并存
# ════════════════════════════════════════════════════════════════════════


class TestTopicMergeOnWrite:
    def test_agent_ns_same_topic_supersedes_old(self):
        """agent 级同 topic_key：新条目写入，旧条目标 superseded（非并存）。"""
        async def run():
            facts = json.dumps([
                {
                    "content": "dim_y = 2*outputs + 2*inputs",
                    "category": "lesson",
                    "confidence": 0.9,
                    "persistence": "global",
                    "topic_key": "dim_y_formula",
                    "evidence": "tested",
                }
            ])
            mw, store = _make_write_middleware(llm_response=facts)
            # 预置旧版（错误公式），同 topic_key，active
            await store.save_agent_fact(
                ThreadFactItem(
                    content="dim_y = 2*outputs + 2.5*inputs",
                    category="lesson",
                    scope="agent",
                    topic_key="dim_y_formula",
                    create_time="2026-01-01T00:00:00",
                )
            )
            await mw.submit_event("t1", "user", "修正 dim_y 公式")
            await mw._aflush_thread("t1")

            # 全量 2 条（审计保留），active 仅 1 条（新版）
            all_facts = await store.query_agent_facts()
            active = await store.query_agent_facts(active_only=True)
            assert len(all_facts) == 2
            assert len(active) == 1
            assert active[0].content == "dim_y = 2*outputs + 2*inputs"
            # 旧条目被标记 superseded
            old = [f for f in all_facts if f.status == FACT_STATUS_SUPERSEDED]
            assert len(old) == 1
            assert old[0].superseded_at != ""

        asyncio.run(run())

    def test_thread_ns_same_topic_supersedes_old(self):
        """thread 级同 topic_key：新条目取代旧条目。"""
        async def run():
            facts = json.dumps([
                {
                    "content": "CLI 用法已改为 --src 参数",
                    "category": "business",
                    "confidence": 0.85,
                    "persistence": "session",
                    "topic_key": "cli_usage",
                }
            ])
            mw, store = _make_write_middleware(llm_response=facts)
            await store.save_fact(
                "t1",
                ThreadFactItem(
                    content="CLI 用法为 --src-root . --out result",
                    category="business",
                    topic_key="cli_usage",
                    create_time="2026-01-01T00:00:00",
                ),
            )
            await mw.submit_event("t1", "user", "CLI 参数改了")
            await mw._aflush_thread("t1")

            active = await store.query_facts("t1", active_only=True)
            assert len(active) == 1
            assert active[0].content == "CLI 用法已改为 --src 参数"

        asyncio.run(run())

    def test_no_topic_key_still_appends(self):
        """无 topic_key（LLM 未输出/归一化为空）→ 保持 append 行为（兼容）。"""
        async def run():
            facts = json.dumps([
                {"content": "fact-a", "category": "conv", "topic_key": ""},
                {"content": "fact-b", "category": "conv"},  # 缺 topic_key
            ])
            mw, store = _make_write_middleware(llm_response=facts)
            await mw.submit_event("t1", "user", "chat")
            await mw._aflush_thread("t1")

            active = await store.query_facts("t1", active_only=True)
            assert len(active) == 2

        asyncio.run(run())

    def test_batch_internal_same_topic_keeps_highest_confidence(self):
        """批内同 topic_key 多条：保留置信度最高者，仅写一条。"""
        async def run():
            facts = json.dumps([
                {
                    "content": "v1 说法",
                    "category": "lesson",
                    "confidence": 0.6,
                    "topic_key": "same_topic",
                },
                {
                    "content": "v2 说法",
                    "category": "lesson",
                    "confidence": 0.95,
                    "topic_key": "same_topic",
                },
            ])
            mw, store = _make_write_middleware(llm_response=facts)
            await mw.submit_event("t1", "user", "chat")
            await mw._aflush_thread("t1")

            # 0.6 < 默认 floor 0.7 也会降级 thread；0.95 global → agent
            active = await store.query_facts("t1", active_only=True)
            agent_active = await store.query_agent_facts(active_only=True)
            assert len(agent_active) == 1
            assert agent_active[0].content == "v2 说法"
            # v1（低置信降级 thread）也写入 thread 级（同 topic 但跨 namespace 不合并）
            assert len(active) == 1
            assert active[0].content == "v1 说法"

        asyncio.run(run())


# ════════════════════════════════════════════════════════════════════════
#  persistence 路由
# ════════════════════════════════════════════════════════════════════════


class TestPersistenceRouting:
    def test_session_persistence_routes_to_thread(self):
        """persistence=session 的 lesson → thread 级（一次性事件不再污染全局）。"""
        async def run():
            facts = json.dumps([
                {
                    "content": "调用内存统计工具前需先初始化内存管理器",
                    "category": "lesson",
                    "confidence": 0.85,
                    "persistence": "session",
                    "evidence": "stated",
                }
            ])
            mw, store = _make_write_middleware(llm_response=facts)
            await mw.submit_event("t1", "user", "工具失败了")
            await mw._aflush_thread("t1")

            assert await store.query_agent_facts(active_only=True) == []
            thread_active = await store.query_facts("t1", active_only=True)
            assert len(thread_active) == 1
            assert thread_active[0].scope == "thread"

        asyncio.run(run())

    def test_global_persistence_routes_to_agent(self):
        async def run():
            facts = json.dumps([
                {
                    "content": "用户偏好中文回复",
                    "category": "user_fact",
                    "confidence": 0.9,
                    "persistence": "global",
                }
            ])
            mw, store = _make_write_middleware(llm_response=facts)
            await mw.submit_event("t1", "user", "记住我喜欢中文")
            await mw._aflush_thread("t1")

            agent_active = await store.query_agent_facts(active_only=True)
            assert len(agent_active) == 1
            assert agent_active[0].scope == "agent"

        asyncio.run(run())

    def test_missing_persistence_falls_back_to_category(self):
        """persistence 缺失 → 回退旧 category 路由（兼容）。"""
        async def run():
            facts = json.dumps([
                {"content": "偏好 Python", "category": "user_fact", "confidence": 0.9},
                {"content": "对话要点", "category": "conv", "confidence": 0.8},
            ])
            mw, store = _make_write_middleware(llm_response=facts)
            await mw.submit_event("t1", "user", "chat")
            await mw._aflush_thread("t1")

            assert len(await store.query_agent_facts(active_only=True)) == 1
            assert len(await store.query_facts("t1", active_only=True)) == 1

        asyncio.run(run())

    def test_low_confidence_global_downgraded_to_thread(self):
        """persistence=global 但 confidence 低于门槛 → 降级 thread 级。"""
        async def run():
            facts = json.dumps([
                {
                    "content": "低置信全局断言",
                    "category": "lesson",
                    "confidence": 0.5,
                    "persistence": "global",
                }
            ])
            mw, store = _make_write_middleware(llm_response=facts)
            await mw.submit_event("t1", "user", "chat")
            await mw._aflush_thread("t1")

            assert await store.query_agent_facts(active_only=True) == []
            thread_active = await store.query_facts("t1", active_only=True)
            assert len(thread_active) == 1
            assert thread_active[0].scope == "thread"

        asyncio.run(run())

    def test_custom_confidence_floor(self):
        """显式传 confidence_floor 覆盖默认门槛。"""
        async def run():
            facts = json.dumps([
                {
                    "content": "0.75 置信断言",
                    "category": "lesson",
                    "confidence": 0.75,
                    "persistence": "global",
                }
            ])
            mw, store = _make_write_middleware(llm_response=facts, confidence_floor=0.9)
            await mw.submit_event("t1", "user", "chat")
            await mw._aflush_thread("t1")

            # 0.75 < 0.9 → 降级 thread
            assert await store.query_agent_facts(active_only=True) == []
            assert len(await store.query_facts("t1", active_only=True)) == 1

        asyncio.run(run())


# ════════════════════════════════════════════════════════════════════════
#  读取 — 状态过滤 / 同主题收敛 / 证据标注
# ════════════════════════════════════════════════════════════════════════


class _FakeRuntime:
    def __init__(self, context):
        self.context = context


class _FakeModelRequest:
    def __init__(self, context, system_message=None, messages=None):
        self.runtime = _FakeRuntime(context)
        self.system_message = system_message
        self.messages = list(messages) if messages is not None else []

    def override(self, system_message=None, messages=None):
        return _FakeModelRequest(
            self.runtime.context, system_message=system_message, messages=messages
        )


def _last_message_text(request) -> str:
    assert request.messages, "记忆块未注入"
    content = request.messages[-1].content
    if isinstance(content, list):
        return "".join(str(b.get("text", "")) if isinstance(b, dict) else str(b) for b in content)
    return str(content)


class TestReadFilteringAndAnnotation:
    def test_superseded_not_injected(self):
        """superseded 条目不注入 prompt。"""
        async def run():
            store = ThreadMemoryStore()
            await store.save_fact(
                "t1", ThreadFactItem(content="旧版说法", category="conv", topic_key="tk")
            )
            # 手动 supersede 它
            facts = await store.query_facts("t1")
            await store.supersede_facts("t1", [facts[0].fact_id])
            await store.save_fact(
                "t1", ThreadFactItem(content="新版说法", category="conv", topic_key="tk")
            )
            mw = ThreadMemoryReadMiddleware(store)

            request = _FakeModelRequest(context={"configurable": {"thread_id": "t1"}})
            captured: list[Any] = []

            async def handler(req):
                captured.append(req)
                return "ok"

            await mw.awrap_model_call(request, handler)
            text = _last_message_text(captured[0])
            assert "新版说法" in text
            assert "旧版说法" not in text

        asyncio.run(run())

    def test_same_topic_converges_to_newest(self):
        """历史存量同 topic_key 双 active → 注入收敛为最新一条。"""
        async def run():
            store = ThreadMemoryStore()
            await store.save_fact(
                "t1",
                ThreadFactItem(
                    content="旧并存版", category="conv", topic_key="tk",
                    create_time="2026-01-01T00:00:00",
                ),
            )
            await store.save_fact(
                "t1",
                ThreadFactItem(
                    content="新并存版", category="conv", topic_key="tk",
                    create_time="2026-02-01T00:00:00",
                ),
            )
            mw = ThreadMemoryReadMiddleware(store)

            request = _FakeModelRequest(context={"configurable": {"thread_id": "t1"}})
            captured: list[Any] = []

            async def handler(req):
                captured.append(req)
                return "ok"

            await mw.awrap_model_call(request, handler)
            text = _last_message_text(captured[0])
            assert "新并存版" in text
            assert "旧并存版" not in text

        asyncio.run(run())

    def test_inferred_evidence_annotated(self):
        """evidence=inferred → 注入时附"（未实测）"标注。"""
        async def run():
            store = ThreadMemoryStore()
            await store.save_fact(
                "t1",
                ThreadFactItem(
                    content="SC_OUT 分支已实现",
                    category="business",
                    evidence="inferred",
                ),
            )
            mw = ThreadMemoryReadMiddleware(store)

            request = _FakeModelRequest(context={"configurable": {"thread_id": "t1"}})
            captured: list[Any] = []

            async def handler(req):
                captured.append(req)
                return "ok"

            await mw.awrap_model_call(request, handler)
            text = _last_message_text(captured[0])
            assert "SC_OUT 分支已实现（未实测）" in text

        asyncio.run(run())

    def test_low_confidence_annotated(self):
        """confidence 低于阈值 → 附"（低置信）"标注；正常置信无标注。"""
        async def run():
            store = ThreadMemoryStore()
            await store.save_fact(
                "t1", ThreadFactItem(content="低置信条目", category="conv", confidence=0.5)
            )
            await store.save_fact(
                "t1", ThreadFactItem(content="高置信条目", category="conv", confidence=0.95)
            )
            mw = ThreadMemoryReadMiddleware(store)

            request = _FakeModelRequest(context={"configurable": {"thread_id": "t1"}})
            captured: list[Any] = []

            async def handler(req):
                captured.append(req)
                return "ok"

            await mw.awrap_model_call(request, handler)
            text = _last_message_text(captured[0])
            assert "低置信条目（低置信）" in text
            assert "高置信条目（低置信）" not in text

        asyncio.run(run())

    def test_recall_limit_applies_after_topic_convergence(self):
        """recall_limit 截取发生在主题收敛之后（同主题只占一个名额）。"""
        async def run():
            store = ThreadMemoryStore()
            # 同 topic 两条 + 两条无主题，recall_limit=3
            await store.save_fact(
                "t1",
                ThreadFactItem(
                    content="topic-old", category="conv", topic_key="tk",
                    create_time="2026-01-01T00:00:00",
                ),
            )
            await store.save_fact(
                "t1",
                ThreadFactItem(
                    content="topic-new", category="conv", topic_key="tk",
                    create_time="2026-02-01T00:00:00",
                ),
            )
            for i in range(2):
                await store.save_fact(
                    "t1",
                    ThreadFactItem(
                        content=f"plain-{i}", category="conv",
                        create_time=f"2026-03-01T00:00:0{i}",
                    ),
                )
            mw = ThreadMemoryReadMiddleware(store, recall_limit=3)

            request = _FakeModelRequest(context={"configurable": {"thread_id": "t1"}})
            captured: list[Any] = []

            async def handler(req):
                captured.append(req)
                return "ok"

            await mw.awrap_model_call(request, handler)
            text = _last_message_text(captured[0])
            # 4 条收敛为 3 条（topic-old 被 topic-new 收敛），limit=3 全注入
            assert "topic-new" in text
            assert "topic-old" not in text
            assert "plain-0" in text and "plain-1" in text

        asyncio.run(run())


# ════════════════════════════════════════════════════════════════════════
#  prune 优先级 — superseded 先于 active LRU
# ════════════════════════════════════════════════════════════════════════


class TestPrunePriority:
    def test_superseded_evicted_before_active(self):
        """超限时 superseded 优先淘汰，active LRU 条目保留。"""
        async def run():
            store = ThreadMemoryStore(max_facts=3)
            # 3 条 active（last_used_at 最旧的排最前）+ 2 条 superseded
            for i in range(3):
                await store.save_fact(
                    "t1",
                    ThreadFactItem(
                        content=f"active-{i}",
                        last_used_at=f"2026-01-0{i+1}T00:00:00",
                    ),
                )
            for i in range(2):
                item = ThreadFactItem(content=f"superseded-{i}")
                await store.save_fact("t1", item)
                await store.supersede_facts("t1", [item.fact_id])
            # 现 5 条 > 3：prune 应淘汰 2 条 superseded，active 全保留
            removed = await store.prune_facts("t1")

            assert removed == 2
            remaining = await store.query_facts("t1")
            assert len(remaining) == 3
            assert all(f.status == "active" for f in remaining)

        asyncio.run(run())

    def test_supersede_facts_skips_non_active(self):
        """supersede 只作用于 active 条目，已 superseded 的保留原时间戳。"""
        async def run():
            store = ThreadMemoryStore()
            item = ThreadFactItem(content="x")
            await store.save_fact("t1", item)
            await store.supersede_facts("t1", [item.fact_id])
            first_ts = (await store.query_facts("t1"))[0].superseded_at
            # 重复 supersede 不覆盖时间戳
            await store.supersede_facts("t1", [item.fact_id])
            again = (await store.query_facts("t1"))[0]
            assert again.superseded_at == first_ts

        asyncio.run(run())


# ════════════════════════════════════════════════════════════════════════
#  MemoryManager.recall — 过滤非 active
# ════════════════════════════════════════════════════════════════════════


class TestManagerRecallFiltering:
    def test_recall_excludes_superseded(self):
        async def run():
            store = ThreadMemoryStore()
            old = ThreadFactItem(content="旧版", category="conv")
            await store.save_fact("t1", old)
            await store.save_fact("t1", ThreadFactItem(content="新版", category="conv"))
            await store.supersede_facts("t1", [old.fact_id])

            mgr = MemoryManager(
                memory_store=store,
                lock_pool=ThreadMemoryLockPool(),
                llm_getter=default_thread_llm_resolver(lambda: None),
            )
            facts = await mgr.recall("t1")
            assert [f.content for f in facts] == ["新版"]

        asyncio.run(run())

    def test_recall_agent_excludes_superseded(self):
        async def run():
            store = ThreadMemoryStore()
            old = ThreadFactItem(content="旧版", category="lesson", scope="agent")
            await store.save_agent_fact(old)
            await store.save_agent_fact(
                ThreadFactItem(content="新版", category="lesson", scope="agent")
            )
            await store.supersede_agent_facts([old.fact_id])

            mgr = MemoryManager(
                memory_store=store,
                lock_pool=ThreadMemoryLockPool(),
                llm_getter=default_thread_llm_resolver(lambda: None),
            )
            facts = await mgr.recall_agent()
            assert [f.content for f in facts] == ["新版"]

        asyncio.run(run())


# ════════════════════════════════════════════════════════════════════════
#  蒸馏 JSON 解析
# ════════════════════════════════════════════════════════════════════════


class TestDistillResponseParsing:
    def test_valid_json(self):
        lesson, persistence, topic = ThreadMemoryWriteMiddleware._parse_distill_response(
            json.dumps({"lesson": "教训", "persistence": "global", "topic_key": "tk_1"})
        )
        assert lesson == "教训"
        assert persistence == "global"
        assert topic == "tk_1"

    def test_invalid_persistence_falls_back_to_session(self):
        _, persistence, _ = ThreadMemoryWriteMiddleware._parse_distill_response(
            json.dumps({"lesson": "教训", "persistence": "weird"})
        )
        assert persistence == "session"

    def test_bare_text_fallback(self):
        lesson, persistence, topic = ThreadMemoryWriteMiddleware._parse_distill_response(
            "纯文本教训，没有 JSON"
        )
        assert lesson == "纯文本教训，没有 JSON"
        assert persistence == "session"
        assert topic == ""

    def test_empty_lesson_dropped(self):
        result = ThreadMemoryWriteMiddleware._parse_distill_response(
            json.dumps({"lesson": "", "persistence": "session"})
        )
        assert result == ("", "", "")

    def test_empty_sentinels(self):
        for sentinel in ("", "[]", "无", "{}"):
            assert ThreadMemoryWriteMiddleware._parse_distill_response(sentinel) == (
                "", "", "",
            )

    def test_json_in_text_extracted(self):
        lesson, persistence, _ = ThreadMemoryWriteMiddleware._parse_distill_response(
            '前置说明 {"lesson": "真教训", "persistence": "global"} 后缀'
        )
        assert lesson == "真教训"
        assert persistence == "global"
