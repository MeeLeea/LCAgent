"""forget_memory 工具测试。

覆盖：
- 成功路径（long / agent 作用域）：返回 forgotten_count + preview，且 preview 内容精确
- preview 截断：最多 3 条、每条 ≤80 字符
- 非法 scope / 空 query / 纯空白 query → 干净错误（success=False，不抛异常）
- MemoryManager 抛 ValueError → 映射为 {"success": False, "error": ...}
- 未注入 MemoryManager → 干净错误
"""
from __future__ import annotations

import asyncio
from importlib import import_module
from typing import Any

import pytest

from memory.models import ThreadFactItem
from memory.store import ThreadMemoryStore

# 注意：``tools/__init__.py`` 导出名为 forget_memory 的 **工具对象**，会遮蔽同名子模块，
# 因此这里用 import_module 明确取得模块对象，以便调用 configure()。
forget_tool_module = import_module("tools.forget_memory")


class _FakeMemoryManager:
    """在真实 ThreadMemoryStore 之上伪造 MemoryManager 的 forget 接口。

    保留真实查询/标记语义（store 层），只替换 MemoryManager 的串行化外壳，
    因此测试可精确断言"哪条内容被遗忘"。
    """

    def __init__(self, store: ThreadMemoryStore, *, raise_value_error: bool = False) -> None:
        self.store = store
        self._raise = raise_value_error
        self.calls: list[tuple[str, Any]] = []

    async def forget(
        self,
        thread_id: str,
        keyword: str | None = None,
        topic_key: str | None = None,
        fact_id: str | None = None,
    ) -> int:
        self.calls.append(("forget", (thread_id, keyword, topic_key)))
        if self._raise:
            raise ValueError("遗忘条件不能全为空：需提供 fact_id / topic_key / keyword 之一")
        return await self.store.forget_facts(
            thread_id, keyword=keyword, topic_key=topic_key, fact_id=fact_id
        )

    async def forget_agent(
        self,
        keyword: str | None = None,
        topic_key: str | None = None,
        fact_id: str | None = None,
    ) -> int:
        self.calls.append(("forget_agent", (keyword, topic_key)))
        if self._raise:
            raise ValueError("遗忘条件不能全为空：需提供 fact_id / topic_key / keyword 之一")
        return await self.store.forget_agent_facts(
            keyword=keyword, topic_key=topic_key, fact_id=fact_id
        )


@pytest.fixture
def configured_tool():
    """注入 fake MemoryManager + 固定 thread_id，返回（工具, fake manager）。"""
    store = ThreadMemoryStore()

    async def _setup(facts: list[ThreadFactItem], *, agent_scope: bool = False) -> _FakeMemoryManager:
        if agent_scope:
            await store.save_agent_facts_batch(facts)
        else:
            await store.save_facts_batch("thread-1", facts)
        manager = _FakeMemoryManager(store)
        forget_tool_module.configure(
            memory_provider=lambda: manager,
            thread_id_provider=lambda: "thread-1",
        )
        return manager

    yield store, _setup
    forget_tool_module.configure(memory_provider=None, thread_id_provider=None)


def test_forget_memory_long_scope_reports_preview(configured_tool):
    """long 作用域：返回计数 + 被遗忘内容预览（精确断言内容）。"""
    _unused_store, setup = configured_tool

    async def run() -> dict[str, Any]:
        await setup([
            ThreadFactItem(content="用户偏好深色主题"),
            ThreadFactItem(content="无关条目"),
        ])
        return await forget_tool_module.forget_memory.ainvoke({"query": "深色主题"})

    result = asyncio.run(run())
    assert result["success"] is True
    assert result["scope"] == "long"
    assert result["forgotten_count"] == 1
    assert result["preview"] == ["用户偏好深色主题"]
    assert result["message"]


def test_forget_memory_agent_scope(configured_tool):
    """agent 作用域：调用 forget_agent 并返回 agent 级预览。"""
    _unused_store, setup = configured_tool

    async def run() -> dict[str, Any]:
        manager = await setup(
            [
                ThreadFactItem(content="用户偏好中文回复", category="user_fact", scope="agent"),
                ThreadFactItem(content="用户使用 Vim", category="user_fact", scope="agent"),
            ],
            agent_scope=True,
        )
        result = await forget_tool_module.forget_memory.ainvoke(
            {"query": "中文", "scope": "agent"}
        )
        assert manager.calls == [("forget_agent", ("中文", None))]
        return result

    result = asyncio.run(run())
    assert result["success"] is True
    assert result["scope"] == "agent"
    assert result["forgotten_count"] == 1
    assert result["preview"] == ["用户偏好中文回复"]


def test_forget_memory_with_topic(configured_tool):
    """topic 参数透传为 topic_key（归一化后匹配）。"""
    _unused_store, setup = configured_tool

    async def run() -> dict[str, Any]:
        await setup([ThreadFactItem(content="公式 v1", topic_key="dim_y_formula")])
        return await forget_tool_module.forget_memory.ainvoke(
            {"query": "", "topic": "Dim-Y Formula"}
        )

    result = asyncio.run(run())
    assert result["success"] is True
    assert result["forgotten_count"] == 1
    assert result["preview"] == ["公式 v1"]


def test_forget_memory_preview_truncation(configured_tool):
    """preview 最多 3 条，每条 ≤80 字符。"""
    _unused_store, setup = configured_tool
    long_text = "长" * 200

    async def run() -> dict[str, Any]:
        await setup(
            [ThreadFactItem(content=f"命中-{i}") for i in range(4)]
            + [ThreadFactItem(content=f"命中长文本{long_text}")]
        )
        return await forget_tool_module.forget_memory.ainvoke({"query": "命中"})

    result = asyncio.run(run())
    assert result["success"] is True
    assert result["forgotten_count"] == 5
    assert len(result["preview"]) == 3
    assert all(len(text) <= 80 for text in result["preview"])


@pytest.mark.parametrize("scope", ["short", "all", "session", "longterm"])
def test_forget_memory_invalid_scope(configured_tool, scope):
    """非法 scope → success=False + error，不抛异常。"""
    _unused_store, setup = configured_tool

    async def run() -> dict[str, Any]:
        await setup([ThreadFactItem(content="x")])
        return await forget_tool_module.forget_memory.ainvoke(
            {"query": "x", "scope": scope}
        )

    result = asyncio.run(run())
    assert result["success"] is False
    assert "scope" in result["error"].lower() or "作用域" in result["error"]


@pytest.mark.parametrize("query", ["", "   ", None])
def test_forget_memory_empty_query(configured_tool, query):
    """空/纯空白/缺失 query → success=False，不抛异常。"""
    _unused_store, setup = configured_tool

    async def run() -> dict[str, Any]:
        await setup([ThreadFactItem(content="x")])
        payload: dict[str, Any] = {"query": query}
        return await forget_tool_module.forget_memory.ainvoke(payload)

    result = asyncio.run(run())
    assert result["success"] is False
    assert result["error"]


def test_forget_memory_value_error_mapped(configured_tool):
    """MemoryManager 抛 ValueError → 映射为 success=False，不向上抛。"""
    _unused_store, _unused_setup = configured_tool

    async def run() -> dict[str, Any]:
        store_ref, _ = configured_tool
        manager = _FakeMemoryManager(store_ref, raise_value_error=True)
        forget_tool_module.configure(
            memory_provider=lambda: manager,
            thread_id_provider=lambda: "thread-1",
        )
        return await forget_tool_module.forget_memory.ainvoke({"query": "任意"})

    result = asyncio.run(run())
    assert result["success"] is False
    assert "遗忘条件" in result["error"]


def test_forget_memory_without_configured_manager():
    """未注入 MemoryManager → 干净错误，不抛异常。"""
    forget_tool_module.configure(memory_provider=None, thread_id_provider=None)
    result = asyncio.run(forget_tool_module.forget_memory.ainvoke({"query": "任意"}))
    assert result["success"] is False
    assert "记忆" in result["error"]


def test_forget_memory_registered_in_all_tools():
    """工具已注册到 tools.all_tools。"""
    from tools import all_tools

    assert any(t.name == "forget_memory" for t in all_tools)


# ════════════════════════════════════════════════════════════════════════
#  configure_for_agent — 入口程序接线（main / api / scheduler 共用）
# ════════════════════════════════════════════════════════════════════════


class _StubAgent:
    """最小 Agent 替身：暴露 session_manager.memory 与 session.current_session_id。"""

    def __init__(self, manager: Any, thread_id: str = "thread-1") -> None:
        self._manager = manager
        self.session_manager = type("SM", (), {"memory": manager})()
        self.session = type("S", (), {"current_session_id": thread_id})()


def test_configure_for_agent_wires_tool_dependencies():
    """configure_for_agent 注入后，工具可调用 forget 并返回成功（非"未初始化"）。"""
    store = ThreadMemoryStore()
    manager = _FakeMemoryManager(store)

    async def run() -> dict[str, Any]:
        await store.save_facts_batch("thread-1", [ThreadFactItem(content="用户偏好深色主题")])
        forget_tool_module.configure_for_agent(_StubAgent(manager, "thread-1"))
        return await forget_tool_module.forget_memory.ainvoke({"query": "深色主题"})

    result = asyncio.run(run())
    assert result["success"] is True
    assert result["forgotten_count"] == 1
    assert result["preview"] == ["用户偏好深色主题"]
    forget_tool_module.configure(memory_provider=None, thread_id_provider=None)


def test_configure_for_agent_memory_disabled_is_graceful():
    """记忆系统未启用（memory=None）→ 工具返回干净错误，不抛异常。"""

    async def run() -> dict[str, Any]:
        forget_tool_module.configure_for_agent(_StubAgent(None, "thread-1"))
        return await forget_tool_module.forget_memory.ainvoke({"query": "x"})

    result = asyncio.run(run())
    assert result["success"] is False
    assert "记忆系统未初始化" in result["error"]
    forget_tool_module.configure(memory_provider=None, thread_id_provider=None)


def test_entrypoints_call_configure_for_agent():
    """main.py / api/server.py / scheduler/run.py 均接线 configure_for_agent。"""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    for rel in ("main.py", "api/server.py", "scheduler/run.py"):
        source = (root / rel).read_text(encoding="utf-8")
        assert "configure_for_agent" in source, f"{rel} 未接线 configure_for_agent"