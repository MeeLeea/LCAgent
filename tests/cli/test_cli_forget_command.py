"""CLI forget 命令测试（沿用 tests/cli/test_cli_commands.py 的 Harness 构造约定）。

覆盖：
- ``forget <关键词>`` → thread 级（long）遗忘，输出含计数与内容预览
- ``forget agent <关键词>`` → agent 级遗忘
- 缺少关键词 → 打印用法，不调用 memory
- 断言"哪条内容被遗忘"（不只断言计数）
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pytest

from cli.commands.dispatcher import dispatch_command
from cli.commands.types import HANDLED, CommandContext
from memory.models import ThreadFactItem
from memory.store import ThreadMemoryStore


class _FakeMemoryManager:
    """在真实 ThreadMemoryStore 上伪造 MemoryManager 的 forget 接口。"""

    def __init__(self) -> None:
        self.store = ThreadMemoryStore()
        self.calls: list[tuple[str, Any]] = []

    async def forget(
        self,
        thread_id: str,
        keyword: str | None = None,
        topic_key: str | None = None,
        fact_id: str | None = None,
    ) -> int:
        self.calls.append(("forget", (thread_id, keyword, topic_key)))
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
        return await self.store.forget_agent_facts(
            keyword=keyword, topic_key=topic_key, fact_id=fact_id
        )


@dataclass
class _FakeSessionManager:
    memory: _FakeMemoryManager = field(default_factory=_FakeMemoryManager)
    calls: list[tuple[str, Any]] = field(default_factory=list)


@dataclass
class _FakeSession:
    current_session_id: str = "thread-1"


@dataclass
class _FakeAgent:
    session_manager: _FakeSessionManager = field(default_factory=_FakeSessionManager)
    session: _FakeSession = field(default_factory=_FakeSession)

    def set_current_session(self, session_id: str) -> None:
        self.session.current_session_id = session_id


@dataclass
class _Harness:
    agent: _FakeAgent
    printed: list[str]
    dispatch: Callable[[str], Any]

    @property
    def output(self) -> str:
        return "\n".join(self.printed)


@pytest.fixture
def harness(tmp_path) -> _Harness:
    agent = _FakeAgent()
    printed: list[str] = []

    def _dispatch(command: str) -> Any:
        """构造 CommandContext 并返回待 await 的 dispatch 协程。

        调用方在事件循环内 ``await``（同一 loop 内既有 store 的异步访问，
        也需要命令分发，故不能再嵌套 ``asyncio.run``）。
        """
        context = CommandContext(
            agent=agent,
            base_dir=str(tmp_path),
            config_file=str(tmp_path / "llm.json"),
            mcp_config_file=str(tmp_path / "mcp.json"),
            print_fn=printed.append,
            input_fn=lambda prompt="": "y",
            select_menu=lambda *args, **kwargs: "x",
            create_llm=lambda provider: None,
            list_providers=dict,
            run_structured_until_completion=lambda a, t: "",
            chat_until_completion=lambda a, t: "",
            safety_backend=object(),
        )
        return dispatch_command(context, command)

    return _Harness(agent=agent, printed=printed, dispatch=_dispatch)


def test_forget_command_thread_scope(harness: _Harness):
    """``forget <关键词>`` → thread 级命中项被遗忘，输出计数与内容预览。"""

    async def run() -> None:
        await harness.agent.session_manager.memory.store.save_facts_batch(
            "thread-1",
            [
                ThreadFactItem(content="用户偏好深色主题"),
                ThreadFactItem(content="无关条目"),
            ],
        )

        outcome = await harness.dispatch("forget 深色主题")

        assert outcome == HANDLED
        assert harness.agent.session_manager.memory.calls == [
            ("forget", ("thread-1", "深色主题", None))
        ]
        assert "1" in harness.output
        assert "深色主题" in harness.output
        # 断言"哪条被遗忘"：仅命中的那条离开 active 集合
        active = await harness.agent.session_manager.memory.store.query_facts(
            "thread-1", active_only=True
        )
        assert [f.content for f in active] == ["无关条目"]

    asyncio.run(run())


def test_forget_agent_command(harness: _Harness):
    """``forget agent <关键词>`` → agent 级命中项被遗忘。"""

    async def run() -> None:
        await harness.agent.session_manager.memory.store.save_agent_facts_batch(
            [
                ThreadFactItem(content="用户偏好中文回复", category="user_fact", scope="agent"),
                ThreadFactItem(content="用户使用 Vim", category="user_fact", scope="agent"),
            ]
        )

        outcome = await harness.dispatch("forget agent 中文")

        assert outcome == HANDLED
        assert harness.agent.session_manager.memory.calls == [
            ("forget_agent", ("中文", None))
        ]
        assert "agent" in harness.output
        assert "中文" in harness.output
        active = await harness.agent.session_manager.memory.store.query_agent_facts(
            active_only=True
        )
        assert [f.content for f in active] == ["用户使用 Vim"]

    asyncio.run(run())


def test_forget_command_without_keyword_prints_usage(harness: _Harness):
    """缺少关键词 → 打印用法且不调用 memory。"""

    async def run() -> None:
        outcome = await harness.dispatch("forget")

        assert outcome == HANDLED
        assert harness.agent.session_manager.memory.calls == []
        assert "用法" in harness.output

    asyncio.run(run())