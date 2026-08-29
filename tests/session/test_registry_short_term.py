"""SessionRegistry short_term_size 测试 - 验证短期上下文窗口截取逻辑。

覆盖配置键 ``latest_msg_cnt`` 的生效路径：
构造时传入的窗口大小作为 ``aget_short_term`` 的默认截取条数。

运行：
  pytest tests/session/test_registry_short_term.py -v
"""
import asyncio
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

from session import SessionRegistry, SessionStore


class _FakeCheckpointer:
    """返回固定消息列表的假 checkpointer（同步 get_tuple 契约）。"""

    def __init__(self, messages):
        self._messages = list(messages)

    def get_tuple(self, config):
        return SimpleNamespace(
            checkpoint={"channel_values": {"messages": self._messages}}
        )


def _make_messages(count: int):
    """生成 count 条交替 Human/AI 消息，content 为 msg-<i>。"""
    return [
        HumanMessage(content=f"msg-{i}") if i % 2 == 0 else AIMessage(content=f"msg-{i}")
        for i in range(count)
    ]


def _make_registry(messages, short_term_size: int = 10) -> SessionRegistry:
    return SessionRegistry(
        _FakeCheckpointer(messages),
        SessionStore(),
        short_term_size=short_term_size,
    )


def test_aget_short_term_uses_configured_window():
    """构造时设定的 short_term_size 生效：5 条消息取最近 3 条。"""

    async def run():
        reg = _make_registry(_make_messages(5), short_term_size=3)
        return await reg.aget_short_term("t1")

    msgs = asyncio.run(run())
    assert len(msgs) == 3
    assert [m["content"] for m in msgs] == ["msg-2", "msg-3", "msg-4"]


def test_aget_short_term_default_window_all_messages():
    """默认 short_term_size=10 时，消息数不足则全部返回。"""

    async def run():
        reg = _make_registry(_make_messages(5))
        return await reg.aget_short_term("t1")

    msgs = asyncio.run(run())
    assert len(msgs) == 5


def test_aget_short_term_explicit_short_term_size_overrides():
    """显式传参 short_term_size 覆盖构造时设定的窗口。"""

    async def run():
        reg = _make_registry(_make_messages(5), short_term_size=3)
        return await reg.aget_short_term("t1", short_term_size=2)

    msgs = asyncio.run(run())
    assert len(msgs) == 2
    assert [m["content"] for m in msgs] == ["msg-3", "msg-4"]


def test_aget_short_term_limit_wins_over_window():
    """limit 优先于 short_term_size。"""

    async def run():
        reg = _make_registry(_make_messages(5), short_term_size=3)
        return await reg.aget_short_term("t1", limit=2)

    msgs = asyncio.run(run())
    assert len(msgs) == 2


def test_aget_short_term_role_mapping():
    """role 映射：HumanMessage→user，AIMessage→assistant。"""

    async def run():
        reg = _make_registry(_make_messages(4), short_term_size=10)
        return await reg.aget_short_term("t1")

    msgs = asyncio.run(run())
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"


def test_aget_short_term_zero_window_returns_all():
    """short_term_size=0 视为不截取，返回全部消息。"""

    async def run():
        reg = _make_registry(_make_messages(4), short_term_size=0)
        return await reg.aget_short_term("t1")

    msgs = asyncio.run(run())
    assert len(msgs) == 4
