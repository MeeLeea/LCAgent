"""
测试工作流节点级 compaction（graph/common._compaction_wrapper 等）

覆盖：
- 消息通道超阈值时触发压缩并合并 messages/summary 到节点返回
- 未超阈值时不调用压缩
- mw 为 None 时 wrap 原样返回节点
- _build_compaction_middleware 在 agent 无 llm / 构造失败时禁用
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, SystemMessage

from graph.common import (
    _build_compaction_middleware,
    _compaction_wrapper,
    wrap_node_with_compaction,
)


class FakeMw:
    """模拟 compaction 中间件(不联网),记录调用并返回固定压缩结果。"""

    def __init__(self, threshold: int = 3, update: dict | None = None) -> None:
        self.config = SimpleNamespace(max_messages=threshold)
        self.calls: list[tuple[list, str, bool]] = []
        self.update = update or {
            "messages": [SystemMessage(content="压缩后摘要")],
            "summary": "压缩后摘要",
        }

    async def arun_compaction(
        self,
        messages: list,
        existing_summary: str = "",
        force: bool = False,
    ) -> dict | None:
        self.calls.append((messages, existing_summary, force))
        return self.update


async def _fake_node(state: dict, *, tag: str = "x") -> dict:
    """模拟被包装的节点:追加一条 AIMessage。"""
    return {"output": tag, "messages": [AIMessage(content=f"节点产出 {tag}")]}


def test_wrapper_compacts_when_over_threshold():
    """累计消息数超阈值时触发压缩,合并 messages/summary 到节点返回。"""
    mw = FakeMw(threshold=2)
    state = {
        "messages": [AIMessage(content="历史1"), AIMessage(content="历史2")],
        "summary": "",
    }

    result = asyncio.run(_compaction_wrapper(_fake_node, mw, state, tag="a"))

    # 节点产出保留
    assert result["output"] == "a"
    # 压缩结果合并进返回 dict
    assert result["summary"] == "压缩后摘要"
    assert isinstance(result["messages"][0], SystemMessage)
    # 压缩调用:2 历史 + 1 新增 = 3 > 阈值 2
    assert len(mw.calls) == 1
    msgs, existing_summary, force = mw.calls[0]
    assert len(msgs) == 3
    assert existing_summary == ""
    assert force is True


def test_wrapper_skips_when_under_threshold():
    """未超阈值时不调用压缩,节点返回原样透传。"""
    mw = FakeMw(threshold=10)
    state = {"messages": [AIMessage(content="历史1")], "summary": "已有摘要"}

    result = asyncio.run(_compaction_wrapper(_fake_node, mw, state, tag="b"))

    assert result["output"] == "b"
    assert result["messages"][0].content == "节点产出 b"
    assert "summary" not in result  # 未压缩,不写 summary
    assert mw.calls == []


def test_wrapper_passes_existing_summary():
    """压缩调用透传 state 中已有的 summary 作增量摘要基础。"""
    mw = FakeMw(threshold=1)
    state = {"messages": [AIMessage(content="历史1")], "summary": "上一轮摘要"}

    asyncio.run(_compaction_wrapper(_fake_node, mw, state, tag="c"))

    _, existing_summary, _ = mw.calls[0]
    assert existing_summary == "上一轮摘要"


def test_wrap_returns_original_when_mw_none():
    """mw 为 None 时 wrap 原样返回节点函数(不启用压缩)。"""
    node = _fake_node
    assert wrap_node_with_compaction(node, None) is node


def test_wrap_wraps_when_mw_present():
    """mw 非 None 时 wrap 返回包装后的可调用对象,行为正确。"""
    mw = FakeMw(threshold=0)
    wrapped = wrap_node_with_compaction(_fake_node, mw)
    result = asyncio.run(wrapped({"messages": []}, tag="d"))
    assert result["summary"] == "压缩后摘要"
    assert len(mw.calls) == 1


def test_build_middleware_none_without_llm():
    """agent 无 llm 属性时返回 None(压缩禁用)。"""
    agent = SimpleNamespace(name="fake", response="x")
    assert _build_compaction_middleware(agent) is None


def test_build_middleware_none_when_get_chat_model_missing():
    """agent.llm 无 get_chat_model 时返回 None。"""
    agent = SimpleNamespace(name="fake", llm=SimpleNamespace())
    assert _build_compaction_middleware(agent) is None


def test_build_middleware_constructs_with_llm():
    """agent 有 llm.get_chat_model 时构造中间件(阈值来自配置)。"""
    from unittest.mock import MagicMock

    agent = SimpleNamespace(name="real", llm=SimpleNamespace(get_chat_model=lambda: MagicMock()))
    mw = _build_compaction_middleware(agent)
    assert mw is not None
    assert mw.config.max_messages == 50  # 默认配置


def test_build_middleware_disables_on_construction_error():
    """构造失败(如 get_chat_model 抛异常)时返回 None,不向外抛。"""
    agent = SimpleNamespace(name="broken", llm=SimpleNamespace(get_chat_model=lambda: (_ for _ in ()).throw(RuntimeError("boom"))))
    assert _build_compaction_middleware(agent) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])