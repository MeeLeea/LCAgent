"""
测试工作流运行器层的 active_skills 透传（arun_compiled_workflow / arun_simple_workflow）

覆盖：
- active_skills 非空时写入 initial_state（节点可读取并注入技能）
- active_skills 为空（默认）时不写入该键，避免覆盖 checkpoint 已持久化的值
- arun_simple_workflow 把 active_skills 转发给 arun_compiled_workflow
"""
from __future__ import annotations

import asyncio

from graph.common.workflow_runner import arun_compiled_workflow
from graph.simple import arun_simple_workflow


class _CapturingGraph:
    """假图：记录 ainvoke 收到的初始状态，aget_state 返回 None（无历史）。"""

    def __init__(self) -> None:
        self.received: dict | None = None

    async def aget_state(self, config):
        return None

    async def ainvoke(self, state, config=None):
        self.received = state
        return {"final_answer": "完成"}


def test_runner_seeds_active_skills_when_non_empty():
    """active_skills 非空时写入 initial_state.active_skills。"""
    graph = _CapturingGraph()

    asyncio.run(arun_compiled_workflow(graph, "任务", active_skills=["git-helper"]))

    assert graph.received is not None
    assert graph.received["active_skills"] == ["git-helper"]


def test_runner_omits_active_skills_when_empty():
    """active_skills 缺省时不写入该键（保留 checkpoint 中的持久化值）。"""
    graph = _CapturingGraph()

    asyncio.run(arun_compiled_workflow(graph, "任务"))

    assert graph.received is not None
    assert "active_skills" not in graph.received


def test_runner_omits_active_skills_when_explicit_empty():
    """显式传空序列同样不写入该键（不覆盖 checkpoint 值）。"""
    graph = _CapturingGraph()

    asyncio.run(arun_compiled_workflow(graph, "任务", active_skills=[]))

    assert graph.received is not None
    assert "active_skills" not in graph.received


def test_simple_workflow_forwards_active_skills():
    """arun_simple_workflow 把 active_skills 转发给通用运行器。"""
    graph = _CapturingGraph()

    asyncio.run(arun_simple_workflow(graph, "任务", active_skills=["pptx"]))

    assert graph.received is not None
    assert graph.received["active_skills"] == ["pptx"]


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
