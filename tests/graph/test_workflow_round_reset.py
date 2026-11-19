"""测试新任务开跑时的轮次计数器清零（reset_round_counters）。

背景：workflow 的 checkpoint 按 thread 持久化，``round`` 会跨任务累积，
导致同会话第二条任务一开局就接近 ``max_rounds``（表现为"迭代环只走一次"
或"仿真失败一次即交付"）。本模块守护"新任务入口清零"这一修复。

覆盖：
- 命名规则：以 ``round`` 结尾的通道清零，``max_rounds`` 上限字段保留
- ``arun_compiled_workflow``：不传 initial_fields 时也归零（跨任务累计回归）
- 图未声明轮次通道时不写入多余键（不污染 state）
- schema 自省失败时 fail-open（退化为 state 键，不抛异常）
"""
from __future__ import annotations

import asyncio

from graph.common.workflow_runner import arun_compiled_workflow, reset_round_counters


class _CapturingGraph:
    """假图：记录 ainvoke 收到的初始状态，aget_state 返回 None（无历史）。"""

    def __init__(self, channels: set[str] | None = None) -> None:
        self._channels = channels or set()
        self.received: dict | None = None

    def get_input_jsonschema(self) -> dict:
        return {"properties": {name: {} for name in self._channels}}

    async def aget_state(self, config):
        return None

    async def ainvoke(self, state, config=None):
        self.received = state
        return {"final_answer": "完成"}


class _NoSchemaGraph(_CapturingGraph):
    """schema 自省抛异常且无 channels 属性的图（验证 fail-open）。"""

    def get_input_jsonschema(self) -> dict:
        raise RuntimeError("schema 不可用")


def test_reset_round_counters_zeros_counters_but_keeps_limit():
    """以 round 结尾的通道清零，max_rounds 保留。"""
    graph = _CapturingGraph({"task", "round", "file_round", "max_rounds"})
    state = {"round": 2, "file_round": 1, "max_rounds": 3}

    reset_round_counters(graph, state)

    assert state["round"] == 0
    assert state["file_round"] == 0
    assert state["max_rounds"] == 3


def test_runner_resets_round_without_initial_fields():
    """不传 initial_fields 时 round 也归零（跨任务累计回归）。"""
    graph = _CapturingGraph({"task", "round", "max_rounds"})

    asyncio.run(arun_compiled_workflow(graph, "第二条任务"))

    assert graph.received is not None
    assert graph.received["round"] == 0


def test_runner_resets_round_with_initial_fields():
    """initial_fields 给出 round/max_rounds 时，round 归零、上限不变。"""
    graph = _CapturingGraph({"task", "round", "max_rounds"})

    asyncio.run(
        arun_compiled_workflow(
            graph, "第二条任务", initial_fields={"round": 0, "max_rounds": 3}
        )
    )

    assert graph.received is not None
    assert graph.received["round"] == 0
    assert graph.received["max_rounds"] == 3


def test_runner_does_not_inject_round_for_graphs_without_counter():
    """图未声明轮次通道时不写入 round（不污染无关工作流的 state）。"""
    graph = _CapturingGraph({"task"})

    asyncio.run(arun_compiled_workflow(graph, "任务"))

    assert graph.received is not None
    assert "round" not in graph.received


def test_reset_round_counters_fail_open_when_schema_unavailable():
    """schema 自省不可用时退化为 state 键，仍完成清零且不抛异常。"""
    graph = _NoSchemaGraph()
    state = {"round": 2, "max_rounds": 3}

    reset_round_counters(graph, state)

    assert state["round"] == 0
    assert state["max_rounds"] == 3
