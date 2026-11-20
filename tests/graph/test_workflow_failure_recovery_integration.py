"""工作流节点失败 → checkpoint 现场保存 → 断点重试恢复 全链路集成测试。

验证 ``节点异常 → MemorySaver checkpoint 保留 pending task → ainvoke(None)
从失败节点续跑`` 全链路在真实 LangGraph 运行时下可行：

- 三节点链 a → b → c，b 首次执行抛瞬时异常（ConnectionError）：
  checkpoint 停留在上一成功 superstep 末，``get_state(config).next == ("b",)``
  且 a 的产出保留在 ``state.values``（现场保存）；
- ``ainvoke(None, config)`` 恢复：LangGraph 判定 resuming，失败节点 b 因
  versions_seen 未更新被重新调度，a/c 不重跑（执行次数精确断言）；
- 图跑完后 ``next`` 为空、再 ainvoke(None) 无重复执行 —— 佐证生产代码
  ``WorkflowAdapter.aretry_events`` 前置 aget_state 校验断点的必要性。

与 ``tests/graph/test_workflow_interrupt_integration.py`` 同构（真实
StateGraph + MemorySaver），但聚焦异常失败恢复而非 interrupt/resume：
interrupt 挂起必须 Command(resume=payload) 恢复，异常失败必须
ainvoke(None) 恢复 —— 两者是互补的恢复路径。
"""
from __future__ import annotations

import asyncio
import os
import sys
from typing import TypedDict

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

# 项目根注入 sys.path（与既有 tests/ 模块同构）
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ==================== 共享 fixture 与辅助 ====================


class _WorkflowState(TypedDict, total=False):
    """简化工作流状态：各节点产出文本，b 产出 final_answer。"""

    a_out: str
    b_out: str
    c_out: str
    final_answer: str


# 各节点执行次数记录（模块级，测试内重置；b 的失败开关也在此）
_exec_counts = {"a": 0, "b": 0, "c": 0}
_b_should_fail = True


async def _node_a(state: _WorkflowState) -> _WorkflowState:
    _exec_counts["a"] += 1
    return {"a_out": "a 的产出"}


async def _node_b(state: _WorkflowState) -> _WorkflowState:
    _exec_counts["b"] += 1
    if _b_should_fail:
        raise ConnectionError("Error code: 503 - 服务暂时不可用")
    return {"b_out": f"b 基于 {state['a_out']} 完成", "final_answer": "b 成功后的最终答案"}


async def _node_c(state: _WorkflowState) -> _WorkflowState:
    _exec_counts["c"] += 1
    return {"c_out": "c 收尾", "final_answer": f"{state['b_out']} → c 收尾"}


def _build_failure_workflow():
    """构建三节点链 a → b → c，带 MemorySaver 持久化。

    复刻生产 workflow 的线性链结构（simple: summarize → manager_plan →
    worker_exec → terminator_final），b 模拟 LLM 调用瞬时失败的节点。
    """
    builder = StateGraph(_WorkflowState)
    builder.add_node("a", _node_a)
    builder.add_node("b", _node_b)
    builder.add_node("c", _node_c)
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("b", "c")
    builder.add_edge("c", END)
    return builder.compile(checkpointer=MemorySaver())


@pytest.fixture
def failure_env():
    """重置执行计数与失败开关，测试结束再还原（防模块级状态泄漏）。"""
    _exec_counts.update({"a": 0, "b": 0, "c": 0})
    global _b_should_fail
    _b_should_fail = True
    yield
    _b_should_fail = True


# ==================== 测试 ====================


def test_node_failure_preserves_scene_in_checkpoint(failure_env):
    """节点异常后 checkpoint 保存现场：next=失败节点，前序产出保留。

    Given: 三节点链 a → b → c + MemorySaver，b 首次执行抛 ConnectionError。
    When:  ainvoke 初次执行 → a 成功 → b 异常 → graph 执行终止。
    Then:  ``get_state(config).next == ("b",)``（失败节点为 pending task），
           a 的产出保留在 ``state.values["a_out"]``（现场保存成功）——
           生产代码 aretry_events 的断点校验与恢复正依赖这两点。
    """
    graph = _build_failure_workflow()
    config = {"configurable": {"thread_id": "thread-failure-scene"}}

    # When: 初次执行，a 成功后 b 抛瞬时异常
    with pytest.raises(ConnectionError):
        asyncio.run(graph.ainvoke({}, config))

    # Then: 失败节点保留为 pending task（next 指向 b）
    state = graph.get_state(config)
    assert state is not None
    assert state.next == ("b",), "失败节点 b 应保留为 checkpoint pending task"
    # And: a 的产出已持久化（现场保存）
    assert state.values["a_out"] == "a 的产出"
    # And: a 仅执行一次，b 执行过一次（失败），c 未执行
    assert _exec_counts == {"a": 1, "b": 1, "c": 0}


def test_ainvoke_none_resumes_from_failed_node_only(failure_env):
    """ainvoke(None) 从失败节点续跑：仅 b 重跑，a/c 各执行一次。

    Given: b 首次失败（同上），随后翻 flag 使 b 第二次成功。
    When:  ``ainvoke(None, config)`` 恢复执行。
    Then:  最终 final_answer 正确（c 收尾产出），执行次数 a=1 / b=2 / c=1
           —— a 的成功产出不重跑（versions_seen 已对齐），b 从头重跑成功，
           c 首次执行收尾。这正是 WorkflowAdapter 自动重试/手动重试的底层语义。
    """
    graph = _build_failure_workflow()
    config = {"configurable": {"thread_id": "thread-failure-resume"}}

    # When: 初次执行 b 失败
    with pytest.raises(ConnectionError):
        asyncio.run(graph.ainvoke({}, config))
    assert _exec_counts == {"a": 1, "b": 1, "c": 0}

    # And: 故障恢复后（flag 翻转），ainvoke(None) 从失败节点续跑
    global _b_should_fail
    _b_should_fail = False
    result = asyncio.run(graph.ainvoke(None, config))

    # Then: 最终产出正确
    assert result["final_answer"] == "b 基于 a 的产出 完成 → c 收尾"
    assert result["c_out"] == "c 收尾"
    # And: 仅失败节点 b 重跑（a 不重跑、c 首跑）
    assert _exec_counts == {"a": 1, "b": 2, "c": 1}
    # And: 图推进到 END，无 pending task
    state_after = graph.get_state(config)
    assert state_after.next == ()


def test_completed_graph_ainvoke_none_is_noop(failure_env):
    """已跑完的图 ainvoke(None) 不重跑任何节点（aretry_events 前置校验的依据）。

    Given: 三节点链正常跑完（b 不失败）。
    When:  再次 ``ainvoke(None, config)``。
    Then:  返回值与首次一致，无节点重复执行（执行次数不变），
           ``get_state(config).next`` 为空 —— 生产代码 aretry_events 对
           next 为空的图直接 yield ERROR，防止这种空转。
    """
    global _b_should_fail
    _b_should_fail = False
    graph = _build_failure_workflow()
    config = {"configurable": {"thread_id": "thread-failure-completed"}}

    # Given: 正常跑完
    first = asyncio.run(graph.ainvoke({}, config))
    assert _exec_counts == {"a": 1, "b": 1, "c": 1}

    # When: 已完结的图再 ainvoke(None)
    second = asyncio.run(graph.ainvoke(None, config))

    # Then: 返回现有 values，无节点重复执行
    assert second["final_answer"] == first["final_answer"]
    assert _exec_counts == {"a": 1, "b": 1, "c": 1}
    assert graph.get_state(config).next == ()
