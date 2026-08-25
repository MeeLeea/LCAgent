"""工作流 interrupt/resume 全链路集成测试。

验证 ``interrupt_confirm → MemorySaver checkpoint → Command(resume) → 恢复``
全链路在真实 LangGraph 运行时下可行：

- ``tools/safety.interrupt_confirm`` 经 ``set_confirm_backend`` 注入 ``confirm``，
  节点内 ``confirm(prompt)`` 调用 ``interrupt()`` 暂停外层 graph；
- 外层 graph 用 ``MemorySaver`` 编译，interrupt 时 checkpoint 记录 pending
  interrupts，``ainvoke`` 返回 ``{"__interrupt__": [Interrupt(...)], ...}``；
- ``Command(resume={"choice_id": "approve"|"deny"})`` 恢复，``interrupt()``
  返回 resume payload，节点据此产出不同结果；
- ``graph.get_state(config)`` 可读取 pending interrupts（checkpoint 持久化）。

与 ``tests/agent/test_human_input.py::test_real_state_graph_dangerous_command_confirm_interrupts_and_resumes``
同构，但本测试聚焦 graph 层：覆盖 approve/deny 双路径 + checkpoint 持久化
状态读取，验证工作流节点内嵌 interrupt 的全链路可行性。
"""
from __future__ import annotations

import asyncio
import os
import sys
from typing import TypedDict

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from tools import safety

# 项目根注入 sys.path（与既有 tests/ 模块同构）
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ==================== 共享 fixture 与辅助 ====================


class _WorkflowState(TypedDict, total=False):
    """简化工作流状态：guard 节点产出 final_answer。"""

    final_answer: str


async def _guard_node(state: _WorkflowState) -> _WorkflowState:
    """模拟工作流内危险命令确认节点。

    调 ``safety.confirm(prompt)`` —— 在 ``set_confirm_backend(interrupt_confirm)``
    下会经 ``interrupt_confirm`` 触发外层 graph 的 ``interrupt()``，把危险命令
    信息抛给前端；resume 后 ``interrupt()`` 返回用户选择，``confirm`` 据此
    返回 bool，节点产出对应 final_answer。
    """
    prompt = (
        "⚠ 检测到危险命令 [匹配危险模式: \\brm\\b]\n"
        "待执行命令：rm x\n"
        "确认执行? [y/N]: "
    )
    approved = safety.confirm(prompt)
    result = "executed:approved" if approved else "cancelled:denied"
    return {"final_answer": result}


def _build_interrupt_workflow() -> StateGraph:
    """构建简化工作流：START → guard → END，带 MemorySaver 持久化。

    复刻生产 simple 工作流的"节点内 confirm → interrupt"链路结构，但剥离
    多节点串联，聚焦 interrupt/resume 本身的全链路验证。
    """
    builder = StateGraph(_WorkflowState)
    builder.add_node("guard", _guard_node)
    builder.add_edge(START, "guard")
    builder.add_edge("guard", END)
    return builder.compile(checkpointer=MemorySaver())


@pytest.fixture
def interrupt_backend():
    """启用 interrupt_confirm 后端，测试结束自动还原。

    与 ``tests/agent/test_human_input.py`` 的 try/finally 还原模式同构，
    防止全局 ``_confirm_backend`` 污染其他测试。
    """
    safety.set_confirm_backend(safety.interrupt_confirm)
    yield safety.interrupt_confirm
    safety.set_confirm_backend(None)


# ==================== 测试 ====================


def test_real_state_graph_workflow_dangerous_command_interrupts_and_resumes(interrupt_backend):
    """真实 LangGraph + MemorySaver + interrupt_confirm → interrupt → resume(approve) → 完成。

    Given: 简化工作流（START → guard → END）+ MemorySaver + interrupt_confirm 后端。
    When:  ainvoke 初次执行 → guard 节点内 confirm 调 interrupt_confirm →
           interrupt() 暂停外层 graph，checkpoint 记录 pending interrupt。
    Then:  返回值含 ``__interrupt__``，其 value 含 ``kind="dangerous_command"``、
           choices 含 approve/deny；resume(approve) 后 confirm 返回 True，
           节点产出 ``final_answer="executed:approved"``，无 ``__interrupt__``。
    """
    graph = _build_interrupt_workflow()
    config = {"configurable": {"thread_id": "thread-workflow-approve"}}

    # When: 初次执行被 interrupt 暂停
    interrupted = asyncio.run(graph.ainvoke({}, config))

    # Then: 返回 __interrupt__，value 含危险命令信息
    assert "__interrupt__" in interrupted
    interrupt = interrupted["__interrupt__"][0]
    assert interrupt.value["kind"] == "dangerous_command"
    assert "检测到危险命令" in interrupt.value["prompt"]
    assert [c["id"] for c in interrupt.value["choices"]] == ["approve", "deny"]

    # When: resume(approve) → confirm 返回 True → 节点产出 executed:approved
    result = asyncio.run(graph.ainvoke(Command(resume={"choice_id": "approve"}), config))

    # Then: 不含 __interrupt__，含 final_answer
    assert "__interrupt__" not in result
    assert result["final_answer"] == "executed:approved"


def test_real_state_graph_workflow_dangerous_command_deny(interrupt_backend):
    """interrupt → resume(deny) → confirm 返回 False → cancelled/DONE 语义。

    Given: 同 approve 测试的工作流 + MemorySaver + interrupt_confirm 后端。
    When:  ainvoke 初次执行 → interrupt 暂停；resume({"choice_id": "deny"}) →
           ``interrupt()`` 返回 deny payload，``confirm`` 据此返回 False。
    Then:  节点产出 ``final_answer="cancelled:denied"``，无 ``__interrupt__``，
           表明危险命令被拒绝、工作流以"已拒绝"语义完成（对照生产路径中
           run_shell 在 deny 时 raise UserRejectedCommandError，此处节点短路
           返回 cancelled 文本，验证 resume payload 的 choice_id 被正确消费）。
    """
    graph = _build_interrupt_workflow()
    config = {"configurable": {"thread_id": "thread-workflow-deny"}}

    # When: 初次执行被 interrupt 暂停
    interrupted = asyncio.run(graph.ainvoke({}, config))
    assert "__interrupt__" in interrupted

    # When: resume(deny) → confirm 返回 False → 节点产出 cancelled:denied
    result = asyncio.run(graph.ainvoke(Command(resume={"choice_id": "deny"}), config))

    # Then: 不含 __interrupt__，final_answer 表明命令被拒绝
    assert "__interrupt__" not in result
    assert result["final_answer"] == "cancelled:denied"


def test_workflow_with_checkpointer_persists_interrupt_state(interrupt_backend):
    """interrupt 后 checkpoint 记录 pending interrupts（经 get_state 可读）。

    Given: 工作流 + MemorySaver + interrupt_confirm 后端。
    When:  ainvoke 初次执行 → interrupt 暂停；调 ``graph.get_state(config)``。
    Then:  ``state.tasks`` 中存在 pending interrupts，其 value 含
           ``kind="dangerous_command"``、choices 含 approve/deny —— 表明
           MemorySaver 在 interrupt 时正确持久化中断状态，可被外层
           （如 CLI human_input loop、scheduler）读取并渲染给用户。
    """
    graph = _build_interrupt_workflow()
    config = {"configurable": {"thread_id": "thread-workflow-checkpoint"}}

    # When: 初次执行被 interrupt 暂停
    asyncio.run(graph.ainvoke({}, config))

    # Then: get_state 可读取 pending interrupts（checkpoint 持久化）
    state = graph.get_state(config)
    assert state is not None
    pending = [i.value for t in state.tasks for i in t.interrupts]
    assert pending, "interrupt 后 checkpoint 应记录 pending interrupts"
    assert pending[0]["kind"] == "dangerous_command"
    assert "检测到危险命令" in pending[0]["prompt"]
    assert [c["id"] for c in pending[0]["choices"]] == ["approve", "deny"]

    # And: resume(approve) 后 pending interrupts 被清除（checkpoint 更新为完成态）
    asyncio.run(graph.ainvoke(Command(resume={"choice_id": "approve"}), config))
    state_after = graph.get_state(config)
    pending_after = [i.value for t in state_after.tasks for i in t.interrupts]
    assert not pending_after, "resume 完成后不应残留 pending interrupts"
