"""
测试工作流运行器层记忆接入（arun_compiled_workflow 的 memory 参数）

覆盖：
- 执行前 recall_text 长期记忆注入 raw_context
- 执行后 final_answer 构造 DONE AgentEvent 提交 consume_event
- memory / memory_thread_id 缺省时静默跳过
- final_answer 为空时不提交
- is_run_mode 控制 DONE 事件的 is_important
- 与 checkpoint 跨轮次摘要（_aget_previous_workflow_summary）并存
"""
from __future__ import annotations

import asyncio

from graph.simple import arun_simple_workflow, build_simple_workflow
from tests.test_workflow import FakeAgent
from utils.events import EventType


class FakeMemory:
    """模拟 MemoryManager(不联网),记录召回与消费调用。"""

    def __init__(self, recalled: str = "【长期记忆】\n- [事实] 用户偏好中文\n") -> None:
        self.recalled = recalled
        self.recall_calls: list[str] = []
        self.consumed_events: list = []

    async def recall_text(self, thread_id: str, limit=None) -> str:
        self.recall_calls.append(thread_id)
        return self.recalled

    async def consume_event(self, event) -> None:
        self.consumed_events.append(event)


def _build_graph():
    """构建 simple 工作流(Fake agents,无 llm → 不启用 compaction)。"""
    manager = FakeAgent(name="manager", response="计划", summary_response="摘要")
    worker = FakeAgent(name="worker", response="结果")
    terminator = FakeAgent(name="terminator", response="最终答案: 完成")
    graph = build_simple_workflow(
        {"manager": manager, "worker": worker, "terminator": terminator}
    )
    return graph, manager, terminator


def test_recall_injected_into_raw_context():
    """执行前召回长期记忆并注入 raw_context(summarize 节点收到记忆文本)。"""
    graph, manager, _ = _build_graph()
    memory = FakeMemory()

    asyncio.run(
        arun_simple_workflow(
            graph, "测试任务", memory=memory, memory_thread_id="wf-mem-1"
        )
    )

    assert memory.recall_calls == ["wf-mem-1"]
    assert manager.calls[0][0] == "summarize"
    assert "【长期记忆】" in manager.calls[0][1]
    assert "用户偏好中文" in manager.calls[0][1]


def test_done_event_consumed():
    """执行后将 final_answer 构造 DONE 事件提交 consume_event。"""
    graph, _, _ = _build_graph()
    memory = FakeMemory()

    asyncio.run(
        arun_simple_workflow(
            graph, "测试任务", memory=memory, memory_thread_id="wf-mem-2"
        )
    )

    assert len(memory.consumed_events) == 1
    event = memory.consumed_events[0]
    assert event.event_type == EventType.DONE
    assert event.content == "最终答案: 完成"
    assert event.thread_id == "wf-mem-2"
    assert event.role == "assistant"
    assert event.is_important is False  # 非运行模式


def test_is_run_mode_marks_important():
    """运行模式下 DONE 事件标记为重要记忆。"""
    graph, _, _ = _build_graph()
    memory = FakeMemory()

    asyncio.run(
        arun_simple_workflow(
            graph, "测试任务", memory=memory, memory_thread_id="wf-mem-3", is_run_mode=True
        )
    )

    assert memory.consumed_events[0].is_important is True


def test_memory_none_skips_all():
    """不传 memory 时无召回无提交,运行正常。"""
    graph, manager, _ = _build_graph()

    asyncio.run(arun_simple_workflow(graph, "测试任务"))

    assert manager.calls[0][0] == "summarize"
    assert "【长期记忆】" not in manager.calls[0][1]


def test_memory_without_thread_id_skips():
    """memory_thread_id 为 None 时不召回也不提交。"""
    graph, manager, _ = _build_graph()
    memory = FakeMemory()

    asyncio.run(arun_simple_workflow(graph, "测试任务", memory=memory))

    assert memory.recall_calls == []
    assert memory.consumed_events == []
    assert "【长期记忆】" not in manager.calls[0][1]


def test_empty_final_answer_skips_consume():
    """final_answer 为空时不提交 DONE 事件(避免空记忆入库)。"""
    graph, _, terminator = _build_graph()
    terminator.response = ""
    memory = FakeMemory()

    asyncio.run(
        arun_simple_workflow(
            graph, "测试任务", memory=memory, memory_thread_id="wf-mem-4"
        )
    )

    assert memory.consumed_events == []


def test_coexists_with_checkpoint_summary():
    """长期记忆与 checkpoint 跨轮次摘要并存(raw_context 同时含两段)。"""
    from langgraph.checkpoint.memory import MemorySaver

    manager = FakeAgent(name="manager", response="计划2", summary_response="摘要2")
    worker = FakeAgent(name="worker", response="结果2")
    terminator = FakeAgent(name="terminator", response="答案2")
    checkpointer = MemorySaver()
    graph = build_simple_workflow(
        {"manager": manager, "worker": worker, "terminator": terminator},
        checkpointer=checkpointer,
    )
    memory = FakeMemory()

    # 第一轮:产生 checkpoint 历史
    asyncio.run(
        arun_simple_workflow(
            graph, "第一轮任务", thread_id="wf-mem-5", memory=memory, memory_thread_id="wf-mem-5"
        )
    )

    # 第二轮:raw_context 同时含 checkpoint 摘要与长期记忆
    manager.calls.clear()
    asyncio.run(
        arun_simple_workflow(
            graph, "第二轮任务", thread_id="wf-mem-5", memory=memory, memory_thread_id="wf-mem-5"
        )
    )

    raw = manager.calls[0][1]
    assert "【上一轮工作流记录】" in raw
    assert "第一轮任务" in raw
    assert "【长期记忆】" in raw
    assert "用户偏好中文" in raw


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])