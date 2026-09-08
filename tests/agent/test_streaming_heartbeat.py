"""streaming.py 心跳机制测试 - 工具执行期间发 tool_running 事件重置前端 watchdog。

验证：
1. 工具执行期间（on_tool_start 后长时间无事件）发心跳事件
2. on_tool_end 后心跳停止（active_tool_call_ids 已清空）
3. 非工具执行期间不发心跳（无活跃工具）
4. 心跳携带正确的 tool_call_id + tool_name
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.streaming import Streaming
from utils.events import EventType


def _event(
    event_name: str,
    *,
    run_id: str = "r1",
    tc_id: str = "tc1",
    name: str = "run_shell",
    output=None,
) -> dict:
    """构造 LangGraph astream_events 事件 dict。"""
    data: dict = {}
    if tc_id:
        data["tool_call_id"] = tc_id
    if output is not None:
        data["output"] = output
    return {
        "event": event_name,
        "name": name,
        "run_id": run_id,
        "data": data,
        "metadata": {},
    }


class _QueueGraph:
    """mock graph：astream_events 从 asyncio.Queue 读事件。

    测试通过 put 控制事件时序，put 之间的等待模拟工具执行卡顿。
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue = asyncio.Queue()

    async def put(self, event: object) -> None:
        await self._queue.put(event)

    def astream_events(self, *args: object, **kwargs: object):
        """返回 async generator，从 queue 读事件直到收到 None sentinel。"""
        queue = self._queue

        async def _gen():
            while True:
                ev = await queue.get()
                if ev is None:
                    break
                yield ev

        return _gen()


class _StubStreaming(Streaming):
    """绕过 AgentCore 依赖的最小 Streaming stub。"""

    def __init__(self, graph: _QueueGraph) -> None:
        self.agent_executor = graph
        self.metrics = MagicMock()
        self.llm = MagicMock()
        self.verbose = False


async def _collect_events(
    streaming: Streaming,
    message: str = "hi",
    thread_id: str = "t1",
) -> list:
    """收集 _arun_graph_events 产出的全部 AgentEvent。"""
    result: list = []
    config = {"configurable": {"thread_id": thread_id}}
    async for ev in streaming._arun_graph_events(
        {"role": "user", "content": message},
        config,
        thread_id,
        "trace1",
    ):
        result.append(ev)
    return result


def test_heartbeat_during_tool_execution(monkeypatch):
    """工具执行期间长时间无事件，应发心跳重置前端 watchdog。"""
    monkeypatch.setattr("agent.streaming.HEARTBEAT_INTERVAL", 0.05)

    graph = _QueueGraph()
    streaming = _StubStreaming(graph)

    async def _scenario() -> list:
        collect_task = asyncio.create_task(_collect_events(streaming))
        # 发 on_tool_start，登记活跃工具
        await graph.put(_event("on_tool_start", tc_id="tc1", name="run_shell"))
        # 等待足够时间让心跳发出（0.2s ≈ 4 次心跳间隔）
        await asyncio.sleep(0.2)
        # 发 on_tool_end，工具执行结束
        output = SimpleNamespace(content="done", tool_call_id="tc1")
        await graph.put(_event("on_tool_end", output=output))
        # 结束事件流
        await graph.put(None)
        return await collect_task

    events = asyncio.run(_scenario())

    heartbeats = [e for e in events if e.event_type == EventType.TOOL_RUNNING]
    assert len(heartbeats) >= 2, f"应至少 2 次心跳，实际 {len(heartbeats)}"
    assert all(h.tool_call_id == "tc1" for h in heartbeats)
    assert all(h.tool_name == "run_shell" for h in heartbeats)

    # on_tool_end 后不应再有心跳
    result_idx = next(
        i for i, e in enumerate(events) if e.event_type == EventType.TOOL_RESULT
    )
    after = events[result_idx + 1:]
    assert not any(e.event_type == EventType.TOOL_RUNNING for e in after)


def test_no_heartbeat_without_active_tool(monkeypatch):
    """非工具执行期间（无 active_tool_call_ids）不发心跳。"""
    monkeypatch.setattr("agent.streaming.HEARTBEAT_INTERVAL", 0.05)

    graph = _QueueGraph()
    streaming = _StubStreaming(graph)

    async def _scenario() -> list:
        collect_task = asyncio.create_task(_collect_events(streaming))
        # 发非工具事件（on_chat_model_stream 无 chunk，会被跳过）
        await graph.put(_event("on_chat_model_stream"))
        # 等待，但无活跃工具，不应发心跳
        await asyncio.sleep(0.2)
        await graph.put(None)
        return await collect_task

    events = asyncio.run(_scenario())

    heartbeats = [e for e in events if e.event_type == EventType.TOOL_RUNNING]
    assert len(heartbeats) == 0, f"无活跃工具不应发心跳，实际 {len(heartbeats)}"


def test_heartbeat_stops_after_tool_end(monkeypatch):
    """on_tool_end 后即使再有无事件延迟，也不发心跳（active 已清空）。"""
    monkeypatch.setattr("agent.streaming.HEARTBEAT_INTERVAL", 0.05)

    graph = _QueueGraph()
    streaming = _StubStreaming(graph)

    async def _scenario() -> list:
        collect_task = asyncio.create_task(_collect_events(streaming))
        # 工具执行
        await graph.put(_event("on_tool_start", tc_id="tc1"))
        await asyncio.sleep(0.15)  # 工具执行期间发心跳
        output = SimpleNamespace(content="done", tool_call_id="tc1")
        await graph.put(_event("on_tool_end", output=output))
        # 工具结束后延迟，不应发心跳
        await asyncio.sleep(0.15)
        await graph.put(None)
        return await collect_task

    events = asyncio.run(_scenario())

    heartbeats = [e for e in events if e.event_type == EventType.TOOL_RUNNING]
    assert len(heartbeats) >= 1, "工具执行期间应有心跳"
    # 所有心跳都在 tool_result 之前
    result_idx = next(
        i for i, e in enumerate(events) if e.event_type == EventType.TOOL_RESULT
    )
    for i, e in enumerate(events):
        if e.event_type == EventType.TOOL_RUNNING:
            assert i < result_idx, "心跳不应出现在 tool_result 之后"
