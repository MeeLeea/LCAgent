"""断流重连（attach）机制单元测试。

覆盖：
- _ActiveStream 的发布/订阅/重放/收尾语义
- _forward_stream_with_cancel 在响应任务被取消（模拟 Starlette 检测到客户端
  断开后取消响应任务）时转移后台执行：runner 不被取消、流保持注册可 attach、
  后台完成后清理注册表

不依赖真实 LLM 与运行中的服务器。
"""
import asyncio

from api.server import (
    _active_streams,
    _ActiveStream,
    _DetachState,
    _forward_stream_with_cancel,
)


class _FakeRequest:
    """最小 Request 替身：is_disconnected 恒 False（断开通过任务取消注入）。"""

    async def is_disconnected(self) -> bool:
        return False


def test_active_stream_subscribe_replays_log_then_live() -> None:
    async def main() -> None:
        stream = _ActiveStream()
        await stream.publish({"type": "token", "content": "a"})
        q = await stream.subscribe()  # 订阅时应先重放已有事件日志
        await stream.publish({"type": "token", "content": "b"})
        await stream.finish()

        assert await q.get() == {"type": "token", "content": "a"}  # 重放
        assert await q.get() == {"type": "token", "content": "b"}  # 实时
        assert await q.get() is None  # 结束哨兵

    asyncio.run(main())


def test_active_stream_subscribe_after_finish_gets_replay_and_sentinel() -> None:
    async def main() -> None:
        stream = _ActiveStream()
        await stream.publish({"type": "done"})
        await stream.finish()

        q = await stream.subscribe()
        assert await q.get() == {"type": "done"}
        assert await q.get() is None
        assert stream.subscribers == []

    asyncio.run(main())


def test_active_stream_unsubscribe() -> None:
    async def main() -> None:
        stream = _ActiveStream()
        q = await stream.subscribe()
        await stream.unsubscribe(q)
        await stream.publish({"type": "token", "content": "x"})
        assert q.empty()  # 退订后不再接收

    asyncio.run(main())


def test_forward_stream_cancellation_detaches_to_background() -> None:
    """核心回归：取消响应任务（客户端断开）不得杀死执行，内容完整产出。"""
    async def main() -> None:
        thread_id = "test-reattach-detach"
        stream = _ActiveStream()
        _active_streams[thread_id] = stream
        detach = _DetachState()
        cancel_event = asyncio.Event()
        produced: list[dict] = []

        async def source():
            for i in range(5):
                ev = {"type": "token", "content": str(i)}
                await stream.publish(ev)
                produced.append(ev)
                yield ev
                await asyncio.sleep(0.01)

        async def consume() -> list[str]:
            return [
                sse
                async for sse in _forward_stream_with_cancel(
                    source(), cancel_event, _FakeRequest(), thread_id,
                    stream=stream, detach=detach,
                )
            ]

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.05)  # 让流产出部分事件
        task.cancel()  # 模拟 Starlette 检测 http.disconnect 后取消响应任务
        try:
            await task
            raise AssertionError("被取消的任务应抛出 CancelledError")
        except asyncio.CancelledError:
            pass

        # 取消瞬间：已转移后台，流保持注册（attach 可用）
        assert detach.detached is True
        assert thread_id in _active_streams

        # 后台执行继续直到完成：runner 未被取消，全部事件产出
        for _ in range(200):
            if thread_id not in _active_streams:
                break
            await asyncio.sleep(0.05)
        assert thread_id not in _active_streams, "后台完成后注册表应清理"
        assert len(produced) == 5, "runner 不应被取消，事件应全部产出"
        assert stream.finished is True

    asyncio.run(main())


def test_forward_stream_cancel_event_still_cancels() -> None:
    """回归：用户主动停止（cancel_event 置位）路径不受影响，runner 被取消。"""
    async def main() -> None:
        thread_id = "test-reattach-stop"
        stream = _ActiveStream()
        _active_streams[thread_id] = stream
        detach = _DetachState()
        cancel_event = asyncio.Event()
        produced: list[dict] = []

        async def source():
            for i in range(100):
                ev = {"type": "token", "content": str(i)}
                produced.append(ev)
                yield ev
                await asyncio.sleep(0.01)

        async def consume() -> list[str]:
            return [
                sse
                async for sse in _forward_stream_with_cancel(
                    source(), cancel_event, _FakeRequest(), thread_id,
                    stream=stream, detach=detach,
                )
            ]

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.05)
        cancel_event.set()  # 用户点击停止
        sses = await task  # 正常结束（非取消）

        assert detach.detached is False  # 停止不是断开，不转移后台
        assert len(produced) < 100, "停止后 runner 应被取消，事件不完整产出"
        assert any('"cancelled"' in s for s in sses), "应下发 cancelled 事件"
        # 非转移路径由 event_stream 收尾（此处测试未经过端点，手动模拟其 finally）
        if thread_id in _active_streams:
            _active_streams.pop(thread_id)

    asyncio.run(main())
