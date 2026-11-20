"""可取消、可超时的异步等待（planner / executor 共享）。

``planner/worker.py`` 与 ``executor/worker.py`` 的 ``_await_with_cancel`` 实现
逐行相同，唯一差异是两类自有取消异常与超时文案（planner「LLM 调用超时」/
executor「模型调用超时」）——文案被子串断言锁死，故异常类与文案均由调用方
传入。子程序侧保留同名薄 wrapper（``_await_with_cancel``）注入自身参数。

本模块只依赖标准库。
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

__all__ = ["await_with_cancel"]


async def await_with_cancel(
    awaitable: Awaitable[Any],
    *,
    timeout: float,
    cancel_check: Callable[[], bool],
    poll_seconds: float,
    cancel_exception: type[Exception],
    timeout_message: str,
    cancel_message: str = "用户取消",
) -> Any:
    """在可取消、可超时的前提下等待协程完成，不泄漏挂起任务。

    Args:
        awaitable: 要等待的协程/可等待对象；超时或取消时会被 cancel，
            无论其是否已启动。
        timeout: 总超时秒数，自本次调用起计算。
        cancel_check: 每个轮询间隔调用一次的取消判定；返回真值时立即取消。
        poll_seconds: 取消轮询间隔（``<= 0`` 时回退 0.5s）。
        cancel_exception: 命中取消时抛出的异常类
            （``PlannerCancelled`` / ``ExecutorCancelled`` 等子程序自有类型）。
        timeout_message: ``TimeoutError`` 携带的消息模板；模板内的
            ``{timeout}`` 会被替换为 ``timeout`` 参数值（用 ``str.replace``
            而非 ``str.format``，兼容文案中其余花括号），如
            ``"LLM 调用超时（{timeout}s）"``。
        cancel_message: 取消异常携带的消息。

    Returns:
        ``awaitable`` 的结果。

    Raises:
        TimeoutError: 超过 ``timeout``（消息为 ``timeout_message``）。
        cancel_exception: ``cancel_check`` 命中（消息为 ``cancel_message``）。
    """
    work = asyncio.ensure_future(awaitable)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    effective_poll = poll_seconds if poll_seconds > 0 else 0.5
    try:
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError(timeout_message.replace("{timeout}", str(timeout)))
            if cancel_check():
                raise cancel_exception(cancel_message)
            done, _pending = await asyncio.wait({work}, timeout=min(effective_poll, remaining))
            if work in done:
                return work.result()
    finally:
        if not work.done():
            work.cancel()
            await asyncio.gather(work, return_exceptions=True)
