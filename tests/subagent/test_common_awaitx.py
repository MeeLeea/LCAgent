"""``subagent.common`` 可取消 / 可超时异步等待的单元测试。

设计约束：
    - 仓库未安装 ``pytest-asyncio``，统一用 ``asyncio.run`` 在同步测试函数中
      驱动协程（与 ``test_worker.py`` / ``test_executor_worker.py`` 一致）；
    - 覆盖：正常完成无悬挂任务、取消（异常类/默认与自定义消息）、超时
      （``{timeout}`` 模板注入且其余花括号保留）、deadline 先到时超时优先于
      取消的次序锁、``poll_seconds<=0`` 回退。

自测辅助模块：``tests.subagent._fakes.open_tasks_excluding_current``。
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from subagent.common.awaitx import await_with_cancel
from tests.subagent._fakes import open_tasks_excluding_current

__all__ = ["_Cancelled"]


class _Cancelled(Exception):
    """测试用取消异常（模拟子程序自有取消类型，如 PlannerCancelled）。"""


async def _await(
    awaitable: Awaitable[Any],
    *,
    timeout: float,
    cancel_check: Callable[[], bool],
    poll_seconds: float = 0.01,
    cancel_exception: type[Exception] = _Cancelled,
    timeout_message: str = "超时（{timeout}s）",
) -> Any:
    """以公共参数形态驱动一次 ``await_with_cancel``（测试辅助）。"""
    return await await_with_cancel(
        awaitable,
        timeout=timeout,
        cancel_check=cancel_check,
        poll_seconds=poll_seconds,
        cancel_exception=cancel_exception,
        timeout_message=timeout_message,
    )


def test_completes_normally_without_dangling_tasks() -> None:
    """协程正常完成时返回结果，且不留下悬挂任务。"""

    async def scenario() -> Any:
        async def coro() -> int:
            return 42

        result = await _await(coro(), timeout=1.0, cancel_check=lambda: False)
        assert open_tasks_excluding_current() == 0
        return result

    assert asyncio.run(scenario()) == 42


def test_cancel_raises_injected_exception_with_default_message() -> None:
    """cancel_check 命中时抛调用方注入的异常类，消息为默认「用户取消」。"""

    async def scenario() -> None:
        async def coro() -> int:
            await asyncio.sleep(10)
            return 1

        with pytest.raises(_Cancelled, match="用户取消"):
            await _await(coro(), timeout=1.0, cancel_check=lambda: True)
        assert open_tasks_excluding_current() == 0

    asyncio.run(scenario())


def test_cancel_custom_message_and_exception_type() -> None:
    """cancel_message 与 cancel_exception 可完全自定义。"""

    async def scenario() -> None:
        async def coro() -> int:
            await asyncio.sleep(10)
            return 1

        caught: Exception | None = None
        try:
            await await_with_cancel(
                coro(),
                timeout=1.0,
                cancel_check=lambda: True,
                poll_seconds=0.01,
                cancel_exception=RuntimeError,
                timeout_message="超时",
                cancel_message="自定义取消原因",
            )
        except RuntimeError as exc:
            caught = exc
        assert caught is not None
        assert str(caught) == "自定义取消原因"
        assert open_tasks_excluding_current() == 0

    asyncio.run(scenario())


def test_timeout_message_replaces_timeout_template() -> None:
    """``{timeout}`` 模板经 ``str.replace`` 注入实际秒数。"""

    async def scenario() -> None:
        async def coro() -> int:
            await asyncio.sleep(10)
            return 1

        with pytest.raises(TimeoutError, match=r"超时（0\.05s）"):
            await _await(
                coro(),
                timeout=0.05,
                timeout_message="超时（{timeout}s）",
                cancel_check=lambda: False,
            )
        assert open_tasks_excluding_current() == 0

    asyncio.run(scenario())


def test_timeout_only_replaces_timeout_token() -> None:
    """仅 ``{timeout}`` 被替换，文案中其余花括号原样保留。"""

    async def scenario() -> None:
        async def coro() -> int:
            await asyncio.sleep(10)
            return 1

        with pytest.raises(TimeoutError) as exc_info:
            await _await(
                coro(),
                timeout=0.05,
                timeout_message="超时 {timeout} {keep}",
                cancel_check=lambda: False,
            )
        assert str(exc_info.value) == "超时 0.05 {keep}"
        assert open_tasks_excluding_current() == 0

    asyncio.run(scenario())


def test_timeout_wins_over_cancel_when_deadline_first() -> None:
    """deadline 先到即抛 TimeoutError（先查超时再查取消的次序锁）。"""

    async def scenario() -> None:
        start = asyncio.get_running_loop().time()

        def late_cancel() -> bool:
            # 仅在 deadline（0.05s）之后（阈值 0.2s 留抖动余量）才命中取消；
            # 轮询在 deadline 前的若干次检查（elapsed < 0.05+ε）必须全为 False
            return (asyncio.get_running_loop().time() - start) > 0.2

        async def coro() -> int:
            await asyncio.sleep(10)
            return 1

        with pytest.raises(TimeoutError):
            await _await(
                coro(),
                timeout=0.05,
                cancel_check=late_cancel,
                poll_seconds=0.02,
            )
        assert open_tasks_excluding_current() == 0

    asyncio.run(scenario())


def test_poll_seconds_non_positive_falls_back() -> None:
    """``poll_seconds<=0`` 回退默认轮询间隔（0.5s），协程完成不被延误。"""

    async def scenario() -> Any:
        async def coro() -> str:
            await asyncio.sleep(0.01)
            return "ok"

        return await _await(
            coro(), timeout=1.0, cancel_check=lambda: False, poll_seconds=0
        )

    assert asyncio.run(scenario()) == "ok"
