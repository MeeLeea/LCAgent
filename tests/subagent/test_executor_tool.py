"""Executor Agent 工具（三个后台执行工具）的离线单元测试。

覆盖范围：
    - 注册：恰好 3 个 executor 工具、均为 BaseTool 且为协程并已进入 all_tools
    - start_executor_task：happy path、空任务、preflight 失败、spawn 失败、
      workspace 非法（项目根拒绝清单）
    - query_executor_task：排队/运行中提示、done 短路、结果缺失、失败/取消、
      缺失任务、wait_seconds 裁剪
    - cancel_executor_task：排队取消、缺失任务

所有测试均不触网、不拉起真实子进程（preflight/spawn 全部 monkeypatch）；
``asyncio.run`` 直接驱动异步工具。
"""
from __future__ import annotations

import asyncio
import inspect
import json
import os
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.tools import BaseTool

import tools as tools_pkg
from subagent.common.config import PROJECT_ROOT
from subagent.common.proc import LauncherError
from subagent.executor import tool as executor_tool
from subagent.executor.store import (
    STATUS_CANCELLED,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_RUNNING,
    ExecutorStore,
    resolve_workspace_at_start,
)

# 当前有效的三个 executor 工具名
EXECUTOR_TOOL_NAMES: tuple[str, ...] = (
    "start_executor_task",
    "query_executor_task",
    "cancel_executor_task",
)


@pytest.fixture
def executor_store(tmp_path: Path) -> Iterator[ExecutorStore]:
    """注入临时根目录的 ExecutorStore，并在测试结束后重置模块单例。"""
    store = ExecutorStore(str(tmp_path))
    executor_tool.configure(store)
    try:
        yield store
    finally:
        executor_tool._store = None


def _invoke(tool: BaseTool, args: dict[str, Any]) -> dict[str, Any]:
    """同步驱动单个异步工具调用并解析其 JSON 返回值。"""
    raw = asyncio.run(tool.ainvoke(args))
    assert isinstance(raw, str)
    return json.loads(raw)


def _task_dirs(store: ExecutorStore) -> list[str]:
    """列出存储根目录下的任务子目录名。"""
    root = Path(store.root)
    return [entry.name for entry in root.iterdir() if entry.is_dir()]


# ===================================================================== 注册


def test_exactly_three_executor_tools_registered() -> None:
    """恰好 3 个 executor 工具，均为 BaseTool 且协程，并已注册到 __all__/all_tools。"""
    for name in EXECUTOR_TOOL_NAMES:
        tool = getattr(tools_pkg, name)
        assert isinstance(tool, BaseTool)
        assert name in tools_pkg.__all__
        assert any(tool is candidate for candidate in tools_pkg.all_tools)
        arun = getattr(tool, "_arun", None)
        coroutine = getattr(tool, "coroutine", None)
        assert inspect.iscoroutinefunction(arun) or inspect.iscoroutinefunction(coroutine)

    executor_tools = [t for t in tools_pkg.all_tools if t.name in EXECUTOR_TOOL_NAMES]
    assert len(executor_tools) == 3


# ===================================================================== 启动


def test_start_happy_path(
    executor_store: ExecutorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """preflight 与 spawn 正常时返回 pid/queued，workspace 为隔离任务目录。"""
    monkeypatch.setattr(executor_tool, "preflight", lambda config: ("fake", "m1"))
    monkeypatch.setattr(
        executor_tool,
        "spawn_worker",
        lambda task_id, root=None, config=None: 4321,
    )

    result = _invoke(executor_tool.start_executor_task, {"task": "把模块实现掉"})

    assert result["success"] is True
    assert result["pid"] == 4321
    assert result["status"] == "queued"
    assert Path(result["workspace"]).name == "workspace"

    task = executor_store.get(result["task_id"])
    assert task is not None
    assert task["status"] == "queued"
    assert task["provider"] == "fake"
    assert task["model"] == "m1"
    assert task["workspace"] == result["workspace"]


def test_start_blank_task(executor_store: ExecutorStore) -> None:
    """空任务直接失败，且不创建任务。"""
    result = _invoke(executor_tool.start_executor_task, {"task": "   "})

    assert result["success"] is False
    assert _task_dirs(executor_store) == []


def test_start_preflight_value_error(
    executor_store: ExecutorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """preflight 抛 ValueError 时失败，不创建任务、不拉起子进程。"""
    spawn_calls = {"count": 0}

    def boom(config: object) -> tuple[str, str]:
        raise ValueError("缺少 API Key")

    def fake_spawn(task_id: str, root: str | None = None, config: object = None) -> int:
        spawn_calls["count"] += 1
        return 1

    monkeypatch.setattr(executor_tool, "preflight", boom)
    monkeypatch.setattr(executor_tool, "spawn_worker", fake_spawn)

    result = _invoke(executor_tool.start_executor_task, {"task": "需求"})

    assert result["success"] is False
    assert "API Key" in result["error"]
    assert result.get("message") == "未启动子进程"
    assert _task_dirs(executor_store) == []
    assert spawn_calls["count"] == 0


def test_start_invalid_workspace_marks_failed(
    executor_store: ExecutorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """workspace 命中拒绝清单（项目根）时任务标记 failed，不拉起子进程。"""
    spawn_calls = {"count": 0}

    def fake_spawn(task_id: str, root: str | None = None, config: object = None) -> int:
        spawn_calls["count"] += 1
        return 1

    monkeypatch.setattr(executor_tool, "preflight", lambda config: ("fake", "m1"))
    monkeypatch.setattr(executor_tool, "spawn_worker", fake_spawn)

    result = _invoke(executor_tool.start_executor_task, {
        "task": "需求", "workspace": PROJECT_ROOT,
    })

    assert result["success"] is False
    assert "workspace 非法" in result["error"]
    assert spawn_calls["count"] == 0
    task = executor_store.get(result["task_id"])
    assert task is not None
    assert task["status"] == STATUS_FAILED


def test_start_spawn_error_marks_task_failed(
    executor_store: ExecutorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """spawn_worker 抛 LauncherError 时任务被标记 failed。"""
    monkeypatch.setattr(executor_tool, "preflight", lambda config: ("fake", "m1"))

    def fake_spawn(task_id: str, root: str | None = None, config: object = None) -> int:
        raise LauncherError("启动子进程失败")

    monkeypatch.setattr(executor_tool, "spawn_worker", fake_spawn)

    result = _invoke(executor_tool.start_executor_task, {"task": "需求"})

    assert result["success"] is False
    task = executor_store.get(result["task_id"])
    assert task is not None
    assert task["status"] == STATUS_FAILED


# ===================================================================== 查询


def test_query_pending_tasks(executor_store: ExecutorStore) -> None:
    """queued/running 任务返回 done=False 并提示继续轮询、不要编造。"""
    queued_id = executor_store.create("排队任务")

    running_id = executor_store.create("运行任务")
    executor_store.update(running_id, status=STATUS_RUNNING, pid=os.getpid())

    for task_id in (queued_id, running_id):
        result = _invoke(
            executor_tool.query_executor_task, {"task_id": task_id, "wait_seconds": 0}
        )
        assert result["success"] is True
        assert result["done"] is False
        assert "轮询" in result["message"]
        assert "编造" in result["message"]

    running = _invoke(
        executor_tool.query_executor_task, {"task_id": running_id, "wait_seconds": 0}
    )
    assert running["pid"] == os.getpid()
    assert "elapsed_seconds" in running


def test_query_done_short_circuit(executor_store: ExecutorStore) -> None:
    """已完成任务：wait_seconds=999 也快速返回，返回结果正文与字符数。"""
    task_id = executor_store.create("任务")
    executor_store.write_text(task_id, "result.md", "执行结果正文")
    executor_store.update(task_id, status=STATUS_DONE, has_result=True)

    begin = time.perf_counter()
    result = _invoke(
        executor_tool.query_executor_task, {"task_id": task_id, "wait_seconds": 999}
    )
    elapsed = time.perf_counter() - begin

    assert elapsed < 0.5
    assert result["success"] is True
    assert result["done"] is True
    assert result["result"] == "执行结果正文"
    assert result["chars"] == len("执行结果正文")
    assert result["truncated"] is False


def test_query_done_missing_result(executor_store: ExecutorStore) -> None:
    """任务已完成但 result.md 缺失：success=False、done=True、提示结果文件缺失。"""
    task_id = executor_store.create("任务")
    executor_store.update(task_id, status=STATUS_DONE, has_result=True)

    result = _invoke(
        executor_tool.query_executor_task, {"task_id": task_id, "wait_seconds": 0}
    )

    assert result["success"] is False
    assert result["done"] is True
    assert "结果文件缺失" in result["error"]


def test_query_failed_or_cancelled(executor_store: ExecutorStore) -> None:
    """failed/cancelled 任务返回 success=False、done=False 并带错误信息。"""
    failed_id = executor_store.create("失败任务")
    executor_store.update(failed_id, status=STATUS_FAILED, error="boom")

    cancelled_id = executor_store.create("取消任务")
    executor_store.update(cancelled_id, status=STATUS_CANCELLED)

    for task_id in (failed_id, cancelled_id):
        result = _invoke(
            executor_tool.query_executor_task, {"task_id": task_id, "wait_seconds": 0}
        )
        assert result["success"] is False
        assert result["done"] is False
        assert result["error"]


def test_query_missing_task(executor_store: ExecutorStore) -> None:
    """查询不存在任务返回 success=False。"""
    result = _invoke(executor_tool.query_executor_task, {"task_id": "missing-id"})
    assert result["success"] is False
    assert "不存在" in result["error"]


def test_query_task_fields_returned_on_done(executor_store: ExecutorStore) -> None:
    """done 响应附带 workspace 与 steps_used 字段（workspace 从 task.json 读取）。"""
    task_id = executor_store.create("任务")
    resolve_workspace_at_start(executor_store, task_id, None)
    expected_workspace = executor_store.get(task_id)["workspace"]  # type: ignore[index]
    executor_store.write_text(task_id, "result.md", "结果")
    executor_store.update(task_id, status=STATUS_DONE, has_result=True, steps_used=7)

    result = _invoke(
        executor_tool.query_executor_task, {"task_id": task_id, "wait_seconds": 0}
    )

    assert result["steps_used"] == 7
    assert result["workspace"] == expected_workspace


def test_query_wait_seconds_clamped(
    executor_store: ExecutorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """wait_seconds 被裁剪到 30，总休眠不超过 30.5 秒且能够终止。"""
    task_id = executor_store.create("运行中任务")
    snapshot = executor_store.get(task_id)
    assert snapshot is not None
    running_task = dict(snapshot)
    running_task["status"] = STATUS_RUNNING
    running_task["pid"] = os.getpid()

    monkeypatch.setattr(executor_store, "reap", lambda tid: running_task)

    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    clock = {"now": 0.0}

    def fake_monotonic() -> float:
        clock["now"] += 1.0
        return clock["now"]

    monkeypatch.setattr(executor_tool, "asyncio", SimpleNamespace(sleep=fake_sleep))
    monkeypatch.setattr(executor_tool, "time", SimpleNamespace(monotonic=fake_monotonic))

    result = _invoke(
        executor_tool.query_executor_task, {"task_id": task_id, "wait_seconds": 999}
    )

    assert result["success"] is True
    assert result["done"] is False
    assert sleeps
    assert sum(sleeps) <= 30.5


# ===================================================================== 取消


def test_cancel_queued_and_missing(executor_store: ExecutorStore) -> None:
    """排队任务取消后为 cancelled；取消缺失任务返回失败。"""
    task_id = executor_store.create("任务")
    result = _invoke(executor_tool.cancel_executor_task, {"task_id": task_id})
    assert result["success"] is True
    assert result["status"] == STATUS_CANCELLED
    assert "不会被回滚" in result["message"]

    missing = _invoke(executor_tool.cancel_executor_task, {"task_id": "missing-id"})
    assert missing["success"] is False
    assert "不存在" in missing["error"]
