"""Planner Agent 工具（三个一次性工具）的离线单元测试。

覆盖范围：
    - 注册：恰好 3 个 planner 工具、均为 BaseTool 且为协程、旧工具名已移除
    - start_planner_task：happy path、空需求、preflight 失败、spawn 失败
    - query_planner_task：排队/运行中提示、completed 短路、方案缺失、
      失败/取消、缺失任务、wait_seconds 裁剪
    - cancel_planner_task：排队取消、缺失任务

所有测试均不触网、不拉起真实子进程；``asyncio.run`` 直接驱动异步工具。
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
from subagent.planner import tool as planner_tool
from subagent.planner.launcher import LauncherError
from subagent.planner.store import (
    STATUS_CANCELLED,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_RUNNING,
    PlannerStore,
)

# 当前有效的三个 planner 工具名
PLANNER_TOOL_NAMES: tuple[str, ...] = (
    "start_planner_task",
    "query_planner_task",
    "cancel_planner_task",
)
# 一次性重写后已被移除的旧工具名
REMOVED_TOOL_NAMES: tuple[str, ...] = (
    "get_planner_task_status",
    "get_planner_task_result",
    "list_planner_tasks",
)


@pytest.fixture
def planner_store(tmp_path: Path) -> Iterator[PlannerStore]:
    """注入临时根目录的 PlannerStore，并在测试结束后重置模块单例。"""
    store = PlannerStore(str(tmp_path))
    planner_tool.configure(store)
    try:
        yield store
    finally:
        planner_tool._store = None


def _invoke(tool: BaseTool, args: dict[str, Any]) -> dict[str, Any]:
    """同步驱动单个异步工具调用并解析其 JSON 返回值。"""
    raw = asyncio.run(tool.ainvoke(args))
    assert isinstance(raw, str)
    return json.loads(raw)


def _task_dirs(store: PlannerStore) -> list[str]:
    """列出存储根目录下的任务子目录名。"""
    root = Path(store.root)
    return [entry.name for entry in root.iterdir() if entry.is_dir()]


# ===================================================================== 注册


def test_exactly_three_planner_tools_registered() -> None:
    """恰好 3 个 planner 工具，均为 BaseTool 且协程；旧工具名已移除。"""
    for name in PLANNER_TOOL_NAMES:
        tool = getattr(tools_pkg, name)
        assert isinstance(tool, BaseTool)
        assert name in tools_pkg.__all__
        assert any(tool is candidate for candidate in tools_pkg.all_tools)
        arun = getattr(tool, "_arun", None)
        coroutine = getattr(tool, "coroutine", None)
        assert inspect.iscoroutinefunction(arun) or inspect.iscoroutinefunction(coroutine)

    planner_tools = [t for t in tools_pkg.all_tools if t.name in PLANNER_TOOL_NAMES]
    assert len(planner_tools) == 3

    registered_names = {t.name for t in tools_pkg.all_tools}
    for removed in REMOVED_TOOL_NAMES:
        assert not hasattr(tools_pkg, removed)
        assert removed not in tools_pkg.__all__
        assert removed not in registered_names


# ===================================================================== 启动


def test_start_happy_path(planner_store: PlannerStore, monkeypatch: pytest.MonkeyPatch) -> None:
    """preflight 与 spawn 正常时返回 pid、状态 queued 且任务已入库。"""
    monkeypatch.setattr(planner_tool, "preflight", lambda config: ("fake", "m1"))
    monkeypatch.setattr(
        planner_tool,
        "spawn_worker",
        lambda task_id, root=None, config=None: 4321,
    )

    result = _invoke(planner_tool.start_planner_task, {"requirement": "做一个完整方案"})

    assert result["success"] is True
    assert result["pid"] == 4321
    assert result["status"] == "queued"
    task = planner_store.get(result["task_id"])
    assert task is not None
    assert task["status"] == "queued"


def test_start_blank_requirement(planner_store: PlannerStore) -> None:
    """空需求直接失败，且不创建任务。"""
    result = _invoke(planner_tool.start_planner_task, {"requirement": "   "})

    assert result["success"] is False
    assert _task_dirs(planner_store) == []


def test_start_preflight_value_error(
    planner_store: PlannerStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """preflight 抛 ValueError 时失败，不创建任务、不拉起子进程。"""
    spawn_calls = {"count": 0}

    def boom(config: object) -> tuple[str, str]:
        raise ValueError("缺少 API Key")

    def fake_spawn(task_id: str, root: str | None = None, config: object = None) -> int:
        spawn_calls["count"] += 1
        return 1

    monkeypatch.setattr(planner_tool, "preflight", boom)
    monkeypatch.setattr(planner_tool, "spawn_worker", fake_spawn)

    result = _invoke(planner_tool.start_planner_task, {"requirement": "需求"})

    assert result["success"] is False
    assert "API Key" in result["error"]
    assert _task_dirs(planner_store) == []
    assert spawn_calls["count"] == 0


def test_start_spawn_error_marks_task_failed(
    planner_store: PlannerStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """spawn_worker 抛 LauncherError 时任务被标记 failed。"""
    monkeypatch.setattr(planner_tool, "preflight", lambda config: ("fake", "m1"))

    def fake_spawn(task_id: str, root: str | None = None, config: object = None) -> int:
        raise LauncherError("启动子进程失败")

    monkeypatch.setattr(planner_tool, "spawn_worker", fake_spawn)

    result = _invoke(planner_tool.start_planner_task, {"requirement": "需求"})

    assert result["success"] is False
    task = planner_store.get(result["task_id"])
    assert task is not None
    assert task["status"] == STATUS_FAILED


# ===================================================================== 查询


def test_query_pending_tasks(planner_store: PlannerStore) -> None:
    """queued/running 任务返回 done=False 并提示继续轮询、不要编造。"""
    queued_id = planner_store.create("排队需求")

    running_id = planner_store.create("运行需求")
    planner_store.update(running_id, status=STATUS_RUNNING, pid=os.getpid())

    for task_id in (queued_id, running_id):
        result = _invoke(
            planner_tool.query_planner_task, {"task_id": task_id, "wait_seconds": 0}
        )
        assert result["success"] is True
        assert result["done"] is False
        assert "轮询" in result["message"]
        assert "编造" in result["message"]


def test_query_done_short_circuit(planner_store: PlannerStore) -> None:
    """已完成任务：wait_seconds=999 也快速返回，返回方案正文与字符数。"""
    task_id = planner_store.create("需求")
    planner_store.write_text(task_id, "plan.md", "最终方案正文")
    planner_store.update(task_id, status=STATUS_DONE, has_result=True)

    begin = time.perf_counter()
    result = _invoke(
        planner_tool.query_planner_task, {"task_id": task_id, "wait_seconds": 999}
    )
    elapsed = time.perf_counter() - begin

    assert elapsed < 0.5
    assert result["success"] is True
    assert result["done"] is True
    assert result["plan"] == "最终方案正文"
    assert result["chars"] == len("最终方案正文")
    assert result["truncated"] is False


def test_query_done_missing_plan(planner_store: PlannerStore) -> None:
    """任务已完成但 plan.md 缺失：success=False、done=True、提示方案文件缺失。"""
    task_id = planner_store.create("需求")
    planner_store.update(task_id, status=STATUS_DONE, has_result=True)

    result = _invoke(
        planner_tool.query_planner_task, {"task_id": task_id, "wait_seconds": 0}
    )

    assert result["success"] is False
    assert result["done"] is True
    assert "方案文件缺失" in result["error"]


def test_query_failed_or_cancelled(planner_store: PlannerStore) -> None:
    """failed/cancelled 任务返回 success=False、done=False 并带错误信息。"""
    failed_id = planner_store.create("失败需求")
    planner_store.update(failed_id, status=STATUS_FAILED, error="boom")

    cancelled_id = planner_store.create("取消需求")
    planner_store.update(cancelled_id, status=STATUS_CANCELLED)

    for task_id in (failed_id, cancelled_id):
        result = _invoke(
            planner_tool.query_planner_task, {"task_id": task_id, "wait_seconds": 0}
        )
        assert result["success"] is False
        assert result["done"] is False
        assert result["error"]


def test_query_missing_task(planner_store: PlannerStore) -> None:
    """查询不存在任务返回 success=False。"""
    result = _invoke(planner_tool.query_planner_task, {"task_id": "missing-id"})
    assert result["success"] is False
    assert "不存在" in result["error"]


def test_query_wait_seconds_clamped(
    planner_store: PlannerStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """wait_seconds 被裁剪到 30，总休眠不超过 30.5 秒且能够终止。"""
    task_id = planner_store.create("运行中需求")
    snapshot = planner_store.get(task_id)
    assert snapshot is not None
    running_task = dict(snapshot)
    running_task["status"] = STATUS_RUNNING
    running_task["pid"] = os.getpid()

    monkeypatch.setattr(planner_store, "reap", lambda tid: running_task)

    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    clock = {"now": 0.0}

    def fake_monotonic() -> float:
        clock["now"] += 1.0
        return clock["now"]

    monkeypatch.setattr(planner_tool, "asyncio", SimpleNamespace(sleep=fake_sleep))
    monkeypatch.setattr(planner_tool, "time", SimpleNamespace(monotonic=fake_monotonic))

    result = _invoke(
        planner_tool.query_planner_task, {"task_id": task_id, "wait_seconds": 999}
    )

    assert result["success"] is True
    assert result["done"] is False
    assert sleeps
    assert sum(sleeps) <= 30.5


# ===================================================================== 取消


def test_cancel_queued_and_missing(planner_store: PlannerStore) -> None:
    """排队任务取消后为 cancelled；取消缺失任务返回失败。"""
    task_id = planner_store.create("需求")
    result = _invoke(planner_tool.cancel_planner_task, {"task_id": task_id})
    assert result["success"] is True
    assert result["status"] == STATUS_CANCELLED

    missing = _invoke(planner_tool.cancel_planner_task, {"task_id": "missing-id"})
    assert missing["success"] is False
    assert "不存在" in missing["error"]
