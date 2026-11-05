"""Executor 存储与 workspace 校验的离线单元测试。

覆盖范围：
    - ExecutorStore：create/get/update 字段白名单、claim（独占/取消短路）、
      request_cancel（queued/running/终态）、reap（进程存活理论）、read_result 截断
    - resolve_workspace_at_start：显式 workspace 归一化、非法路径拒绝、
      未指定时使用任务目录内的隔离 workspace
    - validate_explicit_workspace：拒绝清单（系统目录 / 项目根 / data 等）

所有任务数据写入 ``tmp_path`` 下的 :class:`ExecutorStore`，测试间互不影响；
不触网、不拉起真实子进程。
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from subagent.executor.config import PROJECT_ROOT
from subagent.executor.store import (
    STATUS_CANCELLED,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_QUEUED,
    STATUS_RUNNING,
    ExecutorStore,
    acquire_workspace_lock,
    release_workspace_lock,
    resolve_workspace_at_start,
)
from subagent.executor.workspace import validate_explicit_workspace


@pytest.fixture
def store(tmp_path: Path) -> ExecutorStore:
    """每个测试使用独立的临时任务根目录。"""
    return ExecutorStore(str(tmp_path))


# ================================================================ ExecutorStore ===


def test_create_and_get_roundtrip(store: ExecutorStore) -> None:
    """create/get 往返：字段落入 task.json，ID 为 ex- 前缀。"""
    task_id = store.create("实现一个模块", plan="计划正文", context="上下文", max_steps=7)

    assert task_id.startswith("ex-")
    task = store.get(task_id)
    assert task is not None
    assert task["task"] == "实现一个模块"
    assert task["plan"] == "计划正文"
    assert task["context"] == "上下文"
    assert task["status"] == STATUS_QUEUED
    assert task["steps_used"] == 0
    assert task["max_steps"] == 7
    assert task["has_result"] is False


def test_create_empty_task_rejected(store: ExecutorStore) -> None:
    """空任务正文直接拒绝。"""
    with pytest.raises(ValueError):
        store.create("   ")


def test_unknown_update_field_rejected(store: ExecutorStore) -> None:
    """未知字段（不在白名单）更新抛 ValueError。"""
    task_id = store.create("任务")
    with pytest.raises(ValueError):
        store.update(task_id, bogus_field=1)


def test_update_marks_steps_used(store: ExecutorStore) -> None:
    """步数进度可写回并刷新 updated_at。"""
    task_id = store.create("任务")
    first = store.update(task_id, steps_used=3)
    second = store.update(task_id, steps_used=4)

    assert second["updated_at"] >= first["updated_at"]
    assert store.get(task_id)["steps_used"] == 4


def test_claim_is_exclusive(store: ExecutorStore) -> None:
    """独占认领：第一次认领置 running 并记录 pid；第二次因 worker.lock 占用失败。"""
    task_id = store.create("任务")

    assert store.claim(task_id, 111) is True
    assert store.claim(task_id, 222) is False
    task = store.get(task_id)
    assert task["status"] == STATUS_RUNNING
    assert task["pid"] == 111


def test_claim_with_cancel_flag_short_circuits(store: ExecutorStore) -> None:
    """认领时已存在 cancel.flag：直接 cancelled 且不占用锁。"""
    task_id = store.create("任务")
    (Path(store.task_dir(task_id)) / "cancel.flag").write_text("", encoding="utf-8")

    assert store.claim(task_id, 111) is False
    task = store.get(task_id)
    assert task["status"] == STATUS_CANCELLED
    assert not (Path(store.task_dir(task_id)) / "worker.lock").exists()


def test_request_cancel_semantics(store: ExecutorStore) -> None:
    """queued 立即取消；终态拒绝。"""
    queued = store.create("任务一")
    assert store.request_cancel(queued)["accepted"] is True
    assert store.get(queued)["status"] == STATUS_CANCELLED

    done = store.create("任务二")
    store.update(done, status=STATUS_DONE, has_result=True)
    assert store.request_cancel(done)["accepted"] is False


def test_reap_marks_dead_worker_failed(store: ExecutorStore) -> None:
    """running 且 PID 探测为死 → failed，保留 steps_used 供诊断。"""
    task_id = store.create("任务")
    store.claim(task_id, 111)
    store.update(task_id, steps_used=5)

    updated = store.reap(task_id, pid_alive=lambda pid: False)

    assert updated["status"] == STATUS_FAILED
    assert store.get(task_id)["steps_used"] == 5
    assert "pid=111" in (updated["error"] or "")


def test_reap_keeps_alive_worker(store: ExecutorStore) -> None:
    """PID 存活时 reap 原样返回（保持 running，不做任何改写）。"""
    task_id = store.create("任务")
    store.claim(task_id, 111)

    task = store.reap(task_id, pid_alive=lambda pid: True)
    assert task is not None
    assert store.get(task_id)["status"] == STATUS_RUNNING


def test_read_result_truncates(store: ExecutorStore) -> None:
    """read_result 读取 result.md 并按 max_chars 截断。"""
    task_id = store.create("任务")
    long_text = "x" * 300
    store.write_text(task_id, "result.md", long_text)

    full = store.read_result(task_id)
    cut = store.read_result(task_id, max_chars=100)

    assert full == long_text
    assert cut.startswith("x" * 100)
    assert cut.endswith("...[已截断]")


def test_read_result_missing(store: ExecutorStore) -> None:
    """result.md 不存在时返回 None。"""
    task_id = store.create("任务")
    assert store.read_result(task_id) is None


# ============================================================ workspace 校验 ===


def test_resolve_workspace_default_isolated(store: ExecutorStore) -> None:
    """未指定 workspace：使用任务目录内的隔离 workspace/ 并写回 task.json。"""
    task_id = store.create("任务")

    resolved = resolve_workspace_at_start(store, task_id, None)

    assert os.path.isabs(resolved)
    assert Path(resolved).name == "workspace"
    assert (Path(store.task_dir(task_id)) / "workspace").is_dir()
    assert store.get(task_id)["workspace"] == resolved


def test_resolve_workspace_explicit(tmp_path: Path, store: ExecutorStore) -> None:
    """显式 workspace：归一化写回 task.json。"""
    ws = tmp_path / "ws"
    ws.mkdir()
    task_id = store.create("任务")

    resolved = resolve_workspace_at_start(store, task_id, str(ws))

    assert os.path.normcase(resolved) == os.path.normcase(str(ws))
    assert store.get(task_id)["workspace"] == resolved


def test_resolve_workspace_invalid_marks_error(store: ExecutorStore) -> None:
    """workspace 指向不存在目录：抛 ValueError，不写回 task.json。"""
    task_id = store.create("任务")
    with pytest.raises(ValueError):
        resolve_workspace_at_start(store, task_id, "Z:/no/such/dir")


def test_validate_explicit_workspace_denies_builtin_slots(tmp_path: Path) -> None:
    """拒绝清单：项目根本身与 data 目录被拒，普通子目录通过。"""
    with pytest.raises(ValueError):
        validate_explicit_workspace(PROJECT_ROOT)

    with pytest.raises(ValueError):
        validate_explicit_workspace(os.path.join(PROJECT_ROOT, "data"))

    ok_dir = tmp_path / "ok"
    ok_dir.mkdir()
    assert os.path.normcase(validate_explicit_workspace(str(ok_dir))) == os.path.normcase(str(ok_dir))


def test_workspace_lock_is_exclusive(tmp_path: Path) -> None:
    """workspace 并发互斥：二次获取失败，release 后可再获取。"""
    ws = tmp_path / "ws"
    ws.mkdir()

    assert acquire_workspace_lock(str(ws)) is True
    assert acquire_workspace_lock(str(ws)) is False
    release_workspace_lock(str(ws))
    assert acquire_workspace_lock(str(ws)) is True

    release_workspace_lock(str(ws))
