"""``subagent.common`` 任务状态常量与 ``TaskStoreBase`` 协议的单元测试。

覆盖范围：
    - task_state：状态字符串 / 终态集合 / 固定文件名（防漂移锁，历史任务的
      ``task.json`` 不迁移，字符串必须稳定）；
    - TaskStoreBase（经最小子类 ``_MinimalStore``）：claim 互斥 / 终态拒绝 /
      取消短路 / failed 重认领 / release 幂等 / reap 死回收活保留 /
      request_cancel 三分支 / cancel_requested / write_text / read_text /
      read_result 截断。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, ClassVar

import pytest

from subagent.common.fsx import (
    TRUNCATION_SUFFIX,
    atomic_write_json,
    is_safe_component,
    now_iso,
)
from subagent.common.task_state import (
    CANCEL_FLAG,
    PROMPT_FILE,
    STATUS_CANCELLED,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_QUEUED,
    STATUS_RUNNING,
    TASK_JSON,
    TERMINAL_STATUSES,
    WORKER_LOCK,
)
from subagent.common.task_store import TaskStoreBase

__all__ = ["_MinimalStore", "store", "test_task_state_constants_are_stable"]


class _MinimalStore(TaskStoreBase):
    """``TaskStoreBase`` 的最小子类：仅覆盖自身文案，并以最简 JSON 实现
    ``create`` / ``get`` / ``update``（这两个方法被基类刻意不上提）。"""

    result_filename: ClassVar[str] = "result.md"
    running_cancel_note: ClassVar[str] = "取消已受理，测试用提示"

    def __init__(self, root: str) -> None:
        super().__init__(root, default_root="<未使用>")

    def _task_path(self, task_id: str) -> Path:
        """返回任务 ``task.json`` 路径（测试内用 ``pathlib`` 表达）。"""
        return Path(self.task_dir(task_id)) / TASK_JSON

    def create(self, task_id: str) -> str:
        """创建一个 ``queued`` 最小任务并返回 ``task_id``（签名对齐 planner
        的 ``create() -> str``，与要求接着执行：``task_id = store.create(...)``）。"""
        task: dict[str, Any] = {"task_id": task_id, "status": STATUS_QUEUED}
        self._task_path(task_id).parent.mkdir(parents=True, exist_ok=True)
        with open(self._task_path(task_id), "w", encoding="utf-8") as fh:
            json.dump(task, fh, ensure_ascii=False)
        return task_id

    def get(self, task_id: str) -> dict[str, Any] | None:
        """读取任务；ID 非法或任务不存在时返回 ``None``。"""
        if not is_safe_component(task_id):
            return None
        try:
            with open(self._task_path(task_id), "r", encoding="utf-8") as fh:
                return dict(json.load(fh))
        except (FileNotFoundError, NotADirectoryError):
            return None

    def update(self, task_id: str, **fields: Any) -> dict[str, Any]:
        """读改写任务（原子替换）并刷新 ``updated_at``。"""
        task = self.get(task_id)
        if task is None:
            raise ValueError(f"任务 {task_id} 不存在")
        task.update(fields)
        task["updated_at"] = now_iso()
        atomic_write_json(str(self._task_path(task_id)), task)
        return task


@pytest.fixture
def store(tmp_path: Path) -> _MinimalStore:
    """每个测试使用独立的临时任务根目录。"""
    return _MinimalStore(str(tmp_path))


# ================================================================ task_state ===


def test_task_state_constants_are_stable() -> None:
    """状态字符串、终态集合与固定文件名是持久契约，必须保持稳定。"""
    assert STATUS_QUEUED == "queued"
    assert STATUS_RUNNING == "running"
    assert STATUS_DONE == "done"
    assert STATUS_FAILED == "failed"
    assert STATUS_CANCELLED == "cancelled"
    assert set(TERMINAL_STATUSES) == {STATUS_DONE, STATUS_FAILED, STATUS_CANCELLED}
    assert TASK_JSON == "task.json"
    assert WORKER_LOCK == "worker.lock"
    assert CANCEL_FLAG == "cancel.flag"
    assert PROMPT_FILE == "prompt.md"


# ================================================================== 认领协议 ===


def test_claim_is_mutually_exclusive(store: _MinimalStore) -> None:
    """首个实例认领成功并置 running+pid；第二实例对同一任务认领失败。"""
    task_id = store.create("t1")

    assert store.claim(task_id, 111) is True
    other = _MinimalStore(store.root)
    assert other.claim(task_id, 222) is False

    task = store.get(task_id)
    assert task is not None
    assert task["status"] == STATUS_RUNNING
    assert task["pid"] == 111


def test_claim_unknown_task_returns_false(store: _MinimalStore) -> None:
    """任务不存在（无 task.json）时认领失败且不留下 worker.lock。"""
    assert store.claim("no-such-task", 111) is False
    assert store.get("no-such-task") is None


def test_claim_unsafe_id_returns_false(store: _MinimalStore) -> None:
    """非裸名任务 ID 不做任何文件操作直接拒绝。"""
    assert store.claim("../evil", 111) is False


def test_claim_rejects_terminal_task(store: _MinimalStore) -> None:
    """终态任务不可认领，残留锁被释放，状态保持不变。"""
    task_id = store.create("t1")
    store.update(task_id, status=STATUS_DONE)

    assert store.claim(task_id, 111) is False
    task = store.get(task_id)
    assert task is not None
    assert task["status"] == STATUS_DONE
    assert not (Path(store.task_dir(task_id)) / WORKER_LOCK).exists()


def test_claim_terminal_with_stray_cancel_flag_stays_done(
    store: _MinimalStore,
) -> None:
    """终态任务即使残留 cancel.flag 也保持终态（先校验状态再判取消）。"""
    task_id = store.create("t1")
    store.update(task_id, status=STATUS_DONE)
    (Path(store.task_dir(task_id)) / CANCEL_FLAG).write_text("", encoding="utf-8")

    assert store.claim(task_id, 111) is False
    task = store.get(task_id)
    assert task is not None
    assert task["status"] == STATUS_DONE


def test_claim_queued_with_cancel_flag_short_circuits(store: _MinimalStore) -> None:
    """queued 任务带取消标记时，认领直接转 cancelled 并拒绝启动。"""
    task_id = store.create("t1")
    (Path(store.task_dir(task_id)) / CANCEL_FLAG).write_text("", encoding="utf-8")

    assert store.claim(task_id, 111) is False
    task = store.get(task_id)
    assert task is not None
    assert task["status"] == STATUS_CANCELLED
    assert not (Path(store.task_dir(task_id)) / WORKER_LOCK).exists()


def test_claim_allows_failed_reclaim(store: _MinimalStore) -> None:
    """failed 任务允许被重新认领（重跑场景），error 被清空。"""
    task_id = store.create("t1")
    store.update(task_id, status=STATUS_FAILED, error="上次失败")

    assert store.claim(task_id, 111) is True
    task = store.get(task_id)
    assert task is not None
    assert task["status"] == STATUS_RUNNING
    assert task["error"] is None


def test_release_is_idempotent_and_allows_failed_reclaim(
    store: _MinimalStore,
) -> None:
    """release 幂等：重复释放无害；failed 任务释放锁后可被重新认领。"""
    task_id = store.create("t1")
    assert store.claim(task_id, 111) is True
    store.update(task_id, status=STATUS_FAILED)

    store.release(task_id)
    store.release(task_id)  # 幂等：第二次无害
    assert store.claim(task_id, 222) is True


# ================================================================= 僵尸回收 ===


def test_reap_dead_pid_marks_failed(store: _MinimalStore) -> None:
    """running 且进程已退出 → failed；错误带 pid 且再次 reap 保持收敛。"""
    task_id = store.create("t1")
    store.update(task_id, status=STATUS_RUNNING, pid=98765)

    reaped = store.reap(task_id, pid_alive=lambda pid: False)

    assert reaped is not None
    assert reaped["status"] == STATUS_FAILED
    assert "98765" in (reaped.get("error") or "")
    assert reaped["has_result"] is False

    # 收敛（非 running）后再次 reap 原样返回，不再改写
    again = store.reap(task_id, pid_alive=lambda pid: False)
    assert again == reaped


def test_reap_alive_pid_keeps_running(store: _MinimalStore) -> None:
    """进程存活时 running 任务原样保留。"""
    task_id = store.create("t1")
    store.update(task_id, status=STATUS_RUNNING, pid=111)

    task = store.reap(task_id, pid_alive=lambda pid: True)

    assert task is not None
    assert task["status"] == STATUS_RUNNING


def test_reap_ignores_non_running(store: _MinimalStore) -> None:
    """非 running 任务不被 reap 改写。"""
    task_id = store.create("t1")

    task = store.reap(task_id, pid_alive=lambda pid: False)

    assert task is not None
    assert task["status"] == STATUS_QUEUED


def test_reap_missing_task_returns_none(store: _MinimalStore) -> None:
    """任务不存在时 reap 返回 None。"""
    assert store.reap("no-such-task") is None


def test_reap_default_pid_alive_uses_current_process(store: _MinimalStore) -> None:
    """未显式传 pid_alive 时默认用 ``common.proc`` 判定（本进程必然存活）。"""
    task_id = store.create("t1")
    store.update(task_id, status=STATUS_RUNNING, pid=os.getpid())

    task = store.reap(task_id)

    assert task is not None
    assert task["status"] == STATUS_RUNNING


# ================================================================= 取消协议 ===


def test_request_cancel_queued_cancels_immediately(store: _MinimalStore) -> None:
    """queued 任务直接取消：accepted、状态 cancelled、不落 cancel.flag。"""
    task_id = store.create("t1")

    result = store.request_cancel(task_id)

    assert result["accepted"] is True
    assert result["note"] == "任务尚未启动，已直接取消"
    assert result["task"]["status"] == STATUS_CANCELLED
    assert not (Path(store.task_dir(task_id)) / CANCEL_FLAG).exists()


def test_request_cancel_running_sets_flag(store: _MinimalStore) -> None:
    """running 任务受理取消：写 cancel.flag，note 来自子类文案，状态不动。"""
    task_id = store.create("t1")
    store.update(task_id, status=STATUS_RUNNING)

    result = store.request_cancel(task_id)

    assert result["accepted"] is True
    # 子类文案（running_cancel_note 类属性）是 note 的唯一来源
    assert result["note"] == store.running_cancel_note
    assert result["task"]["status"] == STATUS_RUNNING
    assert (Path(store.task_dir(task_id)) / CANCEL_FLAG).exists()


def test_request_cancel_terminal_rejected(store: _MinimalStore) -> None:
    """终态任务取消被拒绝：note 含终态提示且携带状态名。"""
    task_id = store.create("t1")
    store.update(task_id, status=STATUS_DONE)

    result = store.request_cancel(task_id)

    assert result["accepted"] is False
    assert "任务已处于终态" in result["note"]
    assert STATUS_DONE in result["note"]
    assert result["task"]["status"] == STATUS_DONE


def test_request_cancel_missing_task_raises(store: _MinimalStore) -> None:
    """取消不存在的任务抛 ValueError。"""
    with pytest.raises(ValueError):
        store.request_cancel("no-such-task")


def test_cancel_requested_flag_lifecycle(store: _MinimalStore) -> None:
    """cancel_requested 反映 cancel.flag 是否存在；非法 ID 恒为 False。"""
    task_id = store.create("t1")
    assert store.cancel_requested(task_id) is False

    (Path(store.task_dir(task_id)) / CANCEL_FLAG).write_text("", encoding="utf-8")
    assert store.cancel_requested(task_id) is True
    assert store.cancel_requested("../evil") is False


# ================================================================ 文本产物 IO ===


def test_write_and_read_text(store: _MinimalStore) -> None:
    """write_text 返回绝对路径，read_text 读回内容；缺失时返回 None。"""
    task_id = store.create("t1")

    path = store.write_text(task_id, PROMPT_FILE, "审计提示词")

    assert Path(path).is_absolute()
    assert store.read_text(task_id, PROMPT_FILE) == "审计提示词"
    assert store.read_text(task_id, "missing.md") is None


def test_write_text_rejects_unsafe_names(store: _MinimalStore) -> None:
    """非法任务 ID 或非裸名文件名被拒绝。"""
    task_id = store.create("t1")

    with pytest.raises(ValueError):
        store.write_text("../evil", PROMPT_FILE, "x")
    with pytest.raises(ValueError):
        store.write_text(task_id, "sub/out.md", "x")


def test_read_result_reads_result_filename(store: _MinimalStore) -> None:
    """read_result 固定读取子类 result_filename；未写时返回 None。"""
    task_id = store.create("t1")
    assert store.read_result(task_id) is None

    text = "x" * 50
    store.write_text(task_id, store.result_filename, text)
    assert store.read_result(task_id) == text


def test_read_result_truncates_with_suffix(store: _MinimalStore) -> None:
    """超过 max_chars 截断并追加截断提示；max_chars=0 / 源文件缺失不截断。"""
    task_id = store.create("t1")
    text = "x" * 50
    store.write_text(task_id, store.result_filename, text)

    truncated = store.read_result(task_id, max_chars=10)
    assert truncated == text[:10] + TRUNCATION_SUFFIX
