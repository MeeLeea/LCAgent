"""Planner 文件系统任务存储（PlannerStore）的离线单元测试。

覆盖范围（对应一次性方案设计实现契约）：
    - create：完整契约字段写入、默认值、provider/model 透传、空白需求校验
    - get：读取一致性、缺失任务、目录穿越等非法 ID 返回 None
    - update：已知字段合并 + updated_at 刷新、未知字段拒绝
    - claim：跨实例独占、已有 worker.lock、终态残留 cancel.flag、排队取消标记
    - release：幂等、释放后 failed 任务可再次认领
    - reap：进程退出回收、存活不回收、非 running 不回收、缺失任务返回 None
    - request_cancel：排队/运行/终态/缺失任务语义
    - cancel_requested：取消标记生命周期
    - write_text / read_text / read_result：文本往返、裸文件名校验、结果截断

全部用例离线运行，仅使用 ``tmp_path`` 存放文件系统状态。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from subagent.planner.store import (
    STATUS_CANCELLED,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_QUEUED,
    STATUS_RUNNING,
    PlannerStore,
)

# ================================================================ Fixtures ===


@pytest.fixture
def store(tmp_path: Path) -> PlannerStore:
    """每个测试使用独立的临时根目录，避免共享状态。"""
    return PlannerStore(root=str(tmp_path / "planner"))


def _lock_path(store: PlannerStore, task_id: str) -> str:
    """返回任务的 worker 锁文件路径。"""
    return os.path.join(store.task_dir(task_id), "worker.lock")


def _cancel_flag_path(store: PlannerStore, task_id: str) -> str:
    """返回任务的取消标记文件路径。"""
    return os.path.join(store.task_dir(task_id), "cancel.flag")


def _task_json_path(store: PlannerStore, task_id: str) -> str:
    """返回任务的 task.json 路径。"""
    return os.path.join(store.task_dir(task_id), "task.json")


def _rewrite_field(store: PlannerStore, task_id: str, key: str, value: object) -> None:
    """直接改写 task.json 的单个字段，用于构造底层测试前置状态。"""
    path = _task_json_path(store, task_id)
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    data[key] = value
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False)


# ================================================================ create ===


def test_create_writes_contract_fields_and_defaults(store: PlannerStore) -> None:
    """create 写入完整契约字段，初始状态 queued 且默认值为 None/False。"""
    task_id = store.create(requirement="设计一个缓存模块", context="上下文说明")
    task = store.get(task_id)
    assert task is not None
    assert task["task_id"] == task_id
    assert task["requirement"] == "设计一个缓存模块"
    assert task["context"] == "上下文说明"
    assert task["status"] == STATUS_QUEUED
    assert isinstance(task["created_at"], str) and task["created_at"]
    assert isinstance(task["updated_at"], str) and task["updated_at"]
    assert task["started_at"] is None
    assert task["finished_at"] is None
    assert task["pid"] is None
    assert task["error"] is None
    assert task["has_result"] is False
    assert task["provider"] is None
    assert task["model"] is None

    # * 关键字参数 provider/model 会被原样写入任务
    explicit_id = store.create("需求", provider="fake", model="m1")
    explicit = store.get(explicit_id)
    assert explicit is not None
    assert explicit["provider"] == "fake"
    assert explicit["model"] == "m1"


def test_create_rejects_blank_requirement(store: PlannerStore) -> None:
    """空白需求抛出 ValueError。"""
    for requirement in ("", "   ", "\t\n "):
        with pytest.raises(ValueError):
            store.create(requirement=requirement)


# ================================================================ get ===


def test_get_returns_same_dict_and_missing_none(store: PlannerStore) -> None:
    """get 两次返回一致的字典；缺失任务返回 None。"""
    task_id = store.create("需求")
    first = store.get(task_id)
    second = store.get(task_id)
    assert first is not None and second is not None
    assert first == second
    assert store.get("pl-not-exist") is None


def test_get_rejects_path_traversal_ids(store: PlannerStore) -> None:
    """目录穿越/非法任务 ID 返回 None。"""
    for bad_id in ("../x", "a/b", ".."):
        assert store.get(bad_id) is None


# ================================================================ update ===


def test_update_merges_known_fields_and_refreshes_updated_at(store: PlannerStore) -> None:
    """update 合并已知字段并刷新 updated_at（即便底层值被人为改旧）。"""
    task_id = store.create("需求")
    _rewrite_field(store, task_id, "updated_at", "2000-01-01T00:00:00+00:00")

    updated = store.update(task_id, status=STATUS_RUNNING, pid=123)
    assert updated["status"] == STATUS_RUNNING
    assert updated["pid"] == 123
    assert updated["updated_at"] != "2000-01-01T00:00:00+00:00"

    reread = store.get(task_id)
    assert reread is not None
    assert reread["status"] == STATUS_RUNNING


def test_update_unknown_field_raises(store: PlannerStore) -> None:
    """未知字段抛出 ValueError。"""
    task_id = store.create("需求")
    with pytest.raises(ValueError):
        store.update(task_id, not_a_field=1)


# ================================================================ claim / release ===


def test_claim_is_exclusive_across_instances(store: PlannerStore) -> None:
    """两个实例对同一任务认领：仅第一个成功，且留下 worker.lock。"""
    task_id = store.create("需求")
    other = PlannerStore(root=store.root)
    assert store.claim(task_id, 111) is True
    assert other.claim(task_id, 222) is False
    assert os.path.exists(_lock_path(store, task_id))


def test_claim_with_existing_lock_returns_false(store: PlannerStore) -> None:
    """已存在 worker.lock 时认领失败（O_EXCL），任务状态保持不变。"""
    task_id = store.create("需求")
    Path(_lock_path(store, task_id)).write_text("held", encoding="utf-8")

    assert store.claim(task_id, 111) is False
    task = store.get(task_id)
    assert task is not None
    assert task["status"] == STATUS_QUEUED


def test_claim_terminal_with_stray_cancel_flag_stays_done(store: PlannerStore) -> None:
    """终态（done）任务即使残留 cancel.flag 也不可认领，且状态保持 done。"""
    task_id = store.create("需求")
    store.update(task_id, status=STATUS_DONE, has_result=True)
    Path(_cancel_flag_path(store, task_id)).write_text("", encoding="utf-8")

    assert store.claim(task_id, 111) is False
    task = store.get(task_id)
    assert task is not None
    assert task["status"] == STATUS_DONE
    assert task["has_result"] is True


def test_claim_queued_with_cancel_flag_cancels(store: PlannerStore) -> None:
    """queued 任务的取消标记存在时，认领转为 cancelled 并返回 False。"""
    task_id = store.create("需求")
    Path(_cancel_flag_path(store, task_id)).write_text("", encoding="utf-8")

    assert store.claim(task_id, 111) is False
    task = store.get(task_id)
    assert task is not None
    assert task["status"] == STATUS_CANCELLED
    assert task["finished_at"] is not None
    assert not os.path.exists(_lock_path(store, task_id))


def test_release_is_idempotent_and_allows_failed_reclaim(store: PlannerStore) -> None:
    """release 幂等；释放后 failed 任务可被另一实例再次认领。"""
    task_id = store.create("需求")
    other = PlannerStore(root=store.root)
    assert store.claim(task_id, 111) is True
    store.release(task_id)
    store.release(task_id)  # 第二次释放不抛异常
    assert not os.path.exists(_lock_path(store, task_id))

    store.update(task_id, status=STATUS_FAILED)
    assert other.claim(task_id, 222) is True


# ================================================================ reap ===


def test_reap_dead_pid_marks_failed(store: PlannerStore) -> None:
    """进程已退出的 running 任务被标记为 failed。"""
    task_id = store.create("需求")
    assert store.claim(task_id, 111) is True

    reaped = store.reap(task_id, pid_alive=lambda _pid: False)
    assert reaped is not None
    assert reaped["status"] == STATUS_FAILED
    assert "worker 进程已退出" in (reaped["error"] or "")
    assert reaped["finished_at"] is not None


def test_reap_alive_pid_keeps_running(store: PlannerStore) -> None:
    """进程存活时 running 任务保持不变。"""
    task_id = store.create("需求")
    assert store.claim(task_id, 111) is True

    reaped = store.reap(task_id, pid_alive=lambda _pid: True)
    assert reaped is not None
    assert reaped["status"] == STATUS_RUNNING


def test_reap_ignores_non_running_and_missing(store: PlannerStore) -> None:
    """非 running 状态（done/queued）即使进程已死也不回收；缺失任务返回 None。"""
    for status in (STATUS_DONE, STATUS_QUEUED):
        task_id = store.create("需求")
        store.update(task_id, status=status)

        reaped = store.reap(task_id, pid_alive=lambda _pid: False)
        assert reaped is not None
        assert reaped["status"] == status

    assert store.reap("pl-missing", pid_alive=lambda _pid: False) is None


# ================================================================ request_cancel ===


def test_request_cancel_state_semantics(store: PlannerStore) -> None:
    """排队直接取消、运行写标记保持 running、终态拒绝受理。"""
    queued_id = store.create("排队需求")
    queued = store.request_cancel(queued_id)
    assert queued["accepted"] is True
    assert queued["task"]["status"] == STATUS_CANCELLED
    assert not os.path.exists(_cancel_flag_path(store, queued_id))

    running_id = store.create("运行需求")
    assert store.claim(running_id, 111) is True
    running = store.request_cancel(running_id)
    assert running["accepted"] is True
    assert os.path.exists(_cancel_flag_path(store, running_id))
    running_task = store.get(running_id)
    assert running_task is not None
    assert running_task["status"] == STATUS_RUNNING

    done_id = store.create("完成需求")
    store.update(done_id, status=STATUS_DONE)
    done = store.request_cancel(done_id)
    assert done["accepted"] is False
    assert "终态" in done["note"]


def test_request_cancel_missing_raises(store: PlannerStore) -> None:
    """取消缺失任务抛出 ValueError。"""
    with pytest.raises(ValueError):
        store.request_cancel("pl-missing")


def test_cancel_requested_flag_lifecycle(store: PlannerStore) -> None:
    """cancel_requested 初始为 False，对运行任务取消后为 True。"""
    task_id = store.create("需求")
    assert store.cancel_requested(task_id) is False

    assert store.claim(task_id, 111) is True
    store.request_cancel(task_id)
    assert store.cancel_requested(task_id) is True


# ================================================================ 文本读写 ===


def test_write_read_text_roundtrip_with_chinese(store: PlannerStore) -> None:
    """write_text / read_text 往返一致，含中文。"""
    task_id = store.create("需求")
    content = "方案正文：包含中文与换行\n第二行"
    path = store.write_text(task_id, "notes.md", content)
    assert os.path.isfile(path)
    assert store.read_text(task_id, "notes.md") == content


def test_write_text_rejects_non_bare_filenames(store: PlannerStore) -> None:
    """非法裸文件名（含分隔符或 ..）抛出 ValueError。"""
    task_id = store.create("需求")
    for bad_name in ("a/b.md", "../x"):
        with pytest.raises(ValueError):
            store.write_text(task_id, bad_name, "x")


def test_read_result_missing_full_and_truncated(store: PlannerStore) -> None:
    """plan.md 缺失返回 None；存在时返回全文；超 max_chars 截断并追加标记。"""
    missing_id = store.create("需求")
    assert store.read_result(missing_id) is None

    task_id = store.create("需求")
    content = "abcdefghij" * 10
    store.write_text(task_id, "plan.md", content)
    assert store.read_result(task_id) == content

    truncated = store.read_result(task_id, max_chars=10)
    assert truncated is not None
    assert truncated.startswith(content[:10])
    assert truncated.endswith("...[已截断]")
