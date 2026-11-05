"""Executor 子程序的文件系统任务存储（多步执行，``result.md`` 为最终交付）。

并发模型（与 ``subagent.planner.store`` 相同）：

- **单写者**：每个任务目录同一时刻只有一个 worker 进程持锁写入；写者通过
  ``worker.lock``（``O_CREAT|O_EXCL`` 原子创建）独占认领。
- **原子替换**：``task.json`` 永远是「写临时文件 + ``os.replace``」整体替换。
- **取消标记**：取消通过 ``cancel.flag`` 空文件表达，由 worker 在阶段边界检查。

与 planner 的差异：
- 任务字段增加 ``task`` / ``plan`` / ``context`` / ``workspace`` / ``steps_used``
  / ``max_steps`` / ``marker_present``；
- 任务 ID 前缀为 ``ex-``；
- 最终产物为 ``result.md``（``read_result`` 固定读取）。

本模块只依赖标准库与 ``subagent.common``，不在模块级导入 ``langchain`` / ``tools````
等重依赖，避免主进程的工具导入链被拉重。
"""
from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from subagent.common.fsx import (
    atomic_write_json,
    atomic_write_text,
    is_safe_component,
    load_task_json,
    new_task_id,
    now_iso,
    truncate_text,
    validate_bare_name,
)
from subagent.common.paths import check_contained, normalize_path

# 任务状态常量
STATUS_QUEUED: str = "queued"
STATUS_RUNNING: str = "running"
STATUS_DONE: str = "done"
STATUS_FAILED: str = "failed"
STATUS_CANCELLED: str = "cancelled"

# 终态集合：处于这些状态的任务不再被 worker 认领
TERMINAL_STATUSES: frozenset[str] = frozenset(
    {STATUS_DONE, STATUS_FAILED, STATUS_CANCELLED}
)

# task.json 中允许通过 update 修改的字段（task_id 为不可变主键，不在此列）
_EDITABLE_FIELDS: frozenset[str] = frozenset(
    {
        "task", "plan", "context", "workspace", "status", "created_at", "updated_at",
        "started_at", "finished_at", "pid", "error", "has_result", "marker_present",
        "steps_used", "max_steps", "provider", "model",
    }
)


def _now() -> str:
    """返回带本地时区信息的 ISO 时间戳（秒级精度）。"""
    return now_iso()


class ExecutorStore:
    """Executor 任务的文件系统存储；``<root>/<task_id>/task.json`` 为唯一事实源。"""

    def __init__(self, root: str | None = None) -> None:
        """初始化存储；``root`` 为 ``None`` 时使用默认输出根目录。"""
        root = root or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data",
            "subagent",
            "executor",
        )
        os.makedirs(root, exist_ok=True)
        self.root: str = os.path.normpath(root)

    def task_dir(self, task_id: str) -> str:
        """返回指定任务目录的绝对路径。"""
        return os.path.join(self.root, task_id)

    def _task_json_path(self, task_id: str) -> str:
        """返回任务 ``task.json`` 的路径。"""
        return os.path.join(self.task_dir(task_id), "task.json")

    def _lock_path(self, task_id: str) -> str:
        """返回 ``worker.lock`` 路径。"""
        return os.path.join(self.task_dir(task_id), "worker.lock")

    def _cancel_flag_path(self, task_id: str) -> str:
        """返回 ``cancel.flag`` 路径。"""
        return os.path.join(self.task_dir(task_id), "cancel.flag")

    def new_task_id(self) -> str:
        """生成新的 executor 任务 ID（``ex-`` 前缀）。"""
        return new_task_id("ex")

    def create(
        self,
        task: str,
        plan: str = "",
        context: str = "",
        workspace: str = "",
        *,
        max_steps: int = 24,
        provider: str | None = None,
        model: str | None = None,
    ) -> str:
        """创建一个排队中的新任务并返回任务 ID；``task`` 为空时抛 ``ValueError``。"""
        if not task.strip():
            raise ValueError("task 不能为空")

        task_id = self.new_task_id()
        now = _now()
        data: dict[str, Any] = {
            "task_id": task_id,
            "task": task,
            "plan": plan,
            "context": context,
            "workspace": workspace,
            "status": STATUS_QUEUED,
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "finished_at": None,
            "pid": None,
            "error": None,
            "has_result": False,
            "marker_present": False,
            "steps_used": 0,
            "max_steps": max_steps,
            "provider": provider,
            "model": model,
        }
        os.makedirs(self.task_dir(task_id), exist_ok=True)
        atomic_write_json(self._task_json_path(task_id), data)
        return task_id

    def get(self, task_id: str) -> dict[str, Any] | None:
        """读取任务；ID 非法、任务不存在或 JSON 损坏时返回 ``None``。"""
        if not is_safe_component(task_id):
            return None
        return load_task_json(self._task_json_path(task_id))

    def update(self, task_id: str, **fields: Any) -> dict[str, Any]:
        """读改写任务（原子替换）并刷新 ``updated_at``；未知字段抛 ``ValueError``。"""
        if not is_safe_component(task_id):
            raise ValueError(f"非法任务 ID: {task_id!r}")
        unknown = set(fields) - _EDITABLE_FIELDS
        if unknown:
            raise ValueError(f"未知字段: {', '.join(sorted(unknown))}")

        path = self._task_json_path(task_id)
        current = load_task_json(path)
        if current is None:
            raise ValueError(f"任务 {task_id} 不存在或 task.json 损坏")

        current.update(fields)
        current["updated_at"] = _now()
        atomic_write_json(path, current)
        return current

    def claim(self, task_id: str, pid: int) -> bool:
        """以独占 ``worker.lock`` 认领任务并置为运行态；成功返回 ``True``。"""
        if not is_safe_component(task_id):
            return False

        try:
            fd = os.open(self._lock_path(task_id), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except (FileExistsError, FileNotFoundError):
            return False
        os.close(fd)

        try:
            task = self.get(task_id)
            if task is None:
                self.release(task_id)
                return False

            # 先校验状态再做取消判定：终态任务的残留 cancel.flag 不得改写其状态。
            if task.get("status") not in (STATUS_QUEUED, STATUS_FAILED):
                self.release(task_id)
                return False

            if self.cancel_requested(task_id):
                self.update(
                    task_id,
                    status=STATUS_CANCELLED,
                    finished_at=_now(),
                    has_result=False,
                )
                self.release(task_id)
                return False

            self.update(
                task_id,
                status=STATUS_RUNNING,
                pid=pid,
                started_at=_now(),
                error=None,
            )
            return True
        except BaseException:
            self.release(task_id)
            raise

    def release(self, task_id: str) -> None:
        """释放任务锁（幂等，删除 ``worker.lock``）。"""
        if not is_safe_component(task_id):
            return
        try:
            os.remove(self._lock_path(task_id))
        except (FileNotFoundError, NotADirectoryError):
            return

    def reap(
        self,
        task_id: str,
        pid_alive: Callable[[int | None], bool] | None = None,
    ) -> dict[str, Any] | None:
        """回收僵尸任务：仅 ``running`` 且进程已退出时置为 ``failed``，否则原样返回。"""
        task = self.get(task_id)
        if task is None:
            return None
        if task.get("status") != STATUS_RUNNING:
            return task

        if pid_alive is None:
            # 惰性导入：保持 store 轻量并规避模块循环
            from subagent.common.proc import pid_is_alive

            pid_alive = pid_is_alive

        pid = task.get("pid")
        if not pid_alive(pid):
            return self.update(
                task_id,
                status=STATUS_FAILED,
                error=f"worker 进程已退出（pid={pid}）",
                finished_at=_now(),
                has_result=False,
            )

        return task

    def request_cancel(self, task_id: str) -> dict[str, Any]:
        """请求取消任务，返回 ``accepted`` / ``note`` / ``task``；任务不存在抛 ``ValueError``。"""
        task = self.get(task_id)
        if task is None:
            raise ValueError(f"任务 {task_id} 不存在")

        status = task.get("status")
        if status == STATUS_QUEUED:
            updated = self.update(
                task_id,
                status=STATUS_CANCELLED,
                finished_at=_now(),
                has_result=False,
            )
            return {
                "accepted": True,
                "note": "任务尚未启动，已直接取消",
                "task": updated,
            }

        if status == STATUS_RUNNING:
            fd = os.open(self._cancel_flag_path(task_id), os.O_CREAT | os.O_WRONLY, 0o644)
            os.close(fd)
            return {
                "accepted": True,
                "note": "取消已受理，子程序将在阶段边界（下一次模型调用前）退出；"
                "工作区已写入的文件不会被回滚",
                "task": task,
            }

        return {
            "accepted": False,
            "note": f"任务已处于终态: {status}",
            "task": task,
        }

    def cancel_requested(self, task_id: str) -> bool:
        """判断取消标记 ``cancel.flag`` 是否存在。"""
        if not is_safe_component(task_id):
            return False
        return os.path.exists(self._cancel_flag_path(task_id))

    def write_text(self, task_id: str, name: str, text: str) -> str:
        """原子写入任务目录下的文本文件并返回绝对路径；文件名须为裸名。"""
        if not is_safe_component(task_id):
            raise ValueError(f"非法任务 ID: {task_id!r}")
        validate_bare_name(name)
        path = os.path.join(self.task_dir(task_id), name)
        atomic_write_text(path, text)
        return path

    def read_text(self, task_id: str, name: str) -> str | None:
        """读取任务目录下的文本文件；文件不存在时返回 ``None``。"""
        if not is_safe_component(task_id):
            return None
        validate_bare_name(name)
        path = os.path.join(self.task_dir(task_id), name)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return fh.read()
        except (FileNotFoundError, NotADirectoryError):
            return None

    def read_result(self, task_id: str, max_chars: int = 0) -> str | None:
        """读取最终结果 ``result.md``；不存在返回 ``None``，超 ``max_chars`` 截断并追加提示。"""
        text = self.read_text(task_id, "result.md")
        if text is None:
            return None
        return truncate_text(text, max_chars)


def resolve_workspace_at_start(
    store: ExecutorStore,
    task_id: str,
    workspace: str | None,
) -> str:
    """在主进程 ``start`` 阶段解析 workspace 并写回 task.json（存归一化绝对路径）。

    Args:
        store: 任务存储。
        task_id: 任务 ID（任务目录必须已由 ``create`` 创建）。
        workspace: 用户指定的工作区；``None``/空 → 使用任务目录内的独立 ``workspace/``。

    Returns:
        归一化（realpath+normcase）后的绝对路径。

    Raises:
        ValueError: 显式 workspace 非法（不存在 / 非目录 / 命中拒绝清单）。
    """
    from subagent.executor.workspace import validate_explicit_workspace

    if workspace is None or not workspace.strip():
        resolved = os.path.join(store.task_dir(task_id), "workspace")
        os.makedirs(resolved, exist_ok=True)
    else:
        resolved = validate_explicit_workspace(workspace)

    store.update(task_id, workspace=resolved)
    return resolved


def workspace_lock_path(workspace: str) -> str:
    """返回工作区并发互斥锁文件路径（``.lcagent-executor.lock``）。"""
    return os.path.join(normalize_path(workspace), ".lcagent-executor.lock")


def acquire_workspace_lock(workspace: str) -> bool:
    """获取工作区互斥锁（``O_CREAT|O_EXCL``）；已被占用时返回 ``False``。"""
    lock_path = workspace_lock_path(workspace)
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except (FileExistsError, FileNotFoundError, NotADirectoryError):
        return False
    os.close(fd)
    return True


def release_workspace_lock(workspace: str) -> None:
    """释放工作区互斥锁（幂等）。"""
    try:
        os.remove(workspace_lock_path(workspace))
    except (FileNotFoundError, NotADirectoryError):
        return


def tool_path_error(workspace: str, target: str, base: str | None = None) -> str | None:
    """worker 工具调用前的路径包含校验；返回错误信息或 ``None``。"""
    try:
        check_contained(target, workspace, base=base)
    except ValueError as exc:
        return str(exc)
    return None


__all__ = [
    "STATUS_CANCELLED",
    "STATUS_DONE",
    "STATUS_FAILED",
    "STATUS_QUEUED",
    "STATUS_RUNNING",
    "TERMINAL_STATUSES",
    "ExecutorStore",
    "acquire_workspace_lock",
    "release_workspace_lock",
    "resolve_workspace_at_start",
    "tool_path_error",
    "workspace_lock_path",
]
