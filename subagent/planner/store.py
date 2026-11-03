"""Planner 子程序的文件系统任务存储（一次性方案设计）。

并发模型（关键）：

- **单写者**：每个任务目录同一时刻只有一个 worker 进程持锁写入；写者通过
  ``worker.lock``（``O_CREAT|O_EXCL`` 原子创建）独占认领，读线程与写线程解耦。
- **原子替换**：``task.json`` 永远是「写临时文件 + ``os.replace``」的方式整体替换，
  读者只会看到替换前或替换后的完整 JSON，绝不会读到半截文件。
- **取消标记**：取消通过 ``cancel.flag`` 空文件表达，由 worker 在 LLM 调用结束后检查。

本模块只依赖标准库与 ``subagent.planner.config``，不在模块级导入
``langchain`` / ``team`` 等重依赖，避免主进程的工具导入链被拉重。
"""
from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable
from datetime import datetime
from typing import Any

from subagent.planner.config import DEFAULT_OUTPUT_ROOT

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
        "requirement", "context", "status", "created_at", "updated_at", "started_at",
        "finished_at", "pid", "error", "has_result", "provider", "model",
    }
)

_TRUNCATION_SUFFIX: str = "\n\n...[已截断]"


def _now() -> str:
    """返回带本地时区信息的 ISO 时间戳（秒级精度）。"""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def new_task_id() -> str:
    """生成任务 ID，格式 ``pl-YYYYmmdd-HHMMSS-<8 位十六进制>``。"""
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    return f"pl-{stamp}-{uuid.uuid4().hex[:8]}"


def _is_safe_component(value: str) -> bool:
    """判断 ``value`` 是否为安全的单层文件名/目录名片段（拒绝 ``..`` 与路径分隔符）。"""
    if not value or value in (".", "..") or ".." in value:
        return False
    for sep in (os.sep, os.altsep, "/", "\\"):
        if sep and sep in value:
            return False
    return True


def _validate_bare_name(name: str) -> None:
    """校验 ``name`` 为裸文件名；非法时抛出 ``ValueError``。"""
    if not _is_safe_component(name):
        raise ValueError(f"非法文件名: {name!r}")


def _atomic_write_json(path: str, data: dict[str, Any]) -> None:
    """原子写入 JSON 文件：同目录临时文件 + ``os.replace``。"""
    tmp = f"{path}.tmp-{uuid.uuid4().hex}"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _atomic_write_text(path: str, text: str) -> None:
    """原子写入文本文件：同目录临时文件 + ``os.replace``。"""
    tmp = f"{path}.tmp-{uuid.uuid4().hex}"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _load_task_json(path: str) -> dict[str, Any] | None:
    """读取 task.json；文件缺失、非文件或 JSON 损坏时返回 ``None``。"""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            loaded = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(loaded, dict):
        return None
    return loaded


class PlannerStore:
    """Planner 任务的文件系统存储；``<root>/<task_id>/task.json`` 为唯一事实源。"""

    def __init__(self, root: str | None = None) -> None:
        """初始化存储；``root`` 为 ``None`` 时使用默认输出根目录。"""
        root = root or DEFAULT_OUTPUT_ROOT
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

    def create(
        self,
        requirement: str,
        context: str = "",
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> str:
        """创建一个排队中的新任务并返回任务 ID；``requirement`` 为空时抛 ``ValueError``。"""
        if not requirement.strip():
            raise ValueError("requirement 不能为空")

        task_id = new_task_id()
        now = _now()
        data: dict[str, Any] = {
            "task_id": task_id,
            "requirement": requirement,
            "context": context,
            "status": STATUS_QUEUED,
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "finished_at": None,
            "pid": None,
            "error": None,
            "has_result": False,
            "provider": provider,
            "model": model,
        }
        os.makedirs(self.task_dir(task_id), exist_ok=True)
        _atomic_write_json(self._task_json_path(task_id), data)
        return task_id

    def get(self, task_id: str) -> dict[str, Any] | None:
        """读取任务；ID 非法、任务不存在或 JSON 损坏时返回 ``None``。"""
        if not _is_safe_component(task_id):
            return None
        return _load_task_json(self._task_json_path(task_id))

    def update(self, task_id: str, **fields: Any) -> dict[str, Any]:
        """读改写任务（原子替换）并刷新 ``updated_at``；未知字段抛 ``ValueError``。"""
        if not _is_safe_component(task_id):
            raise ValueError(f"非法任务 ID: {task_id!r}")
        unknown = set(fields) - _EDITABLE_FIELDS
        if unknown:
            raise ValueError(f"未知字段: {', '.join(sorted(unknown))}")

        path = self._task_json_path(task_id)
        with open(path, "r", encoding="utf-8") as fh:
            current = json.load(fh)
        if not isinstance(current, dict):
            raise TypeError(f"任务 {task_id} 的 task.json 格式非法")

        current.update(fields)
        current["updated_at"] = _now()
        _atomic_write_json(path, current)
        return current

    def claim(self, task_id: str, pid: int) -> bool:
        """以独占 ``worker.lock`` 认领任务并置为运行态；成功返回 ``True``。"""
        if not _is_safe_component(task_id):
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
        if not _is_safe_component(task_id):
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
            from subagent.planner.launcher import pid_is_alive

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
                "note": "取消已受理，子程序将在当前 LLM 调用结束后退出",
                "task": task,
            }

        return {
            "accepted": False,
            "note": f"任务已处于终态: {status}",
            "task": task,
        }

    def cancel_requested(self, task_id: str) -> bool:
        """判断取消标记 ``cancel.flag`` 是否存在。"""
        if not _is_safe_component(task_id):
            return False
        return os.path.exists(self._cancel_flag_path(task_id))

    def write_text(self, task_id: str, name: str, text: str) -> str:
        """原子写入任务目录下的文本文件并返回绝对路径；文件名须为裸名。"""
        if not _is_safe_component(task_id):
            raise ValueError(f"非法任务 ID: {task_id!r}")
        _validate_bare_name(name)
        path = os.path.join(self.task_dir(task_id), name)
        _atomic_write_text(path, text)
        return path

    def read_text(self, task_id: str, name: str) -> str | None:
        """读取任务目录下的文本文件；文件不存在时返回 ``None``。"""
        if not _is_safe_component(task_id):
            return None
        _validate_bare_name(name)
        path = os.path.join(self.task_dir(task_id), name)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return fh.read()
        except (FileNotFoundError, NotADirectoryError):
            return None

    def read_result(self, task_id: str, max_chars: int = 0) -> str | None:
        """读取最终方案 ``plan.md``；不存在返回 ``None``，超 ``max_chars`` 截断并追加提示。"""
        text = self.read_text(task_id, "plan.md")
        if text is None:
            return None
        if max_chars > 0 and len(text) > max_chars:
            return text[:max_chars] + _TRUNCATION_SUFFIX
        return text


__all__ = [
    "STATUS_CANCELLED",
    "STATUS_DONE",
    "STATUS_FAILED",
    "STATUS_QUEUED",
    "STATUS_RUNNING",
    "TERMINAL_STATUSES",
    "PlannerStore",
    "new_task_id",
]
