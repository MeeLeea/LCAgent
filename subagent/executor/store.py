"""Executor 子程序的文件系统任务存储（多步执行，``result.md`` 为最终交付）。

并发模型（与 ``subagent.planner.store`` 相同；协议方法统一继承自
:class:`~subagent.common.task_store.TaskStoreBase`）：

- **单写者**：每个任务目录同一时刻只有一个 worker 进程持锁写入；写者通过
  ``worker.lock``（``O_CREAT|O_EXCL`` 原子创建）独占认领。
- **原子替换**：``task.json`` 永远是「写临时文件 + ``os.replace``」整体替换。
- **取消标记**：取消通过 ``cancel.flag`` 空文件表达，由 worker 在阶段边界检查。

与 planner 的差异：
- 任务字段增加 ``task`` / ``plan`` / ``context`` / ``workspace`` / ``steps_used``
  / ``max_steps`` / ``marker_present``；
- 任务 ID 前缀为 ``ex-``；
- 最终产物为 ``result.md``（``read_result`` 固定读取）；
- ``update`` 在 task.json 损坏/缺失时抛 ``ValueError``（经 ``load_task_json``
  判空转换；planner 用裸 ``json.load`` 抛 ``JSONDecodeError``）——因此
  ``get`` / ``update`` / ``create`` 不上提到公共基类。

此外提供 workspace 相关模块级助手（``resolve_workspace_at_start`` / 工作区
互斥锁 / ``tool_path_error``）。本模块只依赖标准库与 ``subagent.common``，
不在模块级导入 ``langchain`` / ``tools`` 等重依赖，避免主进程的工具导入链被拉重。
"""
from __future__ import annotations

import os
from typing import Any, ClassVar

from subagent.common.fsx import (
    atomic_write_json,
    is_safe_component,
    load_task_json,
    new_task_id,
    now_iso,
)
from subagent.common.paths import check_contained, normalize_path
from subagent.common.task_state import (
    STATUS_CANCELLED,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_QUEUED,
    STATUS_RUNNING,
    TERMINAL_STATUSES,
)
from subagent.common.task_store import TaskStoreBase

# task.json 中允许通过 update 修改的字段（task_id 为不可变主键，不在此列）
_EDITABLE_FIELDS: frozenset[str] = frozenset(
    {
        "task", "plan", "context", "workspace", "status", "created_at", "updated_at",
        "started_at", "finished_at", "pid", "error", "has_result", "marker_present",
        "steps_used", "max_steps", "provider", "model",
    }
)

# 默认输出根目录：<项目根>/data/subagent/executor（store.py 位于 subagent/executor/ 下）
_DEFAULT_OUTPUT_ROOT: str = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "subagent",
    "executor",
)


class ExecutorStore(TaskStoreBase):
    """Executor 任务的文件系统存储；``<root>/<task_id>/task.json`` 为唯一事实源。"""

    # 终稿文件名（``read_result`` 固定读取）
    result_filename: ClassVar[str] = "result.md"

    # ``status=running`` 时的取消提示文案（``request_cancel`` 使用；唯一文案差异点）
    running_cancel_note: ClassVar[str] = (
        "取消已受理，子程序将在阶段边界（下一次模型调用前）退出；"
        "工作区已写入的文件不会被回滚"
    )

    def __init__(self, root: str | None = None) -> None:
        """初始化存储；``root`` 为 ``None`` 时使用默认输出根目录。"""
        super().__init__(root or _DEFAULT_OUTPUT_ROOT, default_root=_DEFAULT_OUTPUT_ROOT)

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
        now = now_iso()
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
        current["updated_at"] = now_iso()
        atomic_write_json(path, current)
        return current


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
