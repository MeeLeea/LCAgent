"""Planner 子程序的文件系统任务存储（一次性方案设计）。

并发模型（关键；协议方法统一继承自 :class:`~subagent.common.task_store.TaskStoreBase`）：

- **单写者**：每个任务目录同一时刻只有一个 worker 进程持锁写入；写者通过
  ``worker.lock``（``O_CREAT|O_EXCL`` 原子创建）独占认领，读线程与写线程解耦。
- **原子替换**：``task.json`` 永远是「写临时文件 + ``os.replace``」的方式整体替换，
  读者只会看到替换前或替换后的完整 JSON，绝不会读到半截文件。
- **取消标记**：取消通过 ``cancel.flag`` 空文件表达，由 worker 在 LLM 调用结束后检查。

本模块只保留 planner 专属部分：``create`` / ``get`` / ``update``（字段契约）与
``new_task_id``（``pl-`` 前缀）；``get`` / ``update`` / ``create`` **刻意不上提**到
公共基类——planner 的 ``update`` 用裸 ``json.load``（损坏 JSON 抛
``JSONDecodeError``），语义与 executor 不同，收敛会丢掉这一测试锁定的行为。

本模块只依赖标准库与 ``subagent.planner.config``，不在模块级导入
``langchain`` / ``team`` 等重依赖，避免主进程的工具导入链被拉重。
"""
from __future__ import annotations

import json
import os
from typing import Any, ClassVar

from subagent.common.fsx import (
    atomic_write_json,
    is_safe_component,
    load_task_json,
    now_iso,
)
from subagent.common.fsx import (
    new_task_id as _fsx_new_task_id,
)
from subagent.common.task_state import (
    STATUS_CANCELLED,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_QUEUED,
    STATUS_RUNNING,
    TERMINAL_STATUSES,
)
from subagent.common.task_store import TaskStoreBase
from subagent.planner.config import DEFAULT_OUTPUT_ROOT

# task.json 中允许通过 update 修改的字段（task_id 为不可变主键，不在此列）
_EDITABLE_FIELDS: frozenset[str] = frozenset(
    {
        "requirement", "context", "status", "created_at", "updated_at", "started_at",
        "finished_at", "pid", "error", "has_result", "provider", "model",
    }
)


def new_task_id() -> str:
    """生成任务 ID，格式 ``pl-YYYYmmdd-HHMMSS-<8 位十六进制>``。"""
    return _fsx_new_task_id("pl")


class PlannerStore(TaskStoreBase):
    """Planner 任务的文件系统存储；``<root>/<task_id>/task.json`` 为唯一事实源。"""

    # 终稿文件名（``read_result`` 固定读取）
    result_filename: ClassVar[str] = "plan.md"

    # ``status=running`` 时的取消提示文案（``request_cancel`` 使用；唯一文案差异点）
    running_cancel_note: ClassVar[str] = "取消已受理，子程序将在当前 LLM 调用结束后退出"

    def __init__(self, root: str | None = None) -> None:
        """初始化存储；``root`` 为 ``None`` 时使用默认输出根目录。"""
        super().__init__(root or DEFAULT_OUTPUT_ROOT, default_root=DEFAULT_OUTPUT_ROOT)

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
        now = now_iso()
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
        atomic_write_json(self._task_json_path(task_id), data)
        return task_id

    def get(self, task_id: str) -> dict[str, Any] | None:
        """读取任务；ID 非法、任务不存在或 JSON 损坏时返回 ``None``。"""
        if not is_safe_component(task_id):
            return None
        return load_task_json(self._task_json_path(task_id))

    def update(self, task_id: str, **fields: Any) -> dict[str, Any]:
        """读改写任务（原子替换）并刷新 ``updated_at``；未知字段抛 ``ValueError``。

        刻意沿用裸 ``json.load``：损坏的 task.json 在 update 场景下应抛
        ``JSONDecodeError`` 暴露写端异常，而非像 ``get`` 那样静默返回 ``None``——
        该行为被子类测试锁定（基类不实现 ``update`` 以保留两个子程序的差异）。
        """
        if not is_safe_component(task_id):
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
        current["updated_at"] = now_iso()
        atomic_write_json(path, current)
        return current


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
