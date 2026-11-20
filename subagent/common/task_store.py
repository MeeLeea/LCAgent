"""后台子程序共享的任务存储基类（文件布局 + 认领/回收/取消协议）。

``claim`` / ``release`` / ``reap`` / ``request_cancel`` / ``cancel_requested`` /
``write_text`` / ``read_text`` / ``read_result`` 与路径辅助在 ``PlannerStore``
与 ``ExecutorStore`` 中逐字节等价，统一收口到这里；子类职责只剩：

- ``create`` / ``get`` / ``update`` —— 字段集与 JSON 损坏语义不同，**刻意不上提**
  （planner 的 ``update`` 用裸 ``json.load``，损坏 JSON 抛 ``JSONDecodeError``；
  executor 抛 ``ValueError``），由子类自行实现；
- 两个类属性：``result_filename``（终稿文件名，``read_result`` 固定读取）与
  ``running_cancel_note``（``running`` 状态下的取消提示文案，唯一文案差异点）。

并发模型（不得改变）：``worker.lock`` 以 ``O_CREAT|O_EXCL`` 原子创建独占认领；
``task.json`` 永远走「临时文件 + ``os.replace``」原子替换；``update`` 的原子性
依赖「单写者持锁」不变式，基类保持「先持锁后读改写」的顺序。

本模块只依赖标准库（经 ``subagent.common``）；禁止导入 ``langchain`` /
``tools`` 等重依赖，也**禁止反向 import** ``planner`` / ``executor``。
"""
from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, ClassVar

from subagent.common.fsx import (
    atomic_write_text,
    is_safe_component,
    now_iso,
    truncate_text,
    validate_bare_name,
)
from subagent.common.proc import pid_is_alive
from subagent.common.task_state import (
    CANCEL_FLAG,
    STATUS_CANCELLED,
    STATUS_FAILED,
    STATUS_QUEUED,
    STATUS_RUNNING,
    TASK_JSON,
    WORKER_LOCK,
)

_PROMPT_NOTE_TERMINAL: str = "任务已处于终态: {status}"


class TaskStoreBase:
    """任务存储骨架；``<root>/<task_id>/task.json`` 为唯一事实源。"""

    # 子类必须覆盖：终稿文件名（``read_result`` 固定读取，如 plan.md / result.md）
    result_filename: ClassVar[str]

    # 子类可覆盖：``status=running`` 时的取消提示文案
    running_cancel_note: ClassVar[str] = "取消已受理，子程序将在任务边界后退出"

    # 僵尸回收的默认存活性判定（``subagent.common.proc``，与 planner 原实现语义一致；
    # 用 staticmethod 防止经实例访问时被绑定成方法）
    _pid_alive: ClassVar[Callable[[int | None], bool]] = staticmethod(pid_is_alive)

    def __init__(self, root: str | None, *, default_root: str) -> None:
        """初始化存储；``root`` 为 ``None`` 时使用 ``default_root``。"""
        root = root or default_root
        os.makedirs(root, exist_ok=True)
        self.root: str = os.path.normpath(root)

    # ---------------------------------------------------------------- 路径辅助

    def task_dir(self, task_id: str) -> str:
        """返回指定任务目录的绝对路径。"""
        return os.path.join(self.root, task_id)

    def _task_json_path(self, task_id: str) -> str:
        """返回任务 ``task.json`` 的路径。"""
        return os.path.join(self.task_dir(task_id), TASK_JSON)

    def _lock_path(self, task_id: str) -> str:
        """返回 ``worker.lock`` 路径。"""
        return os.path.join(self.task_dir(task_id), WORKER_LOCK)

    def _cancel_flag_path(self, task_id: str) -> str:
        """返回 ``cancel.flag`` 路径。"""
        return os.path.join(self.task_dir(task_id), CANCEL_FLAG)

    # ------------------------------------------------- 子类必须实现（读改写协议）

    def get(self, task_id: str) -> dict[str, Any] | None:
        """读取任务；ID 非法、任务不存在时返回 ``None``（子类实现）。"""
        raise NotImplementedError

    def update(self, task_id: str, **fields: Any) -> dict[str, Any]:
        """读改写任务（原子替换）并刷新 ``updated_at``（子类实现）。"""
        raise NotImplementedError

    # ------------------------------------------------------------ 独占认领协议

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
                    finished_at=now_iso(),
                    has_result=False,
                )
                self.release(task_id)
                return False

            self.update(
                task_id,
                status=STATUS_RUNNING,
                pid=pid,
                started_at=now_iso(),
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
            pid_alive = self._pid_alive

        pid = task.get("pid")
        if not pid_alive(pid):
            return self.update(
                task_id,
                status=STATUS_FAILED,
                error=f"worker 进程已退出（pid={pid}）",
                finished_at=now_iso(),
                has_result=False,
            )

        return task

    # -------------------------------------------------------------- 取消协议

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
                finished_at=now_iso(),
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
                "note": self.running_cancel_note,
                "task": task,
            }

        return {
            "accepted": False,
            "note": _PROMPT_NOTE_TERMINAL.format(status=status),
            "task": task,
        }

    def cancel_requested(self, task_id: str) -> bool:
        """判断取消标记 ``cancel.flag`` 是否存在。"""
        if not is_safe_component(task_id):
            return False
        return os.path.exists(self._cancel_flag_path(task_id))

    # ------------------------------------------------------------- 文本产物 IO

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
        """读取终稿文件（``result_filename``）；超 ``max_chars`` 截断并追加提示。"""
        text = self.read_text(task_id, self.result_filename)
        if text is None:
            return None
        return truncate_text(text, max_chars)


__all__ = ["TaskStoreBase"]
