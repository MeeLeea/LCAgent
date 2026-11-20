"""后台子程序任务状态常量与固定文件名（planner / executor 共享）。

状态机（``queued → running → done/failed/cancelled``）与终态集合对两个子程序
语义一致；``task.json`` / ``worker.lock`` / ``cancel.flag`` / ``prompt.md`` 等
固定文件名也在此集中定义，避免后续子程序复制时漂移。

注意：``task.json`` 中的状态是唯一事实源，状态字符串表示必须保持稳定
（历史任务目录里的旧 ``task.json`` 不会被迁移）。
"""
from __future__ import annotations

# 任务状态
STATUS_QUEUED: str = "queued"
STATUS_RUNNING: str = "running"
STATUS_DONE: str = "done"
STATUS_FAILED: str = "failed"
STATUS_CANCELLED: str = "cancelled"

# 终态集合：处于这些状态的任务不再被 worker 认领
TERMINAL_STATUSES: frozenset[str] = frozenset(
    {STATUS_DONE, STATUS_FAILED, STATUS_CANCELLED}
)

# 任务目录内固定文件名
TASK_JSON: str = "task.json"
WORKER_LOCK: str = "worker.lock"
CANCEL_FLAG: str = "cancel.flag"
PROMPT_FILE: str = "prompt.md"

__all__ = [
    "CANCEL_FLAG",
    "PROMPT_FILE",
    "STATUS_CANCELLED",
    "STATUS_DONE",
    "STATUS_FAILED",
    "STATUS_QUEUED",
    "STATUS_RUNNING",
    "TASK_JSON",
    "TERMINAL_STATUSES",
    "WORKER_LOCK",
]
