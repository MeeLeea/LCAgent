# -*- coding: utf-8 -*-
"""
定时任务持久化层 - 基于 SQLite

职责：
    - 建表、CRUD
    - 提供到期任务查询（供调度器轮询）
    - 提供周期任务列表（供调度器注册 cron）
    - 原子状态抢占（防止并发重复执行）

设计要点：
    - 每次操作新建连接 + threading.Lock 保护写操作，线程安全
    - execute_time / created_at / executed_at 统一用 ISO 8601 字符串存储
    - 状态机：pending → running → done / failed（失败可重试回退 pending）
"""
import os
import sqlite3
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional


# ------------------------------------------------------------------ SQL ---

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS scheduled_tasks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    task_type     TEXT    NOT NULL,             -- 'one_time' | 'periodic'
    execute_time  TEXT,                          -- ISO 8601, 一次性任务必填
    cron_expr     TEXT,                          -- 5 字段 cron, 周期任务必填
    task_text     TEXT    NOT NULL,             -- 实际要执行的任务描述
    status        TEXT    NOT NULL DEFAULT 'pending',  -- pending|running|done|failed|cancelled
    created_at    TEXT    NOT NULL,
    executed_at   TEXT,
    result        TEXT,
    retry_count   INTEGER NOT NULL DEFAULT 0,
    max_retries   INTEGER NOT NULL DEFAULT 3
);
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_tasks_poll
    ON scheduled_tasks (status, task_type, execute_time);
"""

_VALID_STATUSES = {"pending", "running", "done", "failed", "cancelled"}
_VALID_TYPES = {"one_time", "periodic"}


# ---------------------------------------------------------------- 行转换 ---

def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "task_type": row["task_type"],
        "execute_time": row["execute_time"],
        "cron_expr": row["cron_expr"],
        "task_text": row["task_text"],
        "status": row["status"],
        "created_at": row["created_at"],
        "executed_at": row["executed_at"],
        "result": row["result"],
        "retry_count": row["retry_count"],
        "max_retries": row["max_retries"],
    }


# ---------------------------------------------------------------- TaskStore ---

class TaskStore:
    """SQLite 定时任务存储，线程安全。"""

    def __init__(self, db_path: str):
        """
        Args:
            db_path: SQLite 数据库文件路径。目录不存在时自动创建。
        """
        self.db_path = db_path
        self._lock = threading.Lock()
        self._ensure_dir()
        self._init_db()

    # ---- 内部工具 ----

    def _ensure_dir(self):
        directory = os.path.dirname(os.path.abspath(self.db_path))
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")  # 并发读写更友好
        return conn

    def _init_db(self):
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(_CREATE_TABLE)
                conn.execute(_CREATE_INDEX)
                conn.commit()
            finally:
                conn.close()

    @staticmethod
    def _iso(dt: Optional[datetime]) -> Optional[str]:
        if dt is None:
            return None
        return dt.isoformat()

    # ---- 写操作 ----

    def create_task(
        self,
        task_type: str,
        task_text: str,
        execute_time: Optional[str] = None,
        cron_expr: Optional[str] = None,
        max_retries: int = 3,
    ) -> int:
        """
        插入一条定时任务，返回自增 ID。

        校验规则：
            - task_type 必须是 one_time / periodic
            - one_time  → execute_time 必填
            - periodic  → cron_expr 必填
        """
        if task_type not in _VALID_TYPES:
            raise ValueError(f"task_type 必须是 {sorted(_VALID_TYPES)} 之一，得到: {task_type}")
        if not task_text or not task_text.strip():
            raise ValueError("task_text 不能为空")
        if task_type == "one_time" and not execute_time:
            raise ValueError("一次性任务必须提供 execute_time")
        if task_type == "periodic" and not cron_expr:
            raise ValueError("周期任务必须提供 cron_expr")

        now_iso = self._iso(datetime.now())
        with self._lock:
            conn = self._connect()
            try:
                cursor = conn.execute(
                    """
                    INSERT INTO scheduled_tasks
                        (task_type, execute_time, cron_expr, task_text,
                         status, created_at, retry_count, max_retries)
                    VALUES (?, ?, ?, ?, 'pending', ?, 0, ?)
                    """,
                    (task_type, execute_time, cron_expr, task_text.strip(),
                     now_iso, max_retries),
                )
                conn.commit()
                return cursor.lastrowid
            finally:
                conn.close()

    def claim_task(self, task_id: int) -> bool:
        """
        原子抢占：把 pending → running。

        通过 ``UPDATE ... WHERE status='pending'`` 保证只有一个调用方能成功，
        返回 True 表示抢占成功（该调用方负责执行），False 表示已被他人抢走。
        """
        now_iso = self._iso(datetime.now())
        with self._lock:
            conn = self._connect()
            try:
                cursor = conn.execute(
                    """
                    UPDATE scheduled_tasks
                       SET status = 'running', executed_at = ?
                     WHERE id = ? AND status = 'pending'
                    """,
                    (now_iso, task_id),
                )
                conn.commit()
                return cursor.rowcount == 1
            finally:
                conn.close()

    def mark_done(self, task_id: int, result: str):
        self._update_status(task_id, "done", result=result)

    def mark_failed(self, task_id: int, error: str):
        """标记失败；若仍有重试次数则回退为 pending 并递增 retry_count。"""
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT retry_count, max_retries FROM scheduled_tasks WHERE id = ?",
                    (task_id,),
                ).fetchone()
                if row is None:
                    return
                retry_count = row["retry_count"] + 1
                if retry_count < row["max_retries"]:
                    # 还有重试机会：回退 pending，下次轮询会再次拾取
                    conn.execute(
                        """
                        UPDATE scheduled_tasks
                           SET status = 'pending',
                               retry_count = ?,
                               result = ?
                         WHERE id = ?
                        """,
                        (retry_count, error, task_id),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE scheduled_tasks
                           SET status = 'failed',
                               retry_count = ?,
                               result = ?
                         WHERE id = ?
                        """,
                        (retry_count, error, task_id),
                    )
                conn.commit()
            finally:
                conn.close()

    def cancel_task(self, task_id: int) -> bool:
        """取消尚未完成的任务（pending → cancelled）。返回是否取消成功。"""
        with self._lock:
            conn = self._connect()
            try:
                cursor = conn.execute(
                    """
                    UPDATE scheduled_tasks
                       SET status = 'cancelled'
                     WHERE id = ? AND status = 'pending'
                    """,
                    (task_id,),
                )
                conn.commit()
                return cursor.rowcount == 1
            finally:
                conn.close()

    def _update_status(
        self,
        task_id: int,
        status: str,
        result: Optional[str] = None,
    ):
        if status not in _VALID_STATUSES:
            raise ValueError(f"status 必须是 {sorted(_VALID_STATUSES)} 之一")
        with self._lock:
            conn = self._connect()
            try:
                if result is not None:
                    conn.execute(
                        "UPDATE scheduled_tasks SET status = ?, result = ? WHERE id = ?",
                        (status, result, task_id),
                    )
                else:
                    conn.execute(
                        "UPDATE scheduled_tasks SET status = ? WHERE id = ?",
                        (status, task_id),
                    )
                conn.commit()
            finally:
                conn.close()

    # ---- 读操作 ----

    def get_task(self, task_id: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM scheduled_tasks WHERE id = ?",
                    (task_id,),
                ).fetchone()
                return _row_to_dict(row) if row else None
            finally:
                conn.close()

    def get_due_tasks(self, now: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        查询到期的一次性任务：status=pending + task_type=one_time + execute_time <= now。
        """
        now_iso = self._iso(now or datetime.now())
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """
                    SELECT * FROM scheduled_tasks
                     WHERE status = 'pending'
                       AND task_type = 'one_time'
                       AND execute_time <= ?
                     ORDER BY execute_time ASC
                    """,
                    (now_iso,),
                ).fetchall()
                return [_row_to_dict(r) for r in rows]
            finally:
                conn.close()

    def list_periodic_tasks(self, status: str = "pending") -> List[Dict[str, Any]]:
        """查询周期任务（供调度器启动时同步注册 cron job）。"""
        with self._lock:
            conn = self._connect()
            try:
                if status:
                    rows = conn.execute(
                        """
                        SELECT * FROM scheduled_tasks
                         WHERE task_type = 'periodic' AND status = ?
                         ORDER BY id ASC
                        """,
                        (status,),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT * FROM scheduled_tasks
                         WHERE task_type = 'periodic'
                         ORDER BY id ASC
                        """,
                    ).fetchall()
                return [_row_to_dict(r) for r in rows]
            finally:
                conn.close()

    def list_tasks(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出任务，可按状态过滤。"""
        with self._lock:
            conn = self._connect()
            try:
                if status:
                    rows = conn.execute(
                        "SELECT * FROM scheduled_tasks WHERE status = ? ORDER BY id DESC",
                        (status,),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM scheduled_tasks ORDER BY id DESC"
                    ).fetchall()
                return [_row_to_dict(r) for r in rows]
            finally:
                conn.close()

    def delete_task(self, task_id: int) -> bool:
        """删除一条任务（任意状态）。返回是否删除成功。"""
        with self._lock:
            conn = self._connect()
            try:
                cursor = conn.execute(
                    "DELETE FROM scheduled_tasks WHERE id = ?",
                    (task_id,),
                )
                conn.commit()
                return cursor.rowcount == 1
            finally:
                conn.close()

    def cleanup_finished(self) -> int:
        """
        清理已结束的任务（done / failed / cancelled），返回删除的行数。

        pending 和 running 的任务不会被删除。
        """
        with self._lock:
            conn = self._connect()
            try:
                cursor = conn.execute(
                    "DELETE FROM scheduled_tasks WHERE status IN ('done', 'failed', 'cancelled')"
                )
                conn.commit()
                return cursor.rowcount
            finally:
                conn.close()