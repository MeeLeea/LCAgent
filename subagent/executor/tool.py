"""Executor 任务执行子程序的 Agent 工具（一次性设计，仅三个工具）。

非阻塞契约（关键）：
    ``start_executor_task`` 只拉起一个**分离的后台进程**并立即返回（毫秒级），
    它**不会同步执行任务**。主 Agent 必须改用 ``query_executor_task`` 轮询
    （可带 ``wait_seconds`` 做有界等待），仅在 ``done == true`` 之后才允许读取
    返回的 ``result`` 正文。

一次性设计（no memory / no checkpoint）：
    任务是一次性的——任务正文 / 执行计划 / 上下文 / 工作区在派活时固化为
    ``task.json``，后台 worker 多步执行完毕即产出 ``result.md``。**没有**多轮
    记忆、列表或 checkpoint 续跑；以 ``task_id`` 为唯一句柄。

与 planner 的分工（防误用）：
    ``planner`` 产出方案文档（一次 LLM 调用，零副作用）；``executor`` 是后台
    实现/执行（多步工具循环，**会真实写盘**——工作区文件不会被取消/失败回滚）。
    只有在已有明确「任务 + 可执行计划」时才派 executor，纯方案讨论用 planner。

僵尸回收（reaper）：
    ``query_executor_task`` 每次先调用 ``store.reap``——对 ``running`` 任务检查
    子进程存活，进程意外退出会被标记为 ``failed``。``steps_used`` 字段记录了
    出错前的执行进度，便于诊断。

防编造（anti-fabrication）：
    本模块所有工具**绝不编造、推测或总结执行结果**。在 ``done == true`` 之前，
    任何"执行成果"都是幻觉；工具返回的 message 会反复提醒主 Agent 继续轮询。

进程模型说明：
    本模块只依赖同包的 ``config`` / ``launcher`` / ``store`` / ``workspace``
    以及 stdlib 级 ``subagent.common``，**绝不**在模块级导入 ``agent`` /
    ``worker`` / ``team`` / ``langgraph``，以免主进程因导入工具链而拉重依赖。
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime
from typing import Any

from langchain_core.tools import tool

from subagent.common.proc import LauncherError
from subagent.executor.config import ExecutorConfig
from subagent.executor.launcher import preflight, spawn_worker
from subagent.executor.store import (
    STATUS_CANCELLED,
    STATUS_DONE,
    STATUS_FAILED,
    TERMINAL_STATUSES,
    ExecutorStore,
    resolve_workspace_at_start,
)

# ---- 模块级单例（通过 configure 注入，便于测试替换） ----

_store: ExecutorStore | None = None


def configure(store: ExecutorStore | None = None) -> None:
    """注入 ExecutorStore 单例（测试或自定义根目录时调用）。

    Args:
        store: 已构造好的 :class:`ExecutorStore` 实例；``None`` 时清空，
            之后 ``_get_store`` 会用默认根目录懒初始化。
    """
    global _store
    _store = store


def _get_store() -> ExecutorStore:
    """获取当前 ExecutorStore 单例（懒初始化，默认根目录）。"""
    global _store
    if _store is None:
        _store = ExecutorStore()
    return _store


def _json(data: dict[str, Any]) -> str:
    """把字典序列化为带缩进、保留中文的 JSON 字符串。"""
    return json.dumps(data, ensure_ascii=False, indent=2)


def _now_iso() -> str:
    """返回带本地时区信息的 ISO 时间戳（秒级精度）。"""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _parse_iso(value: object) -> datetime | None:
    """把 ISO 时间字符串解析为带时区的 ``datetime``；失败返回 ``None``。"""
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.now().astimezone().tzinfo)
    return parsed


def _elapsed_seconds(task: dict[str, Any]) -> float:
    """计算任务已耗时（秒）：从创建时间到完成时间或当前时刻。"""
    start = _parse_iso(task.get("created_at"))
    if start is None:
        return 0.0
    end = _parse_iso(task.get("finished_at")) or datetime.now().astimezone()
    return round(max(0.0, (end - start).total_seconds()), 1)


# ================================================================ Tools ===


@tool
async def start_executor_task(
    task: str,
    plan: str = "",
    context: str = "",
    workspace: str = "",
) -> str:
    """在后台启动一个独立的“任务执行”子任务并立即返回（非阻塞，会真实写盘）。

    【使用场景】
    当任务已经有明确的执行目标 + 可执行计划（例如来自 planner 的方案），
    需要后台真实实现/产出时调用。本工具负责"派活"，真正的执行由后台子程序
    在受限工具集（read_file / write_file / list_dir，绑定 workspace）内多步完成。

    【副作用提示（重要）】
    executor 会在工作区内**真实创建/修改文件**；任务取消或失败时已写入的文件
    **不会回滚**。工作区默认是隔离的任务目录，指定 workspace 时必须是一个已
    存在的目录绝对路径（禁止系统目录、项目根目录等受保护路径）。

    【非阻塞语义（重要）】
    本工具只拉起一个**分离的后台进程**便毫秒级返回，**不会同步产出结果**。
    返回的 ``task_id`` 用于后续轮询；必须反复调用 ``query_executor_task``
    （可带 ``wait_seconds`` 有界等待），仅在 ``done == true`` 之后才会拿到
    ``result`` 正文与产出清单。

    【与 planner 的分工】
    纯方案设计/讨论用 ``start_planner_task``（零写盘副作用）；只有计划明确
    要落地时才需 executor，两者不要串联混用于同一目标。

    【防编造（重要）】
    在 ``done == true`` 之前，**不要**编造、推测或总结任何执行结果——那是幻觉。

    Args:
        task: 执行任务正文，不能为空。
        plan: 可选执行计划（如 planner 产出的方案要点；为空时由 executor 自行安排）。
        context: 可选上下文（相关文件说明/约束/背景）。
        workspace: 可选工作区目录（必须已存在的绝对路径）；留空则使用任务自己的
            隔离工作区（推荐）。

    Returns:
        JSON 字符串，包含 task_id / pid / status / workspace / output_dir 等启动信息。
    """
    if not task or not task.strip():
        return _json({"success": False, "error": "task 不能为空"})

    store = _get_store()
    config = ExecutorConfig.load()

    try:
        provider, model = preflight(config)
    except ValueError as exc:
        # preflight 失败：不创建任务、不拉进程
        return _json({
            "success": False,
            "error": str(exc),
            "message": "未启动子进程",
        })

    task_id = store.create(
        task, plan=plan, context=context, max_steps=config.max_steps,
        provider=provider, model=model,
    )

    try:
        resolved_ws = resolve_workspace_at_start(store, task_id, workspace or None)
    except ValueError as exc:
        # workspace 非法：不拉进程，任务标记失败
        try:
            store.update(
                task_id,
                status=STATUS_FAILED,
                finished_at=_now_iso(),
                error=f"workspace 非法: {exc}",
            )
        except (OSError, ValueError):
            pass
        return _json({
            "success": False,
            "task_id": task_id,
            "error": f"workspace 非法: {exc}",
            "message": "未启动子进程",
        })

    try:
        pid = spawn_worker(task_id, root=config.resolved_output_root(), config=config)
    except (LauncherError, OSError) as exc:
        # 尽力把任务标记为失败，避免留下永远 queued 的僵尸任务
        try:
            store.update(
                task_id,
                status=STATUS_FAILED,
                finished_at=_now_iso(),
                error=str(exc),
            )
        except (OSError, ValueError):
            pass
        return _json({
            "success": False,
            "task_id": task_id,
            "error": str(exc),
        })

    task_dir = store.task_dir(task_id)
    return _json({
        "success": True,
        "task_id": task_id,
        "pid": pid,
        "status": "queued",
        "workspace": resolved_ws,
        "output_dir": task_dir,
        "status_file": os.path.join(task_dir, "task.json"),
        "message": (
            "任务执行已在后台启动。请用 query_executor_task(task_id, wait_seconds=30) "
            "轮询；status=done 前不要编造执行结果。"
        ),
    })


@tool
async def query_executor_task(
    task_id: str,
    wait_seconds: int = 0,
    max_chars: int = 12000,
) -> str:
    """查询任务执行进度，并在完成时返回结果文档正文（单次最多等待 30 秒）。

    【使用场景】
    在 ``start_executor_task`` 之后反复调用本工具轮询，直到 ``done == true``。
    这是预期的轮询方式——每次返回后若仍在执行，请继续调用，不要臆测结果。

    【有界等待】
    ``wait_seconds`` 会被限制在 ``[0, 30]`` 秒；因此本工具不会无限阻塞，重复轮询
    是正常行为。建议传 ``wait_seconds=30`` 以减少无效往返。

    【防编造（重要）】
    只有当返回中 ``done == true`` 且 ``status == "done"`` 时才允许使用 ``result``
    字段的具体内容；在此之前**不要**编造、推测或总结任何执行结果。

    Args:
        task_id: 任务 ID（来自 ``start_executor_task``）。
        wait_seconds: 有界等待秒数，会被 clamp 到 [0, 30]。
        max_chars: 结果截断长度；``0`` 表示返回全文，其余会被 clamp 到 [1000, 50000]。

    Returns:
        JSON 字符串；完成时包含 result 正文与产出文件清单，未完成时包含状态信息。
    """
    store = _get_store()
    bounded_wait = max(0, min(30, int(wait_seconds)))
    bounded_chars = 0 if max_chars == 0 else max(1000, min(50000, int(max_chars)))

    start = time.monotonic()
    task: dict[str, Any] | None = None
    while True:
        task = store.reap(task_id)
        if task is None:
            return _json({
                "success": False,
                "task_id": task_id,
                "error": f"任务不存在: {task_id}",
            })
        if task.get("status") in TERMINAL_STATUSES:
            break
        elapsed = time.monotonic() - start
        if elapsed >= bounded_wait:
            break
        remaining = bounded_wait - elapsed
        await asyncio.sleep(min(1.0, remaining))

    status = str(task.get("status") or "")

    if status == STATUS_DONE:
        if task.get("has_result"):
            result = store.read_result(task_id, bounded_chars)
            if result is None:
                return _json({
                    "success": False,
                    "done": True,
                    "status": "done",
                    "error": "结果文件缺失",
                })
            return _json({
                "success": True,
                "done": True,
                "task_id": task_id,
                "status": "done",
                "result": result,
                "chars": len(result),
                "steps_used": task.get("steps_used"),
                "workspace": task.get("workspace"),
                "truncated": result.endswith("...[已截断]"),
                "message": "任务已完成",
            })
        return _json({
            "success": False,
            "done": True,
            "status": "done",
            "error": "任务已完成但结果文件缺失",
        })

    if status in (STATUS_FAILED, STATUS_CANCELLED):
        default_error = "任务失败" if status == STATUS_FAILED else "任务已取消"
        return _json({
            "success": False,
            "done": False,
            "task_id": task_id,
            "status": status,
            "error": task.get("error") or default_error,
            "steps_used": task.get("steps_used"),
            "message": "任务未成功完成，不要编造执行结果",
        })

    # queued / running
    return _json({
        "success": True,
        "done": False,
        "task_id": task_id,
        "status": status,
        "pid": task.get("pid"),
        "steps_used": task.get("steps_used"),
        "elapsed_seconds": _elapsed_seconds(task),
        "message": "仍在执行中，请继续轮询，不要编造执行结果",
    })


@tool
async def cancel_executor_task(task_id: str) -> str:
    """请求取消一个任务执行（在阶段边界生效）。

    【取消语义】
    - ``queued``：立即置为 cancelled；
    - ``running``：写取消标记，子程序在下一个阶段边界（下一次模型调用前 /
      写结果前）退出，**不写 ``result.md``**。

    【副作用提示（重要）】
    取消**不会回滚**工作区中已写入的文件；不要向用户断言"什么都没改"。

    【提示】
    取消是"受理"而非"立即 kill"，返回后**不要立即向用户断言子进程已停止**。

    Args:
        task_id: 要取消的任务 ID。

    Returns:
        JSON 字符串，包含是否受理、当前状态与说明。
    """
    store = _get_store()
    try:
        result = store.request_cancel(task_id)
    except ValueError as exc:
        return _json({
            "success": False,
            "task_id": task_id,
            "error": str(exc),
        })

    return _json({
        "success": bool(result["accepted"]),
        "task_id": task_id,
        "status": result["task"]["status"],
        "note": result["note"],
        "message": "取消在阶段边界生效；工作区已写入的文件不会被回滚",
    })


__all__ = [
    "cancel_executor_task",
    "configure",
    "query_executor_task",
    "start_executor_task",
]
