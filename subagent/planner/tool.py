"""Planner 方案设计子程序的 Agent 工具（一次性最小设计，仅三个工具）。

非阻塞契约（关键）：
    ``start_planner_task`` 只拉起一个**分离的后台进程**并立即返回（毫秒级），
    它**不会同步产出方案**。主 Agent 必须改用 ``query_planner_task`` 轮询
    （可带 ``wait_seconds`` 做有界等待），仅在 ``done == true`` 之后才允许读取
    返回的 ``plan`` 正文。

一次性设计（no memory / no checkpoint）：
    任务是一次性的——需求正文 + 上下文在派活时固化为 ``task.json``，后台 worker
    单次跑完即产出 ``plan.md``。**没有**多轮记忆、去重、列表或 checkpoint 续跑；
    以 ``task_id`` 为唯一句柄。工具面刻意收敛为三个：启动 / 查询 / 取消。

僵尸回收（reaper）：
    ``query_planner_task`` 每次先调用 ``store.reap``——对 ``running`` 任务检查子进程
    存活，进程意外退出会被标记为 ``failed``，避免任务永远卡在 ``running``。

防编造（anti-fabrication）：
    本模块所有工具**绝不编造、推测或总结方案内容**。在 ``done == true`` 之前，
    任何“方案要点”都是幻觉；工具返回的 message 会反复提醒主 Agent 继续轮询。

进程模型说明：
    本模块只依赖同包的 ``config`` / ``launcher`` / ``store`` 三个轻量模块，
    **绝不**在模块级导入 ``agent`` / ``worker`` / ``team``，以免主进程因导入
    工具链而加载 langchain / team 重依赖。
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime
from typing import Any

from langchain_core.tools import tool

from subagent.config import load_planner_config

from .launcher import LauncherError, preflight, spawn_worker
from .store import (
    STATUS_CANCELLED,
    STATUS_DONE,
    STATUS_FAILED,
    TERMINAL_STATUSES,
    PlannerStore,
)

# ---- 模块级单例（通过 configure 注入，便于测试替换） ----

_store: PlannerStore | None = None


def configure(store: PlannerStore | None = None) -> None:
    """注入 PlannerStore 单例（测试或自定义根目录时调用）。

    Args:
        store: 已构造好的 :class:`PlannerStore` 实例；``None`` 时清空，
            之后 ``_get_store`` 会用默认根目录懒初始化。
    """
    global _store
    _store = store


def _get_store() -> PlannerStore:
    """获取当前 PlannerStore 单例（懒初始化，默认根目录）。"""
    global _store
    if _store is None:
        _store = PlannerStore()
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
async def start_planner_task(requirement: str, context: str = "") -> str:
    """在后台启动一个独立的“方案设计”子任务并立即返回（非阻塞）。

    【使用场景】
    当用户提出复杂需求，需要一份包含架构取舍、候选方案对比与落地计划的
    完整方案时调用。本工具负责“派活”，真正的方案由后台子程序一次性设计。

    【非阻塞语义（重要）】
    本工具只拉起一个**分离的后台进程**便毫秒级返回，**不会同步产出方案**。
    返回的 ``task_id`` 用于后续轮询；你必须先反复调用 ``query_planner_task``
    （可带 ``wait_seconds`` 做有界等待），仅在 ``done == true`` 之后才会在返回
    中拿到 ``plan`` 正文。

    【防编造（重要）】
    在 ``done == true`` 之前，**不要**编造、推测或总结任何方案内容——那是幻觉。

    Args:
        requirement: 方案需求正文，不能为空。
        context: 需求上下文（可选，如已有代码/约束/背景）。

    Returns:
        JSON 字符串，包含 task_id / pid / status / output_dir / status_file 等启动信息。
    """
    if not requirement or not requirement.strip():
        return _json({"success": False, "error": "requirement 不能为空"})

    store = _get_store()
    config = load_planner_config()

    try:
        provider, model = preflight(config)
    except ValueError as exc:
        # preflight 失败：不创建任务、不拉进程
        return _json({
            "success": False,
            "error": str(exc),
            "message": "未启动子进程",
        })

    task_id = store.create(requirement, context, provider=provider, model=model)

    try:
        pid = spawn_worker(
            task_id, root=os.path.normpath(config.output_root), config=config
        )
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
        "output_dir": task_dir,
        "status_file": os.path.join(task_dir, "task.json"),
        "message": (
            "方案设计已在后台启动。请用 query_planner_task(task_id, wait_seconds=30) "
            "轮询；status=done 前不要编造方案内容。"
        ),
    })


@tool
async def query_planner_task(
    task_id: str,
    wait_seconds: int = 0,
    max_chars: int = 12000,
) -> str:
    """查询方案设计任务进度，并在完成时返回方案正文（单次最多等待 30 秒）。

    【使用场景】
    在 ``start_planner_task`` 之后反复调用本工具轮询，直到 ``done == true``。
    这是预期的轮询方式——每次返回后若仍在设计，请继续调用，不要臆测结果。

    【有界等待】
    ``wait_seconds`` 会被限制在 ``[0, 30]`` 秒；因此本工具不会无限阻塞，重复轮询
    是正常行为。建议传 ``wait_seconds=30`` 以减少无效往返。

    【防编造（重要）】
    只有当返回中 ``done == true`` 且 ``status == "done"`` 时才允许使用 ``plan``
    字段的具体内容；在此之前**不要**编造、推测或总结任何方案内容。

    Args:
        task_id: 任务 ID（来自 ``start_planner_task``）。
        wait_seconds: 有界等待秒数，会被 clamp 到 [0, 30]。
        max_chars: 结果截断长度；``0`` 表示返回全文，其余会被 clamp 到 [1000, 50000]。

    Returns:
        JSON 字符串；完成时包含 plan 正文与字符数，未完成时包含状态信息。
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
            plan = store.read_result(task_id, bounded_chars)
            if plan is None:
                return _json({
                    "success": False,
                    "done": True,
                    "status": "done",
                    "error": "方案文件缺失",
                })
            return _json({
                "success": True,
                "done": True,
                "task_id": task_id,
                "status": "done",
                "plan": plan,
                "chars": len(plan),
                "truncated": plan.endswith("...[已截断]"),
                "message": "方案已生成",
            })
        return _json({
            "success": False,
            "done": True,
            "status": "done",
            "error": "任务已完成但方案文件缺失",
        })

    if status in (STATUS_FAILED, STATUS_CANCELLED):
        default_error = "任务失败" if status == STATUS_FAILED else "任务已取消"
        return _json({
            "success": False,
            "done": False,
            "task_id": task_id,
            "status": status,
            "error": task.get("error") or default_error,
            "message": "任务未成功完成，不要编造方案内容",
        })

    # queued / running
    return _json({
        "success": True,
        "done": False,
        "task_id": task_id,
        "status": status,
        "pid": task.get("pid"),
        "elapsed_seconds": _elapsed_seconds(task),
        "message": "仍在设计中，请继续轮询，不要编造方案内容",
    })


@tool
async def cancel_planner_task(task_id: str) -> str:
    """请求取消一个方案设计任务（在阶段边界生效）。

    【取消语义】
    - ``queued``：立即置为 cancelled；
    - ``running``：写取消标记，子进程在当前 LLM 调用结束后停止。

    【提示】
    取消是“受理”而非“立即 kill”，返回后**不要立即向用户断言子进程已停止**；
    取消后也不会暴露任何不完整的方案内容。

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
        "message": "取消在阶段边界生效；不要立即断言子进程已停止",
    })


__all__ = [
    "cancel_planner_task",
    "configure",
    "query_planner_task",
    "start_planner_task",
]
