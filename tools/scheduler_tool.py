"""
定时任务 Agent 工具 - 对话阶段调用的 @tool 函数

职责（逻辑 A 的入库端）：
    Agent 解析用户意图 → 调用 schedule_task 登记任务 → 回复"任务已登记"
    Agent 不等待、不阻塞，登记完即结束本轮对话。

设计要点：
    - 工具通过模块级 configure() 注入 TaskStore 和可选的 SchedulerEngine
    - 未 configure 时使用默认 DB 路径（memory/scheduled_tasks.sqlite）
    - 周期任务登记时，若引擎已运行则立即注册 cron job；否则等引擎启动时同步
    - 返回 JSON 字符串（与项目中 get_local_time 等工具风格一致）
"""
import json
import os
from typing import Any

from langchain_core.tools import tool

from scheduler.store import TaskStore

# ---- 默认 DB 路径（锚定项目根的 memory/ 目录） ---

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_DB_PATH = os.path.join(_PROJECT_ROOT, "memory", "scheduled_tasks.sqlite")


# ---- 模块级单例（通过 configure 注入） ---

_store: TaskStore | None = None
_engine: Any = None  # SchedulerEngine 实例（可选）


def configure(
    db_path: str | None = None,
    engine: Any = None,
    task_store: TaskStore | None = None,
):
    """
    初始化工具模块依赖（在程序启动时调用一次）。

    Args:
        db_path:    SQLite 数据库路径（task_store 未提供时使用）
        engine:     SchedulerEngine 实例（可选，用于立即注册周期任务）
        task_store: 已构造好的 TaskStore 实例（优先于 db_path）
    """
    global _store, _engine
    _store = task_store or TaskStore(db_path or _DEFAULT_DB_PATH)
    _engine = engine


def _get_store() -> TaskStore:
    """获取当前 TaskStore 单例（懒初始化）。"""
    global _store
    if _store is None:
        _store = TaskStore(_DEFAULT_DB_PATH)
    return _store


def _get_engine():
    """获取当前 SchedulerEngine（可能为 None）。"""
    return _engine


# ================================================================ Tools ===

@tool
def schedule_task(
    task_text: str,
    task_type: str,
    execute_time: str = "",
    cron_expr: str = "",
) -> str:
    """
    登记一个定时任务，到时间后由后台调度器自动调用 Agent 执行。

    【使用场景】
    当用户要求在指定时间或按周期执行某任务时调用此工具。
    例如："明天下午3点生成报告"、"每天9点发送日报"、"每周一整理项目文件"。

    【参数说明】
    - task_text: 实际要执行的任务描述（去掉时间部分）。如"生成销售报告"、"发送日报到飞书群"。
    - task_type: 任务类型。
      "one_time" = 一次性任务（在某个时间点执行一次）。
      "periodic" = 周期性任务（按固定频率重复执行）。
    - execute_time: 一次性任务的执行时间，ISO 8601 格式。
      如 "2026-07-29T15:00:00"。task_type 为 one_time 时必填。
    - cron_expr: 周期任务的 cron 表达式（5字段：分 时 日 月 周）。
      如 "0 9 * * *" = 每天9点，"0 9 * * 1" = 每周一9点，"30 17 * * 1-5" = 工作日17:30。
      task_type 为 periodic 时必填。

    【时间解析参考】
    如果用户的时间描述较模糊（如"明天"、"下周一"），请先调用 get_local_time 获取当前时间，
    然后自行计算精确时间或 cron 表达式后再调用本工具。

    【重要】
    登记后任务不会立即执行，而是存入数据库等待调度器在到时自动触发。
    你只需回复用户"任务已登记，到时自动执行"即可，不要等待。

    Args:
        task_text: 实际要执行的任务描述
        task_type: "one_time" 或 "periodic"
        execute_time: ISO 8601 时间（一次性任务必填）
        cron_expr: 5字段cron表达式（周期任务必填）

    Returns:
        JSON 字符串，包含任务 ID 和登记结果
    """
    try:
        store = _get_store()

        if task_type not in ("one_time", "periodic"):
            return _json({
                "success": False,
                "error": f"task_type 必须是 one_time 或 periodic，得到: {task_type}",
            })

        if task_type == "one_time" and not execute_time:
            return _json({
                "success": False,
                "error": "一次性任务必须提供 execute_time（ISO 8601 格式）",
            })

        if task_type == "periodic" and not cron_expr:
            return _json({
                "success": False,
                "error": "周期任务必须提供 cron_expr（5字段cron表达式）",
            })

        task_id = store.create_task(
            task_type=task_type,
            task_text=task_text,
            execute_time=execute_time or None,
            cron_expr=cron_expr or None,
        )

        engine = _get_engine()
        if engine and task_type == "periodic":
            task = store.get_task(task_id)
            if task:
                engine.register_periodic_task(task)

        if task_type == "one_time":
            message = f"一次性任务已登记（ID: {task_id}），将在 {execute_time} 自动执行。"
        else:
            message = f"周期任务已登记（ID: {task_id}），cron: {cron_expr}，将按计划自动执行。"

        return _json({
            "success": True,
            "task_id": task_id,
            "task_type": task_type,
            "execute_time": execute_time or None,
            "cron_expr": cron_expr or None,
            "task_text": task_text,
            "message": message,
        })

    except ValueError as exc:
        return _json({"success": False, "error": str(exc)})
    except Exception as exc:
        return _json({"success": False, "error": f"登记任务失败: {exc}"})


@tool
def list_scheduled_tasks(status: str = "") -> str:
    """
    查询已登记的定时任务列表。

    Args:
        status: 按状态过滤，可选值: pending / running / done / failed / cancelled。
                留空则返回所有任务。

    Returns:
        JSON 字符串，包含任务列表
    """
    try:
        store = _get_store()
        tasks = store.list_tasks(status=status or None)

        simplified = []
        for t in tasks:
            simplified.append({
                "id": t["id"],
                "type": t["task_type"],
                "status": t["status"],
                "task_text": t["task_text"][:100],
                "execute_time": t["execute_time"],
                "cron_expr": t["cron_expr"],
                "created_at": t["created_at"],
                "executed_at": t["executed_at"],
                "retry_count": t["retry_count"],
            })

        return _json({
            "success": True,
            "count": len(simplified),
            "tasks": simplified,
        })
    except Exception as exc:
        return _json({"success": False, "error": f"查询失败: {exc}"})


@tool
def cancel_scheduled_task(task_id: int) -> str:
    """
    取消一个尚未执行的定时任务（状态须为 pending）。

    对于周期任务，取消后会同时从调度器移除 cron job，不再重复执行。

    Args:
        task_id: 要取消的任务 ID

    Returns:
        JSON 字符串，包含取消结果
    """
    try:
        store = _get_store()
        task = store.get_task(task_id)
        if task is None:
            return _json({"success": False, "error": f"任务 {task_id} 不存在"})

        cancelled = store.cancel_task(task_id)
        if not cancelled:
            return _json({
                "success": False,
                "error": f"任务 {task_id} 状态为 {task['status']}，无法取消（只能取消 pending 状态的任务）",
            })

        engine = _get_engine()
        if engine and task["task_type"] == "periodic":
            engine.unregister_periodic_task(task_id)

        return _json({
            "success": True,
            "task_id": task_id,
            "message": f"任务 {task_id} 已取消",
        })
    except Exception as exc:
        return _json({"success": False, "error": f"取消失败: {exc}"})


@tool
def delete_scheduled_task(task_id: int) -> str:
    """
    删除一条定时任务（任意状态均可删除）。

    与 cancel_scheduled_task 不同：cancel 只是标记状态为 cancelled，任务记录仍保留；
    delete 则直接从数据库移除任务记录。适合清理已完成或已取消的任务。

    Args:
        task_id: 要删除的任务 ID

    Returns:
        JSON 字符串，包含删除结果
    """
    try:
        store = _get_store()
        deleted = store.delete_task(task_id)
        if not deleted:
            return _json({"success": False, "error": f"任务 {task_id} 不存在"})

        engine = _get_engine()
        if engine:
            engine.unregister_periodic_task(task_id)

        return _json({
            "success": True,
            "task_id": task_id,
            "message": f"任务 {task_id} 已删除",
        })
    except Exception as exc:
        return _json({"success": False, "error": f"删除失败: {exc}"})


@tool
def cleanup_finished_tasks() -> str:
    """
    清理所有已结束的定时任务（状态为 done / failed / cancelled）。

    pending 和 running 的任务不会被删除。适合定期清理历史任务保持列表整洁。

    Returns:
        JSON 字符串，包含清理数量
    """
    try:
        store = _get_store()
        deleted_count = store.cleanup_finished()
        return _json({
            "success": True,
            "deleted_count": deleted_count,
            "message": f"已清理 {deleted_count} 条已结束的任务",
        })
    except Exception as exc:
        return _json({"success": False, "error": f"清理失败: {exc}"})


# ---- 辅助 ----

def _json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)