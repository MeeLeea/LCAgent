#!/usr/bin/env python3
"""显式遗忘长期记忆的工具（forgotten 软删除）。

模型可主动调用本工具，按内容关键词 / 主题遗忘用户明确要求"忘掉"的记忆。
安全边界：**只标记不物理删除**（status → ``forgotten``），因此可恢复；
物理清理由后台 purge 承接。

依赖注入（工具无法自取 Agent/Session，与 ``tools/scheduler_tool.py`` 同一模式）：
本模块通过 :func:`configure` 接收"当前会话记忆依赖"的 provider，由入口程序在
构建 Agent 后注入一次（见 ``api/server.py`` 的 lifespan）。未注入时工具返回
干净的 ``{"success": false, "error": ...}``，不抛异常。
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from langchain_core.tools import tool

from cli.commands.memory import forget_facts_via_manager

logger = logging.getLogger(__name__)

# 合法作用域（术语对齐 clear / DELETE /api/memory：long=thread 级，agent=跨会话）
_VALID_SCOPES: tuple[str, ...] = ("long", "agent")

# 模块级注入：MemoryManager provider + 当前会话 ID provider
_memory_provider: Callable[[], Any] | None = None
_thread_id_provider: Callable[[], str | None] | None = None


def configure(
    memory_provider: Callable[[], Any] | None = None,
    thread_id_provider: Callable[[], str | None] | None = None,
) -> None:
    """注入当前会话的记忆依赖（入口程序在 Agent 构建完成后调用一次）。

    Args:
        memory_provider: 返回 ``MemoryManager``（或其等价 mock）的无参可调用对象；
            传 ``None`` 表示清除注入（测试清理 / 记忆系统未启用）
        thread_id_provider: 返回当前会话 ID 的无参可调用对象；为 None 时按空串处理
    """
    global _memory_provider, _thread_id_provider
    _memory_provider = memory_provider
    _thread_id_provider = thread_id_provider


def configure_for_agent(agent: Any) -> None:
    """按入口程序的 Agent 实例注入工具依赖（main / api / scheduler 复用）。

    Agent 构建完成、绑定 MemoryManager 之后调用一次；记忆系统未启用
    （``session_manager.memory`` 为 None）时无副作用。

    Args:
        agent: 已构建的 AgentCore（经 ``set_memory_manager`` 注入记忆后）
    """
    memory_provider = lambda: getattr(agent.session_manager, "memory", None)
    thread_id_provider = lambda: agent.session.current_session_id
    configure(
        memory_provider=memory_provider,
        thread_id_provider=thread_id_provider,
    )


def _resolve_memory() -> Any | None:
    """取出已注入的 MemoryManager（未注入返回 None）。"""
    if _memory_provider is None:
        return None
    try:
        return _memory_provider()
    except Exception as error:  # pragma: no cover - 防御性
        logger.debug("forget_memory: 获取 MemoryManager 失败: %s", error)
        return None


def _resolve_thread_id() -> str:
    """取出当前会话 ID（未注入返回空串）。"""
    if _thread_id_provider is None:
        return ""
    try:
        return _thread_id_provider() or ""
    except Exception as error:  # pragma: no cover - 防御性
        logger.debug("forget_memory: 获取 thread_id 失败: %s", error)
        return ""


@tool
async def forget_memory(
    query: str | None = None,
    topic: str | None = None,
    scope: str = "long",
) -> dict[str, Any]:
    """显式遗忘（软删除）长期记忆中符合条件的条目。

    只标记不物理删除：被遗忘的记忆不再注入对话、不占容量额度，之后可被重新学习。

    Args:
        query: 记忆内容关键词（不区分大小写子串匹配）；与 topic 至少提供一个
        topic: 记忆主题键（如 ``dim_y_formula``），提供时优先按主题精确匹配
        scope: ``long``（当前会话的长期记忆，默认）或 ``agent``（跨会话共享记忆）

    Returns:
        统一结果结构：``{"success": bool, "scope": str, "forgotten_count": int,
        "preview": list[str], "message"/"error": str}``；``preview`` 最多 3 条、
        每条 ≤80 字符，供模型向用户复述"忘了什么"。
    """
    normalized_scope = (scope or "long").strip().lower()
    if normalized_scope not in _VALID_SCOPES:
        return {
            "success": False,
            "error": "scope 必须为 long（当前会话长期记忆）或 agent（跨会话共享记忆）",
            "forgotten_count": 0,
        }

    effective_query = (query or "").strip()
    effective_topic = (topic or "").strip()
    if not effective_query and not effective_topic:
        return {
            "success": False,
            "error": "query 不能为空：请提供记忆内容关键词，或改用 topic 指定主题",
            "forgotten_count": 0,
        }

    memory = _resolve_memory()
    if memory is None:
        return {
            "success": False,
            "error": "记忆系统未初始化（未注入 MemoryManager），无法执行遗忘",
            "forgotten_count": 0,
        }

    try:
        result = await forget_facts_via_manager(
            memory,
            thread_id=_resolve_thread_id(),
            scope=normalized_scope,
            keyword=effective_query or None,
            topic=effective_topic or None,
        )
    except ValueError as error:
        # MemoryManager 的条件校验错误（如条件全空）→ 结果化，不向上抛
        return {"success": False, "error": str(error), "forgotten_count": 0}
    except Exception as error:
        logger.warning("forget_memory 执行失败: %s", error)
        return {
            "success": False,
            "error": f"遗忘失败: {error}",
            "forgotten_count": 0,
        }

    count = int(result.get("forgotten_count", 0))
    label = "agent 级记忆" if normalized_scope == "agent" else "长期记忆"
    message = (
        f"已遗忘 {count} 条{label}（软删除，不再注入对话）"
        if count
        else f"未找到匹配的{label}"
    )
    return {
        "success": True,
        "scope": normalized_scope,
        "forgotten_count": count,
        "preview": result.get("preview", []),
        "message": message,
    }


if __name__ == "__main__":
    import asyncio

    print(asyncio.run(forget_memory.ainvoke({"query": "示例关键词"})))
