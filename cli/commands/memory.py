"""Memory management commands.

本模块承载记忆管理的**唯一领域实现**：``clear_memory_apply`` /
``compress_memory_apply`` / ``compact_context_apply`` 三个纯逻辑函数供 CLI
处理器与 HTTP API 端点共用，避免同一操作在两处各写一套导致语义漂移
（历史缺陷：``DELETE /api/memory`` 缺少 ``agent`` scope，且 ``all`` 未清
agent 级记忆）。
"""

from __future__ import annotations

from typing import Any, Protocol

from .types import HANDLED, CommandContext, CommandOutcome


class _MemoryAgent(Protocol):
    """记忆命令所需的最小 Agent 接口（CLI / API 的 agent 均满足）。"""

    session_manager: Any
    session: Any

    def set_current_session(self, session_id: str) -> None: ...


#: clear 目标归一化表：英文/中文别名 → 规范作用域。
_CLEAR_TARGETS = {
    "long": "long",
    "长期": "long",
    "short": "short",
    "短期": "short",
    "agent": "agent",
    "全局": "agent",
    "all": "all",
    "全部": "all",
}


def normalize_clear_target(raw_target: str | None) -> str:
    """把 clear 目标归一化为 ``long|short|agent|all``。

    Args:
        raw_target: 用户/请求给出的目标；空值按 ``long`` 处理。

    Returns:
        规范作用域字符串。

    Raises:
        ValueError: 无法识别的目标。
    """
    key = (raw_target or "long").strip().lower()
    normalized = _CLEAR_TARGETS.get(key)
    if normalized is None:
        raise ValueError("clear 目标必须为 long|short|agent|all")
    return normalized


async def clear_memory_apply(agent: _MemoryAgent, raw_target: str | None) -> dict[str, Any]:
    """执行 clear 的领域逻辑（CLI / API 共用唯一实现）。

    Returns:
        结构化结果，调用方负责打印或序列化：
        ``{"scope", "long_cleared", "agent_cleared", "new_thread_id"}``
        （未涉及的字段为 ``None``）。

    Raises:
        ValueError: 无法识别的目标。
    """
    scope = normalize_clear_target(raw_target)
    session_manager = agent.session_manager
    result: dict[str, Any] = {
        "scope": scope,
        "long_cleared": None,
        "agent_cleared": None,
        "new_thread_id": None,
    }
    if scope in ("long", "all"):
        result["long_cleared"] = await session_manager.aclear_long_term_memory()
    if scope in ("agent", "all"):
        result["agent_cleared"] = await session_manager.aclear_agent_memory()
    if scope in ("short", "all"):
        # 短期记忆 = 当前会话 checkpoint；开启新会话替代删除
        tid = agent.session.new_session()
        agent.set_current_session(tid)
        result["new_thread_id"] = tid
    return result


async def compress_memory_apply(
    agent: _MemoryAgent, scope: str = "thread"
) -> dict[str, Any]:
    """压缩长期记忆（CLI / API 共用唯一实现）。

    Args:
        agent: 含 ``session_manager`` 的 Agent。
        scope: ``thread``（会话级，默认）或 ``agent``（跨会话共享）。

    Returns:
        结构化结果：压缩成功时合并 ``acompress_*`` 的返回并附 ``scope``；
        无记忆可压缩时返回 ``{"success": False, "skipped": True, "scope": scope}``。
    """
    session_manager = agent.session_manager
    normalized = "agent" if scope in ("agent", "全局") else "thread"
    summary = await session_manager.aget_memory_summary()
    count_key = "agent_fact_count" if normalized == "agent" else "long_term_count"
    if not summary.get(count_key):
        return {"success": False, "skipped": True, "scope": normalized}
    if normalized == "agent":
        result = await session_manager.acompress_agent_memory()
    else:
        result = await session_manager.acompress_memory()
    return {**result, "scope": normalized, "skipped": False}


async def compact_context_apply(
    agent: _MemoryAgent, *, thread_id: str | None = None
) -> dict[str, Any] | None:
    """手动触发上下文压缩（CLI / API 共用唯一实现）。

    手动触发一律 ``force=True``：跳过 ``max_context_tokens`` 阈值检查，允许用户在
    预估 token 未超阈值时主动压缩（仍需消息数 > ``keep_recent`` 才能安全切割）。

    Returns:
        压缩结果字典；消息过少无法安全切割时返回 ``None``。
    """
    return await agent.session_manager.manually_compact(force=True, thread_id=thread_id)


async def clear_memory(context: CommandContext, user_input: str) -> CommandOutcome:
    parts = user_input.split(None, 1)
    target = parts[1].strip().lower() if len(parts) > 1 else "long"
    try:
        result = await clear_memory_apply(context.agent, target)
    except ValueError:
        context.print("\n用法: clear [long|short|agent|all]  (默认 long)")
        return HANDLED

    scope = result["scope"]
    if scope == "long":
        context.print(f"\n已清空长期记忆 ({result['long_cleared']} 条 facts)")
    elif scope == "short":
        context.print(f"\n已清空短期记忆（新会话: {result['new_thread_id']}）")
    elif scope == "agent":
        context.print(f"\n已清空 agent 级长期记忆 ({result['agent_cleared']} 条 facts)")
    else:  # all
        context.print(
            f"\n已清空全部记忆 "
            f"(长期 {result['long_cleared']} 条 + agent 级 {result['agent_cleared']} 条 + 短期，"
            f"新会话: {result['new_thread_id']})"
        )
    return HANDLED


def _print_compress_result(context: CommandContext, result: dict[str, Any], scope: str) -> None:
    """按作用域打印压缩结果（CLI 展示层）。"""
    if not result.get("success"):
        if result.get("skipped"):
            label = "agent 级长期记忆" if scope == "agent" else "长期记忆"
            context.print(f"\n没有{label}可压缩")
        else:
            context.print(f"\n压缩失败: {result.get('error', '未知错误')}")
        return
    context.print("压缩完成！")
    context.print(f"  原记忆条数:   {result['original_count']} 条")
    context.print(f"  原字符数:     {result['original_chars']} 字符")
    context.print(f"  压缩后字符数: {result['compressed_chars']} 字符")
    ratio = (1 - int(result["compressed_chars"]) / max(int(result["original_chars"]), 1)) * 100
    context.print(f"  压缩率:       {ratio:.1f}%")
    context.print("\n--- 摘要内容 ---")
    context.print(str(result["summary"]))
    suffix = " (agent 级，跨会话共享)" if scope == "agent" else ""
    context.print(f"--- 已保存到长期记忆 Store{suffix} ---")


async def compress_memory(context: CommandContext) -> CommandOutcome:
    result = await compress_memory_apply(context.agent, scope="thread")
    _print_compress_result(context, result, scope="thread")
    return HANDLED


async def compress_agent_memory(context: CommandContext) -> CommandOutcome:
    """压缩 agent 级（跨会话共享）长期记忆。"""
    result = await compress_memory_apply(context.agent, scope="agent")
    _print_compress_result(context, result, scope="agent")
    return HANDLED


async def show_agent_memory(context: CommandContext) -> CommandOutcome:
    """召回并展示 agent 级（跨会话共享）长期记忆。"""
    text = await context.agent.session_manager.arecall_agent_memory()
    if not text:
        context.print("\n暂无 agent 级长期记忆")
    else:
        context.print("\n" + text.rstrip("\n"))
    return HANDLED


async def compact_context(context: CommandContext) -> CommandOutcome:
    """手动触发上下文压缩（增量摘要 + 工具输出 Prune）。"""
    context.print("\n开始压缩当前会话上下文...")
    result = await compact_context_apply(context.agent)
    if result is None:
        context.print("消息过少（少于保留阈值），无法安全切割，无需压缩")
        return HANDLED
    context.print(f"压缩完成！{result['messages_before']} → {result['messages_after']} 条消息")
    context.print("\n--- 摘要内容 ---")
    context.print(result["summary"])
    return HANDLED
