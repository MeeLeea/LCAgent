"""Memory management commands.

本模块承载记忆管理的**唯一领域实现**：``clear_memory_apply`` /
``compress_memory_apply`` / ``compact_context_apply`` 三个纯逻辑函数供 CLI
处理器与 HTTP API 端点共用，避免同一操作在两处各写一套导致语义漂移
（历史缺陷：``DELETE /api/memory`` 缺少 ``agent`` scope，且 ``all`` 未清
agent 级记忆）。
"""

from __future__ import annotations

from typing import Any, Protocol

from memory.models import FACT_STATUS_FORGOTTEN

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


#: forget 目标归一化表：英文/中文别名 → 规范作用域。
#: 术语对齐既有 ``clear`` / ``DELETE /api/memory`` 的 ``long|agent``，
#: 不使用 ``session``（避免前端维护两套映射）。
_FORGET_SCOPES = {
    "long": "long",
    "长期": "long",
    "agent": "agent",
    "全局": "agent",
}

# 遗忘预览上限：最多展示 3 条、每条 ≤80 字符（供模型/用户确认遗忘了什么）
_FORGET_PREVIEW_MAX_ITEMS = 3
_FORGET_PREVIEW_MAX_CHARS = 80


def normalize_forget_scope(raw_scope: str | None) -> str:
    """把 forget 作用域归一化为 ``long|agent``（空值按 ``long``）。

    Raises:
        ValueError: 无法识别的作用域（``short`` / ``all`` 等 clear 专有目标）。
    """
    key = (raw_scope or "long").strip().lower()
    normalized = _FORGET_SCOPES.get(key)
    if normalized is None:
        raise ValueError("forget 作用域必须为 long|agent")
    return normalized


def _truncate_preview(content: str) -> str:
    """把一条记忆内容裁剪为预览文本（超长截断并加省略号）。"""
    text = (content or "").strip()
    if len(text) <= _FORGET_PREVIEW_MAX_CHARS:
        return text
    return text[: _FORGET_PREVIEW_MAX_CHARS - 1] + "…"


async def forget_facts_via_manager(
    memory: Any,
    *,
    thread_id: str,
    scope: str,
    keyword: str | None = None,
    topic: str | None = None,
) -> dict[str, Any]:
    """遗忘的领域核心实现（CLI / HTTP API / forget_memory 工具共用）。

    调用方负责提供已就绪的 ``MemoryManager``（串行化/缓冲清理在其内部完成）；
    本函数只做"遗忘前后快照 diff"以产出被遗忘内容预览，不复制匹配逻辑。

    Args:
        memory: ``MemoryManager`` 实例（需提供 ``forget`` / ``forget_agent`` / ``store``）。
        thread_id: 目标会话 ID（``scope="agent"`` 时仅作日志用途，不参与 namespace）
        scope: 已归一化的 ``long`` 或 ``agent``
        keyword: 内容关键词（不区分大小写子串匹配）
        topic: 主题键（内部由 Store 层归一化后匹配）

    Returns:
        ``{"scope", "forgotten_count", "preview"}``，``preview`` 为被遗忘内容预览列表。
    """
    store = memory.store
    is_agent = scope == "agent"

    # 1. 遗忘前快照：仅统计"本轮之前不是 forgotten"的条目，避免把历史 forgotten
    #    误算进 preview（幂等重复遗忘时 preview 应为空）。
    before = (
        await store.query_agent_facts() if is_agent else await store.query_facts(thread_id)
    )
    candidate_ids = {
        f.fact_id for f in before if f.status != FACT_STATUS_FORGOTTEN
    }

    # 2. 执行遗忘（MemoryManager 内部：缓冲清理 + 锁串行化 + 软删除标记）
    if is_agent:
        count = await memory.forget_agent(keyword=keyword, topic_key=topic)
    else:
        count = await memory.forget(thread_id, keyword=keyword, topic_key=topic)

    # 3. 遗忘后快照 diff → 本轮新变为 forgotten 的条目（保证 preview 精确）
    after = (
        await store.query_agent_facts() if is_agent else await store.query_facts(thread_id)
    )
    newly_forgotten = [
        f
        for f in after
        if f.status == FACT_STATUS_FORGOTTEN and f.fact_id in candidate_ids
    ]
    preview = [
        _truncate_preview(f.content) for f in newly_forgotten
    ][:_FORGET_PREVIEW_MAX_ITEMS]

    return {"scope": scope, "forgotten_count": count, "preview": preview}


async def forget_memory_apply(
    agent: _MemoryAgent,
    scope: str = "long",
    keyword: str | None = None,
    topic: str | None = None,
) -> dict[str, Any]:
    """执行 forget 的领域逻辑（CLI / HTTP API 共用唯一实现）。

    Raises:
        ValueError: 作用域非法，或 keyword 与 topic 同时缺失。

    Returns:
        ``{"scope", "forgotten_count", "preview"}``（记忆系统未启用时附
        ``"skipped": True``，与 SessionManager 其它记忆接口的"空操作"语义一致）。
    """
    normalized = normalize_forget_scope(scope)
    effective_keyword = (keyword or "").strip()
    effective_topic = (topic or "").strip()
    if not effective_keyword and not effective_topic:
        raise ValueError("forget 需要提供关键词（keyword）或主题（topic）")

    memory = getattr(agent.session_manager, "memory", None)
    if memory is None:
        return {
            "scope": normalized,
            "forgotten_count": 0,
            "preview": [],
            "skipped": True,
        }

    return await forget_facts_via_manager(
        memory,
        thread_id=agent.session.current_session_id,
        scope=normalized,
        keyword=effective_keyword or None,
        topic=effective_topic or None,
    )


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


async def forget_memory(context: CommandContext, user_input: str) -> CommandOutcome:
    """CLI ``forget <关键词>`` / ``forget agent <关键词>``：显式遗忘（软删除）。

    作用域术语与 ``clear`` 一致：默认 ``long``（thread 级，会话隔离），
    ``forget agent <关键词>`` 作用于 agent 级（跨会话共享）记忆。
    """
    parts = user_input.split(None, 2)
    if len(parts) < 2:
        context.print("\n用法: forget <关键词>  或  forget agent <关键词>")
        return HANDLED

    if parts[1].strip().lower() == "agent":
        scope = "agent"
        keyword = parts[2].strip() if len(parts) > 2 else ""
    else:
        scope = "long"
        keyword = " ".join(parts[1:]).strip()

    if not keyword:
        context.print("\n用法: forget <关键词>  或  forget agent <关键词>")
        return HANDLED

    try:
        result = await forget_memory_apply(context.agent, scope=scope, keyword=keyword)
    except ValueError as error:
        context.print(f"\n遗忘失败: {error}")
        return HANDLED

    if result.get("skipped"):
        context.print("\n记忆系统未启用，无法遗忘")
        return HANDLED

    label = "agent 级长期记忆" if result["scope"] == "agent" else "长期记忆"
    count = result["forgotten_count"]
    if not count:
        context.print(f"\n未找到匹配「{keyword}」的{label}")
        return HANDLED

    context.print(f"\n已遗忘 {count} 条{label}（软删除，不再注入对话）")
    if result["preview"]:
        context.print("\n--- 已遗忘内容 ---")
        for content in result["preview"]:
            context.print(f"- {content}")
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


async def show_thread_memory(context: CommandContext) -> CommandOutcome:
    """召回并展示当前会话（thread 级）长期记忆。"""
    text = await context.agent.session_manager.arecall_memory()
    if not text:
        context.print("\n暂无会话级长期记忆")
    else:
        context.print("\n" + text.rstrip("\n"))
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
