"""Metrics display commands."""

from __future__ import annotations

from .types import HANDLED, CommandContext, CommandOutcome


def show_metrics(context: CommandContext) -> CommandOutcome:
    """展示当前会话的运行时指标

    包含三类指标：
    - LLM 调用统计（次数、tokens、按 provider 分组）
    - 工具执行统计（次数、耗时、按工具名分组）
    - 压缩统计（触发次数、消息裁剪量）
    """
    metrics = getattr(context.agent, "metrics", None)
    if metrics is None:
        context.print("\n当前 Agent 不支持指标收集")
        return HANDLED

    summary = metrics.get_summary()
    session = summary["session"]
    llm = summary["llm"]
    tools = summary["tools"]
    compaction = summary["compaction"]

    context.print("\n" + "=" * 50)
    context.print("运行时指标")
    context.print("=" * 50)

    # 会话概览
    context.print("\n--- 会话概览 ---")
    context.print(f"运行时长:   {session['duration_seconds']:.1f}s")
    context.print(f"Turn 次数:  {session['turn_count']}")

    # LLM 指标
    context.print("\n--- LLM 调用统计 ---")
    context.print(f"总调用次数:     {llm['total_calls']}")
    context.print(f"总 prompt tokens:    {llm['total_prompt_tokens']:,}")
    context.print(f"总 completion tokens: {llm['total_completion_tokens']:,}")
    context.print(f"总 tokens:      {llm['total_tokens']:,}")
    if llm["total_calls"] > 0:
        avg_tokens = llm["total_tokens"] / llm["total_calls"]
        context.print(f"平均 tokens/次: {avg_tokens:.0f}")
    context.print(f"总耗时:         {llm['total_duration_ms']:.0f}ms")

    if llm["by_provider"]:
        context.print("\n  按 Provider 分组:")
        for provider, stats in llm["by_provider"].items():
            context.print(f"  [{provider}]")
            context.print(f"    调用次数: {stats['count']}")
            context.print(f"    tokens:   {stats['total_tokens']:,} (avg {stats['avg_tokens']:.0f})")
            context.print(f"    耗时:     {stats['total_ms']:.0f}ms")

    # 工具指标
    context.print("\n--- 工具执行统计 ---")
    context.print(f"总调用次数: {tools['total_calls']}")
    context.print(f"总耗时:     {tools['total_duration_ms']:.0f}ms")

    if tools["by_name"]:
        context.print("\n  按工具分组:")
        for name, stats in sorted(tools["by_name"].items(), key=lambda x: x[1]["total_ms"], reverse=True):
            context.print(f"  [{name}]")
            context.print(f"    调用次数: {stats['count']}")
            context.print(f"    总耗时:   {stats['total_ms']:.0f}ms (min {stats['min_ms']:.0f} / max {stats['max_ms']:.0f} / avg {stats['avg_ms']:.0f})")
            if stats["failures"] > 0:
                context.print(f"    失败次数: {stats['failures']}")
            if stats["timeouts"] > 0:
                context.print(f"    超时次数: {stats['timeouts']}")

    # 压缩指标
    context.print("\n--- 压缩统计 ---")
    context.print(f"触发次数:       {compaction['total_count']}")
    if compaction["total_count"] > 0:
        context.print(f"压缩前消息总数: {compaction['total_messages_before']}")
        context.print(f"压缩后消息总数: {compaction['total_messages_after']}")
        context.print(f"节省消息数:     {compaction['messages_saved']}")
        context.print(f"总耗时:         {compaction['total_duration_ms']:.0f}ms")

    context.print()
    return HANDLED


def metrics_command(context: CommandContext, user_input: str) -> CommandOutcome:
    """metrics 命令路由

    用法:
      metrics         展示当前指标
      metrics:status  同上
      metrics:reset   重置所有指标
    """
    parts = user_input.split(":", 1)
    sub = parts[1].strip().lower() if len(parts) > 1 else ""

    if sub in ("", "status"):
        return show_metrics(context)

    if sub == "reset":
        metrics = getattr(context.agent, "metrics", None)
        if metrics is None:
            context.print("\n当前 Agent 不支持指标收集")
            return HANDLED
        metrics.reset()
        context.print("\n指标已重置")
        return HANDLED

    context.print("\n用法: metrics[:status|:reset]")
    return HANDLED
