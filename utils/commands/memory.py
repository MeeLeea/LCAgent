"""Memory management commands."""

from __future__ import annotations

from .types import CommandContext, CommandOutcome, HANDLED


def clear_memory(context: CommandContext, user_input: str) -> CommandOutcome:
    parts = user_input.split(None, 1)
    target = parts[1].strip().lower() if len(parts) > 1 else "long"
    if target in ("long", "长期"):
        context.agent.memory.clear_long_term()
        context.print("\n已清空长期记忆(并删除 memory.json)")
    elif target in ("short", "短期"):
        context.agent.memory.clear_short_term()
        context.print("\n已清空短期记忆")
    elif target in ("all", "全部"):
        context.agent.memory.clear_long_term()
        context.agent.memory.clear_short_term()
        context.print("\n已清空全部记忆(长期+短期)")
    else:
        context.print("\n用法: clear [long|short|all]  (默认 long)")
    return HANDLED


def compress_memory(context: CommandContext) -> CommandOutcome:
    mem = context.agent.get_memory_summary()
    if mem["long_term_count"] == 0:
        context.print("\n没有长期记忆可压缩")
        return HANDLED
    context.print(f"\n开始压缩长期记忆 (共 {mem['long_term_count']} 条)...")
    result = context.agent.compress_memory()
    if result["success"]:
        context.print("压缩完成！")
        context.print(f"  原记忆条数:   {result['original_count']} 条")
        context.print(f"  原字符数:     {result['original_chars']} 字符")
        context.print(f"  压缩后字符数: {result['compressed_chars']} 字符")
        ratio = (1 - int(result["compressed_chars"]) / max(int(result["original_chars"]), 1)) * 100
        context.print(f"  压缩率:       {ratio:.1f}%")
        context.print("\n--- 摘要内容 ---")
        context.print(str(result["summary"]))
        context.print("--- 已保存到 memory.json ---")
    else:
        context.print(f"\n压缩失败: {result.get('error', '未知错误')}")
    return HANDLED
