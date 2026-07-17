"""Ordered interactive command dispatcher."""

from __future__ import annotations

from . import core, execution, mcp, memory, provider, safety, skills, threads
from .types import BREAK, HANDLED, UNHANDLED, CommandContext, CommandOutcome


def dispatch_command(context: CommandContext, user_input: str) -> CommandOutcome:
    low = user_input.lower()
    if not user_input:
        return UNHANDLED
    if low in ["quit", "exit"]:
        context.print("再见!")
        return BREAK
    if low == "help":
        return core.show_help(context)
    if low == "info":
        return core.show_info(context)
    if low == "thread" or low == "threads":
        return threads.manage_threads(context)
    if low == "thread:new":
        return threads.new_thread(context)
    if low.startswith("thread:delete"):
        return threads.delete_thread_command(context, user_input)
    if low == "export" or low.startswith("export:"):
        return threads.export_thread(context, user_input)
    if low == "tools":
        return core.show_tools(context)
    if low.startswith("clear"):
        return memory.clear_memory(context, user_input)
    if low in ("compress", "压缩"):
        return memory.compress_memory(context)
    if low.startswith("switch"):
        return provider.switch_provider(context, user_input)
    if low == "model" or low == "models":
        return provider.choose_model(context)
    if low.startswith("model:") or (low.startswith("model ") and not low.startswith("model:")):
        return provider.switch_model(context, user_input)
    if low == "mcp":
        return mcp.show_mcp(context)
    if low == "mcp:reload":
        return mcp.reload_mcp(context)
    if low.startswith("mcp:add"):
        return mcp.add_mcp(context, user_input)
    if low.startswith("mcp:remove"):
        return mcp.remove_mcp(context, user_input)
    if low.startswith("mcp:toggle"):
        return mcp.toggle_mcp(context, user_input)
    if low == "skill" or low == "skills":
        return skills.list_skills(context)
    if low.startswith("skill:"):
        return skills.skill_command(context, user_input)
    if low == "safety":
        return safety.show_safety(context)
    if low.startswith("safety:"):
        return safety.safety_command(context, user_input)
    if low.startswith("json:"):
        return execution.json_mode(context, user_input)
    if low.startswith("react:"):
        return execution.react_mode(context, user_input)
    if low.startswith("cot:"):
        return execution.cot_mode(context, user_input)
    return execution.chat_mode(context, user_input)


__all__ = ["CommandContext", "CommandOutcome", "dispatch_command", "HANDLED"]
