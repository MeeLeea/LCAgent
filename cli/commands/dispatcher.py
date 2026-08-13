"""Ordered interactive command dispatcher."""

from __future__ import annotations

import inspect

from . import (
    core,
    execution,
    log_level,
    mcp,
    memory,
    metrics,
    provider,
    role,
    safety,
    skills,
    threads,
    workflow,
    workspace,
)
from .types import BREAK, HANDLED, UNHANDLED, CommandContext, CommandOutcome


async def _invoke(handler, *args) -> CommandOutcome:
    result = handler(*args)
    if inspect.isawaitable(result):
        result = await result
    return result


async def dispatch_command(context: CommandContext, user_input: str) -> CommandOutcome:
    low = user_input.lower()
    if not user_input:
        return UNHANDLED
    if low in ["quit", "exit"]:
        context.print("再见!")
        return BREAK
    if low == "help":
        return await _invoke(core.show_help, context)
    if low == "info":
        return await _invoke(core.show_info, context)
    if low == "thread" or low == "threads":
        return await _invoke(threads.manage_threads, context)
    if low == "thread:new":
        return await _invoke(threads.new_thread, context)
    if low.startswith("thread:delete"):
        return await _invoke(threads.delete_thread_command, context, user_input)
    if low == "export" or low.startswith("export:"):
        return await _invoke(threads.export_thread, context, user_input)
    if low == "tools":
        return await _invoke(core.show_tools, context)
    if low.startswith("clear"):
        return await _invoke(memory.clear_memory, context, user_input)
    if low in ("compress", "压缩"):
        return await _invoke(memory.compress_memory, context)
    if low in ("compact", "压缩上下文"):
        return await _invoke(memory.compact_context, context)
    if low.startswith("switch"):
        return await _invoke(provider.switch_provider, context, user_input)
    if low == "model" or low == "models":
        return await _invoke(provider.choose_model, context)
    if low.startswith("model:") or (low.startswith("model ") and not low.startswith("model:")):
        return await _invoke(provider.switch_model, context, user_input)
    if low == "mcp":
        return await _invoke(mcp.show_mcp, context)
    if low.startswith("mcp:reload"):
        return await _invoke(mcp.reload_mcp, context, user_input)
    if low.startswith("mcp:add"):
        return await _invoke(mcp.add_mcp, context, user_input)
    if low.startswith("mcp:remove"):
        return await _invoke(mcp.remove_mcp, context, user_input)
    if low.startswith("mcp:toggle"):
        return await _invoke(mcp.toggle_mcp, context, user_input)
    if low == "skill" or low == "skills":
        return await _invoke(skills.list_skills, context)
    if low.startswith("skill:"):
        return await _invoke(skills.skill_command, context, user_input)
    if low == "role" or low == "roles" or low.startswith("role:"):
        return await _invoke(role.role_command, context, user_input)
    if low == "safety":
        return await _invoke(safety.show_safety, context)
    if low.startswith("safety:"):
        return await _invoke(safety.safety_command, context, user_input)
    if low == "workflow" or low.startswith("workflow:"):
        return await _invoke(workflow.workflow_command, context, user_input)
    if low == "workspace" or low.startswith(("workspace ", "workspace:")):
        return await _invoke(workspace.workspace_command, context, user_input)
    if low == "metrics" or low.startswith("metrics:"):
        return await _invoke(metrics.metrics_command, context, user_input)
    if low == "log" or low.startswith("log:"):
        return await _invoke(log_level.log_command, context, user_input)
    if low.startswith("json:"):
        return await _invoke(execution.json_mode, context, user_input)
    if low.startswith("react:"):
        return await _invoke(execution.react_mode, context, user_input)
    if low.startswith("cot:"):
        return await _invoke(execution.cot_mode, context, user_input)
    return await _invoke(execution.chat_mode, context, user_input)


__all__ = ["HANDLED", "CommandContext", "CommandOutcome", "dispatch_command"]
