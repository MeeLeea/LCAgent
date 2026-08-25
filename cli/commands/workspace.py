"""工作空间(workspace)管理命令 - 绑定/查看/清除当前会话的工作目录。

绑定后，文件类与执行类工具的路径将被限制在 workspace 内，
防止 Agent 越权访问其他目录（由 WorkspaceSecurityMW 拦截）。

子命令:
    workspace            查看当前会话绑定的 workspace
    workspace <path>     设置/修改当前会话的 workspace
    workspace:clear      清除当前会话的 workspace 绑定
    workspace:help       显示帮助
"""

from __future__ import annotations

from .types import HANDLED, CommandContext, CommandOutcome


async def workspace_command(context: CommandContext, user_input: str) -> CommandOutcome:
    """workspace 命令入口，按子命令分发。

    Args:
        context: 命令运行时上下文
        user_input: 用户原始输入（含 ``workspace`` 前缀）

    Returns:
        HANDLED（始终处理，不中断主循环）
    """
    low = user_input.lower().strip()

    # 子命令优先匹配，避免 ``workspace clear`` 被当成绑定路径 "clear"
    if low in ("workspace:clear", "workspace clear"):
        return await _aclear_workspace(context)
    if low in ("workspace:help", "workspace help"):
        return _show_help(context)

    # workspace <path> 或 workspace（无参数=查看）
    parts = user_input.split(None, 1)
    path = parts[1].strip() if len(parts) > 1 else ""
    if not path:
        return await _ashow_workspace(context)
    return await _aset_workspace(context, path)


async def _ashow_workspace(context: CommandContext) -> CommandOutcome:
    """查看当前会话绑定的 workspace。"""
    sid = context.agent.session.current_session_id
    ws = await context.agent.session.aget_workspace(sid)
    context.print(f"\n当前会话: {sid}")
    if ws:
        context.print(f"工作空间: {ws}")
    else:
        context.print("工作空间: (未绑定)")
        context.print("提示: 用 'workspace <path>' 绑定工作目录")
    return HANDLED


async def _aset_workspace(context: CommandContext, path: str) -> CommandOutcome:
    """设置/修改当前会话的 workspace 绑定。"""
    sid = context.agent.session.current_session_id
    try:
        real = await context.agent.session.aset_workspace(path, sid)
    except ValueError as error:
        context.print(f"\n绑定失败: {error}")
        return HANDLED
    context.print(f"\n已为会话 {sid} 绑定工作空间:")
    context.print(f"  {real}")
    context.print("文件/执行类工具调用将被限制在该目录内。")
    return HANDLED


async def _aclear_workspace(context: CommandContext) -> CommandOutcome:
    """清除当前会话的 workspace 绑定。"""
    sid = context.agent.session.current_session_id
    existed = await context.agent.session.aclear_workspace(sid)
    if existed:
        context.print(f"\n已清除会话 {sid} 的工作空间绑定")
    else:
        context.print(f"\n会话 {sid} 原本未绑定工作空间")
    return HANDLED


def _show_help(context: CommandContext) -> CommandOutcome:
    """显示 workspace 命令帮助。"""
    context.print("\n工作空间(workspace)命令:")
    context.print("  workspace            查看当前会话绑定的 workspace")
    context.print("  workspace <path>    设置/修改当前会话的 workspace")
    context.print("  workspace:clear     清除当前会话的 workspace 绑定")
    context.print("  workspace:help      显示本帮助")
    context.print("\n说明:")
    context.print("  绑定后，文件类与执行类工具的路径将被限制在 workspace 内，")
    context.print("  防止 Agent 越权访问其他目录。")
    return HANDLED


__all__ = ["workspace_command"]
