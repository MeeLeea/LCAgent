"""运行时日志级别命令。

在交互式 CLI 中动态调整全局日志级别，无需重启进程：
  - ``log``            查看当前级别 + 方向键菜单选择切换
  - ``log:<level>``    直接切换到指定级别（如 log:debug）
"""

from __future__ import annotations

from agent.logging_config import LOG_LEVELS, get_log_level_name, set_log_level

from .types import HANDLED, CommandContext, CommandOutcome


def log_command(context: CommandContext, user_input: str) -> CommandOutcome:
    """log 命令路由

    用法:
      log            查看当前级别并进入菜单选择
      log:<level>    直接设置级别 (debug|info|warning|error|critical)
    """
    low = user_input.lower()
    # 带冒号参数直接设置；否则进入交互菜单。
    if low.startswith("log:"):
        return _set_level(context, user_input.split(":", 1)[1].strip())
    return _choose_level(context)


def _choose_level(context: CommandContext) -> CommandOutcome:
    current = get_log_level_name()
    context.print(f"\n当前日志级别: {current}")
    selected = context.select_menu(
        "选择日志级别",
        [(name, name) for name in LOG_LEVELS],
        current=current,
    )
    if selected is None:
        return HANDLED
    return _set_level(context, str(selected))


def _set_level(context: CommandContext, level: str) -> CommandOutcome:
    if not level:
        valid = "|".join(name.lower() for name in LOG_LEVELS)
        context.print(f"\n用法: log:<{valid}>  或  log 进入菜单选择")
        return HANDLED
    try:
        set_log_level(level)
    except (ValueError, TypeError) as error:
        context.print(f"\n设置失败: {error}")
        return HANDLED
    context.print(f"\n日志级别已切换为: {get_log_level_name()}")
    return HANDLED
