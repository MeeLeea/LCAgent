"""Safety policy commands."""

from __future__ import annotations

from .types import HANDLED, CommandContext, CommandOutcome, JsonValue


def show_safety(context: CommandContext) -> CommandOutcome:
    config = context.safety_backend.load_config()
    context.print("\n安全策略:")
    context.print("-" * 50)
    context.print(f"  模式:        {config['mode']}")
    context.print(f"  危险确认:    {'开启' if config['confirm_dangerous'] else '关闭'}")
    context.print(f"  追加黑名单:  {config.get('blacklist') or '(空)'}")
    if config["mode"] == "whitelist":
        context.print(f"  白名单命令:  {config.get('whitelist')}")
    context.print("-" * 50)
    return HANDLED


def safety_command(context: CommandContext, user_input: str) -> CommandOutcome:
    rest = user_input[7:].strip().lower()
    parts = rest.split(None, 1)
    sub = parts[0]
    if sub == "mode":
        return _set_mode(context, parts[1].strip() if len(parts) > 1 else "")
    if sub == "confirm":
        return _set_confirm(context, parts[1].strip() if len(parts) > 1 else "")
    context.print("\n用法: safety:mode <blacklist|whitelist> | safety:confirm <on|off>")
    return HANDLED


def _set_mode(context: CommandContext, mode: str) -> CommandOutcome:
    if mode not in ("blacklist", "whitelist"):
        context.print("\n用法: safety:mode <blacklist|whitelist>")
        return HANDLED
    config: dict[str, JsonValue] = context.safety_backend.load_config()
    config["mode"] = mode
    if context.safety_backend.save_config(config):
        context.print(f"\n已切换安全模式: {mode}")
    else:
        context.print("\n保存失败")
    return HANDLED


def _set_confirm(context: CommandContext, value: str) -> CommandOutcome:
    enabled = value in ("on", "true", "1", "启用")
    config: dict[str, JsonValue] = context.safety_backend.load_config()
    config["confirm_dangerous"] = enabled
    if context.safety_backend.save_config(config):
        context.print(f"\n危险命令确认已{'开启' if enabled else '关闭'}")
    else:
        context.print("\n保存失败")
    return HANDLED
