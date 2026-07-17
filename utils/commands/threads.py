"""Conversation thread and export commands."""

from __future__ import annotations

import os

from .types import CommandContext, CommandOutcome, HANDLED


def manage_threads(context: CommandContext) -> CommandOutcome:
    while True:
        threads = context.agent.memory.list_threads()
        current = context.agent.memory.thread_id
        if not threads:
            context.print("\n暂无会话记录")
            break
        options = [_thread_option(context, thread_id, current) for thread_id in threads]
        selected = context.select_menu(
            f"选择会话 (共 {len(threads)} 个,↑↓ 选择,Enter 切换)",
            options,
            current=current,
            action_keys={b"\x04": "delete"},
            hint="  (↑↓ 选择, Enter 切换, Ctrl+D 删除, Esc 取消)",
        )
        if selected is None:
            break
        if isinstance(selected, tuple) and selected[0] == "delete":
            _delete_thread(context, selected[1])
            continue
        if selected == current:
            context.print(f"\n已在当前会话: {current}")
            break
        context.agent.memory.switch_thread(str(selected))
        messages = context.agent.memory.get_messages()
        context.print(f"\n已切换到会话: {selected} (恢复 {len(messages or [])} 条历史消息)")
        break
    return HANDLED


def new_thread(context: CommandContext) -> CommandOutcome:
    old = context.agent.memory.thread_id
    new = context.agent.memory.new_thread()
    context.print(f"\n已开启新会话: {new}")
    context.print(f"原会话 {old} 已保留,可用 'thread' 切回")
    return HANDLED


def delete_thread_command(context: CommandContext, user_input: str) -> CommandOutcome:
    parts = user_input.split(None, 1)
    if len(parts) < 2:
        context.print("用法: thread:delete <thread_id>")
        return HANDLED
    _delete_thread(context, parts[1].strip())
    return HANDLED


def export_thread(context: CommandContext, user_input: str) -> CommandOutcome:
    low = user_input.lower()
    rest = user_input[7:].strip() if low.startswith("export:") else user_input[6:].strip()
    parts = rest.split(None, 1)
    thread_id = parts[0] if parts else None
    path = parts[1] if len(parts) > 1 else None
    text = context.agent.memory.export_thread(thread_id)
    if not text.strip():
        context.print("\n该会话没有可导出的消息")
        return HANDLED
    if not path:
        exports_dir = os.path.join(context.base_dir, "exports")
        os.makedirs(exports_dir, exist_ok=True)
        safe_thread_id = thread_id or context.agent.memory.thread_id
        path = os.path.join(exports_dir, f"{safe_thread_id}.md")
    try:
        with open(path, "w", encoding="utf-8") as file:
            file.write(text)
        context.print(f"\n已导出对话到: {path} ({len(text)} 字符)")
    except OSError as error:
        context.print(f"\n导出失败: {error}")
        context.print("\n--- 对话内容预览 ---")
        context.print(text[:1000])
    return HANDLED


def _thread_option(context: CommandContext, thread_id: str, current: str) -> tuple[str, str]:
    saved = context.agent.memory.thread_id
    try:
        # 只为读取消息数临时切换线程，finally 必须恢复真实的当前会话。
        context.agent.memory.thread_id = thread_id
        messages = context.agent.memory.get_messages() or []
        message_count = len(messages)
    except (AttributeError, OSError, RuntimeError):
        message_count = 0
    finally:
        context.agent.memory.thread_id = saved
    mark = " (当前)" if thread_id == current else ""
    return f"{thread_id}  [{message_count} 条消息]{mark}", thread_id


def _delete_thread(context: CommandContext, thread_id: str) -> None:
    # 会话删除不可恢复，因此命令行入口始终执行二次确认。
    confirm = context.input(f"确认删除会话 '{thread_id}'? 此操作不可恢复 [y/N]: ").strip().lower()
    if confirm not in ("y", "yes"):
        context.print("已取消")
        return
    was_current = thread_id == context.agent.memory.thread_id
    if context.agent.memory.delete_thread(thread_id):
        context.print(f"\n已删除会话: {thread_id}")
        if was_current:
            context.print(f"当前会话已被删除,自动切换到: {context.agent.memory.thread_id}")
        return
    context.print(f"\n删除失败:会话 '{thread_id}' 不存在或数据库错误")
