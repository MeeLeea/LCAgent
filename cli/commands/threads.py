"""Conversation thread and export commands."""

from __future__ import annotations

import asyncio
import os

from langchain_core.messages import HumanMessage

from .types import HANDLED, CommandContext, CommandOutcome


def _write_export_file(path: str, text: str) -> None:
    """把导出文本写入文件(阻塞 IO,由 to_thread 放到线程池执行)"""
    with open(path, "w", encoding="utf-8") as file:
        file.write(text)


async def manage_threads(context: CommandContext) -> CommandOutcome:
    while True:
        threads = await context.agent.session.alist_sessions()
        current = context.agent.session.current_session_id
        if not threads:
            context.print("\n暂无会话记录")
            break
        options = [await _athread_option(context, thread_id, current) for thread_id in threads]
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
            await _adelete_thread(context, selected[1])
            continue
        if selected == current:
            context.print(f"\n已在当前会话: {current}")
            break
        await context.agent.session.aswitch_session(str(selected))
        context.agent.memory.thread_id = str(selected)  # 同步 memory 指针
        messages = await context.agent.session.aget_messages()
        context.print(f"\n已切换到会话: {selected} (恢复 {len(messages or [])} 条历史消息)")
        break
    return HANDLED


def new_thread(context: CommandContext) -> CommandOutcome:
    old = context.agent.session.current_session_id
    new = context.agent.session.new_session()
    context.agent.memory.thread_id = new  # 同步 memory 指针
    context.print(f"\n已开启新会话: {new}")
    context.print(f"原会话 {old} 已保留,可用 'thread' 切回")
    return HANDLED


async def delete_thread_command(context: CommandContext, user_input: str) -> CommandOutcome:
    parts = user_input.split(None, 1)
    if len(parts) < 2:
        context.print("用法: thread:delete <thread_id>")
        return HANDLED
    await _adelete_thread(context, parts[1].strip())
    return HANDLED


async def export_thread(context: CommandContext, user_input: str) -> CommandOutcome:
    low = user_input.lower()
    rest = user_input[7:].strip() if low.startswith("export:") else user_input[6:].strip()
    parts = rest.split(None, 1)
    thread_id = parts[0] if parts else None
    text = await context.agent.session.aexport_session(thread_id)
    if not text.strip():
        context.print("\n该会话没有可导出的消息")
        return HANDLED
    if not path:
        exports_dir = os.path.join(context.base_dir, "exports")
        os.makedirs(exports_dir, exist_ok=True)
        safe_thread_id = thread_id or context.agent.session.current_session_id
        path = os.path.join(exports_dir, f"{safe_thread_id}.md")
    try:
        await asyncio.to_thread(_write_export_file, path, text)
        context.print(f"\n已导出对话到: {path} ({len(text)} 字符)")
    except OSError as error:
        context.print(f"\n导出失败: {error}")
        context.print("\n--- 对话内容预览 ---")
        context.print(text[:1000])
    return HANDLED


async def _athread_option(
    context: CommandContext,
    thread_id: str,
    current: str,
) -> tuple[str, str]:
    try:
        # 直接读取目标会话消息，不再临时变异 thread_id
        messages = await context.agent.session.aget_messages(session_id=thread_id) or []
        message_count = len(messages)
        # 用第一条用户消息作为会话标题,不调用 LLM,避免菜单渲染变慢
        preview = _messages_preview(messages)
    except (AttributeError, OSError, RuntimeError):
        message_count = 0
        preview = ""
    mark = " (当前)" if thread_id == current else ""
    # 有缩略时用缩略作为主标题,否则回退到 thread_id
    label_main = preview if preview else thread_id
    return f"{label_main}  [{message_count} 条消息]{mark}", thread_id


def _messages_preview(messages, max_len: int = 15) -> str:
    """取第一条用户消息作为会话标题,超长截断。

    无用户消息时返回空串,由调用方回退到 thread_id。

    Args:
        messages: 会话消息列表
        max_len: 标题最大长度

    Returns:
        第一条用户消息的摘要;无用户消息返回空串
    """
    for msg in messages:
        if isinstance(msg, HumanMessage):
            content = _message_text(msg.content)
            content = content.replace("\r\n", " ").replace("\n", " ").replace("\r", " ").strip()
            if content:
                if len(content) > max_len:
                    return content[:max_len] + "..."
                return content
    return ""


def _message_text(content: object) -> str:
    """将消息内容归一化为纯文本,支持字符串与内容块列表。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return " ".join(part for part in parts if part)
    return str(content)


async def _adelete_thread(context: CommandContext, thread_id: str) -> None:
    # 会话删除不可恢复，因此命令行入口始终执行二次确认。
    confirm = context.input(f"确认删除会话 '{thread_id}'? 此操作不可恢复 [y/N]: ").strip().lower()
    if confirm not in ("y", "yes"):
        context.print("已取消")
        return
    was_current = thread_id == context.agent.session.current_session_id
    if await context.agent.session.adelete_session(thread_id):
        context.print(f"\n已删除会话: {thread_id}")
        if was_current:
            context.print(f"当前会话已被删除,自动切换到: {context.agent.session.current_session_id}")
        return
    context.print(f"\n删除失败:会话 '{thread_id}' 不存在或数据库错误")
