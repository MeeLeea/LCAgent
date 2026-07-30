"""Conversation thread and export commands."""

from __future__ import annotations

import os

from langchain_core.messages import HumanMessage

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
    try:
        # 直接读取目标会话消息，不再临时变异 thread_id
        messages = context.agent.memory.get_messages(thread_id=thread_id) or []
        message_count = len(messages)
        # 标题缓存挂在 agent 实例上,同一次运行内不重复调 LLM
        cache = getattr(context.agent, "_thread_title_cache", None)
        if cache is None:
            cache = {}
            try:
                setattr(context.agent, "_thread_title_cache", cache)
            except AttributeError:
                pass
        # 对前5条 HumanMessage 调 LLM 生成标题
        preview = _messages_preview(messages, context.llm, cache, thread_id, message_count)
    except (AttributeError, OSError, RuntimeError):
        message_count = 0
        preview = ""
    mark = " (当前)" if thread_id == current else ""
    # 有缩略时用缩略作为主标题,否则回退到 thread_id
    label_main = preview if preview else thread_id
    return f"{label_main}  [{message_count} 条消息]{mark}", thread_id


def _messages_preview(
    messages,
    llm,
    cache: dict,
    thread_id: str,
    message_count: int,
    required: int = 3,
    max_len: int = 15,
) -> str:
    """对前 N 条 HumanMessage 调用 LLM 生成标题,用于会话菜单显示。

    策略:
    - 只取 HumanMessage(用户消息最能体现会话主题)
    - 不足 required 条时返回空(回退到 thread_id)
    - 满 required 条时拼接内容,调用 LLM 生成 max_len 字以内的标题
    - 结果按 thread_id + message_count 缓存,消息数变化时重新生成
    - LLM 调用失败时回退到纯文本截断
    """
    # 命中缓存直接返回(消息数没变就用旧标题)
    if thread_id in cache:
        cached_title, cached_count = cache[thread_id]
        if cached_count == message_count:
            return cached_title

    # 取前 required 条 HumanMessage
    human_contents = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            content = content.replace("\n", " ").replace("\r", " ").strip()
            if content:
                human_contents.append(content)
        if len(human_contents) >= required:
            break

    # 不足 required 条则跳过(返回空,让调用方回退到 thread_id)
    if len(human_contents) < required:
        return ""

    # 调用 LLM 生成标题
    dialog_text = "\n".join(
        f"{i + 1}. {c}" for i, c in enumerate(human_contents[:required])
    )
    try:
        title = llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        f"你是标题生成器。用{max_len}字以内的中文概括对话主题,"
                        "只输出标题文本,不要加引号、标点符号或任何解释。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"以下是用户的多轮提问,请生成一个{max_len}字以内的标题:\n\n"
                        f"{dialog_text}"
                    ),
                },
            ],
            temperature=0.3,
            max_tokens=30,
        )
        # 清理 LLM 输出(去引号、换行、首尾空白)
        title = title.strip().strip("\"'""''").strip()
        if title:
            title = title[:max_len]
            cache[thread_id] = (title, message_count)
            return title
    except Exception:
        pass

    # LLM 失败回退:纯文本截断
    summary = " ".join(human_contents[:required])
    summary = " ".join(summary.split())
    if len(summary) > max_len:
        summary = summary[:max_len] + "..."
    return summary


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
