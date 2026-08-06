"""Skill loading commands."""

from __future__ import annotations

import asyncio

from .types import CommandContext, CommandOutcome, HANDLED


def list_skills(context: CommandContext) -> CommandOutcome:
    skills = context.agent.list_skills()
    if not skills:
        context.print("\n当前没有可用技能(目录 .agents/skills 为空)")
        return HANDLED
    context.print("\n可用技能:")
    context.print("-" * 50)
    for skill in skills:
        context.print(f"  • {skill['name']}")
        if skill["description"]:
            desc = skill["description"]
            if len(desc) > 80:
                desc = desc[:80] + "..."
            context.print(f"      {desc}")
    context.print("-" * 50)
    context.print("使用 'skill:<name>' 加载技能到当前会话, 'skill:clear' 清空已加载技能")
    return HANDLED


def skill_command(context: CommandContext, user_input: str) -> CommandOutcome:
    rest = user_input[6:].strip()
    parts = rest.split(None, 1)
    sub = parts[0].strip().lower() if parts else ""
    task_text = parts[1].strip() if len(parts) > 1 else ""
    if sub in ("clear", "清空", "reset"):
        asyncio.run(context.agent.aclear_skills())
        context.print("\n已清空手动加载的技能")
        return HANDLED
    if not sub:
        context.print("\n用法: skill:<name> [任务]  或  skill (列出所有)")
        return HANDLED
    matched = _match_skill(context, sub)
    if matched is None:
        available = [skill["name"] for skill in context.agent.list_skills()]
        context.print(f"\n未找到技能: {sub}")
        context.print(f"可用: {', '.join(available) or '(无)'}")
        return HANDLED
    if asyncio.run(context.agent.aload_skill(matched)):
        context.print(f"\n已加载技能: {matched} (将注入后续对话的 system prompt)")
        if not context.agent.auto_match_skills:
            context.print("提示: 自动匹配已关闭,本技能仅手动加载生效")
        if task_text:
            result = context.run_structured_until_completion(context.agent, task_text)
            context.print(f"\n助手: {result}")
    else:
        context.print(f"\n加载失败: {matched}")
    return HANDLED


def _match_skill(context: CommandContext, sub: str) -> str | None:
    # 精确匹配优先，避免短名称误命中多个包含关系相近的技能。
    for skill in context.agent.list_skills():
        if skill["name"].lower() == sub:
            return skill["name"]
    # 精确匹配失败后才允许便捷的子串匹配。
    for skill in context.agent.list_skills():
        if sub in skill["name"].lower():
            return skill["name"]
    return None
