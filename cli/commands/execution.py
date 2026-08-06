"""Agent execution command modes."""

from __future__ import annotations

import json

from .types import HANDLED, CommandContext, CommandOutcome


async def json_mode(context: CommandContext, user_input: str) -> CommandOutcome:
    task = user_input[5:].strip()
    if not task:
        context.print("\n用法: json:<任务描述>  (要求 Agent 以 JSON 对象返回结果)")
        return HANDLED
    full = (
        task + "\n\n【输出要求】请只输出一个合法的 JSON 对象,"
        "不要包含 ``` 代码块标记或其它任何解释性文字,"
        "直接用 JSON 表达任务结果。"
    )
    result = await context.run_structured_until_completion(context.agent, full)
    parsed = context.agent.llm.extract_json(result)
    if parsed is not None:
        context.print("\n解析成功,JSON 结构:")
        context.print(json.dumps(parsed, ensure_ascii=False, indent=2))
    else:
        context.print("\n未能解析为 JSON,原始输出:")
        context.print(result)
    return HANDLED


async def react_mode(context: CommandContext, user_input: str) -> CommandOutcome:
    task = user_input[6:].strip()
    result = await context.run_structured_until_completion(context.agent, task)
    context.print(f"\n助手: {result}")
    return HANDLED


def cot_mode(context: CommandContext, user_input: str) -> CommandOutcome:
    task = user_input[4:].strip()
    result = context.agent.cot(task)
    context.print(f"\n助手: {result}")
    return HANDLED


async def chat_mode(context: CommandContext, user_input: str) -> CommandOutcome:
    result = await context.chat_until_completion(context.agent, user_input)
    context.print(f"\n助手: {result}")
    return HANDLED
