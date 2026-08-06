"""团队角色切换命令 - 列出 team/ 可用角色并按用户选择重建主 Agent。

支持命令:
    role / roles         - 列出可用角色(方向键选择切换)
    role:<name>          - 直接切换到指定角色
    role:<name> <task>   - 切换角色并立即执行该任务
"""
from __future__ import annotations

from .types import HANDLED, CommandContext, CommandOutcome


async def role_command(context: CommandContext, user_input: str) -> CommandOutcome:
    """处理 role 相关命令(见模块 docstring)。"""
    low = user_input.strip().lower()

    # 无参数:列出可用角色并进入选择菜单
    if low in ("role", "roles"):
        return await _choose_role(context)

    # 带参数: role:<name> [task]
    rest = user_input[5:].strip()  # 去掉 "role:"
    if not rest:
        context.print("\n用法: role:<name> [任务]  或  role (列出所有)")
        return HANDLED

    parts = rest.split(None, 1)
    role_name = parts[0].strip()
    task_text = parts[1].strip() if len(parts) > 1 else ""
    return await _switch_role(context, role_name, task_text)


async def _choose_role(context: CommandContext) -> CommandOutcome:
    """从 team/ 扫描可用角色,渲染选择菜单后切换到选中角色。"""
    # 函数内延迟导入,避免命令层与 agent 模块的循环依赖
    from agent.role_sw import get_available_team_roles

    roles = get_available_team_roles()
    if not roles:
        context.print("\n当前没有可用团队角色(目录 team/ 为空或缺少 agent_config.json/AGENT.md)")
        return HANDLED

    selected = context.select_menu(
        "选择团队角色",
        [(name, name) for name in roles],
        current=getattr(context.agent, "name", None),
    )
    if selected is None:
        return HANDLED
    return await _switch_role(context, str(selected), "")


async def _switch_role(context: CommandContext, role_name: str, task_text: str) -> CommandOutcome:
    """按角色名重建主 Agent,可选地在切换后立即执行任务。"""
    try:
        # arebuild_from_team_dir 为异步入口,就地把 AgentCore 切换为目标角色
        await context.agent.arebuild_from_team_dir(role_name, task=task_text)
    except KeyError as error:
        # 角色不存在:补充展示可用角色,便于用户重试
        from agent.role_sw import get_available_team_roles

        available = ", ".join(get_available_team_roles()) or "(无)"
        context.print(f"\n切换失败: {error}")
        context.print(f"可用角色: {available}")
        return HANDLED
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        context.print(f"\n切换失败: {error}")
        return HANDLED

    context.print(f"\n已切换到团队角色: {role_name}")

    if task_text:
        result = await context.run_structured_until_completion(context.agent, task_text)
        context.print(f"\n助手: {result}")
    return HANDLED
