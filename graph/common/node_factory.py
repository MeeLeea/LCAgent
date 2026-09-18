"""通用 LLM 节点工厂 — 消除工作流节点重复样板代码。

create_llm_node 生成统一签名的异步节点函数，封装
render_template → 基础提示词前置 → inject_into_prompt → run_team_turn_with_interrupt
→ return 链路，通过回调参数支持各节点的差异化逻辑。
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from graph.common.interrupt_forward import run_team_turn_with_interrupt
from llm.config import load_agent_rules

if TYPE_CHECKING:
    from team.base import TeamAgent


def create_llm_node(
    template_name: str,
    output_field: str,
    template_vars_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    match_text_fn: Callable[[dict[str, Any]], str] | None = None,
    exclude_skills: tuple[str, ...] = (),
    inject_skill: bool = True,
    extra_return_fn: Callable[[dict[str, Any], str], dict[str, Any]] | None = None,
    base_prompts: str | None = None,
) -> Callable[..., dict[str, Any]]:
    """通用 LLM 节点工厂 — 消除重复的样板代码。

    生成的节点函数统一签名: ``async def node(state, agent, injector=None, config=None)``

    Args:
        template_name: 模板名称（传给 agent.get_template）
        output_field: 结果写入的 state 字段名
        template_vars_fn: 从 state 提取模板变量的函数，返回 render_template 的 kwargs;
            为 None 时默认返回 ``{"task": state.get("task", "")}``
        match_text_fn: 从 state 提取技能匹配文本的函数;
            为 None 时默认返回 state.get("task", "")
        exclude_skills: 注入时排除的技能名列表
        inject_skill: 是否注入技能块（False 用于摘要节点等不需技能的场景）
        extra_return_fn: 额外的 state 返回字段函数 ``(state, result) -> dict``;
            用于需要返回 round/output_files 等额外字段的节点
        base_prompts: 基础提示词;默认 None 表示按角色能力**运行时**解析
            （见 ``llm.config.load_agent_rules``）：持有工具的 Agent 注入
            「重要规则」+「工具规则」，纯文本角色的 Agent 仅注入「重要规则」。
            因 ``create_llm_node`` 在模块级定义、构建期拿不到 Agent 实例，
            故在节点执行时按 ``agent.tools`` 判定。传空串可关闭注入，
            传非空字符串则覆盖默认解析结果

    Returns:
        异步节点函数
    """

    async def node_fn(
        state: dict[str, Any],
        agent: TeamAgent,
        injector: Any = None,
        config: Optional[RunnableConfig] = None,  # noqa: UP045 - LangGraph 注解判定仅接受该字符串形态
    ) -> dict[str, Any]:
        if template_vars_fn is not None:
            template_vars = template_vars_fn(state)
        else:
            template_vars = {"task": state.get("task", "")}

        prompt = agent.render_template(
            agent.get_template(template_name), **template_vars
        )

        # 基础提示词按角色能力解析（有工具才附带「工具规则」），空片段自动跳过
        rules = (
            load_agent_rules(include_tool_rules=bool(getattr(agent, "tools", None)))
            if base_prompts is None
            else base_prompts
        )
        prompt = "\n\n".join(part for part in (rules, prompt) if part)

        if inject_skill and injector is not None:
            match_text = match_text_fn(state) if match_text_fn else state.get("task", "")
            prompt = injector.inject_into_prompt(
                prompt, match_text, exclude_skills=exclude_skills
            )

        result = await run_team_turn_with_interrupt(agent, prompt, config)

        ret: dict[str, Any] = {
            output_field: result,
            "messages": [AIMessage(content=result)],
        }
        if extra_return_fn is not None:
            ret.update(extra_return_fn(state, result))
        return ret

    return node_fn
