"""
Manager Agent - 负责拆解任务并生成执行计划
"""
from collections.abc import Sequence
from typing import ClassVar

from graph.registry import register_agent
from team.base import PromptInjector, TeamAgent


@register_agent("manager", "team/manager/agent_config.json", tools=None)
class ManagerAgent(TeamAgent):
    """
    管理者 Agent,负责任务拆解与规划

    继承 TeamAgent 轻量基类,纯文本推理模式(不使用工具)
    """

    # 工作流节点提示词的默认模板(仅 AGENT.md 缺失或未定义小节时兜底)
    default_templates: ClassVar[dict[str, str]] = {
        "manager_plan": (
            "请为以下任务制定详细的执行计划:\n\n"
            "{task}\n\n"
            "记忆上下文摘要:\n"
            "{context_summary}"
        ),
        "summarize_context": (
            "你是一个工作流上下文提炼助手。请从以下对话历史与长期记忆中,"
            "提炼出与当前任务相关的关键背景(用户偏好、已确定的事实、进行中的事项),"
            "形成简洁的中文摘要。要求:\n"
            "1. 只保留与任务相关的信息,去除无关与重复内容\n"
            "2. 按主题分条目组织,使用 '- ' 开头\n"
            "3. 保持事实准确,不要添加推测内容\n"
            "4. 没有相关背景时,直接输出'无相关背景'"
        ),
    }

    async def asummarize_context(self, memory_text: str, config: dict | None = None) -> str:
        """
        异步版 summarize_context(供 summarize 节点直接 await 调用)

        语义与同步版一致:memory_text 为空时短路返回空串,跳过 LLM 调用;
        否则经 ``_astream_messages`` 流式聚合(TOKEN 增量可透传到外层事件流)。
        """
        memory_text = memory_text.strip()
        if not memory_text:
            return ""

        messages = [
            {"role": "system", "content": self.get_template("summarize_context")},
            {"role": "user", "content": memory_text},
        ]
        chunks: list[str] = []
        async for chunk in self._astream_messages(messages, config):
            chunks.append(chunk)
        return "".join(chunks).strip()

    async def aplan_task(
        self,
        task: str,
        context_summary: str = "",
        injector: PromptInjector | None = None,
        config: dict | None = None,
        active_names: Sequence[str] = (),
    ) -> str:
        """
        异步版 plan_task(供 manager_plan 节点直接 await 调用)

        模板渲染/技能注入保持同步(纯 CPU),仅 LLM 调用走异步流式
        (``await self.ainvoke``),token 增量可透传到外层事件流。

        active_names 由节点函数从 state["active_skills"] 取值传入,
        使手动加载的技能在 graph 节点生效。
        """
        template = self.get_template("manager_plan")
        prompt = self.render_template(template, task=task, context_summary=context_summary)
        if injector is not None:
            prompt = injector.inject_into_prompt(prompt, task, active_names)
        return await self.ainvoke(prompt, config)
