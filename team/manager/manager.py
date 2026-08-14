"""
Manager Agent - 负责拆解任务并生成执行计划
"""
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

    def summarize_context(self, memory_text: str) -> str:
        """
        把当前会话记忆提炼成与任务相关的简洁上下文摘要

        供工作流 summarize 节点调用:只消费一次原始记忆,下游各节点
        通过 WorkflowState.context_summary 获取蒸馏后的上下文。

        Args:
            memory_text: 原始记忆文本(当前会话 + 长期记忆)

        Returns:
            上下文摘要字符串;memory_text 为空时直接返回空串,跳过 LLM 调用
        """
        memory_text = memory_text.strip()
        if not memory_text:
            return ""

        messages = [
            {"role": "system", "content": self.get_template("summarize_context")},
            {"role": "user", "content": memory_text},
        ]
        return self.llm.chat(messages).strip()

    def plan_task(self, task: str, context_summary: str = "", injector: PromptInjector | None = None) -> str:
        """
        拆解任务并生成执行计划(结合记忆上下文摘要)

        供工作流 manager_plan 节点调用:渲染 manager_plan 模板 → 可选技能注入
        → LLM 生成计划文本。为同步方法,节点层经 asyncio.to_thread 异步执行。

        Args:
            task: 用户原始任务
            context_summary: 上下文摘要(可为空串)
            injector: 技能注入器;为 None 时跳过技能注入

        Returns:
            生成的执行计划文本
        """
        template = self.get_template("manager_plan")
        prompt = self.render_template(template, task=task, context_summary=context_summary)
        if injector is not None:
            prompt = injector.inject_into_prompt(prompt, task)
        return self.invoke(prompt)
