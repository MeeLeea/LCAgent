"""
Terminator Agent - 负责汇总 Worker 执行结果并返回最终答案
"""
from typing import ClassVar

from graph.registry import register_agent
from team.base import PromptInjector, TeamAgent


@register_agent("terminator", "team/terminator/agent_config.json", tools=None)
class TerminatorAgent(TeamAgent):
    """
    终结者 Agent,负责汇总工作结果并返回最终答案给用户
    
    继承 TeamAgent 轻量基类,纯文本推理模式(不使用工具)
    """

    # 工作流节点提示词的默认模板(仅 AGENT.md 缺失或未定义小节时兜底)
    default_templates: ClassVar[dict[str, str]] = {
        "terminator_final": (
            "原始任务: {task}\n\n"
            "执行计划: {plan}\n\n"
            "执行结果: {worker_result}\n\n"
            "记忆上下文摘要:\n"
            "{context_summary}\n\n"
            "请汇总以上信息,为用户提供清晰的最终答案。"
        ),
    }

    def finalize(
        self,
        task: str,
        plan: str,
        worker_result: str,
        context_summary: str = "",
        injector: PromptInjector | None = None,
    ) -> str:
        """
        汇总执行结果并生成最终答案(结合记忆上下文摘要与技能注入)

        供工作流 terminator_final 节点调用:渲染 terminator_final 模板 → 可选
        技能注入 → LLM 生成最终答案。为同步方法,节点层经 asyncio.to_thread 异步执行。

        Args:
            task: 用户原始任务
            plan: Manager 拆解出的执行计划
            worker_result: Worker 执行结果
            context_summary: 上下文摘要(可为空串)
            injector: 技能注入器;为 None 时跳过技能注入

        Returns:
            最终答案文本
        """
        template = self.get_template("terminator_final")
        prompt = self.render_template(
            template,
            task=task,
            plan=plan,
            worker_result=worker_result,
            context_summary=context_summary,
        )
        if injector is not None:
            prompt = injector.inject_into_prompt(prompt, task)
        return self.invoke(prompt)

    async def afinalize(
        self,
        task: str,
        plan: str,
        worker_result: str,
        context_summary: str = "",
        injector: PromptInjector | None = None,
        config: dict | None = None,
    ) -> str:
        """
        异步版 finalize(供 terminator_final 节点直接 await 调用)

        模板渲染/技能注入保持同步(纯 CPU),仅 LLM 调用走异步流式
        (``await self.ainvoke``),token 增量可透传到外层事件流。
        """
        template = self.get_template("terminator_final")
        prompt = self.render_template(
            template,
            task=task,
            plan=plan,
            worker_result=worker_result,
            context_summary=context_summary,
        )
        if injector is not None:
            prompt = injector.inject_into_prompt(prompt, task)
        return await self.ainvoke(prompt, config)
