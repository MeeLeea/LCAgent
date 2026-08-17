"""
Architect Agent - 芯片架构工程师,负责架构方案设计、权衡分析、评审与规格文档输出
"""
from typing import ClassVar

from graph.registry import register_agent
from team.base import PromptInjector, TeamAgent


@register_agent("architect", "team/architect/agent_config.json", tools=None)
class ArchiAgent(TeamAgent):
    """
    架构师 Agent,负责芯片架构方案设计、PPA 权衡分析、多维度评审与规格文档输出

    继承 TeamAgent 轻量基类,纯文本推理模式(不使用工具);各工作流方法
    渲染对应 `## workflow:*` 小节 → 可选技能注入 → LLM 调用,同步方法由
    节点层经 asyncio.to_thread 异步执行。
    """

    # 工作流节点提示词的默认模板(仅 AGENT.md 缺失或未定义小节时兜底)
    default_templates: ClassVar[dict[str, str]] = {
        "architect_plan": (
            "请为以下芯片架构设计任务制定详细的执行计划:\n\n"
            "{task}\n\n"
        ),
        "architect_design": (
            "请为以下任务完成芯片架构方案设计:\n\n"
            "{task}\n\n"
        ),
        "architect_analyze": (
            "请对以下架构方案进行权衡分析(性能-面积-功耗PPA、风险与瓶颈):\n\n"
            "{task}\n\n"
        ),
        "architect_review": (
            "请从架构、RTL实现、后端物理、软件驱动、验证多维度评审以下方案:\n\n"
            "{task}\n\n"
        ),
        "architect_spec": (
            "请将以下架构方案整理为可直接交付RTL开发的规格文档:\n\n"
            "{task}\n\n"
        ),
    }

    def _run_architect_task(
        self,
        template_name: str,
        task: str,
        context_summary: str,
        injector: PromptInjector | None,
    ) -> str:
        """
        架构工作流方法通用执行体:渲染模板 → 可选技能注入 → LLM 调用

        Args:
            template_name: 工作流小节名(对应 AGENT.md 的 `## workflow:{name}`)
            task: 用户任务
            context_summary: 上下文摘要(可为空串)
            injector: 技能注入器;为 None 时跳过技能注入

        Returns:
            LLM 生成结果文本
        """
        template = self.get_template(template_name)
        prompt = self.render_template(template, task=task, context_summary=context_summary)
        if injector is not None:
            prompt = injector.inject_into_prompt(prompt, task)
        return self.invoke(prompt)

    def plan_task(self, task: str, context_summary: str = "", injector: PromptInjector | None = None) -> str:
        """
        为芯片架构设计任务制定执行计划(结合记忆上下文摘要)

        供工作流 architect_plan 节点调用。
        """
        return self._run_architect_task("architect_plan", task, context_summary, injector)

    def design_task(self, task: str, context_summary: str = "", injector: PromptInjector | None = None) -> str:
        """
        输出芯片架构方案设计(SoC 划分、总线/NoC 拓扑、存储层次、时钟电源域等)

        供工作流 architect_design 节点调用。
        """
        return self._run_architect_task("architect_design", task, context_summary, injector)

    def analyze_task(self, task: str, context_summary: str = "", injector: PromptInjector | None = None) -> str:
        """
        架构权衡分析:性能-面积-功耗 PPA 对比、瓶颈与风险识别

        供工作流 architect_analyze 节点调用。
        """
        return self._run_architect_task("architect_analyze", task, context_summary, injector)

    def review_task(self, task: str, context_summary: str = "", injector: PromptInjector | None = None) -> str:
        """
        架构评审:从架构、RTL 实现、后端物理、软件驱动、验证多维度给出评审意见

        供工作流 architect_review 节点调用。
        """
        return self._run_architect_task("architect_review", task, context_summary, injector)

    def spec_task(self, task: str, context_summary: str = "", injector: PromptInjector | None = None) -> str:
        """
        输出规格文档:架构 Spec、接口 Spec、Pinmap、寄存器规格、架构约束清单

        供工作流 architect_spec 节点调用。
        """
        return self._run_architect_task("architect_spec", task, context_summary, injector)

    # ============ 异步版(供 rtl_graph 节点直接 await 调用) ============

    async def _arun_architect_task_async(
        self,
        template_name: str,
        task: str,
        context_summary: str,
        injector: PromptInjector | None,
    ) -> str:
        """架构工作流方法异步通用执行体:模板渲染/技能注入同步,LLM 调用异步流式"""
        template = self.get_template(template_name)
        prompt = self.render_template(template, task=task, context_summary=context_summary)
        if injector is not None:
            prompt = injector.inject_into_prompt(prompt, task)
        return await self.ainvoke(prompt)

    async def aplan_task(self, task: str, context_summary: str = "", injector: PromptInjector | None = None) -> str:
        """异步版 plan_task(供 architect_plan 节点调用)"""
        return await self._arun_architect_task_async("architect_plan", task, context_summary, injector)

    async def adesign_task(self, task: str, context_summary: str = "", injector: PromptInjector | None = None) -> str:
        """异步版 design_task(供 architect_design 节点调用)"""
        return await self._arun_architect_task_async("architect_design", task, context_summary, injector)

    async def aanalyze_task(self, task: str, context_summary: str = "", injector: PromptInjector | None = None) -> str:
        """异步版 analyze_task(供 architect_analyze 节点调用)"""
        return await self._arun_architect_task_async("architect_analyze", task, context_summary, injector)

    async def areview_task(self, task: str, context_summary: str = "", injector: PromptInjector | None = None) -> str:
        """异步版 review_task(供 architect_review 节点调用)"""
        return await self._arun_architect_task_async("architect_review", task, context_summary, injector)

    async def aspec_task(self, task: str, context_summary: str = "", injector: PromptInjector | None = None) -> str:
        """异步版 spec_task(供 architect_spec 节点调用)"""
        return await self._arun_architect_task_async("architect_spec", task, context_summary, injector)