"""
Architect Agent - 芯片架构工程师,负责架构方案设计、权衡分析、评审与规格文档输出
"""
from collections.abc import Sequence
from typing import ClassVar

from graph.registry import register_agent
from team.base import PromptInjector, TeamAgent


@register_agent(
    "architect",
    "team/architect/agent_config.json",
    tools=None,
    mcp_tools=["write_file","edit_file","list_directory","read_file","delete_file","create_directory","delete_directory"],
)
class ArchiAgent(TeamAgent):
    """
    架构师 Agent,负责芯片架构方案设计、PPA 权衡分析、多维度评审与规格文档输出

    继承 TeamAgent 轻量基类,可选工具调用能力(声明 mcp_tools=["write_file"],
    由 build_workflow 装配期同步拉取;加载失败时降级为纯文本模式);各工作流方法
    渲染对应 `## workflow:*` 小节 → 可选技能注入 → 异步 LLM 调用(TOKEN 级流式)。
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

    # ============ 异步版(供 rtl_graph 节点直接 await 调用) ============

    async def _arun_architect_task_async(
        self,
        template_name: str,
        task: str,
        context_summary: str,
        injector: PromptInjector | None,
        config: dict | None = None,
        active_names: Sequence[str] = (),
    ) -> str:
        """架构工作流方法异步通用执行体:模板渲染/技能注入同步,LLM 调用异步流式

        active_names 由节点函数从 state["active_skills"] 取值传入。
        """
        template = self.get_template(template_name)
        prompt = self.render_template(template, task=task, context_summary=context_summary)
        if injector is not None:
            prompt = injector.inject_into_prompt(prompt, task, active_names)
        return await self.ainvoke(prompt, config)

    async def aplan_task(self, task: str, context_summary: str = "", injector: PromptInjector | None = None, config: dict | None = None, active_names: Sequence[str] = ()) -> str:
        """异步版 plan_task(供 architect_plan 节点调用)"""
        return await self._arun_architect_task_async("architect_plan", task, context_summary, injector, config, active_names)

    async def adesign_task(self, task: str, context_summary: str = "", injector: PromptInjector | None = None, config: dict | None = None, active_names: Sequence[str] = ()) -> str:
        """异步版 design_task(供 architect_design 节点调用)"""
        return await self._arun_architect_task_async("architect_design", task, context_summary, injector, config, active_names)

    async def aanalyze_task(self, task: str, context_summary: str = "", injector: PromptInjector | None = None, config: dict | None = None, active_names: Sequence[str] = ()) -> str:
        """异步版 analyze_task(供 architect_analyze 节点调用)"""
        return await self._arun_architect_task_async("architect_analyze", task, context_summary, injector, config, active_names)

    async def areview_task(self, task: str, context_summary: str = "", injector: PromptInjector | None = None, config: dict | None = None, active_names: Sequence[str] = ()) -> str:
        """异步版 review_task(供 architect_review 节点调用)"""
        return await self._arun_architect_task_async("architect_review", task, context_summary, injector, config, active_names)

    async def aspec_task(self, task: str, context_summary: str = "", injector: PromptInjector | None = None, config: dict | None = None, active_names: Sequence[str] = ()) -> str:
        """异步版 spec_task(供 architect_spec 节点调用)"""
        return await self._arun_architect_task_async("architect_spec", task, context_summary, injector, config, active_names)