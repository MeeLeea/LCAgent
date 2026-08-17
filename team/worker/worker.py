"""
Worker Agent - 负责执行具体子任务
"""
from typing import ClassVar

from graph.registry import register_agent
from team.base import PromptInjector, TeamAgent
from tools import all_tools


@register_agent("worker", "team/worker/agent_config.json", tools=all_tools)
class WorkerAgent(TeamAgent):
    """
    执行者 Agent,负责执行上级分配的子任务
    
    继承 TeamAgent 轻量基类,支持工具调用能力(通过 tools 参数注入)
    """
    
    # 定制 LLM 采样参数:执行型任务用较低温度提升确定性,放宽 token 上限
    temperature = 0.3
    max_tokens = 4096

    # 工作流节点提示词的默认模板(仅 AGENT.md 缺失或未定义小节时兜底)
    default_templates: ClassVar[dict[str, str]] = {
        "worker_exec": (
            "请执行以下计划:\n\n"
            "{plan}"
        ),
    }

    async def aexecute_task(
        self,
        plan: str,
        injector: PromptInjector | None = None,
        config: dict | None = None,
    ) -> str:
        """
        异步版 execute_task(供 worker_exec 节点直接 await 调用)

        模板渲染/技能注入保持同步(纯 CPU),仅 LLM 调用走异步流式
        (``await self.ainvoke``)。config(含 configurable.workspace_path 与
        callbacks)透传给 ainvoke,使工具调用受 workspace 隔离约束、token
        增量可流出到外层事件流。
        """
        template = self.get_template("worker_exec")
        prompt = self.render_template(template, plan=plan)
        if injector is not None:
            prompt = injector.inject_into_prompt(prompt, plan)
        return await self.ainvoke(prompt, config)
