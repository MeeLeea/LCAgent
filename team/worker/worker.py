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

    def execute_task(self, plan: str, injector: PromptInjector | None = None) -> str:
        """
        执行计划中的子任务(结合技能注入)

        供工作流 worker_exec 节点调用:渲染 worker_exec 模板 → 可选技能注入
        → invoke 执行(Worker 具备工具能力,经 ReAct 循环调用工具)。
        为同步方法,节点层经 asyncio.to_thread 异步执行。

        Args:
            plan: Manager 拆解出的执行计划
            injector: 技能注入器;为 None 时跳过技能注入

        Returns:
            子任务执行结果文本
        """
        template = self.get_template("worker_exec")
        prompt = self.render_template(template, plan=plan)
        if injector is not None:
            prompt = injector.inject_into_prompt(prompt, plan)
        return self.invoke(prompt)
