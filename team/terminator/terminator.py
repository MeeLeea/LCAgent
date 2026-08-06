"""
Terminator Agent - 负责汇总 Worker 执行结果并返回最终答案
"""
from typing import ClassVar

from graph.registry import register_agent
from team.base import TeamAgent


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
