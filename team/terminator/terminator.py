"""
Terminator Agent - 负责汇总 Worker 执行结果并返回最终答案
"""
from graph.registry import register_agent
from team.base import TeamAgent


@register_agent("terminator", "team/terminator/agent_config.json", tools=None)
class TerminatorAgent(TeamAgent):
    """
    终结者 Agent,负责汇总工作结果并返回最终答案给用户
    
    继承 TeamAgent 轻量基类,纯文本推理模式(不使用工具)
    """
