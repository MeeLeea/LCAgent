"""
Manager Agent - 负责拆解任务并生成执行计划
"""
from graph.registry import register_agent
from team.base import TeamAgent


@register_agent("manager", "team/manager/agent_config.json", tools=None)
class ManagerAgent(TeamAgent):
    """
    管理者 Agent,负责任务拆解与规划

    继承 TeamAgent 轻量基类,纯文本推理模式(不使用工具)
    """

