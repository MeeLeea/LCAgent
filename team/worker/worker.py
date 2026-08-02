"""
Worker Agent - 负责执行具体子任务
"""
from graph.registry import register_agent
from team.base import TeamAgent
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
