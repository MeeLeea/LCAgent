"""
工作流注册表与构建入口
"""
from __future__ import annotations

import os

from langchain_core.tools import BaseTool

from graph.simple import build_simple_workflow

# 项目根目录(基于本文件位置计算)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 工作流注册表
WORKFLOWS = {
    "simple": build_simple_workflow,
}

# Agent 注册表: name -> {agent_class, config_file, tools}
# 由 team/*/*.py 中的 @register_agent 装饰器在模块加载时填充
AGENT_REGISTRY: dict[str, dict] = {}


def register_agent(name: str, config_file: str, tools: list[BaseTool] | None = None):
    """
    将 Agent 类注册到全局 AGENT_REGISTRY,供 build_workflow 统一构建
    
    Args:
        name: 角色名(如 "manager"/"worker"/"terminator")
        config_file: agent_config.json 路径(相对项目根)
        tools: 该角色的工具列表(纯文本角色传 None)
        
    Returns:
        装饰器函数,原样返回被装饰的类
    """
    def decorator(cls):
        AGENT_REGISTRY[name] = {
            "agent_class": cls,
            "config_file": config_file,
            "tools": tools,
        }
        return cls
    return decorator


def build_workflow(name: str) -> tuple[object, dict]:
    """
    构建指定名称的工作流(静默返回,不打印)
    
    Args:
        name: 工作流名称(如 "simple")
        
    Returns:
        (graph, agents) 元组:
            - graph: 编译好的 LangGraph StateGraph
            - agents: 包含各已注册角色 Agent 实例的字典
            
    Raises:
        KeyError: 工作流名称不存在,或所需角色未注册
    """
    if name not in WORKFLOWS:
        available = ", ".join(WORKFLOWS.keys())
        raise KeyError(f"未知工作流: {name}。可用工作流: {available}")
    
    # 函数内延迟导入 team,触发各 agent 模块的 @register_agent 装饰器执行
    from team import build_team_agent
    
    def _build(role: str):
        if role not in AGENT_REGISTRY:
            available = ", ".join(AGENT_REGISTRY.keys()) or "(空)"
            raise KeyError(f"未注册的角色: {role}。已注册角色: {available}")
        spec = AGENT_REGISTRY[role]
        return build_team_agent(
            spec["agent_class"],
            spec["config_file"],
            BASE_DIR,
            tools=spec["tools"],
        )
    
    # 遍历注册表构建所有已注册角色,不再写死具体角色
    agents = {role: _build(role) for role in AGENT_REGISTRY}
    
    # 调用工作流构建器(统一接收 agents 字典)
    workflow_builder = WORKFLOWS[name]
    graph = workflow_builder(agents)
    
    return graph, agents
