"""
Team Agent 工厂函数 - 统一构建团队 Agent
"""
from __future__ import annotations

import os

from langchain_core.tools import BaseTool

from llm.config import load_agent_config, resolve_path
from team.base import TeamAgent


def build_team_agent(
    agent_class: type[TeamAgent],
    config_file: str,
    base_dir: str,
    tools: list[BaseTool] | None = None,
    **overrides
) -> TeamAgent:
    """
    构建团队 Agent(Manager/Worker/Terminator)
    
    Args:
        agent_class: Agent 类(ManagerAgent/WorkerAgent/TerminatorAgent)
        config_file: agent_config.json 路径(相对项目根)
        base_dir: 项目根目录
        tools: 可选工具列表(Worker 需要,Manager/Terminator 不需要)
        **overrides: 覆盖配置中的参数
        
    Returns:
        初始化好的 Agent 实例
    """
    config_path = os.path.join(base_dir, config_file)
    config = load_agent_config(config_path)
    
    # 应用覆盖参数
    config.update(overrides)
    
    # 采样参数来源：角色级 agent_config.json（load_agent_config 已合并 DEFAULTS，
    # 未显式配置时自动落到 DEFAULTS 默认值），overrides 经 config.update 已优先覆盖
    temperature = config.get("temperature")
    max_tokens = config.get("max_tokens")
    
    # 解析角色 AGENT.md 绝对路径(system_prompt 与工作流模板均由 TeamAgent 自动解析)
    prompt_file = resolve_path(config.get("agent_prompt_file", "agent/AGENT.md"), base_dir)
    # 技能目录(绝对路径):默认 <项目根>/.agents/skills
    skills_dir = (
        resolve_path(config["skills_dir"], base_dir) if config.get("skills_dir") else None
    )
    
    # 构建轻量 TeamAgent(system_prompt 由 __init__ 从 prompt_file 自动解析)
    agent = agent_class(
        name=config["name"],
        tools=tools,
        max_iterations=config["max_iterations"],
        verbose=config.get("verbose", False),
        provider=config.get("provider", "zhipu"),
        model=config.get("model"),
        prompt_file=prompt_file,
        temperature=temperature,
        max_tokens=max_tokens,
        tool_timeout=config.get("tool_timeout"),
        skills_dir=skills_dir,
        auto_match_skills=config.get("auto_match_skills", True),
    )
    
    return agent
