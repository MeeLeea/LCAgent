"""
Team 模块 - 多 Agent 团队协作
"""
from team.architect.Architect import ArchiAgent
from team.factory import build_team_agent
from team.manager.manager import ManagerAgent
from team.terminator.terminator import TerminatorAgent
from team.worker.worker import WorkerAgent

__all__ = [
    "ArchiAgent",
    "ManagerAgent",
    "TerminatorAgent",
    "WorkerAgent",
    "build_team_agent",
]
