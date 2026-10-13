"""
Team 模块 - 多 Agent 团队协作

注意：导入本包会触发 manager/worker/terminator/architect 的 @register_agent
注册副作用（scheduler/run.py 依赖此行为）；rtl_designer/rtl_verification
由 graph/rtl_graph.py 直接 import 角色模块注册。
外部消费方请统一经子模块直接 import（如 from team.manager.manager import
ManagerAgent）；本文件仅保留历史 re-export 与注册副作用，不再新增导出。
"""
from team.architect.Architect import ArchiAgent  # noqa: F401  # 保留:注册副作用
from team.factory import build_team_agent
from team.manager.manager import ManagerAgent
from team.terminator.terminator import TerminatorAgent
from team.worker.worker import WorkerAgent

__all__ = [
    "ManagerAgent",
    "TerminatorAgent",
    "WorkerAgent",
    "build_team_agent",
]
