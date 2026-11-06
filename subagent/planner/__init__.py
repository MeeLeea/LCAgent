"""Planner 子程序包。

复杂需求的后台「方案设计」子程序：主 Agent 通过 ``subagent.planner.tool``
启动独立进程，子程序一次性完成方案设计并产出方案文档。

注意：本文件刻意保持轻量，只导出配置。
禁止在此导入 ``agent`` / ``worker`` / ``team`` —— ``tools`` 包会在
主对话进程启动时 eager import ``subagent.planner.tool``，一旦这里拉起
``team`` / ``langchain`` 依赖，会让主进程的 tools 导入链变重并形成
``tools → subagent → team → tools`` 的脆弱环。
"""
from subagent.common.config import PlannerConfig

__all__ = ["PlannerConfig"]
