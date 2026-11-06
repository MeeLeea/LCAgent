"""Executor 子程序包。

复杂任务的后台「执行」子程序：主 Agent 通过 ``subagent.executor.tool`` 启动独立
进程，worker 在受限工具集（read_file / write_file / list_dir，全部绑定 workspace）
内多步执行任务，最终产出 ``result.md``。

注意：本文件刻意保持轻量，只导出配置。
禁止在此导入 ``agent`` / ``worker`` / ``team`` —— ``tools`` 包会在
主对话进程启动时 eager import ``subagent.executor.tool``，一旦这里拉起
``team`` / ``langchain`` 依赖，会让主进程的 tools 导入链变重并形成脆弱环。
"""
from subagent.config import ExecutorConfig

__all__ = ["ExecutorConfig"]
