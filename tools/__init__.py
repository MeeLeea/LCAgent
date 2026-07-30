# -*- coding: utf-8 -*-
"""
LangChainAgent Tools Package
"""

from .search import search
from .file_tool import read_file, write_file
from .calculator import calculate
from .terminal_tools import run_shell, run_python, run_cmd
from .get_local_time import get_local_time
from .open_file import open_file, open_sqlite
from .skill_tool import read_skill
from .scheduler_tool import schedule_task, list_scheduled_tasks, cancel_scheduled_task, delete_scheduled_task, cleanup_finished_tasks, configure
from cli.human_input import ask_human

# 导出所有本地工具供Agent使用
all_tools = [
    search,
    read_file,
    write_file,
    calculate,
    run_shell,
    run_python,
    run_cmd,
    get_local_time,
    open_file,
    open_sqlite,
    read_skill,
    schedule_task,
    list_scheduled_tasks,
    cancel_scheduled_task,
    delete_scheduled_task,
    cleanup_finished_tasks,
    ask_human,
]

__all__ = [
    'search', 'read_file', 'write_file', 'calculate',
    'run_shell', 'run_python', 'run_cmd',
    'get_local_time',
    'open_file', 'open_sqlite',
    'read_skill',
    'schedule_task', 'list_scheduled_tasks', 'cancel_scheduled_task',
    'delete_scheduled_task', 'cleanup_finished_tasks', 'configure',
    'ask_human',
    'all_tools',
]
