"""
LangChainAgent Tools Package
"""

from cli.human_input import ask_human

from .calculator import calculate
from .create_tools import create_tool
from .file_tool import read_file, write_file
from .get_local_time import get_local_time
from .open_file import open_file, open_sqlite
from .scheduler_tool import (
    cancel_scheduled_task,
    cleanup_finished_tasks,
    configure,
    delete_scheduled_task,
    list_scheduled_tasks,
    schedule_task,
)
from .search import search
from .skill_tool import read_skill
from .terminal_tools import run_cmd, run_python, run_shell

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
    create_tool,
    schedule_task,
    list_scheduled_tasks,
    cancel_scheduled_task,
    delete_scheduled_task,
    cleanup_finished_tasks,
    ask_human,
]

__all__ = [
    'all_tools',
    'ask_human',
    'calculate',
    'cancel_scheduled_task',
    'cleanup_finished_tasks',
    'configure',
    'create_tool',
    'delete_scheduled_task',
    'get_local_time',
    'list_scheduled_tasks',
    'open_file',
    'open_sqlite',
    'read_file',
    'read_skill',
    'run_cmd',
    'run_python',
    'run_shell',
    'schedule_task',
    'search',
    'write_file',
]
