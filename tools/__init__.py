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
]

__all__ = [
    'search', 'read_file', 'write_file', 'calculate',
    'run_shell', 'run_python', 'run_cmd',
    'get_local_time',
    'open_file', 'open_sqlite',
    'read_skill',
    'all_tools',
]
