"""
文件打开工具 - 使用LangChain @tool装饰器
用系统默认程序或指定程序(如 DB Browser for SQLite)打开文件
"""
import os
import shutil
import subprocess
import sys
from typing import Any

from langchain_core.tools import tool

# 默认的程序映射(按文件扩展名)
# 值为 None 表示用系统默认程序打开;指定路径则用该程序打开
# 可通过环境变量覆盖路径
DEFAULT_APPS = {
    ".sqlite": os.environ.get("SQLITE_BROWSER_PATH"),  # DB Browser for SQLite 路径
    ".db": os.environ.get("SQLITE_BROWSER_PATH"),
    ".json": None,   # 系统默认编辑器
    ".txt": None,
    ".md": None,
    ".html": None,   # 系统默认浏览器
    ".pdf": None,
    ".png": None,
    ".jpg": None,
    ".py": None,
}


@tool
def open_file(file_path: str, app_path: str | None = None) -> dict[str, Any]:
    """
    打开文件工具。用系统默认程序或指定程序打开文件。
    常用于:
    - 打开 .sqlite/.db 文件用 DB Browser for SQLite 查看
    - 打开 .md/.txt 用编辑器查看
    - 打开 .html 用浏览器查看
    - 打开文件夹

    Args:
        file_path: 要打开的文件或文件夹路径
        app_path: 指定程序路径(可选)。不传则按扩展名选默认程序;
                  若环境变量 SQLITE_BROWSER_PATH 已配置,打开 .sqlite/.db 时自动使用

    Returns:
        操作结果字典
    """
    try:
        # 路径转换:相对路径转绝对路径
        if not os.path.isabs(file_path):
            file_path = os.path.abspath(file_path)

        # 文件夹:直接用系统文件管理器打开
        if os.path.isdir(file_path):
            _open_with_system(file_path)
            return {
                "success": True,
                "path": file_path,
                "type": "directory",
                "message": "已在文件管理器中打开文件夹"
            }

        # 文件存在性检查
        if not os.path.exists(file_path):
            return {
                "success": False,
                "error": f"文件不存在: {file_path}"
            }

        # 决定用哪个程序打开
        ext = os.path.splitext(file_path)[1].lower()
        used_app = app_path

        if not used_app:
            # 没指定程序,按扩展名查默认映射
            used_app = DEFAULT_APPS.get(ext)

        if used_app:
            # 用指定程序打开
            if not os.path.exists(used_app):
                return {
                    "success": False,
                    "error": f"指定的程序不存在: {used_app}",
                    "hint": "请检查程序路径,或设置环境变量 SQLITE_BROWSER_PATH"
                }
            _open_with_app(used_app, file_path)
            return {
                "success": True,
                "path": file_path,
                "app": used_app,
                "type": "file",
                "message": f"已用 {os.path.basename(used_app)} 打开文件"
            }
        else:
            # 用系统默认程序打开
            _open_with_system(file_path)
            return {
                "success": True,
                "path": file_path,
                "type": "file",
                "message": "已用系统默认程序打开文件"
            }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def _open_with_system(path: str):
    """用系统默认程序打开(跨平台)"""
    if sys.platform == "win32":
        # Windows: os.startfile 最简单
        os.startfile(path)
    elif sys.platform == "darwin":
        # macOS
        subprocess.run(["open", path], check=False)
    else:
        # Linux
        subprocess.run(["xdg-open", path], check=False)


def _open_with_app(app_path: str, file_path: str):
    """用指定程序打开文件(跨平台)"""
    if sys.platform == "win32":
        # Windows: 用 subprocess 避免阻塞
        subprocess.Popen([app_path, file_path])
    else:
        subprocess.Popen([app_path, file_path])


@tool
def open_sqlite(file_path: str) -> dict[str, Any]:
    """
    专门打开 SQLite 数据库文件的工具(用 DB Browser for SQLite)。
    如果未配置 DB Browser 路径,会尝试常见安装位置自动查找。

    Args:
        file_path: SQLite 数据库文件路径(.sqlite/.db)

    Returns:
        操作结果字典
    """
    try:
        if not os.path.isabs(file_path):
            file_path = os.path.abspath(file_path)

        if not os.path.exists(file_path):
            return {
                "success": False,
                "error": f"文件不存在: {file_path}"
            }

        ext = os.path.splitext(file_path)[1].lower()
        if ext not in (".sqlite", ".db", ".sqlite3"):
            return {
                "success": False,
                "error": f"不是 SQLite 文件: {file_path} (扩展名 {ext})"
            }

        # 1. 优先用环境变量
        app_path = os.environ.get("SQLITE_BROWSER_PATH")

        # 2. 没配置则尝试常见安装位置(Windows)
        if not app_path and sys.platform == "win32":
            candidates = [
                r"C:\Program Files\DB Browser for SQLite\DB Browser for SQLite.exe",
                r"C:\Program Files (x86)\DB Browser for SQLite\DB Browser for SQLite.exe",
            ]
            for c in candidates:
                if os.path.exists(c):
                    app_path = c
                    break

        # 3. 查 PATH
        if not app_path:
            found = shutil.which("DB Browser for SQLite.exe") or shutil.which("sqlitebrowser")
            if found:
                app_path = found

        if not app_path:
            return {
                "success": False,
                "error": "未找到 DB Browser for SQLite",
                "hint": "请安装 DB Browser for SQLite,或设置环境变量 SQLITE_BROWSER_PATH 指向其安装路径",
                "download": "https://sqlitebrowser.org/dl/"
            }

        _open_with_app(app_path, file_path)
        return {
            "success": True,
            "path": file_path,
            "app": app_path,
            "message": "已用 DB Browser for SQLite 打开数据库"
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
