"""
终端工具 - 允许智能体运行终端命令、Python 脚本、PowerShell/CMD 脚本
使用 LangChain @tool 装饰器，支持 .py / .ps1 / .bat 文件
"""
from langchain_core.tools import tool
from typing import Dict, Any, Optional
import os
import subprocess
import sys
import platform

from .safety import check_command, check_exec, confirm


# Windows 默认超时（秒），防止命令卡死
DEFAULT_TIMEOUT = 60


def _truncate(text: str, max_chars: int = 4000) -> str:
    """截断超长输出，避免回传给 LLM 时占用过多 token"""
    if not text:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... [输出已截断，共 {len(text)} 字符，仅显示前 {max_chars} 字符]"


def _guard_command(command: str) -> Optional[Dict[str, Any]]:
    """
    命令安全护栏: 在执行前检查
    - deny  -> 返回错误字典(不执行)
    - confirm -> 交互式确认,拒绝则返回错误字典
    - allow -> 返回 None(允许执行)
    """
    status, reason = check_command(command)
    if status == "deny":
        return {
            "success": False,
            "error": f"命令被安全策略拦截: {reason}",
            "command": command,
        }
    if status == "confirm":
        if not confirm(f"⚠ 检测到危险命令 [{reason}]\n确认执行? [y/N]: "):
            return {
                "success": False,
                "error": "用户拒绝执行危险命令",
                "command": command,
            }
    return None


def _guard_exec(file_path: str) -> Optional[Dict[str, Any]]:
    """
    脚本执行安全护栏(运行任意 .py/.ps1/.bat 本质危险)
    """
    status, reason = check_exec()
    if status == "deny":
        return {
            "success": False,
            "error": f"执行被安全策略拦截: {reason}",
            "file_path": file_path,
        }
    if status == "confirm":
        if not confirm(f"⚠ {reason} [{file_path}]\n确认执行? [y/N]: "):
            return {
                "success": False,
                "error": "用户拒绝执行",
                "file_path": file_path,
            }
    return None


@tool
def run_shell(command: str, cwd: Optional[str] = None, timeout: int = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    """
    运行 Shell 命令（跨平台）。
    在 Windows 上使用 PowerShell，在 Linux/Mac 上使用 bash/sh。
    适合执行单条命令、管道、重定向等通用 shell 操作。

    Args:
        command: shell 命令字符串，如 "dir" / "ls -la" / "Get-ChildItem"
        cwd: 工作目录（绝对路径），默认为当前目录
        timeout: 超时时间（秒），默认 60 秒

    Returns:
        包含执行结果的字典：success, returncode, stdout, stderr, command, cwd
    """
    try:
        guard = _guard_command(command)
        if guard:
            return guard

        is_windows = platform.system() == "Windows"
        if is_windows:
            # Windows 用 PowerShell
            shell_args = [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-Command", command
            ]
        else:
            # Unix 用 bash， failing back 到 sh
            shell_bin = "/bin/bash" if os.path.exists("/bin/bash") else "/bin/sh"
            shell_args = [shell_bin, "-c", command]

        proc = subprocess.run(
            shell_args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace"
        )

        return {
            "success": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": _truncate(proc.stdout),
            "stderr": _truncate(proc.stderr) or None,
            "command": command,
            "cwd": cwd
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": f"命令超时（{timeout}秒）",
            "command": command,
            "cwd": cwd
        }
    except FileNotFoundError as e:
        return {
            "success": False,
            "error": f"找不到解释器: {e}",
            "command": command,
            "cwd": cwd
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"执行失败: {e}",
            "command": command,
            "cwd": cwd
        }


@tool
def run_python(file_path: str, script_args: str = "", cwd: Optional[str] = None, timeout: int = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    """
    运行 Python 脚本文件（.py）。
    使用当前 Python 解释器执行，可传递命令行参数。

    Args:
        file_path: Python 脚本路径（相对或绝对），如 "tools/demo.py"
        script_args: 传给脚本的命令行参数字符串，如 "--name test --count 3"，默认为空
        cwd: 工作目录（绝对路径），默认为当前目录
        timeout: 超时时间（秒），默认 60 秒

    Returns:
        包含执行结果的字典：success, returncode, stdout, stderr, file_path, script_args
    """
    try:
        guard = _guard_exec(file_path)
        if guard:
            return guard

        # 规范化路径
        abs_path = os.path.abspath(file_path)
        if not os.path.isfile(abs_path):
            return {
                "success": False,
                "error": f"文件不存在: {abs_path}",
                "file_path": file_path
            }

        if not abs_path.lower().endswith(".py"):
            return {
                "success": False,
                "error": f"不是 Python 文件: {abs_path}",
                "file_path": file_path
            }

        # 构造命令：python <script> <script_args>
        cmd = [sys.executable, abs_path]
        if script_args:
            cmd.extend(script_args.split())

        proc = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace"
        )

        return {
            "success": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": _truncate(proc.stdout),
            "stderr": _truncate(proc.stderr) or None,
            "file_path": abs_path,
            "script_args": script_args or None,
            "cwd": cwd
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": f"Python 脚本执行超时（{timeout}秒）",
            "file_path": file_path,
            "script_args": script_args
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"执行失败: {e}",
            "file_path": file_path,
            "script_args": script_args
        }


@tool
def run_cmd(file_path: str, script_args: str = "", cwd: Optional[str] = None, timeout: int = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    """
    运行 CMD/PowerShell 脚本文件（.bat / .cmd / .ps1）。
    自动根据扩展名选择解释器：
      - .bat / .cmd → cmd.exe /c
      - .ps1        → powershell -ExecutionPolicy Bypass -File

    Args:
        file_path: 脚本路径（相对或绝对），如 "build.ps1" / "deploy.bat"
        script_args: 传给脚本的参数字符串，如 "Release x64"，默认为空
        cwd: 工作目录（绝对路径），默认为当前目录
        timeout: 超时时间（秒），默认 60 秒

    Returns:
        包含执行结果的字典：success, returncode, stdout, stderr, file_path, script_args, script_type
    """
    try:
        guard = _guard_exec(file_path)
        if guard:
            return guard

        abs_path = os.path.abspath(file_path)
        if not os.path.isfile(abs_path):
            return {
                "success": False,
                "error": f"文件不存在: {abs_path}",
                "file_path": file_path
            }

        ext = os.path.splitext(abs_path)[1].lower()

        # 根据扩展名选择解释器
        if ext in (".bat", ".cmd"):
            # CMD 批处理
            cmd = ["cmd", "/c", abs_path]
            if script_args:
                cmd.extend(script_args.split())
            script_type = "bat"
        elif ext == ".ps1":
            # PowerShell 脚本
            cmd = [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy", "Bypass",
                "-File", abs_path
            ]
            if script_args:
                cmd.extend(script_args.split())
            script_type = "ps1"
        else:
            return {
                "success": False,
                "error": f"不支持的脚本类型: {ext}（仅支持 .bat / .cmd / .ps1）",
                "file_path": file_path
            }

        proc = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace"
        )

        return {
            "success": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": _truncate(proc.stdout),
            "stderr": _truncate(proc.stderr) or None,
            "file_path": abs_path,
            "script_args": script_args or None,
            "script_type": script_type,
            "cwd": cwd
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": f"脚本执行超时（{timeout}秒）",
            "file_path": file_path,
            "script_args": script_args
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"执行失败: {e}",
            "file_path": file_path,
            "script_args": script_args
        }


# 工具描述（供文档/查看使用）
TOOL_DESCRIPTION = """
run_shell: 运行 Shell 命令（Windows=PowerShell，Unix=bash）
参数:
    - command (str): shell 命令字符串
    - cwd (str, 可选): 工作目录
    - timeout (int, 可选): 超时秒数，默认 60
返回: 包含 success/returncode/stdout/stderr 的字典

run_python: 运行 Python 脚本文件（.py）
参数:
    - file_path (str): Python 文件路径
    - args (str, 可选): 命令行参数
    - cwd (str, 可选): 工作目录
    - timeout (int, 可选): 超时秒数，默认 60
返回: 包含 success/returncode/stdout/stderr 的字典

run_cmd: 运行 CMD/PowerShell 脚本（.bat / .cmd / .ps1）
参数:
    - file_path (str): 脚本文件路径
    - args (str, 可选): 命令行参数
    - cwd (str, 可选): 工作目录
    - timeout (int, 可选): 超时秒数，默认 60
返回: 包含 success/returncode/stdout/stderr/script_type 的字典
"""


if __name__ == "__main__":
    # 简单自测
    print("=== 测试 run_shell ===")
    r = run_shell.invoke({"command": "echo hello from shell"})
    print(r)

    print("\n=== 测试 run_python ===")
    # 创建临时测试脚本
    test_py = "_terminal_test.py"
    with open(test_py, "w", encoding="utf-8") as f:
        f.write("import sys\nprint('Python version:', sys.version)\nprint('Args:', sys.argv[1:])\n")
    r = run_python.invoke({"file_path": test_py, "script_args": "--foo bar"})
    print(r)
    os.remove(test_py)

    print("\n=== 测试 run_cmd (.bat) ===")
    test_bat = "_terminal_test.bat"
    with open(test_bat, "w", encoding="utf-8") as f:
        f.write("@echo off\necho Hello from BAT\n")
    r = run_cmd.invoke({"file_path": test_bat})
    print(r)
    os.remove(test_bat)
