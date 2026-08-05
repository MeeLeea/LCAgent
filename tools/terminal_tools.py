"""
终端工具 - 允许智能体运行终端命令、Python 脚本、PowerShell/CMD 脚本
使用 LangChain @tool 装饰器，支持 .py / .ps1 / .bat 文件
"""
from langchain_core.tools import tool
from langgraph.errors import GraphInterrupt
from typing import Dict, Any, Optional
import os
import shutil
import subprocess
import sys
import platform
import re

from .safety import check_command, check_exec, confirm


# Windows 默认超时（秒），防止命令卡死
DEFAULT_TIMEOUT = 60

# 覆盖常见命令行参数、环境变量和 Authorization Bearer 形式；只替换值，保留命令结构供用户判断。
SENSITIVE_COMMAND_PATTERNS = (
    re.compile(
        r"(?i)(?P<prefix>(?:--?|/)(?:api[-_]?key|token|access[-_]?token|password|passwd|secret)\s*(?:=|\s)\s*)"
        r"(?P<quote>[\"']?)(?P<value>[^\s\"']+)(?P=quote)"
    ),
    re.compile(
        r"(?i)(?P<prefix>\b(?:api[-_]?key|token|access[-_]?token|password|passwd|secret)\s*=\s*)"
        r"(?P<quote>[\"']?)(?P<value>[^\s\"']+)(?P=quote)"
    ),
    re.compile(r"(?i)(?P<prefix>Authorization\s*:\s*Bearer\s+)(?P<value>[^\s\"']+)"),
)


class UserRejectedCommandError(RuntimeError):
    """用户拒绝危险操作时终止当前 Agent turn，防止模型自动重试。"""

    def __init__(self, command: str):
        self.command = command
        super().__init__("用户拒绝执行危险命令")


def _truncate(text: str, max_chars: int = 4000) -> str:
    """截断超长输出，避免回传给 LLM 时占用过多 token"""
    if not text:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... [输出已截断，共 {len(text)} 字符，仅显示前 {max_chars} 字符]"


def _redact_command(command: str) -> str:
    """隐藏命令中的常见密钥、令牌和密码，同时保留可审查的命令结构。"""
    redacted = command
    for pattern in SENSITIVE_COMMAND_PATTERNS:
        redacted = pattern.sub(lambda match: f"{match.group('prefix')}***", redacted)
    return redacted


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
        if not confirm(
            f"⚠ 检测到危险命令 [{reason}]\n"
            f"待执行命令：{_redact_command(command)}\n"
            "确认执行? [y/N]: "
        ):
            # 普通失败结果会被 ReAct 模型当作可重试错误；异常交给 AgentCore 终止本轮。
            raise UserRejectedCommandError(command)
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
            # 与 shell 命令保持一致：拒绝后终止整轮，而不是返回可重试错误。
            raise UserRejectedCommandError(file_path)
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
            # Windows 用 PowerShell；PATH 中找不到时回退到已知安装路径
            pwsh = shutil.which("powershell") or r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
            shell_args = [
                pwsh,
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
    except UserRejectedCommandError:
        # 用户拒绝不是可恢复的工具失败，必须穿透边界终止当前 Agent turn。
        raise
    except GraphInterrupt:
        # 前端确认（interrupt_confirm）是图中断，必须穿透，让 LangGraph 暂停并等待 resume。
        raise
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
    except UserRejectedCommandError:
        raise
    except GraphInterrupt:
        raise
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
    except UserRejectedCommandError:
        raise
    except GraphInterrupt:
        raise
    except Exception as e:
        return {
            "success": False,
            "error": f"执行失败: {e}",
            "file_path": file_path,
            "script_args": script_args
        }
