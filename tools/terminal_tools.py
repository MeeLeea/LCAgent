"""
终端工具 - 允许智能体运行终端命令、Python 脚本、PowerShell/CMD 脚本
使用 LangChain @tool 装饰器，支持 .py / .ps1 / .bat 文件
"""
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
from typing import Any

from langchain_core.tools import tool
from langgraph.errors import GraphInterrupt

from .safety import check_command, check_exec, confirm

# Windows 默认超时（秒），防止命令卡死
DEFAULT_TIMEOUT = 60
MAX_OUTPUT_CHARS = 10000  # 超长输出截断，避免回传给 LLM 时占用过多 token

# 超时后 ctrl+c 软中断的 grace period（秒）：给子进程时间刷出缓冲输出并清理
_GRACE_PERIOD = 5

# 超时原因分类（供主模型判断如何修改命令重试）
TIMEOUT_REASONS: dict[str, str] = {
    "interactive": "交互式命令（等待用户输入）",
    "network": "网络连接阻塞",
    "io_block": "IO 阻塞（如 tail -f 等持续监听）",
    "command_error": "命令错误",
    "dead_loop": "死循环或计算密集",
    "unknown": "未知原因",
}

# 交互式命令特征（未加非交互标志时视为交互式等待输入）
_INTERACTIVE_PATTERN = re.compile(
    r"(?i)\b(python3?|ipython|node|psql|mysql|sqlite3|ssh|telnet|ftp|bc|redis-cli|mongo)\b"
)
# 非交互标志（出现则不算交互式等待输入）
_NON_INTERACTIVE_FLAGS = (
    "-c ", "--non-interactive", "-noninteractive", "</dev/null",
    "-command ", "-NoProfile", "-NonInteractive", "-file ", "-f ",
)
# 网络命令特征
_NETWORK_PATTERN = re.compile(
    r"(?i)\b(curl|wget|ping|telnet|nc|netcat|scp|rsync|"
    r"git\s+(clone|fetch|pull|push)|pip\s+install|npm\s+install|cargo\s+build)\b"
)
# 网络命令自带超时标志（有则不算网络阻塞）
_NETWORK_TIMEOUT_FLAG = re.compile(r"(?i)--(?:timeout|max-time|connect-timeout|deadline)")
# IO 持续监听特征
_IO_BLOCK_PATTERN = re.compile(r"(?i)\b(tail\s+-f|less\b|more\b|watch\b|tail\s+--follow)\b")


def _classify_timeout(command: str, partial_output: str, returncode: int | None) -> str:
    """启发式分类超时原因，供主模型判断如何修改命令重试。

    分类维度：交互式命令 / 网络阻塞 / IO 阻塞 / 命令错误 / 死循环。
    """
    cmd_lower = (command or "").lower()
    partial = partial_output or ""

    # 交互式命令：匹配交互解释器且未加非交互标志
    if _INTERACTIVE_PATTERN.search(cmd_lower) and not any(
        flag in cmd_lower for flag in _NON_INTERACTIVE_FLAGS
    ):
        return "interactive"

    # 网络阻塞：匹配网络命令且未自带超时标志
    if _NETWORK_PATTERN.search(cmd_lower) and not _NETWORK_TIMEOUT_FLAG.search(cmd_lower):
        return "network"

    # IO 持续监听
    if _IO_BLOCK_PATTERN.search(cmd_lower):
        return "io_block"

    # 命令错误：有输出且 returncode 异常
    if returncode is not None and returncode != 0 and partial.strip():
        return "command_error"

    # 兜底：无明显特征，归为死循环 / 计算密集
    return "dead_loop"


def _send_ctrl_c(proc: subprocess.Popen, is_windows: bool) -> None:
    """发送 ctrl+c 软中断信号，让子进程有机会清理并刷出缓冲输出。

    Windows 用 CTRL_BREAK_EVENT（跨进程组生效，CTRL_C_EVENT 不可靠）；
    Unix 给整个进程组发 SIGINT（需配合 start_new_session=True 创建的会话组）。
    """
    try:
        if is_windows:
            proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    except (OSError, ProcessLookupError, ValueError, AttributeError):
        # 进程可能已退出或不支持该信号，忽略
        pass


def _kill_process_tree(proc: subprocess.Popen, is_windows: bool) -> None:
    """强杀进程树，确保子进程不会成为孤儿。

    Windows 用 taskkill /T /F 递归终止整树；Unix 用 killpg SIGKILL。
    """
    try:
        if is_windows:
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                capture_output=True,
                timeout=5,
                check=False,
            )
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (OSError, ProcessLookupError, subprocess.TimeoutExpired):
        pass
    finally:
        try:
            proc.kill()
        except OSError:
            pass


def _run_with_timeout(
    cmd: list[str],
    cwd: str | None,
    timeout: int,
) -> tuple[int | None, str, str, bool]:
    """用 Popen + ctrl+c 软中断执行命令，超时先发中断信号再强杀。

    与 subprocess.run(timeout=) 的区别：后者超时直接 kill（SIGKILL/terminate），
    子进程没机会刷出已缓冲的输出；本函数先发 ctrl+c 软中断，等 grace period
    收集 partial 输出（供 LLM 判断超时原因），再强杀整树。

    Args:
        cmd: 命令参数列表（如 ["powershell", "-Command", "dir"]）
        cwd: 工作目录
        timeout: 超时秒数

    Returns:
        (returncode, stdout, stderr, timed_out)：timed_out 为 True 表示发生超时
    """
    is_windows = platform.system() == "Windows"
    popen_kwargs: dict[str, Any] = {
        "cwd": cwd,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
    }
    if is_windows:
        # 新进程组：CTRL_BREAK_EVENT 才能跨组送达子进程
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        # 新会话：os.killpg 才能终止整个进程树
        popen_kwargs["start_new_session"] = True

    proc = subprocess.Popen(cmd, **popen_kwargs)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return proc.returncode, stdout or "", stderr or "", False
    except subprocess.TimeoutExpired:
        # 超时：先发 ctrl+c 软中断，让子进程刷出缓冲输出
        _send_ctrl_c(proc, is_windows)
        try:
            stdout, stderr = proc.communicate(timeout=_GRACE_PERIOD)
        except subprocess.TimeoutExpired:
            # grace period 也超时，强杀整树
            _kill_process_tree(proc, is_windows)
            try:
                stdout, stderr = proc.communicate(timeout=3)
            except Exception:
                stdout, stderr = stdout or "", stderr or ""
        return proc.returncode, stdout or "", stderr or "", True

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


def _truncate(text: str, max_chars: int = MAX_OUTPUT_CHARS) -> str:
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


def _guard_command(command: str) -> dict[str, Any] | None:
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
    if status == "confirm" and not confirm(
        f"⚠ 检测到危险命令 [{reason}]\n"
        f"待执行命令：{_redact_command(command)}\n"
        "确认执行? [y/N]: "
    ):
        # 普通失败结果会被 ReAct 模型当作可重试错误；异常交给 AgentCore 终止本轮。
        raise UserRejectedCommandError(command)
    return None


def _guard_exec(file_path: str) -> dict[str, Any] | None:
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
    if status == "confirm" and not confirm(f"⚠ {reason} [{file_path}]\n确认执行? [y/N]: "):
        # 与 shell 命令保持一致：拒绝后终止整轮，而不是返回可重试错误。
        raise UserRejectedCommandError(file_path)
    return None


@tool
def run_shell(command: str, cwd: str | None = None, timeout: int = DEFAULT_TIMEOUT) -> dict[str, Any]:
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

        returncode, stdout, stderr, timed_out = _run_with_timeout(shell_args, cwd, timeout)
        if timed_out:
            reason = _classify_timeout(command, stdout, returncode)
            return {
                "success": False,
                "error_type": "timeout",
                "error": f"命令超时（{timeout}秒）：{TIMEOUT_REASONS.get(reason, '未知原因')}",
                "timeout_reason": reason,
                "partial_stdout": _truncate(stdout),
                "partial_stderr": _truncate(stderr) or None,
                "command": command,
                "cwd": cwd,
            }

        return {
            "success": returncode == 0,
            "returncode": returncode,
            "stdout": _truncate(stdout),
            "stderr": _truncate(stderr) or None,
            "command": command,
            "cwd": cwd,
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
def run_python(file_path: str, script_args: str = "", cwd: str | None = None, timeout: int = DEFAULT_TIMEOUT) -> dict[str, Any]:
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

        returncode, stdout, stderr, timed_out = _run_with_timeout(cmd, cwd, timeout)
        if timed_out:
            reason = _classify_timeout(file_path, stdout, returncode)
            return {
                "success": False,
                "error_type": "timeout",
                "error": f"Python 脚本执行超时（{timeout}秒）：{TIMEOUT_REASONS.get(reason, '未知原因')}",
                "timeout_reason": reason,
                "partial_stdout": _truncate(stdout),
                "partial_stderr": _truncate(stderr) or None,
                "file_path": file_path,
                "script_args": script_args,
            }

        if returncode != 0:
            return {
                "success": False,
                "error": f"脚本执行失败 (exit {returncode}): {stderr.strip()}",
                "stdout": stdout,
                "file_path": file_path,
            }

        return {
            "success": True,
            "returncode": returncode,
            "stdout": _truncate(stdout),
            "stderr": _truncate(stderr) or None,
            "file_path": file_path,
            "script_args": script_args,
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
def run_cmd(file_path: str, script_args: str = "", cwd: str | None = None, timeout: int = DEFAULT_TIMEOUT) -> dict[str, Any]:
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
        else:
            return {
                "success": False,
                "error": f"不支持的脚本类型: {ext}（仅支持 .bat / .cmd / .ps1）",
                "file_path": file_path
            }

        returncode, stdout, stderr, timed_out = _run_with_timeout(cmd, cwd, timeout)
        if timed_out:
            reason = _classify_timeout(file_path, stdout, returncode)
            return {
                "success": False,
                "error_type": "timeout",
                "error": f"脚本执行超时（{timeout}秒）：{TIMEOUT_REASONS.get(reason, '未知原因')}",
                "timeout_reason": reason,
                "partial_stdout": _truncate(stdout),
                "partial_stderr": _truncate(stderr) or None,
                "file_path": file_path,
                "script_args": script_args,
            }

        if returncode != 0:
            return {
                "success": False,
                "error": f"命令执行失败 (exit {returncode}): {stderr.strip()}",
                "stdout": stdout,
            }

        return {
            "success": True,
            "returncode": returncode,
            "stdout": _truncate(stdout),
            "stderr": _truncate(stderr) or None,
            "file_path": file_path,
            "script_args": script_args,
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
