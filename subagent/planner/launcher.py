"""Planner 子程序启动器：以「分离进程」模型启动后台 worker，并提供跨平台存活性检查。

进程模型说明：
    - 主对话进程只负责调用 :func:`spawn_worker` 拉起子进程，**不等待、不通信**，
      因此调用毫秒级返回，主进程可立即继续响应用户。
    - 子进程以 ``python -m subagent.planner.worker --task-id <id>`` 形式运行，工作
      目录固定为项目根 ``PROJECT_ROOT``，保证 ``PYTHONPATH`` 与包导入一致。
    - 子进程的标准输出与标准错误统一重定向到 ``<task_dir>/worker.log``（追加写入、
      行缓冲），主进程不持有管道读端，避免父进程退出后子进程因写管道失败而崩溃。
    - Windows 使用 ``DETACHED_PROCESS`` 让子进程脱离父进程的控制台（不继承 Ctrl+C、
      不依附父控制台），POSIX 使用 ``start_new_session=True`` 创建独立会话。

为什么禁止在 Windows 上使用 ``os.kill(pid, 0)``：
    ``os.kill`` 在 POSIX 上以信号 0 做「存在性探测」，但在 Windows 上 Python 的
    实现会调用 ``TerminateProcess``——``signal 0`` 不是探测而是**真正终止目标进程**。
    因此本模块在 Windows 分支改用 ``ctypes`` 调用 ``kernel32.OpenProcess`` /
    ``GetExitCodeProcess`` 判断存活，绝不触碰 ``os.kill``。

句柄管理：
    ``OpenProcess`` 打开的进程句柄必须显式 ``CloseHandle``，否则会泄漏内核句柄。
    存活性轮询是高频热路径，本模块在 ``try/finally`` 中**始终**关闭句柄。
    ``kernel32`` 的函数签名（``argtypes`` / ``restype``）在首次使用时缓存定义，
    避免每次检查都重复解析，也保证 64 位下 HANDLE 按 ``c_void_p`` 正确处理。
"""
from __future__ import annotations

import ctypes
import logging
import os
import subprocess
import sys

from subagent.common.config import PROJECT_ROOT, resolve_llm_settings
from subagent.config import PlannerConfig, load_planner_config

logger = logging.getLogger(__name__)

__all__ = ["LauncherError", "pid_is_alive", "preflight", "spawn_worker"]

# 平台判定：Windows 与 POSIX 的存活探测实现完全不同
_IS_WINDOWS: bool = sys.platform == "win32"

# Windows 进程访问权限：仅查询有限信息即可判断存活，权限最小化
PROCESS_QUERY_LIMITED_INFORMATION: int = 0x1000
# GetExitCodeProcess 对仍在运行的进程返回的「伪退出码」
STILL_ACTIVE: int = 259
# OpenProcess 失败时表示「进程存在但当前权限无法查询」
ERROR_ACCESS_DENIED: int = 5

# 子进程以模块方式运行，命令固定为 [sys.executable, "-m", _WORKER_MODULE, ...]
_WORKER_MODULE: str = "subagent.planner.worker"
# 子进程日志文件名（位于任务目录内）
_LOG_FILENAME: str = "worker.log"

# 缓存的 kernel32 动态库句柄（仅在 Windows 上惰性初始化）
_kernel32: ctypes.WinDLL | None = None


class LauncherError(RuntimeError):
    """启动 planner 子进程失败时抛出。"""


def _get_kernel32() -> ctypes.WinDLL:
    """惰性获取并缓存 ``kernel32``，同时声明关键函数签名。

    仅在 Windows 上调用。缓存后每次存活检查复用同一句柄，避免重复定义签名。

    Returns:
        配置好 ``argtypes`` / ``restype`` 的 ``kernel32`` 动态库对象。
    """
    global _kernel32
    if _kernel32 is None:
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        # OpenProcess(DWORD dwDesiredAccess, BOOL bInheritHandle, DWORD dwProcessId) -> HANDLE
        kernel32.OpenProcess.argtypes = [
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        ]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        # GetExitCodeProcess(HANDLE hProcess, LPDWORD lpExitCode) -> BOOL
        kernel32.GetExitCodeProcess.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        # CloseHandle(HANDLE hObject) -> BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        _kernel32 = kernel32
    return _kernel32


def _pid_is_alive_windows(pid: int) -> bool:
    """Windows 平台存活探测（基于 ``kernel32`` 进程句柄，绝不使用 ``os.kill``）。"""
    from ctypes import wintypes

    kernel32 = _get_kernel32()
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        # 打开失败：ACCESS_DENIED 表示进程存在但不可查询；其余错误视为不存在
        return ctypes.get_last_error() == ERROR_ACCESS_DENIED
    try:
        exit_code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return False
        return exit_code.value == STILL_ACTIVE
    finally:
        # 高频轮询下必须关闭句柄，否则内核句柄泄漏
        kernel32.CloseHandle(handle)


def _pid_is_alive_posix(pid: int) -> bool:
    """POSIX 平台存活探测（``os.kill(pid, 0)`` 仅发信号 0，不产生副作用）。"""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # 进程存在但当前用户无权限发信号
        return True
    except OSError:
        return False
    return True


def pid_is_alive(pid: int | None) -> bool:
    """判断给定 PID 对应的进程是否仍然存活。

    Args:
        pid: 目标进程 ID；``None``、``0`` 或负数一律视为不存活。

    Returns:
        ``True`` 表示进程存活，``False`` 表示不存在、已退出或输入非法。
    """
    if pid is None or pid <= 0:
        return False
    if _IS_WINDOWS:
        return _pid_is_alive_windows(pid)
    return _pid_is_alive_posix(pid)


def _prompt_file_is_usable(path: str) -> bool:
    """判断规划者提示词文件是否存在且非空。"""
    try:
        return bool(path) and os.path.isfile(path) and os.path.getsize(path) > 0
    except OSError:
        return False


def preflight(config: PlannerConfig | None = None) -> tuple[str, str]:
    """启动前校验配置，返回实际生效的 ``(provider, model)``。

    在真正拉起子进程前同步校验以下条件，避免「缺提示词 / 缺 API Key」变成
    一个永远 ``running`` 的僵尸任务：
        1. 规划者提示词文件存在且非空；
        2. ``resolve_llm_settings`` 能解析出可用 provider 与 API Key。

    Args:
        config: 待校验的配置；``None`` 时加载默认配置文件。

    Returns:
        ``(provider, model)`` 二元组。

    Raises:
        ValueError: 提示词文件缺失/为空，或 provider / API Key / model 配置非法。
    """
    resolved = config or load_planner_config()

    prompt_file = resolved.agent_prompt_file
    if not _prompt_file_is_usable(prompt_file):
        raise ValueError(f"规划者提示词文件缺失或为空: {prompt_file}")

    return resolve_llm_settings(resolved)


def _validate_task_id(task_id: str) -> None:
    """校验 ``task_id`` 是纯粹的目录名（禁止路径分隔符与 ``..``）。"""
    if not task_id:
        raise ValueError("任务 ID 不能为空")
    if task_id in (".", "..") or ".." in task_id or "/" in task_id or "\\" in task_id:
        raise ValueError(f"非法任务 ID（不能包含路径分隔符或 '..'）: {task_id}")


def _build_worker_env() -> dict[str, str]:
    """构造子进程环境变量：继承父环境并补齐导入与编码相关变量。"""
    env = dict(os.environ)
    existing_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        PROJECT_ROOT if not existing_path else PROJECT_ROOT + os.pathsep + existing_path
    )
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def spawn_worker(
    task_id: str,
    *,
    root: str | None = None,
    config: PlannerConfig | None = None,
) -> int:
    """以分离进程方式启动 planner worker，返回子进程 PID。

    子进程命令固定为 ``[sys.executable, "-m", "subagent.planner.worker", "--task-id", task_id]``，
    工作目录为项目根，stdout/stderr 追加写入 ``<task_dir>/worker.log``。本函数
    **不等待**子进程，成功 fork 后立即返回。

    Args:
        task_id: 任务目录名（裸目录名，禁止路径分隔符与 ``..``）。
        root: 输出根目录；``None`` 时取配置 ``output_root`` 的规范化路径。
        config: Planner 配置；``None`` 时加载默认配置。

    Returns:
        已启动子进程的 PID。

    Raises:
        ValueError: ``task_id`` 非法。
        LauncherError: 任务目录不存在，或 ``Popen`` 启动失败。
    """
    _validate_task_id(task_id)

    resolved_config = config or load_planner_config()
    output_root = root or os.path.normpath(resolved_config.output_root)
    task_dir = os.path.join(output_root, task_id)
    if not os.path.isdir(task_dir):
        raise LauncherError(f"任务目录不存在: {task_dir}")

    command = [sys.executable, "-m", _WORKER_MODULE, "--task-id", task_id]
    log_path = os.path.join(task_dir, _LOG_FILENAME)

    popen_kwargs: dict[str, object] = {
        "cwd": PROJECT_ROOT,
        "env": _build_worker_env(),
        "stdin": subprocess.DEVNULL,
        "stderr": subprocess.STDOUT,
    }
    if _IS_WINDOWS:
        # DETACHED_PROCESS 已提供「无控制台」语义，不可再叠加 CREATE_NO_WINDOW（两者冲突）
        popen_kwargs["creationflags"] = subprocess.DETACHED_PROCESS
    else:
        popen_kwargs["start_new_session"] = True

    # 用 with 确保父进程副本在 Popen 返回后立即关闭
    # （子进程已 dup 自己的句柄；父进程若泄漏会随长驻进程不断累积）
    with open(log_path, "a", encoding="utf-8", buffering=1) as log_handle:
        popen_kwargs["stdout"] = log_handle
        try:
            proc = subprocess.Popen(command, **popen_kwargs)  # type: ignore[arg-type]
        except (OSError, ValueError) as exc:
            raise LauncherError(f"启动 planner 子进程失败: {exc}") from exc

    return proc.pid
