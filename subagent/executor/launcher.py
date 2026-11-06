"""Executor 子程序启动器：preflight 校验 + 复用公共分离进程启动。

与 planner 的分工：
- 进程启动与 PID 探测由 :mod:`subagent.common.proc` 承担（不 import planner）。
- 本模块只做*启动前同步校验*，保证「缺提示词 / 缺 API Key / 工作区非法」这类
  问题在 ``start_executor_task`` 阶段同步暴露，而不是变成一个永远 ``running``
  的僵尸任务。
"""
from __future__ import annotations

import logging
import os

from subagent.common.config import (
    ExecutorConfig,
    load_executor_config,
    resolve_llm_settings,
)
from subagent.common.proc import spawn_detached

logger = logging.getLogger(__name__)

__all__ = ["preflight", "spawn_worker"]

# 子进程以模块方式运行的入口模块
_WORKER_MODULE: str = "subagent.executor.worker"
# 子进程日志文件名（位于任务目录内）
_LOG_FILENAME: str = "worker.log"


def _prompt_file_is_usable(path: str) -> bool:
    """判断提示词文件是否存在且非空。"""
    try:
        return bool(path) and os.path.isfile(path) and os.path.getsize(path) > 0
    except OSError:
        return False


def preflight(config: ExecutorConfig | None = None) -> tuple[str, str]:
    """启动前校验配置，返回实际生效的 ``(provider, model)``。

    在真正拉起子进程前同步校验以下条件，避免「缺提示词 / 缺 API Key」变成
    一个永远 ``running`` 的僵尸任务：
        1. 提示词文件存在且非空；
        2. ``resolve_llm_settings`` 能解析出可用 provider 与 API Key。

    Args:
        config: 待校验的配置；``None`` 时加载默认配置文件。

    Returns:
        ``(provider, model)`` 二元组。

    Raises:
        ValueError: 提示词文件缺失/为空，或 provider / API Key / model 配置非法。
    """
    resolved = config or load_executor_config()

    prompt_file = resolved.agent_prompt_file
    if not _prompt_file_is_usable(prompt_file):
        raise ValueError(f"执行者提示词文件缺失或为空: {prompt_file}")

    return resolve_llm_settings(resolved)


def spawn_worker(
    task_id: str,
    *,
    root: str | None = None,
    config: ExecutorConfig | None = None,
) -> int:
    """以分离进程方式启动 executor worker，返回子进程 PID。

    子进程命令固定为 ``[sys.executable, "-m", "subagent.executor.worker", "--task-id", task_id]``，
    工作目录为项目根，stdout/stderr 追加写入 ``<task_dir>/worker.log``。本函数
    **不等待**子进程，成功启动后立即返回。

    Args:
        task_id: 任务目录名（裸目录名，禁止路径分隔符与 ``..``）。
        root: 输出根目录；``None`` 时取配置 ``output_root`` 的规范化路径。
        config: Executor 配置；``None`` 时加载默认配置。

    Returns:
        已启动子进程的 PID。

    Raises:
        LauncherError: 任务目录不存在，或 ``Popen`` 启动失败。
    """
    resolved_config = config or load_executor_config()
    output_root = root or os.path.normpath(resolved_config.output_root)
    return spawn_detached(_WORKER_MODULE, task_id, root=output_root, log_file=_LOG_FILENAME)
