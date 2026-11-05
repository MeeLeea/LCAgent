"""Executor 工作区的解析与校验（安全边界的一部分）。

职责边界：
- ``validate_explicit_workspace``：主进程 ``start`` 阶段校验用户显式指定的
  workspace（存在性 + 绝对路径 + 拒绝清单），返回归一化绝对路径。
- 工作区并发互斥（``.lcagent-executor.lock``）与单次工具调用的路径包含校验
  分别位于 :mod:`subagent.executor.store` 与 :mod:`subagent.common.paths`。

安全原则（与 Oracle 评审结论一致）：
- workspace 在主进程 ``start`` 阶段就解析归一化并写入 task.json，worker 不解析
  用户输入；
- 拒绝清单覆盖：卷根 / 系统目录 / 用户主目录本身 / 项目根本身及其关键子目录 /
  子程序运行时数据目录；
- v1 不提供 ``run_shell``：detached 进程没有人机确认兜底，shell 无法用「路径限制」
  边界约束，默认禁用而非配置开关。
"""
from __future__ import annotations

from subagent.common.paths import normalize_path, workspace_denial_reason

__all__ = ["WORKSPACE_LOCK_FILENAME", "validate_explicit_workspace"]

# 工作区并发互斥锁的固定文件名（位于 workspace 根目录内）
WORKSPACE_LOCK_FILENAME: str = ".lcagent-executor.lock"


def validate_explicit_workspace(workspace: str) -> str:
    """校验显式指定的 workspace，返回归一化后的绝对路径。

    Args:
        workspace: 用户传入的工作区路径。

    Returns:
        realpath+normcase 归一化后的绝对路径。

    Raises:
        ValueError: 空串 / 非绝对路径 / 目录不存在 / 命中拒绝清单。
    """
    if not isinstance(workspace, str) or not workspace.strip():
        raise ValueError("workspace 不能为空")
    reason = workspace_denial_reason(workspace)
    if reason is not None:
        raise ValueError(reason)
    return normalize_path(workspace)
