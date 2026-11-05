"""路径安全公共工具：归一化 / 包含关系校验 / 工作区拒绝清单。

只依赖标准库。归一化逻辑与 ``tools/safety.py`` 的 ``_normalize_path`` 保持一致
（abspath → realpath → normcase；realpath 解析 symlink/junction，normcase 消除
Windows 大小写差异，``ntpath.commonpath`` 本身不做大小写归一故必须显式归一化）。

两个用途：

1. ``check_contained(target, root)`` — 每次 worker 工具调用前的路径包含校验。
2. ``workspace_denial_reason(path)`` — 显式 workspace 的黑名单校验（在主进程
   ``start`` 阶段执行）。

注意：本模块刻意放在 ``subagent/common`` 而不是 ``utils/``，因为 worker 进程
只允许导入 stdlib 级模块；而它也不放进 ``tools/`` —— ``import tools.<任何子模块>``
会先执行 ``tools/__init__.py``，把 langchain / cli 等重依赖拉进后台子进程。
"""
from __future__ import annotations

import os

__all__ = ["check_contained", "normalize_path", "workspace_denial_reason"]


def normalize_path(path: str) -> str:
    """把路径归一化为绝对、解析 symlink/junction、大小写规范化的形式。"""
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def check_contained(
    target: str | os.PathLike[str],
    root: str,
    *,
    base: str | None = None,
) -> str:
    """校验 ``target`` 解析后位于 ``root`` 之内，返回规范化后的绝对路径。

    相对路径先基于 ``base``（默认 ``root``）拼接。校验包含三步：拒绝空串/NUL、
    realpath+normcase 归一化、``os.path.commonpath`` 包含判定（两边均归一化）。

    Args:
        target: 待校验路径（允许绝对路径，但解析后必须仍在 ``root`` 内）。
        root: 约束根目录（应是已归一化的绝对路径）。
        base: 相对路径的解析基准；``None`` 时使用 ``root``。

    Returns:
        归一化后的绝对路径。

    Raises:
        ValueError: 路径非法或越出 ``root``。
    """
    if not isinstance(target, str) or not target.strip():
        raise ValueError("路径不能为空")
    if "\x00" in target:
        raise ValueError("路径包含非法字符 NUL")

    root_norm = normalize_path(root)
    raw = target if os.path.isabs(target) else os.path.join(base or root_norm, target)
    target_norm = normalize_path(raw)

    try:
        common = os.path.commonpath((root_norm, target_norm))
    except ValueError as exc:
        # Windows 跨盘符（不同 drive）等情况下 commonpath 抛 ValueError
        raise ValueError(f"路径 {target!r} 无法与工作区 {root!r} 确定共同前缀: {exc}") from exc
    if common != root_norm:
        raise ValueError(f"路径越出工作区边界: {target!r} -> {target_norm}")
    return target_norm


def _candidate_denied_roots(path_norm: str) -> list[str]:
    """构造与 ``path_norm`` 同盘的拒绝清单（全部 realpath+normcase 归一化后比较）。"""
    denied: list[str] = []

    def _add(raw: str) -> None:
        if raw and os.path.isdir(raw):
            denied.append(normalize_path(raw))

    # 所在盘符的卷根
    drive, _ = os.path.splitdrive(path_norm)
    if drive:
        _add(drive + os.sep)

    # Windows 系统目录
    for env_key in ("SystemRoot", "WinDir"):
        _add(os.environ.get(env_key, ""))
    _add(os.path.join(os.environ.get("SystemRoot", ""), "System32"))
    for env_key in ("ProgramFiles", "ProgramFiles(x86)", "ProgramData"):
        _add(os.environ.get(env_key, ""))

    # 用户主目录本身（其子目录不在此列，由调用方按需扩展）
    _add(os.path.expanduser("~"))

    # 项目根目录本身及其关键子目录（executor 修改 agent/config 等于逃逸）
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    _add(project_root)
    for child in ("agent", "config", ".git", ".venv", "data", "subagent"):
        _add(os.path.join(project_root, child))

    # 子程序任务产物根（planner 与 executor 的运行时数据目录）
    _add(os.path.join(project_root, "data", "subagent"))

    return denied


def workspace_denial_reason(path: str) -> str | None:
    """判断绝对路径 ``path`` 是否命中显式 workspace 的拒绝清单。

    Returns:
        命中时返回拒绝原因（人类可读）；合法时返回 ``None``。
    """
    if not os.path.isabs(path):
        return "workspace 必须是绝对路径"
    if not os.path.isdir(path):
        return f"workspace 目录不存在: {path}"
    path_norm = normalize_path(path)
    for denied in _candidate_denied_roots(path_norm):
        if path_norm == denied:
            return f"workspace 不允许使用受保护的系统/项目路径: {path}"
    return None
