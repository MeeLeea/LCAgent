"""Executor worker 的受限工具集（LangChain 工具工厂，仅 3 个文件类工具）。

安全边界（Oracle 评审确认的最小强制集）：
- 所有路径经 :func:`subagent.common.paths.check_contained` 校验，必须落在
  workspace 之内（含 realpath+normcase 归一与 symlink/junction 解析）；
- ``write_file`` 对**父目录**再做一次 realpath 包含校验后原子写入
  （临时文件与目标同目录 + ``os.replace``），防止写入时 symlink 换向；
- ``read_file`` 有字节数上限（分窗读取），``list_dir`` 有条目数上限（防大目录
  卡死与上下文爆炸）；
- v1 **不提供** run_shell / delete / rename：detached 进程无确认 UI，删除不可逆，
  shell 无法用路径边界约束。

本模块依赖 ``langchain_core.tools``（worker 进程本来就依赖 langchain-core），
但**不导入** ``langchain`` / ``langgraph`` / ``agent`` / ``tools`` 包。
"""
from __future__ import annotations

import os
import uuid
from collections.abc import Callable

from langchain_core.tools import BaseTool, StructuredTool

from subagent.common.paths import check_contained

__all__ = ["build_executor_tools"]


def _read_file(
    workspace: str,
    *,
    max_bytes: int,
) -> Callable[[str, int, int], str]:
    """构造 ``read_file`` 闭包：按字符窗读取，上限 ``max_bytes`` 字节。"""

    def read_file(path: str, start: int = 0, length: int = 0) -> str:
        """读取工作区内文件的文本内容（受 max_read_bytes 限制）。

        Args:
            path: 工作区内的相对路径（绝对路径也必须位于工作区内）。
            start: 起始字符偏移（0-based），用于分段读取超大文件。
            length: 本次最多读取的字符数；0 表示读取到 ``max_read_bytes`` 上限。

        Returns:
            文本内容；读取失败时返回以「[错误]」开头的提示（LLM 可据此自纠）。
        """
        try:
            resolved = check_contained(path, workspace)
        except ValueError as exc:
            return f"[错误] {exc}"
        try:
            size = os.path.getsize(resolved)
            if size > max_bytes and length <= 0:
                return (
                    f"[错误] 文件过大（{size} 字节 > 上限 {max_bytes}），"
                    "请用 start/length 分段读取，不要一次读整个文件"
                )
            with open(resolved, "r", encoding="utf-8", errors="replace") as fh:
                fh.seek(min(max(start, 0), size))
                window = length if length > 0 else max_bytes
                return fh.read(window)
        except (OSError, UnicodeError) as exc:
            return f"[错误] 读取失败（{type(exc).__name__}）: {exc}"

    return read_file


def _write_file(workspace: str) -> Callable[[str, str], str]:
    """构造 ``write_file`` 闭包：父目录二次校验 + 原子替换写入。"""

    def write_file(path: str, content: str) -> str:
        """把文本内容写入工作区内文件（覆盖写，父目录自动创建）。

        Args:
            path: 工作区内的相对路径（绝对路径也必须位于工作区内）。
            content: 要写入的完整文本内容（整体覆盖原文件）。

        Returns:
            成功时返回写入摘要；失败时返回以「[错误]」开头的提示。
        """
        try:
            resolved = check_contained(path, workspace)
        except ValueError as exc:
            return f"[错误] {exc}"
        parent = os.path.dirname(resolved)
        try:
            # 对父目录再做一次包含校验：防止解析后的父目录经 symlink 逃出工作区
            check_contained(parent, workspace)
            os.makedirs(parent, exist_ok=True)
            tmp = f"{resolved}.tmp-{uuid.uuid4().hex}"
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(content)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, resolved)
        except (OSError, UnicodeError) as exc:
            return f"[错误] 写入失败（{type(exc).__name__}）: {exc}"
        return f"已写入 {os.path.relpath(resolved, workspace)}（{len(content)} 字符）"

    return write_file


def _list_dir(workspace: str, *, max_entries: int) -> Callable[[str, int], str]:
    """构造 ``list_dir`` 闭包：单层目录列举，条目数上限 ``max_entries``。"""

    def list_dir(path: str = ".", limit: int = 0) -> str:
        """列出工作区内目录的单层内容（不递归）。

        Args:
            path: 工作区内的目录路径，默认为工作区根（``.``）。
            limit: 本次最多返回的条目数；0 表示使用系统默认上限。

        Returns:
            每行一个条目（``[目录]`` / ``[文件]`` 前缀 + 名称）；超出上限时以
            提示行截断。失败时返回以「[错误]」开头的提示。
        """
        try:
            resolved = check_contained(path, workspace)
        except ValueError as exc:
            return f"[错误] {exc}"
        try:
            names = sorted(os.listdir(resolved))
        except (OSError, ValueError) as exc:
            return f"[错误] 列举失败（{type(exc).__name__}）: {exc}"

        capped = max_entries if limit <= 0 else min(limit, max_entries)
        lines: list[str] = []
        for name in names[:capped]:
            kind = "[目录]" if os.path.isdir(os.path.join(resolved, name)) else "[文件]"
            lines.append(f"{kind} {name}")
        truncated = len(names) - len(lines)
        if truncated > 0:
            lines.append(f"（仅显示前 {len(lines)} 条，剩余 {truncated} 条未显示）")
        return "\n".join(lines) if lines else "（空目录）"

    return list_dir


def build_executor_tools(workspace: str, config: object) -> list[BaseTool]:
    """构建绑定到 ``workspace`` 的受限工具集（3 个工具）。

    Args:
        workspace: 归一化后的工作区绝对路径（worker 已在认领后归一化）。
        config: 配置对象（``ExecutorConfig``）；仅读取上限字段，避免包级
            循环依赖故标注为 ``object``，运行时会按鸭子类型访问以下字段：
            ``max_read_bytes`` / ``max_list_entries``。

    Returns:
        LangChain 工具列表，可直接交给 ``model.bind_tools(tools)``。
    """
    read_file = StructuredTool.from_function(
        _read_file(workspace, max_bytes=int(config.max_read_bytes)),
        name="read_file",
        description=(
            "读取工作区内文件的文本内容，支持 start/length 分段读取超大文件。"
            "修改文件前必须先用它读取原文件。"
        ),
    )
    write_file = StructuredTool.from_function(
        _write_file(workspace),
        name="write_file",
        description=(
            "把文本内容覆盖写入工作区内文件（父目录自动创建，不存在的目录、"
            "工作区外的路径一律拒绝）。"
        ),
    )
    list_dir_tool = StructuredTool.from_function(
        _list_dir(workspace, max_entries=int(config.max_list_entries)),
        name="list_dir",
        description="列出工作区内目录的单层内容（不递归），用于了解目录结构。",
    )
    return [read_file, write_file, list_dir_tool]
