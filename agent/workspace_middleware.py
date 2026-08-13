"""工作空间安全中间件 - 拦截文件/执行类工具调用，强制 workspace 隔离。

四层架构的 layer 4（安全隔离层）：
- 文件类工具（MCP-Filesystem 提供：read_file/write_file/list_directory 等）：
  从 config 读 workspace_path → 将 LLM 传入的相对路径解析为 workspace 内绝对路径
  → commonpath 校验防逃逸 → 注入绝对路径到 args → 放行 MCP 执行
- 执行类工具（run_shell/run_python/run_cmd）：
  从 config 读 workspace_path → 强制 cwd=workspace → 校验 LLM 显式路径不逃逸

设计要点：
- workspace_path 从 config.configurable 读取（Session 固有属性，Agent 只读消费）
- 无 workspace 绑定时（兼容旧会话）不拦截，直接放行
- 中间件无状态，workspace 按 session 隔离（从 config 读），天然并发安全
- 所有会话共享同一编译图，隔离完全由 config + 中间件保证
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.messages import ToolMessage
from langchain.tools.tool_node import ToolCallRequest

logger = logging.getLogger(__name__)

# MCP-Filesystem 提供的文件类工具名（需做路径解析 + 逃逸校验）
_FILESYSTEM_TOOLS: frozenset[str] = frozenset({
    "read_file",
    "write_file",
    "list_directory",
    "create_directory",
    "move_file",
    "search_files",
    "get_file_info",
    "list_allowed_directories",
    "read_text_file",
    "write_text_file",
    "read_multiple_files",
    "edit_file",
    "directory_tree",
})

# 执行类工具名（需强制 cwd 对齐 + 路径逃逸校验）
_EXEC_TOOLS: frozenset[str] = frozenset({
    "run_shell",
    "run_python",
    "run_cmd",
})

# 执行类工具中含路径参数的字段名
_EXEC_PATH_FIELDS: dict[str, tuple[str, ...]] = {
    "run_shell": ("cwd",),
    "run_python": ("cwd", "file_path"),
    "run_cmd": ("cwd", "file_path"),
}


def _get_workspace(config: Any) -> str | None:
    """从 RunnableConfig 读取当前会话的 workspace_path。

    Args:
        config: LangChain 运行时配置

    Returns:
        workspace 绝对路径，未绑定时返回 None
    """
    if config is None:
        return None
    configurable = config.get("configurable", {})
    if not isinstance(configurable, dict):
        return None
    return configurable.get("workspace_path")


def _resolve_in_workspace(path: str, workspace: str) -> str:
    """将路径解析为 workspace 内的绝对路径，并做逃逸校验。

    委托给 tools.safety.check_workspace_escape，复用 safety.py 的路径规范化
    能力（_normalize_path），保证与安全护栏的路径处理逻辑一致。

    Args:
        path: LLM 传入的路径（相对或绝对）
        workspace: 当前会话的 workspace 绝对路径

    Returns:
        解析后的绝对路径

    Raises:
        ValueError: 路径逃逸 workspace 边界
    """
    from tools.safety import check_workspace_escape

    resolved, error = check_workspace_escape(path, workspace)
    if error:
        raise ValueError(error)
    return resolved


def _validate_exec_path(path: str, workspace: str) -> str:
    """校验执行类工具的路径参数不逃逸 workspace。

    与 _resolve_in_workspace 一致，委托给 check_workspace_escape。
    """
    return _resolve_in_workspace(path, workspace)


class WorkspaceSecurityMiddleware(AgentMiddleware):
    """工作空间安全隔离中间件。

    拦截文件类和执行类工具调用，从 config 读取 workspace_path，
    解析相对路径为 workspace 内绝对路径，commonpath 校验防逃逸。

    无 workspace 绑定时（兼容旧会话）不拦截，直接放行。
    """

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage],
    ) -> ToolMessage:
        """同步版本：拦截工具调用，注入 workspace 路径 + 校验逃逸。"""
        tool_name = request.tool_call["name"]
        workspace = _get_workspace(request.runtime.config)

        # 无 workspace 绑定，直接放行（兼容旧会话）
        if workspace is None:
            return handler(request)

        # 文件类工具：解析路径参数 + 逃逸校验
        if tool_name in _FILESYSTEM_TOOLS:
            return self._handle_filesystem_tool(request, handler, workspace)

        # 执行类工具：强制 cwd + 路径校验
        if tool_name in _EXEC_TOOLS:
            return self._handle_exec_tool(request, handler, workspace)

        # 其他工具：直接放行
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ) -> Any:
        """异步版本：拦截工具调用，注入 workspace 路径 + 校验逃逸。"""
        tool_name = request.tool_call["name"]
        workspace = _get_workspace(request.runtime.config)

        if workspace is None:
            return await handler(request)

        if tool_name in _FILESYSTEM_TOOLS:
            return await self._ahandle_filesystem_tool(request, handler, workspace)

        if tool_name in _EXEC_TOOLS:
            return await self._ahandle_exec_tool(request, handler, workspace)

        return await handler(request)

    # ============ 文件类工具处理 ============

    def _handle_filesystem_tool(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage],
        workspace: str,
    ) -> ToolMessage:
        """解析文件工具的路径参数为 workspace 内绝对路径。"""
        try:
            new_args = self._resolve_file_args(request.tool_call["args"], workspace)
        except ValueError as e:
            return self._build_error_tool_message(request, str(e))
        new_request = request.override(tool_call={
            "name": request.tool_call["name"],
            "args": new_args,
            "id": request.tool_call.get("id", ""),
        })
        return handler(new_request)

    async def _ahandle_filesystem_tool(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
        workspace: str,
    ) -> Any:
        """异步版本：解析文件工具的路径参数。"""
        try:
            new_args = self._resolve_file_args(request.tool_call["args"], workspace)
        except ValueError as e:
            return self._build_error_tool_message(request, str(e))
        new_request = request.override(tool_call={
            "name": request.tool_call["name"],
            "args": new_args,
            "id": request.tool_call.get("id", ""),
        })
        return await handler(new_request)

    @staticmethod
    def _resolve_file_args(
        args: dict[str, Any], workspace: str
    ) -> dict[str, Any]:
        """解析文件工具 args 中的路径字段为 workspace 内绝对路径。

        MCP-Filesystem 工具的路径参数名：
        - read_file/write_file/edit_file/get_file_info: path
        - list_directory/create_directory/search_files: path (或 dir_path)
        - move_file: source_path + destination_path
        - read_multiple_files: paths (列表)
        """
        new_args = dict(args)
        # 单路径字段
        for field in ("path", "file_path", "dir_path"):
            if field in new_args and isinstance(new_args[field], str):
                new_args[field] = _resolve_in_workspace(new_args[field], workspace)
        # move_file 的双路径
        for field in ("source_path", "destination_path", "src_path", "dst_path"):
            if field in new_args and isinstance(new_args[field], str):
                new_args[field] = _resolve_in_workspace(new_args[field], workspace)
        # read_multiple_files 的路径列表
        if "paths" in new_args and isinstance(new_args[field], list):
            new_args["paths"] = [
                _resolve_in_workspace(p, workspace) if isinstance(p, str) else p
                for p in new_args["paths"]
            ]
        return new_args

    # ============ 执行类工具处理 ============

    def _handle_exec_tool(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage],
        workspace: str,
    ) -> ToolMessage:
        """强制 cwd=workspace + 校验显式路径不逃逸。"""
        try:
            new_args = self._resolve_exec_args(
                request.tool_call["name"],
                request.tool_call["args"],
                workspace,
            )
        except ValueError as e:
            return self._build_error_tool_message(request, str(e))
        new_request = request.override(tool_call={
            "name": request.tool_call["name"],
            "args": new_args,
            "id": request.tool_call.get("id", ""),
        })
        return handler(new_request)

    async def _ahandle_exec_tool(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
        workspace: str,
    ) -> Any:
        """异步版本：强制 cwd + 路径校验。"""
        try:
            new_args = self._resolve_exec_args(
                request.tool_call["name"],
                request.tool_call["args"],
                workspace,
            )
        except ValueError as e:
            return self._build_error_tool_message(request, str(e))
        new_request = request.override(tool_call={
            "name": request.tool_call["name"],
            "args": new_args,
            "id": request.tool_call.get("id", ""),
        })
        return await handler(new_request)

    @staticmethod
    def _resolve_exec_args(
        tool_name: str,
        args: dict[str, Any],
        workspace: str,
    ) -> dict[str, Any]:
        """解析执行类工具 args：强制 cwd + 校验路径字段。

        - cwd 字段：强制设为 workspace（覆盖 LLM 传入的任何值）
        - file_path 字段：校验不逃逸 workspace
        """
        new_args = dict(args)
        path_fields = _EXEC_PATH_FIELDS.get(tool_name, ())
        for field in path_fields:
            if field == "cwd":
                # cwd 强制对齐 workspace，不接受 LLM 传入值
                new_args[field] = workspace
            elif field in new_args and isinstance(new_args[field], str):
                # file_path 等路径字段：校验不逃逸
                new_args[field] = _validate_exec_path(new_args[field], workspace)
        return new_args

    # ============ 错误处理 ============

    @staticmethod
    def _build_error_tool_message(
        request: ToolCallRequest,
        error: str,
    ) -> ToolMessage:
        """构建逃逸校验失败的 ToolMessage。"""
        tool_call_id = request.tool_call.get("id", "")
        tool_name = request.tool_call["name"]
        logger.warning("工作空间安全拦截 [%s]: %s", tool_name, error)
        return ToolMessage(
            content=f"操作被拒绝：{error}",
            tool_call_id=tool_call_id,
            name=tool_name,
        )


__all__ = ["WorkspaceSecurityMiddleware"]
