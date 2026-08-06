"""LCAgent 异常层次结构

统一异常分层，便于上层精准 catch 和错误处理。

层次:
    LCAgentError                    ← 所有 LCAgent 异常的基类
    ├── MCPConnectionError          ← MCP server 连接失败/断连
    ├── ToolTimeoutError            ← 工具执行超时
    ├── CompressError               ← 上下文压缩失败
    ├── InterruptTimeoutError       ← 中断会话超时（过期未恢复）
    └── AgentStateError             ← AgentCore 状态错误（如已关闭后调用）
"""
from __future__ import annotations


class LCAgentError(Exception):
    """所有 LCAgent 异常的基类"""

    def __init__(self, message: str = "", *, detail: str | None = None):
        super().__init__(message)
        self.detail = detail or message


class MCPConnectionError(LCAgentError):
    """MCP server 连接失败或断连

    Attributes:
        server_name: 出错的 MCP server 名称
    """

    def __init__(self, server_name: str, message: str = "", *, detail: str | None = None):
        msg = message or f"MCP server '{server_name}' 连接失败"
        super().__init__(msg, detail=detail)
        self.server_name = server_name


class ToolTimeoutError(LCAgentError):
    """工具执行超时

    Attributes:
        tool_name: 超时的工具名称
        timeout: 超时时间（秒）
    """

    def __init__(self, tool_name: str, timeout: float, *, detail: str | None = None):
        msg = f"工具 '{tool_name}' 执行超时（{timeout}秒）"
        super().__init__(msg, detail=detail)
        self.tool_name = tool_name
        self.timeout = timeout


class CompressError(LCAgentError):
    """上下文压缩失败

    Attributes:
        stage: 失败阶段（'summarize' / 'prune' / 'inject'）
    """

    def __init__(self, stage: str, message: str = "", *, detail: str | None = None):
        msg = message or f"上下文压缩失败（阶段: {stage}）"
        super().__init__(msg, detail=detail)
        self.stage = stage


class InterruptTimeoutError(LCAgentError):
    """中断会话超时（过期未恢复）

    Attributes:
        thread_id: 超时的会话线程 ID
    """

    def __init__(self, thread_id: str, message: str = "", *, detail: str | None = None):
        msg = message or f"中断会话 '{thread_id}' 已超时"
        super().__init__(msg, detail=detail)
        self.thread_id = thread_id


class AgentStateError(LCAgentError):
    """AgentCore 状态错误（如在已关闭后调用方法）"""



__all__ = [
    "AgentStateError",
    "CompressError",
    "InterruptTimeoutError",
    "LCAgentError",
    "MCPConnectionError",
    "ToolTimeoutError",
]
