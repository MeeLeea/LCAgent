"""LCAgent 异常层次结构

统一异常分层，便于上层精准 catch 和错误处理。

层次:
    LCAgentError                    ← 所有 LCAgent 异常的基类
    └── AgentStateError             ← AgentCore 状态错误（如已关闭后调用）
"""
from __future__ import annotations


class LCAgentError(Exception):
    """所有 LCAgent 异常的基类"""

    def __init__(self, message: str = "", *, detail: str | None = None):
        super().__init__(message)
        self.detail = detail or message


class AgentStateError(LCAgentError):
    """AgentCore 状态错误（如在已关闭后调用方法）"""



__all__ = [
    "AgentStateError",
    "LCAgentError",
]
