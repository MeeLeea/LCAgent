"""工具参数校验中间件 - 在工具执行前拦截违反语义约束的参数组合。

四层架构的 layer 4（安全隔离层）补充，与 ``WorkspaceSecurityMW`` 同层，
职责互补：

- ``WorkspaceSecurityMW``：路径解析 + 逃逸校验（参数值的合法性）
- ``ToolArgValidatorMW``：参数语义约束校验（参数组合的合法性，如互斥）

背景：MCP server 的工具 schema 声明了参数类型，但不声明语义约束
（如 read_file 的 head 和 tail 互斥）。pydantic args_schema 只校验
类型/必填，不查语义约束，放行了互斥参数组合，调用到 MCP server 才报错。

本中间件在工具执行前按规则校验参数语义约束，冲突时返回 error ToolMessage
（不调 handler），LLM 下一轮 ReAct 反思修正。

设计要点：
- 无状态中间件，所有会话共享同一编译图
- 规则用 Python 声明式注册（ArgRule 基类 + 子类），加规则不改中间件逻辑
- 冲突时返回 error ToolMessage（不抛异常、不改 args），与 TerminalRetryCapMW 范式一致
- 错误信息必须含"违反了什么约束 + 怎么改"，利于 LLM 自愈
- 扩展新约束类型 = 加 ArgRule 子类 + 注册规则，中间件调度逻辑不变
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.messages import ToolMessage
from langchain.tools.tool_node import ToolCallRequest

logger = logging.getLogger(__name__)


class ArgRule(ABC):
    """工具参数校验规则基类。

    扩展新约束类型 = 加子类 + 注册规则到 ``_RULES``，中间件调度逻辑不变。

    Attributes:
        tool_name: 工具名（与 MCP server 注册名一致）
    """

    tool_name: str

    @abstractmethod
    def check(self, args: dict[str, Any]) -> str | None:
        """校验参数，返回错误信息（给 LLM 看）或 None（通过）。

        Args:
            args: LLM 生成的工具调用参数

        Returns:
            错误信息字符串（含修正提示）或 None（校验通过）
        """
        ...


class MutexRule(ArgRule):
    """参数互斥规则：指定的参数组中最多只能有一个被设置（非 None）。

    适用场景：
    - read_file 的 head/tail 互斥
    - write_file 的 mode/append 互斥
    - search 的 regex/glob 互斥

    注意：``None`` 值视为"未指定"，不计入互斥判定。LLM 可能显式传
    ``head=None, tail=5``，此时应视为只指定了 tail。
    """

    def __init__(
        self,
        tool_name: str,
        params: tuple[str, ...],
        message: str,
    ) -> None:
        """初始化互斥规则。

        Args:
            tool_name: 工具名
            params: 互斥参数名元组（组内最多一个被设置）
            message: 返回给 LLM 的错误信息（必须含修正提示）
        """
        self.tool_name = tool_name
        self.params = params
        self.message = message

    def check(self, args: dict[str, Any]) -> str | None:
        present = [p for p in self.params if p in args and args[p] is not None]
        if len(present) > 1:
            return self.message
        return None


# ============ 规则注册表 ============
# 加规则 = 加一行注册条目，中间件逻辑不变
# 错误信息必须含"违反了什么约束 + 怎么改"，利于 LLM 自愈
_RULES: list[ArgRule] = [
    MutexRule(
        tool_name="read_file",
        params=("head", "tail"),
        message=(
            "参数冲突：read_file 的 head 和 tail 互斥，不能同时指定。"
            "head 读取文件开头 N 行，tail 读取末尾 N 行，"
            "请移除其中一个后重试。"
        ),
    ),
    MutexRule(
        tool_name="read_text_file",
        params=("head", "tail"),
        message=(
            "参数冲突：read_text_file 的 head 和 tail 互斥，不能同时指定。"
            "head 读取文件开头 N 行，tail 读取末尾 N 行，"
            "请移除其中一个后重试。"
        ),
    ),
]


class ToolArgValidatorMW(AgentMiddleware):
    """工具参数校验中间件。

    在 LLM 生成的工具调用执行前，按规则校验参数语义约束（如互斥参数）。
    冲突时返回 error ToolMessage（不调 handler），LLM 下一轮 ReAct 反思修正。

    与 ``WorkspaceSecurityMW`` 互补：workspace 负责路径解析+逃逸校验，
    本中间件负责参数语义约束（schema 不声明但实际互斥的约束）。
    """

    def __init__(self, rules: list[ArgRule] | None = None) -> None:
        """初始化中间件。

        Args:
            rules: 校验规则列表，None 时使用默认 ``_RULES``
        """
        self._rules = rules if rules is not None else _RULES

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage],
    ) -> ToolMessage:
        """同步版本：校验参数约束，冲突时返回 error ToolMessage。"""
        tool_name = request.tool_call.get("name", "")
        args = request.tool_call.get("args", {})

        for rule in self._rules:
            if rule.tool_name != tool_name:
                continue
            error = rule.check(args)
            if error is not None:
                logger.warning(
                    "参数校验拦截 [%s]: %s", tool_name, error,
                )
                return self._build_error_message(request, error)

        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ) -> Any:
        """异步版本：校验参数约束，冲突时返回 error ToolMessage。"""
        tool_name = request.tool_call.get("name", "")
        args = request.tool_call.get("args", {})

        for rule in self._rules:
            if rule.tool_name != tool_name:
                continue
            error = rule.check(args)
            if error is not None:
                logger.warning(
                    "参数校验拦截 [%s]: %s", tool_name, error,
                )
                return self._build_error_message(request, error)

        return await handler(request)

    @staticmethod
    def _build_error_message(
        request: ToolCallRequest,
        error: str,
    ) -> ToolMessage:
        """构建参数校验失败的 ToolMessage。"""
        tool_call_id = request.tool_call.get("id", "")
        tool_name = request.tool_call.get("name", "")
        return ToolMessage(
            content=error,
            tool_call_id=tool_call_id,
            name=tool_name,
            status="error",
        )


__all__ = ["ArgRule", "MutexRule", "ToolArgValidatorMW"]
