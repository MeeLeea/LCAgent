"""工具执行错误纠错中间件 - 将工具异常转换为 LLM 可读的反思提示。

四层架构的 layer 4（安全隔离层）补充：与 ``WorkspaceSecurityMiddleware`` 同层，
职责互补：

- ``WorkspaceSecurityMiddleware``：拦截文件/执行类工具，解析路径 + 逃逸校验
- ``ToolErrorMiddleware``：捕获工具执行抛出的异常，转换为 ``ToolMessage(status="error")``
  进入图状态，使 LLM 在下一轮 ReAct 循环中读到错误并反思修正

背景：langgraph 的 ToolNode 对非 ``ToolInvocationError`` 异常默认直接 re-raise
（见 ``_default_handle_tool_errors``），异常逃逸会导致整个流终止（发 ``ERROR`` 事件），
LLM 没有机会在同一轮 ReAct 循环中根据报错调整参数。本中间件在工具执行层拦截异常，
转换为错误 ``ToolMessage``，让图继续执行。

设计要点：
- 所有工具执行异常统一转换为 ``ToolMessage``（用户确认方案：所有工具错误都转成 toolMessage）
- 错误内容按 langchain 官方建议点名异常类型（而非裸异常消息，避免泄露内部细节）
- 从 ``request.runtime.config`` 动态读取 workspace 根目录并附加到提示中（非硬编码路径）
- 附加反思指令，引导 LLM 分析失败原因后重试（ReAct 反思）
- 不捕获 ``GraphBubbleUp`` 控制流信号（中断/父命令必须继续传播）
- 与 ``WorkspaceSecurityMiddleware`` 组合：外层工具错误 → 内层路径逃逸拦截（返回
  ToolMessage 而非抛异常，不受本中间件影响）
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain.agents.middleware import ToolErrorMiddleware

if TYPE_CHECKING:
    from langgraph.prebuilt.tool_node import ToolCallRequest


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


def _format_error_message(
    exc: Exception,
    request: ToolCallRequest,
) -> str:
    """构建工具执行失败的错误 ToolMessage 内容。

    按 langchain 官方建议点名异常类型（而非裸异常消息）；从 runtime config
    动态读取 workspace 根目录，附加路径反思提示；末尾附加通用反思指令。

    Args:
        exc: 工具执行抛出的异常
        request: 工具调用请求（含工具名、args、runtime config）

    Returns:
        面向 LLM 的错误纠错提示文本
    """
    tool_name = request.tool_call.get("name", "未知工具")
    exc_type = type(exc).__name__
    exc_msg = str(exc).strip() or "无详细信息"

    parts: list[str] = [f"[工具执行失败] {tool_name} 抛出了 {exc_type}: {exc_msg}"]

    workspace = _get_workspace(request.runtime.config)
    if workspace:
        parts.append(f"工作空间根目录为 {workspace}")
        parts.append(
            "文件类工具请基于工作空间根目录使用相对路径，"
            "若相对路径首段与工作空间目录名重复会导致路径重复拼接，应去除该前缀"
        )

    parts.append("请反思失败原因（参数是否正确、路径是否有效、前置条件是否满足），修正后重试")
    return "。".join(parts)


class ToolExecutionErrorMiddleware(ToolErrorMiddleware):
    """将工具执行异常转换为带反思指令的 ToolMessage。

    继承 langchain 内置 ``ToolErrorMiddleware``，注入自定义错误格式化逻辑。
    默认对所有工具生效（``tools=None``），所有工具执行异常都会转换为
    ``ToolMessage(status="error")``，使 ReAct 循环得以继续，LLM 能读到报错并反思修正。
    """

    def __init__(self) -> None:
        """初始化中间件，注入异步错误格式化处理器。"""
        super().__init__(aon_error=self._aon_error)

    async def _aon_error(
        self,
        exc: Exception,
        request: ToolCallRequest,
    ) -> str:
        """异步错误处理器：格式化错误消息并返回。

        Args:
            exc: 工具执行抛出的异常
            request: 工具调用请求

        Returns:
            错误 ToolMessage 的内容
        """
        return _format_error_message(exc, request)


__all__ = ["ToolExecutionErrorMiddleware"]
