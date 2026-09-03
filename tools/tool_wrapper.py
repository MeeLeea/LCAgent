"""工具超时包装器

为本地工具和 MCP 工具提供统一的超时保护，防止 Agent 因工具卡死而永久阻塞。

设计:
- wrap_tool_with_timeout: 包装单个工具，在其 _arun 上叠加 asyncio.wait_for
- wrap_tools_with_timeout: 批量包装工具列表
- 超时后返回 JSON 错误消息（而非抛异常），让 Agent 能继续推理

工具超时配置:
- 全局默认: DEFAULT_TIMEOUT (60秒)
- 按工具名覆盖: TOOL_TIMEOUTS 字典
- 排除列表: NO_TIMEOUT_TOOLS（如 ask_human 需要无限等待用户输入）
"""
from __future__ import annotations

import asyncio
import json
from inspect import signature as _signature
from typing import Any

from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import BaseTool

# 全局默认超时（秒）
DEFAULT_TIMEOUT: float = 60.0

# 按工具名覆盖超时（优先于全局默认）
TOOL_TIMEOUTS: dict[str, float] = {
    "ask_human": 600.0,       # 人工交互，给 10 分钟
    "schedule_task": 120.0,   # 调度器可能需要更长
    "search": 90.0,           # 搜索可能需要多次请求
    "run_shell": 600.0,       # shell 命令（vivado/xsim 批处理耗时 3-10 分钟）
}

# 完全排除超时的工具（无限等待）
# 目前为空，ask_human 通过 TOOL_TIMEOUTS 给了 600 秒上限
NO_TIMEOUT_TOOLS: set[str] = set()


def _get_tool_timeout(tool: BaseTool, default_timeout: float | None = None) -> float | None:
    """获取工具的超时时间

    优先级: NO_TIMEOUT_TOOLS > TOOL_TIMEOUTS > default_timeout > DEFAULT_TIMEOUT
    """
    name = getattr(tool, "name", "")
    if name in NO_TIMEOUT_TOOLS:
        return None
    if name in TOOL_TIMEOUTS:
        return TOOL_TIMEOUTS[name]
    if default_timeout is not None:
        return default_timeout
    return DEFAULT_TIMEOUT


def wrap_tool_with_timeout(tool: BaseTool, timeout: float | None = None) -> BaseTool:
    """为工具添加超时保护

    通过替换工具的 _arun 方法实现：
    - 如果工具有原生 _arun，包裹 asyncio.wait_for
    - 如果工具只有 _run（同步），先 to_thread 再 wait_for
    - 超时后返回 JSON 错误字符串，不抛异常

    Args:
        tool: 原始工具
        timeout: 超时秒数（None=按配置/默认值）

    Returns:
        传入的 tool 对象（已就地修改 _arun）
    """
    effective_timeout = timeout if timeout is not None else _get_tool_timeout(tool)

    # 无需超时的工具，直接返回
    if effective_timeout is None:
        return tool

    tool_name = getattr(tool, "name", "unknown")

    # 保存原始方法（已绑定，不需要再传 self）
    original_run = tool._run
    original_arun = getattr(tool, "_arun", None)

    # 检查是否已经包装过（避免重复包裹）
    if getattr(tool, "_timeout_wrapped", False):
        return tool

    # langchain-core >= 1.0: BaseTool.arun() 通过 _get_runnable_config_param() 按类型注解
    # 检测 _arun 是否声明了 `config: RunnableConfig` 参数，只有声明了才会注入 config。
    # 因此包装函数必须显式保留该注解（不能写成 *args/**kwargs），否则原始
    # StructuredTool._arun(config 必填) 会报 “missing 1 required keyword-only argument: 'config'”。
    # run_manager 同理显式透传，保持与原始 _arun 的调用契约一致。
    async def _arun_with_timeout(
        *args: Any,
        config: RunnableConfig = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> str:
        try:
            if original_arun is not None:
                coro = original_arun(*args, config=config, run_manager=run_manager, **kwargs)
            else:
                # 同步工具放到线程池跑，再加超时
                run_kwargs = dict(kwargs)
                if "config" in _signature(original_run).parameters:
                    run_kwargs["config"] = config
                coro = asyncio.to_thread(original_run, *args, **run_kwargs)
            return await asyncio.wait_for(coro, timeout=effective_timeout)
        except TimeoutError:
            # 不导入 ToolTimeoutError 以避免循环依赖（tool_wrapper ← agent ← tools）
            # 错误消息格式与 ToolTimeoutError 保持一致
            timeout_msg = f"工具 '{tool_name}' 执行超时（{effective_timeout}秒）"
            return json.dumps({
                "error": "tool_timeout",
                "tool": tool_name,
                "timeout": effective_timeout,
                "message": timeout_msg,
            }, ensure_ascii=False)
        except Exception:
            # 非 timeout 异常照常抛出，由 LangGraph 的错误处理接管
            raise

    # 标记已包装
    tool._timeout_wrapped = True
    # 就地替换 _arun（实例属性，不会自动绑定 self）
    tool._arun = _arun_with_timeout
    return tool


def wrap_tools_with_timeout(
    tools: list[BaseTool],
    default_timeout: float | None = None,
) -> list[BaseTool]:
    """批量包装工具列表

    Args:
        tools: 原始工具列表
        default_timeout: 全局默认超时（None=使用 DEFAULT_TIMEOUT / TOOL_TIMEOUTS）

    Returns:
        包装后的工具列表（原地修改，返回同一批对象引用）
    """
    # default_timeout 直接透传,不再修改全局默认(避免并发包装时全局状态串值)
    return [wrap_tool_with_timeout(t, default_timeout) for t in tools]


__all__ = [
    "DEFAULT_TIMEOUT",
    "NO_TIMEOUT_TOOLS",
    "TOOL_TIMEOUTS",
    "wrap_tool_with_timeout",
    "wrap_tools_with_timeout",
]
