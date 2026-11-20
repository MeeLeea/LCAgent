"""tests/subagent 共享的假对象与辅助函数（planner/executor worker 测试复用）。

设计约束：
    - 假模型绝不触网：``ainvoke`` 按调用序号执行注入的 handler，或返回占位文本；
    - 同时模拟 ``bind(**kwargs)`` 与 ``bind_tools(tools)``（记录并返回自身），
      覆盖 planner worker（``bind`` 采样参数）与 executor worker（``bind_tools``
      受限工具集）两条模型绑定链路；
    - ``default_content`` 为未提供 handler 时的占位回复模板，子类以
      ``{call}`` 占位符注入调用序号。

被 ``tests/subagent/test_worker.py`` 与 ``tests/subagent/test_executor_worker.py``
引用（``tests`` 为带 ``__init__.py`` 的包，经 ``tests.subagent._fakes`` 导入）。
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, ClassVar

from langchain_core.messages import AIMessage, BaseMessage

from subagent.executor.worker import RESULT_MARKER

# 模型调用处理器：入参为（调用序号, 消息列表），返回一条 AIMessage 或抛异常。
ModelHandler = Callable[[int, list[BaseMessage]], Awaitable[AIMessage]]


class FakeModel:
    """可编排的假聊天模型：记录绑定信息、按调用序号执行 handler。"""

    # 未提供 handler 时的占位回复模板（``{call}`` 替换为调用序号）
    default_content: ClassVar[str] = "占位输出-{call}"

    def __init__(self, handler: ModelHandler | None = None) -> None:
        self.calls: int = 0
        self.handler = handler
        self.bind_kwargs: dict[str, Any] = {}
        self.bound_tools: list[Any] | None = None

    def bind(self, **kwargs: Any) -> FakeModel:
        """记录绑定的采样参数并返回自身，模拟可链式绑定的模型。"""
        self.bind_kwargs = dict(kwargs)
        return self

    def bind_tools(self, tools: list[Any]) -> FakeModel:
        """记录绑定的工具并返回自身（模拟可链式绑定的模型）。"""
        self.bound_tools = list(tools)
        return self

    async def ainvoke(self, messages: list[BaseMessage]) -> AIMessage:
        """按调用序号执行处理器；未提供处理器时返回占位文本。"""
        self.calls += 1
        if self.handler is not None:
            return await self.handler(self.calls, messages)
        return AIMessage(content=self.default_content.format(call=self.calls))


class PlannerFakeModel(FakeModel):
    """planner 占位模型：默认回复模拟方案输出（无结束标记约定）。"""

    default_content = "方案输出-{call}"


class ExecutorFakeModel(FakeModel):
    """executor 占位模型：默认回复模拟带结束标记的最终答复。"""

    default_content = f"执行结果-{{call}}\n\n{RESULT_MARKER}"


def tool_call_message(name: str, args: dict[str, Any], call_index: int) -> AIMessage:
    """构造一条带单个 tool_call 的 AIMessage。"""
    return AIMessage(
        content="",
        tool_calls=[{
            "name": name,
            "args": args,
            "id": f"call_{call_index}",
            "type": "tool_call",
        }],
    )


def open_tasks_excluding_current() -> int:
    """返回当前事件循环中除本协程外尚未完成的任务数量（防任务泄漏断言）。"""
    current = asyncio.current_task()
    return sum(1 for task in asyncio.all_tasks() if task is not current and not task.done())
