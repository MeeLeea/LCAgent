"""标准化 Agent 执行事件模型 — 三层架构的唯一通信载体。

AgentCore 产生 ``AgentEvent`` 事件流 → SessionManager 和 MemoryManager
各自消费同一事件流，互不耦合。

事件类型覆盖 LangGraph ``astream_events`` 的全部关键节点：
- TOKEN: LLM 文本增量
- TOOL_CALL: 工具调用开始
- TOOL_RESULT: 工具执行结果
- INTERRUPT: 被 ask_human / 危险命令确认中断
- CANCELLED: 用户拒绝执行危险命令
- ERROR: 执行异常
- DONE: 本轮执行结束

每个事件携带记忆判定字段（role / is_important / thread_id），
MemoryManager 可直接从事件流中提取需要沉淀的长期记忆，
无需二次查询 Agent 内部状态。

设计原则：
- frozen dataclass：事件不可变，安全在多消费者间传递
- to_sse_dict()：向后兼容现有 SSE 事件格式
- is_terminal()：标识流终止事件
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class EventType(str, Enum):
    """Agent 执行事件类型。"""

    TOKEN = "token"
    """LLM 文本增量"""

    TOOL_CALL = "tool_call"
    """工具调用开始"""

    TOOL_RESULT = "tool_result"
    """工具执行结果"""

    INTERRUPT = "interrupt"
    """被 ask_human / 危险命令确认中断"""

    CANCELLED = "cancelled"
    """用户拒绝执行危险命令"""

    ERROR = "error"
    """执行异常"""

    DONE = "done"
    """本轮执行结束"""


def _naive_now_iso() -> str:
    """返回本地 naive ISO 时间字符串（兼容历史数据格式）。"""
    return datetime.now().isoformat()  # noqa: DTZ005


def make_interrupt_dict(prompt: str, choices: list[Any]) -> dict[str, Any]:
    """构造中断事件的前端 SSE dict（前端协议格式的唯一来源）。

    ``AgentEvent.to_sse_dict`` 的 INTERRUPT 分支与 ``message_utils.build_interrupt_event``
    共用此函数，避免同一 {type, prompt, choices} 结构在多处重复书写。
    """
    return {"type": "interrupt", "prompt": prompt, "choices": choices}


@dataclass(frozen=True, slots=True)
class AgentEvent:
    """标准化 Agent 执行事件。

    所有层（Agent / Session / Memory）只通过此结构通信，不依赖 raw message。

    Attributes:
        event_type: 事件类型（见 EventType 枚举）
        content: 文本内容（TOKEN 的增量文本 / ERROR 的错误信息 / CANCELLED 的取消说明 / DONE 时为最终输出）
        thread_id: 会话线程 ID（记忆判定 & 会话隔离必需）
        role: 消息角色（user / assistant），仅 TOKEN 和 DONE 事件有意义，供 MemoryManager 判定
        is_important: 是否用户显式标记为重要（记忆判定字段）
        tool_call_id: 工具调用 ID（TOOL_CALL / TOOL_RESULT 事件）
        tool_name: 工具名称（TOOL_CALL / TOOL_RESULT 事件）
        tool_args: 工具输入参数（TOOL_CALL 事件）
        interrupt_prompt: 中断提示文本（INTERRUPT 事件）
        interrupt_choices: 中断可选项列表（INTERRUPT 事件）
        timestamp: 事件产生时间（naive ISO 字符串）
        trace_id: 追踪 ID（用于日志关联）
    """

    event_type: EventType
    content: str = ""
    thread_id: str = ""
    role: str = ""  # user / assistant / system
    is_important: bool = False
    tool_call_id: str = ""
    tool_name: str = ""
    tool_args: Any = None
    interrupt_prompt: str = ""
    interrupt_choices: list[Any] = field(default_factory=list)
    timestamp: str = field(default_factory=_naive_now_iso)
    trace_id: str = ""

    # ============ 工厂方法 ============

    @classmethod
    def token(
        cls,
        text: str,
        *,
        thread_id: str = "",
        role: str = "",
        trace_id: str = "",
    ) -> AgentEvent:
        """创建 LLM 文本增量事件。"""
        return cls(
            event_type=EventType.TOKEN,
            content=text,
            thread_id=thread_id,
            role=role,
            trace_id=trace_id,
        )

    @classmethod
    def tool_call(
        cls,
        *,
        tool_call_id: str,
        name: str,
        args: Any = None,
        thread_id: str = "",
        trace_id: str = "",
    ) -> AgentEvent:
        """创建工具调用开始事件。"""
        return cls(
            event_type=EventType.TOOL_CALL,
            tool_call_id=tool_call_id,
            tool_name=name,
            tool_args=args,
            thread_id=thread_id,
            trace_id=trace_id,
        )

    @classmethod
    def tool_result(
        cls,
        *,
        tool_call_id: str,
        name: str = "",
        content: str = "",
        thread_id: str = "",
        trace_id: str = "",
    ) -> AgentEvent:
        """创建工具执行结果事件。"""
        return cls(
            event_type=EventType.TOOL_RESULT,
            content=content,
            tool_call_id=tool_call_id,
            tool_name=name,
            thread_id=thread_id,
            trace_id=trace_id,
        )

    @classmethod
    def interrupt(
        cls,
        *,
        prompt: str,
        choices: list[Any] | None = None,
        thread_id: str = "",
        trace_id: str = "",
    ) -> AgentEvent:
        """创建中断事件（ask_human / 危险命令确认）。"""
        return cls(
            event_type=EventType.INTERRUPT,
            interrupt_prompt=prompt,
            interrupt_choices=choices or [],
            thread_id=thread_id,
            trace_id=trace_id,
        )

    @classmethod
    def cancelled(
        cls,
        content: str,
        *,
        thread_id: str = "",
        trace_id: str = "",
    ) -> AgentEvent:
        """创建用户拒绝危险命令事件。"""
        return cls(
            event_type=EventType.CANCELLED,
            content=content,
            thread_id=thread_id,
            trace_id=trace_id,
        )

    @classmethod
    def error(
        cls,
        content: str,
        *,
        thread_id: str = "",
        trace_id: str = "",
    ) -> AgentEvent:
        """创建执行异常事件。"""
        return cls(
            event_type=EventType.ERROR,
            content=content,
            thread_id=thread_id,
            trace_id=trace_id,
        )

    @classmethod
    def done(
        cls,
        *,
        content: str = "",
        thread_id: str = "",
        role: str = "assistant",
        is_important: bool = False,
        trace_id: str = "",
    ) -> AgentEvent:
        """创建本轮执行结束事件。content 携带最终输出文本。"""
        return cls(
            event_type=EventType.DONE,
            content=content,
            thread_id=thread_id,
            role=role,
            is_important=is_important,
            trace_id=trace_id,
        )

    # ============ 查询方法 ============

    @property
    def is_terminal(self) -> bool:
        """是否为流终止事件（DONE / ERROR / CANCELLED）。"""
        return self.event_type in (
            EventType.DONE,
            EventType.ERROR,
            EventType.CANCELLED,
        )

    @property
    def is_memory_worthy(self) -> bool:
        """是否值得提交给 MemoryManager 评估（携带角色和内容的事件）。

        MEMORY_WORTHY 事件类型：
        - DONE: 最终输出（role=assistant）
        - TOOL_RESULT: 工具执行结果（可能包含经验教训）
        - TOKEN 不单独提交（已聚合在 DONE 的 content 中）
        """
        return self.event_type in (
            EventType.DONE,
            EventType.TOOL_RESULT,
        ) and bool(self.content.strip())

    # ============ 序列化 ============

    def to_sse_dict(self) -> dict[str, Any]:
        """转换为 SSE 推送用的 dict 格式（向后兼容现有前端协议）。

        输出格式（向后兼容前端 SSE 协议）：
        - token: {"type": "token", "content": str}
        - tool_call: {"type": "tool_call", "id", "name", "args"}
        - tool_result: {"type": "tool_result", "id", "name", "content"}
        - interrupt: {"type": "interrupt", "prompt", "choices"}
        - cancelled: {"type": "cancelled", "content"}
        - error: {"type": "error", "content"}
        - done: {"type": "done"}
        """
        if self.event_type == EventType.TOKEN:
            return {"type": "token", "content": self.content}
        elif self.event_type == EventType.TOOL_CALL:
            return {
                "type": "tool_call",
                "id": self.tool_call_id,
                "name": self.tool_name,
                "args": self.tool_args,
            }
        elif self.event_type == EventType.TOOL_RESULT:
            return {
                "type": "tool_result",
                "id": self.tool_call_id,
                "name": self.tool_name,
                "content": self.content,
            }
        elif self.event_type == EventType.INTERRUPT:
            return make_interrupt_dict(self.interrupt_prompt, self.interrupt_choices)
        elif self.event_type == EventType.CANCELLED:
            return {"type": "cancelled", "content": self.content}
        elif self.event_type == EventType.ERROR:
            return {"type": "error", "content": self.content}
        elif self.event_type == EventType.DONE:
            return {"type": "done"}
        return {"type": "unknown"}


__all__ = [
    "AgentEvent",
    "EventType",
]
