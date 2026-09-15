"""长期记忆数据结构与事件分类判定。

定义长期记忆的核心数据模型：
- ``ThreadFactItem``: 单条长期记忆（存储到 LangGraph Store）
- ``MemoryInputEvent``: 待评估是否写入长期记忆的 Agent 事件（原 AgentEvent，重命名以释放 AgentEvent 给执行事件模型）
- ``MemoryCategory``: 记忆分类枚举
- ``judge_long_term_memory``: 事件分类判定函数

设计参照 ``docs/长期事件触发.md`` 中的伪代码规范。
"""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


def _naive_now() -> datetime:
    """返回本地 naive 时间（无时区），兼容历史数据格式。"""
    return datetime.now()  # noqa: DTZ005


class MemoryCategory(str, Enum):
    """长期记忆分类。"""

    USER_FACT = "user_fact"
    """用户事实偏好"""

    LESSON_EXPERIENCE = "lesson"
    """经验教训：工具踩坑、稳定推理结论、不可行方案"""

    BUSINESS_ENTITY = "business"
    """业务实体信息：项目配置、接口、长期目标、角色定义"""

    IMPORTANT_CONVERSATION = "conv"
    """用户显式标记 / 重要技术决策"""

    SKIP = "skip"
    """不写入长期记忆，仅保留在 checkpoint 短期会话记忆"""


@dataclass
class ThreadFactItem:
    """单条长期记忆条目（存储到 LangGraph Store）。

    Attributes:
        fact_id: 唯一标识（uuid hex）
        thread_id: 所属会话线程 ID。agent 作用域下自动设为 agent_key（process_type 或 "default"），用于溯源
        scope: 记忆作用域。``"thread"`` 表示按 thread 隔离（conv/business 类），``"agent"`` 表示跨会话共享（user_fact/lesson 类）。默认 ``"thread"``
        content: 记忆文本内容
        category: 记忆分类（MemoryCategory 值）
        confidence: 置信度 0.0-1.0
        create_time: 创建时间（naive ISO 字符串）
        last_used_at: 最后使用时间（naive ISO 字符串，读取时更新）
    """

    fact_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    thread_id: str = ""
    scope: str = "thread"
    content: str = ""
    category: str = MemoryCategory.IMPORTANT_CONVERSATION.value
    confidence: float = 0.8
    create_time: str = field(default_factory=lambda: _naive_now().isoformat())
    last_used_at: str = field(default_factory=lambda: _naive_now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """序列化为 dict（用于 Store.aput value）。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ThreadFactItem:
        """从 dict 反序列化（用于 Store.aget/asearch 结果）。"""
        return cls(**data)


@dataclass
class MemoryInputEvent:
    """待评估是否写入长期记忆的事件。

    由写中间件的 Fact 处理流水线构建，传入
    :func:`judge_long_term_memory` 做确定性预判定。
    语义字段（是否猜想/是否一次性任务等）不再以确定性规则判断，
    统一交由 LLM 抽取阶段依据对话内容自行分类。

    Attributes:
        event_type: 事件类型（``"message"`` 或 ``"tool_result"``）
        content: 原始文本片段
        is_user_explicit_remember: 用户明确说"记住这个"
        failure_repeat_count: 同类工具失败历史出现次数（由写中间件维护）
    """

    event_type: str = "message"
    content: str = ""
    is_user_explicit_remember: bool = False
    failure_repeat_count: int = 0


def judge_long_term_memory(event: MemoryInputEvent) -> MemoryCategory | None:
    """确定性预判定一条事件是否值得沉淀长期记忆。

    只处理可确定性判定的信号（失败次数 / 用户显式记住），语义字段
    （是否猜想、是否临时、是否技术决策等）不再以规则判断，统一交由
    LLM 抽取阶段依据对话内容自行分类。

    返回值语义：
    - :attr:`MemoryCategory.SKIP`：确定性丢弃，不进入 LLM 抽取
    - :attr:`MemoryCategory.LESSON_EXPERIENCE`：同类失败 ≥2 次，确定性记为经验教训
    - :attr:`MemoryCategory.IMPORTANT_CONVERSATION`：用户显式标记，提高 LLM 抽取优先级
    - ``None``：值得评估，分类交由 LLM 抽取决定
    """
    # 1. 单次失败的工具结果且非显式记住 → 确定性丢弃（不值得沉淀）
    if (
        event.event_type == "tool_result"
        and event.failure_repeat_count <= 1
        and not event.is_user_explicit_remember
    ):
        return MemoryCategory.SKIP

    # 2. 同类失败 ≥2 次 → 确定性记为经验教训（绕过 LLM 分类）
    if event.event_type == "tool_result" and event.failure_repeat_count >= 2:
        return MemoryCategory.LESSON_EXPERIENCE

    # 3. 用户显式标记 → 提高优先级（LLM 抽取时标注）
    if event.is_user_explicit_remember:
        return MemoryCategory.IMPORTANT_CONVERSATION

    # 4. 其余 → 交 LLM 抽取
    return None


__all__ = [
    "MemoryCategory",
    "MemoryInputEvent",
    "ThreadFactItem",
    "judge_long_term_memory",
]
