"""长期记忆数据结构与事件分类判定。

定义长期记忆的核心数据模型：
- ``ThreadFactItem``: 单条长期记忆（存储到 LangGraph Store）
- ``AgentEvent``: 待评估是否写入长期记忆的 Agent 事件
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
        thread_id: 所属会话线程 ID
        content: 记忆文本内容
        category: 记忆分类（MemoryCategory 值）
        confidence: 置信度 0.0-1.0
        create_time: 创建时间（naive ISO 字符串）
        last_used_at: 最后使用时间（naive ISO 字符串，读取时更新）
    """

    fact_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    thread_id: str = ""
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
class AgentEvent:
    """待评估是否写入长期记忆的事件。

    由中间件在每轮 Agent 执行结束后构建，传入
    :func:`judge_long_term_memory` 判定分类。

    Attributes:
        event_type: 事件类型 (message / tool_result / reasoning / user_command)
        content: 原始文本片段
        is_user_explicit_remember: 用户明确说"记住这个"
        is_one_shot_task: 是否为本轮临时一次性子任务
        is_hypothesis: 是否未确认的猜想、试探方案
        failure_repeat_count: 同类失败历史出现次数
        is_reusable: 是否跨会话可复用
        is_project_long_term_goal: 是否项目长期目标
        is_temp_resource: 临时路径、临时变量、本轮才有效
        is_technical_decision: 技术选型、方案取舍、架构约定
    """

    event_type: str = "message"
    content: str = ""
    is_user_explicit_remember: bool = False
    is_one_shot_task: bool = False
    is_hypothesis: bool = False
    failure_repeat_count: int = 0
    is_reusable: bool = False
    is_project_long_term_goal: bool = False
    is_temp_resource: bool = False
    is_technical_decision: bool = False


def judge_long_term_memory(event: AgentEvent) -> MemoryCategory:
    """判断一条 Agent 事件是否下沉长期记忆。

    返回分类；返回 :attr:`MemoryCategory.SKIP` 表示只保存在 checkpoint 短期会话记忆。

    判定逻辑参照 ``docs/长期事件触发.md``：
    1. 前置过滤：临时资源、未确认猜想、单纯一次性子任务、单次未标记失败 → SKIP
    2. 经验教训：同类报错重复 >= 2 次、稳定可复用推理结论 → LESSON_EXPERIENCE
    3. 业务实体：项目长期目标、可复用且非一次性非临时的实体信息 → BUSINESS_ENTITY
    4. 用户显式标记 / 技术决策 → IMPORTANT_CONVERSATION
    5. 其余 → SKIP
    """
    # --------------------------
    # 前置过滤：直接跳过的条件
    # --------------------------
    if any([
        event.is_temp_resource,
        event.is_hypothesis,
        event.is_one_shot_task and not event.is_technical_decision,
        (
            event.event_type == "tool_result"
            and event.failure_repeat_count <= 1
            and not event.is_user_explicit_remember
        ),
    ]):
        return MemoryCategory.SKIP

    # --------------------------
    # 条件 1：经验 & 教训
    # --------------------------
    is_lesson_case = any([
        event.event_type == "tool_result" and event.failure_repeat_count >= 2,
        event.event_type == "reasoning" and event.is_reusable and not event.is_hypothesis,
    ])
    if is_lesson_case:
        return MemoryCategory.LESSON_EXPERIENCE

    # --------------------------
    # 条件 2：业务实体信息
    # --------------------------
    is_business_entity_case = any([
        event.is_project_long_term_goal,
        event.is_reusable and not event.is_one_shot_task and not event.is_temp_resource,
    ])
    if is_business_entity_case:
        return MemoryCategory.BUSINESS_ENTITY

    # --------------------------
    # 条件 3：用户显式标记 / 重要技术决策
    # --------------------------
    if any([
        event.is_user_explicit_remember,
        event.is_technical_decision,
    ]):
        return MemoryCategory.IMPORTANT_CONVERSATION

    # --------------------------
    # 其余全部跳过
    # --------------------------
    return MemoryCategory.SKIP


__all__ = [
    "AgentEvent",
    "MemoryCategory",
    "ThreadFactItem",
    "judge_long_term_memory",
]
