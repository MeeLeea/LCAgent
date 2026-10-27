"""长期记忆数据结构与事件分类判定。

定义长期记忆的核心数据模型：
- ``ThreadFactItem``: 单条长期记忆（存储到 LangGraph Store）
- ``MemoryInputEvent``: 待评估是否写入长期记忆的 Agent 事件（原 AgentEvent，重命名以释放 AgentEvent 给执行事件模型）
- ``MemoryCategory``: 记忆分类枚举
- ``judge_long_term_memory``: 事件分类判定函数
- ``normalize_topic_key``: 主题键归一化（写入合并的稳定性前提）

记忆生命周期字段（status / evidence / topic_key）：
- ``status``: ``active`` 生效 / ``superseded`` 被同主题新条目取代（保留审计） / ``stale`` 人工标记过时
- ``evidence``: 证据来源 ``tested`` 实测 / ``stated`` 陈述 / ``inferred`` 推断
- ``topic_key``: 规范化主题键，非空时同主题新记忆取代旧记忆（而非追加并存）

设计参照 ``docs/长期事件触发.md`` 中的伪代码规范。
"""
from __future__ import annotations

import re
import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from enum import Enum
from typing import Any


def _naive_now() -> datetime:
    """返回本地 naive 时间（无时区），兼容历史数据格式。"""
    return datetime.now()  # noqa: DTZ005


# ── 记忆状态（status 字段取值） ──
FACT_STATUS_ACTIVE = "active"
"""生效中：读取注入 prompt 的唯一状态"""

FACT_STATUS_SUPERSEDED = "superseded"
"""已被同主题（topic_key 相同）新条目取代，保留供审计，读取注入时过滤"""

FACT_STATUS_STALE = "stale"
"""人工/机制标记过时，读取注入时过滤"""

VALID_FACT_STATUSES: frozenset[str] = frozenset({
    FACT_STATUS_ACTIVE,
    FACT_STATUS_SUPERSEDED,
    FACT_STATUS_STALE,
})

# ── 证据来源（evidence 字段取值） ──
EVIDENCE_TESTED = "tested"
EVIDENCE_STATED = "stated"
EVIDENCE_INFERRED = "inferred"

VALID_EVIDENCES: frozenset[str] = frozenset({
    EVIDENCE_TESTED,
    EVIDENCE_STATED,
    EVIDENCE_INFERRED,
})

# ── 持久层级（persistence 判定，决定写入作用域，与 category 解耦） ──
PERSISTENCE_SESSION = "session"
"""仅本次会话成立（环境状态 / 初始化顺序 / 任务特有约束）→ thread 级 namespace"""

PERSISTENCE_GLOBAL = "global"
"""跨会话普适（通用契约 / 用户长期偏好 / 跨项目教训）→ agent 级 namespace"""

VALID_PERSISTENCES: frozenset[str] = frozenset({PERSISTENCE_SESSION, PERSISTENCE_GLOBAL})

# 主题键长度上限（归一化后超长截断）
TOPIC_KEY_MAX_CHARS = 50

# 主题键归一化：仅允许 [a-z0-9_]，其余字符替换为下划线
_TOPIC_KEY_INVALID_CHARS = re.compile(r"[^a-z0-9_]+")
_TOPIC_KEY_CONSECUTIVE_UNDERSCORES = re.compile(r"_+")


def normalize_topic_key(raw: str) -> str:
    """归一化 LLM 输出的主题键为稳定 snake_case 形式。

    写入合并依赖 topic_key 精确匹配，LLM 输出必须先经确定性归一化：
    小写、非法字符折叠为单个下划线、去首尾下划线、截断至 50 字符。
    归一化后为空（如纯中文/纯符号输入）返回空串——空 topic_key 视为
    "无主题"，写入退化为追加（兼容旧数据与 LLM 未输出场景）。

    Args:
        raw: LLM 抽取输出的原始主题键（可能大小写混杂、含中文/空格/连字符）

    Returns:
        规范化主题键；无效输入返回空串
    """
    if not raw:
        return ""
    lowered = raw.strip().lower()
    folded = _TOPIC_KEY_INVALID_CHARS.sub("_", lowered)
    folded = _TOPIC_KEY_CONSECUTIVE_UNDERSCORES.sub("_", folded).strip("_")
    return folded[:TOPIC_KEY_MAX_CHARS]


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
        topic_key: 规范化主题键（snake_case，≤50 字符）。非空时同主题新记忆
            写入会将旧条目标记 ``superseded``（取代而非并存）；为空表示无主题
            （旧数据 / LLM 未输出），写入与读取均退化为 append 行为
        evidence: 证据来源。``"tested"`` 实测验证 / ``"stated"`` 对话陈述（默认） /
            ``"inferred"`` 未经实测的推断——读取注入时推断类附"（未实测）"标注
        status: 生命周期状态。``"active"`` 生效（默认）/ ``"superseded"`` 被同主题
            新条目取代（保留审计）/ ``"stale"`` 标记过时。读取注入时过滤非 active
        superseded_at: 被取代时间（naive ISO 字符串，仅 status=superseded 时非空）
    """

    fact_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    thread_id: str = ""
    scope: str = "thread"
    content: str = ""
    category: str = MemoryCategory.IMPORTANT_CONVERSATION.value
    confidence: float = 0.8
    create_time: str = field(default_factory=lambda: _naive_now().isoformat())
    last_used_at: str = field(default_factory=lambda: _naive_now().isoformat())
    topic_key: str = ""
    evidence: str = EVIDENCE_STATED
    status: str = FACT_STATUS_ACTIVE
    superseded_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        """序列化为 dict（用于 Store.aput value）。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ThreadFactItem:
        """从 dict 反序列化（用于 Store.aget/asearch 结果）。

        过滤未知字段：新版本写入的行若被旧代码读取（或反之），多余字段
        不再导致 ``TypeError``，保证 schema 演进的双向兼容。旧数据缺失的
        新字段由 dataclass 默认值兜底（active / 无主题 / 陈述型证据）。
        """
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


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
    - :attr:`MemoryCategory.LESSON_EXPERIENCE`：同类失败 ≥2 次，分类确定性锁定
      为经验教训（内容仍由写中间件交 LLM 蒸馏成简短教训后入库，不逐字存原文）
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
    "EVIDENCE_INFERRED",
    "EVIDENCE_STATED",
    "EVIDENCE_TESTED",
    "FACT_STATUS_ACTIVE",
    "FACT_STATUS_STALE",
    "FACT_STATUS_SUPERSEDED",
    "PERSISTENCE_GLOBAL",
    "PERSISTENCE_SESSION",
    "VALID_EVIDENCES",
    "VALID_FACT_STATUSES",
    "VALID_PERSISTENCES",
    "MemoryCategory",
    "MemoryInputEvent",
    "ThreadFactItem",
    "judge_long_term_memory",
    "normalize_topic_key",
]
