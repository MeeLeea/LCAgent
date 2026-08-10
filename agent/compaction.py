"""自定义压缩中间件：增量摘要 + 工具输出 Prune + 保留近期消息

三层压缩策略:
1. 增量摘要: 已有 state.summary + 旧消息 -> 更新后的 summary（避免每次全量重做）
2. 工具输出 Prune: 保留消息中的老工具输出替换为占位符，无损释放大量 token
3. 保留近期 N 条原始消息，不修改

摘要存入 LangGraph state.summary 字段，随 checkpoint 自动持久化，
天然实现 per-thread 隔离（每个 thread 有独立 summary），彻底消除
self.compaction_summary 的跨会话污染问题。

触发方式:
- 自动: before_model 中间件，每次 model 调用前检查消息数是否超阈值
- 手动: AgentCore.manually_compact() / CLI 命令 compact
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.agents.middleware.types import OmitFromInput
from langchain_core.messages import (
    AnyMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.utils import get_buffer_string
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime
from typing_extensions import NotRequired

logger = logging.getLogger(__name__)


class LCAgentState(AgentState):
    """扩展 AgentState，添加增量摘要 + 活跃技能字段（随 checkpoint 持久化）。

    - ``summary``: 当前 thread 的历史对话摘要，由 CompactionMiddleware 增量更新。
    - ``active_skills``: 当前 thread 手动加载的技能名列表，由
      ``SkillInjectionMiddleware`` 在 model 调用时读取并注入提示词。
      使用 ``OmitFromInput`` 防止用户输入覆盖此字段。

    每个 thread 拥有独立的 state，天然隔离。
    """

    summary: str
    active_skills: Annotated[NotRequired[list[str]], OmitFromInput]


@dataclass(frozen=True, slots=True)
class CompactionConfig:
    """压缩配置"""

    max_messages: int = 50
    """触发压缩的消息数阈值（消息总数超过此值时触发）"""

    keep_recent: int = 20
    """保留最近 N 条消息（不参与摘要，原样保留）"""

    max_tool_output_chars: int = 200
    """工具输出超过此长度则触发 Prune"""

    tool_prune_preview: int = 100
    """Prune 后保留的预览字符数"""

    @classmethod
    def from_kwargs(cls, max_context_messages: int = 0, context_trim_keep: int = 12) -> CompactionConfig:
        """从 AgentCore 现有配置参数构建 CompactionConfig。

        Args:
            max_context_messages: 旧配置中的消息阈值（0=关闭）。0 时使用默认值 50。
            context_trim_keep: 旧配置中的保留消息数。
        """
        return cls(
            max_messages=max_context_messages if max_context_messages > 0 else 50,
            keep_recent=max(context_trim_keep, 4),
        )


class LCAgentCompactionMiddleware(AgentMiddleware):
    """三层压缩中间件：增量摘要 + 工具输出 Prune + 保留近期消息

    在 before_model / abefore_model 中自动触发：
    - 消息总数 <= max_messages 时不压缩
    - 超过阈值时：
      1. 找安全切割点（不拆开 AIMessage(tool_calls) + ToolMessage 对）
      2. 旧消息与已有 summary 增量合并 -> 新 summary
      3. 保留消息中的长工具输出 Prune 为占位符
      4. 重建消息列表：SystemMessage(摘要) + Pruned 近期消息
      5. 更新 state.summary

    state.summary 随 checkpoint 持久化，每个 thread 独立隔离。
    """

    SUMMARY_HEADER = "【历史对话摘要（上文因过长已被自动压缩）】\n"

    def __init__(
        self,
        model: Any,
        config: CompactionConfig | None = None,
        on_compaction: Callable[[str, int, int, int, float], None] | None = None,
    ):
        """初始化压缩中间件

        Args:
            model: LLM 模型，用于生成摘要
            config: 压缩配置
            on_compaction: 压缩完成回调，签名 (trigger, messages_before, messages_after, summary_length, duration_ms)
                           用于将自动触发的压缩记录到 MetricsCollector
        """
        self.model = model
        self.config = config or CompactionConfig()
        self._on_compaction = on_compaction

    # ============ 自动触发（中间件接口） ============

    def before_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """同步版本：在 model 调用前检查并执行压缩"""
        messages = state["messages"]
        if len(messages) <= self.config.max_messages:
            return None
        return self._do_compact_sync(state)

    async def abefore_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """异步版本：在 model 调用前检查并执行压缩"""
        messages = state["messages"]
        if len(messages) <= self.config.max_messages:
            return None
        return await self._do_compact_async(state)

    # ============ 手动触发（供 AgentCore 调用） ============

    async def arun_compaction(
        self,
        messages: list[AnyMessage],
        existing_summary: str = "",
        force: bool = False,
    ) -> dict[str, Any] | None:
        """手动执行一次压缩，返回状态更新字典（或 None 表示无需压缩）。

        供 AgentCore.manually_compact() 调用，不依赖 LangGraph 中间件上下文。

        Args:
            messages: 当前 thread 的完整消息列表
            existing_summary: 当前已有的摘要文本
            force: 为 True 时跳过 max_messages 阈值检查，允许在消息数
                   未超阈值时强制压缩。仍受 _find_safe_cutoff 约束
                   （消息数 <= keep_recent 时无法安全切割，返回 None）。

        Returns:
            {"messages": [...], "summary": str} 或 None（消息不足或摘要失败时）
        """
        if not force and len(messages) <= self.config.max_messages:
            return None

        cutoff = self._find_safe_cutoff(messages)
        if cutoff <= 0:
            return None

        to_summarize = messages[:cutoff]
        to_keep = messages[cutoff:]

        new_summary = await self._aincremental_summary(existing_summary, to_summarize)
        if not new_summary:
            return None

        result, _ = self._build_compact_result(new_summary, to_keep)
        return result

    # ============ 核心压缩逻辑 ============

    def _do_compact_sync(self, state: dict[str, Any]) -> dict[str, Any] | None:
        """同步执行压缩"""
        _start = time.time()
        messages = list(state.get("messages", []))
        cutoff = self._find_safe_cutoff(messages)
        if cutoff <= 0:
            return None

        to_summarize = messages[:cutoff]
        to_keep = messages[cutoff:]

        existing_summary = state.get("summary", "") or ""
        new_summary = self._create_summary_sync(existing_summary, to_summarize)
        if not new_summary:
            return None

        result, messages_after = self._build_compact_result(new_summary, to_keep)
        self._notify_compaction_metric(len(messages), messages_after, len(new_summary), _start)
        return result

    async def _do_compact_async(self, state: dict[str, Any]) -> dict[str, Any] | None:
        """异步执行压缩"""
        _start = time.time()
        messages = list(state.get("messages", []))
        cutoff = self._find_safe_cutoff(messages)
        if cutoff <= 0:
            return None

        to_summarize = messages[:cutoff]
        to_keep = messages[cutoff:]

        existing_summary = state.get("summary", "") or ""
        new_summary = await self._aincremental_summary(existing_summary, to_summarize)
        if not new_summary:
            return None

        result, messages_after = self._build_compact_result(new_summary, to_keep)
        self._notify_compaction_metric(len(messages), messages_after, len(new_summary), _start)
        return result

    def _build_compact_result(
        self, new_summary: str, to_keep: list[AnyMessage]
    ) -> tuple[dict[str, Any], int]:
        """构造压缩结果：REMOVE_ALL 标记 + 摘要 SystemMessage + Prune 后的保留消息。

        Args:
            new_summary: 生成的增量摘要文本
            to_keep: 待保留的近期消息列表

        Returns:
            (状态更新字典, 压缩后消息数 messages_after)
        """
        pruned_keep = self._prune_tool_outputs(to_keep)
        return (
            {
                "messages": [
                    # REMOVE_ALL_MESSAGES 先清空，再写入压缩后的消息
                    # 这样 checkpoint 中旧消息被彻底移除，不再占用存储
                    _make_remove_all(),
                    SystemMessage(content=self.SUMMARY_HEADER + new_summary),
                    *pruned_keep,
                ],
                "summary": new_summary,
            },
            len(pruned_keep) + 1,  # +1 for summary SystemMessage
        )

    def _notify_compaction_metric(
        self,
        messages_before: int,
        messages_after: int,
        summary_length: int,
        start: float,
    ) -> None:
        """自动触发的压缩指标回调（失败不影响压缩主流程，记录后继续）。"""
        if self._on_compaction is None:
            return
        duration_ms = (time.time() - start) * 1000
        try:
            self._on_compaction("auto", messages_before, messages_after, summary_length, duration_ms)
        except Exception as error:
            logger.warning("压缩指标回调失败: %s", error)

    # ============ 增量摘要 ============

    def _create_summary_sync(
        self, existing: str, messages: list[AnyMessage]
    ) -> str:
        """同步生成增量摘要"""
        formatted = get_buffer_string(messages, format="xml")
        prompt = self._build_summary_prompt(existing, formatted)
        try:
            response = self.model.invoke(prompt)
            return response.text.strip()
        except Exception as error:
            logger.warning("增量摘要生成失败，跳过压缩: %s", error, exc_info=True)
            return ""

    async def _aincremental_summary(
        self, existing: str, messages: list[AnyMessage]
    ) -> str:
        """异步生成增量摘要：已有摘要 + 新消息 -> 更新后的摘要

        如果已有摘要，则请求 LLM 将新内容合并到已有摘要中（增量更新）；
        如果没有已有摘要，则请求 LLM 生成首次摘要。

        摘要失败时返回空字符串，调用方据此决定不压缩（保留原消息）。
        """
        formatted = get_buffer_string(messages, format="xml")
        prompt = self._build_summary_prompt(existing, formatted)
        try:
            response = await self.model.ainvoke(prompt)
            return response.text.strip()
        except Exception as error:
            logger.warning("增量摘要生成失败，跳过压缩: %s", error, exc_info=True)
            return ""

    @staticmethod
    def _build_summary_prompt(existing: str, formatted_messages: str) -> str:
        """构建摘要 prompt：有已有摘要时增量合并，无则首次生成"""
        if existing:
            return (
                "你是对话摘要助手。以下是已有的对话摘要和新增的对话内容。\n"
                "请将新增内容合并到已有摘要中，更新摘要。\n"
                "保留所有关键决策、用户意图、事实和文件操作记录。\n"
                "按主题分条组织，不要添加推测内容。用中文输出。\n\n"
                f"【已有摘要】\n{existing}\n\n"
                f"【新增对话】\n{formatted_messages}"
            )
        return (
            "请将以下对话历史压缩成一份简洁的中文摘要，\n"
            "保留关键决策、用户意图与事实，按主题分条列出，不要添加推测内容：\n\n"
            f"{formatted_messages}"
        )

    # ============ 工具输出 Prune ============

    def _prune_tool_outputs(self, messages: list[AnyMessage]) -> list[AnyMessage]:
        """Prune：保留消息中的长工具输出替换为占位符

        工具输出（如文件内容、搜索结果、命令输出）往往占 70%+ token。
        Prune 后只保留预览字符，大幅释放 token，同时保留语义完整性
        （Agent 仍能知道工具执行了什么操作、大致返回了什么类型的结果）。

        AIMessage 中的 tool_calls 不受影响（工具调用的参数通常很短）。
        """
        pruned: list[AnyMessage] = []
        for msg in messages:
            if isinstance(msg, ToolMessage):
                content = str(msg.content)
                if len(content) > self.config.max_tool_output_chars:
                    preview = content[: self.config.tool_prune_preview]
                    original_len = len(content)
                    pruned_msg = ToolMessage(
                        content=(
                            f"[工具输出已裁剪 {original_len}→{self.config.tool_prune_preview}字符] "
                            f"{preview}..."
                        ),
                        tool_call_id=msg.tool_call_id,
                        name=getattr(msg, "name", "") or "",
                        status=getattr(msg, "status", "success"),
                    )
                    pruned.append(pruned_msg)
                    continue
            pruned.append(msg)
        return pruned

    # ============ 安全切割 ============

    def _find_safe_cutoff(self, messages: list[AnyMessage]) -> int:
        """找出安全切割点：不拆开 AIMessage(tool_calls) + ToolMessage 对

        目标是保留最近 keep_recent 条消息，但如果切割点落在 ToolMessage 上，
        需要向前找到对应的 AIMessage（包含 tool_calls 的那条），
        确保 AI 调用与工具结果成对出现在同一侧。

        Returns:
            切割点索引（0..len(messages)），0 表示无法安全切割
        """
        keep = self.config.keep_recent
        if len(messages) <= keep:
            return 0

        target = len(messages) - keep
        if target <= 0:
            return 0

        # 如果切割点是 ToolMessage，向前找到对应的 AIMessage
        while target > 0 and isinstance(messages[target], ToolMessage):
            target -= 1

        # 如果回退到 0，说明整个历史都是工具消息对，无法安全切割
        return target


# ============ 辅助函数 ============


def _make_remove_all():
    """创建 RemoveAllMessages 标记，用于清空 checkpoint 中的旧消息"""
    from langchain_core.messages import RemoveMessage

    return RemoveMessage(id=REMOVE_ALL_MESSAGES)


__all__ = [
    "CompactionConfig",
    "LCAgentCompactionMiddleware",
    "LCAgentState",
]
