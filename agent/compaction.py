"""自定义压缩中间件：增量摘要 + 工具输出 Prune + 保留近期消息

三层压缩策略:
1. 增量摘要: 已有 state.summary + 旧消息 -> 更新后的 summary（避免每次全量重做）
2. 工具输出 Prune: 保留消息中的老工具输出替换为占位符，无损释放大量 token
3. 保留近期 N 条原始消息，不修改

摘要存入 LangGraph state.summary 字段，随 checkpoint 自动持久化，
天然实现 per-thread 隔离（每个 thread 有独立 summary），彻底消除
self.compaction_summary 的跨会话污染问题。

触发方式:
- 自动: before_model 中间件，每次 model 调用前按预估 token 是否超阈值触发
- 手动: AgentCore.manually_compact() / CLI 命令 compact

工具输出 Prune 独立于压缩触发：只要历史工具输出裁剪收益足够，就在每次
model 调用前独立执行（保护最后 keep_recent 条消息），与 token 阈值无关。
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, NotRequired

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.agents.middleware.types import OmitFromInput
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.messages.utils import get_buffer_string
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)


class LCAgentState(AgentState):
    """扩展 AgentState，添加增量摘要 + 活跃技能字段（随 checkpoint 持久化）。

    - ``summary``: 当前 thread 的历史对话摘要，由 CompactionMiddleware 增量更新。
    - ``active_skills``: 当前 thread 手动加载的技能名列表，由
      ``SkillInjectionMW`` 在 model 调用时读取并注入提示词。
      使用 ``OmitFromInput`` 防止用户输入覆盖此字段。

    每个 thread 拥有独立的 state，天然隔离。
    """

    summary: str
    active_skills: Annotated[NotRequired[list[str]], OmitFromInput]


def estimate_messages_tokens(messages: list[AnyMessage]) -> int:
    """字符数 /4 粗估预估 token；含 tool_calls 参数文本。无新依赖。"""
    total_chars = 0
    for msg in messages:
        content = str(msg.content) if msg.content is not None else ""
        total_chars += len(content)
        if isinstance(msg, AIMessage):
            for tc in getattr(msg, "tool_calls", None) or []:
                total_chars += len(str(tc.get("args", "")))
    return max(1, total_chars // 4)


@dataclass(frozen=True, slots=True)
class CompactionConfig:
    """压缩配置"""

    max_context_tokens: int = 100_000
    """触发压缩的预估 token 阈值（0 = 关闭自动触发）。每次 model 调用前检查。"""

    keep_recent: int = 20
    """保留最近 N 条消息（不参与摘要）"""

    max_tool_output_chars: int = 200
    """工具输出超过此长度则触发 Prune"""

    tool_prune_preview: int = 100
    """Prune 后保留的预览字符数"""

    @classmethod
    def from_kwargs(cls, max_context_tokens: int = 0, context_trim_keep: int = 12) -> CompactionConfig:
        """从 AgentCore 现有配置参数构建 CompactionConfig。

        触发阈值按预估 token（字符数 /4 粗估）计算，而非消息条数：
        巨型工具输出会让少量消息占据绝大部分 token，仅看条数会漏判。

        Args:
            max_context_tokens: 预估 token 阈值（0=关闭自动触发）。0 时使用默认值 100000。
            context_trim_keep: 保留的最近消息数（下限 4）。
        """
        return cls(
            max_context_tokens=max_context_tokens if max_context_tokens > 0 else 100_000,
            keep_recent=max(context_trim_keep, 4),
        )


class LCAgentCompactionMiddleware(AgentMiddleware):
    """三层压缩中间件：增量摘要 + 工具输出 Prune + 保留近期消息

    在 before_model / abefore_model 中自动触发，触发条件是**预估 token**（字符数 /4
    粗估）达到 ``max_context_tokens`` 阈值，而非消息条数——巨型工具输出会让少量消息
    占据绝大部分 token，仅看条数会漏判（真实会话曾出现 117 条消息占 185K prompt tokens
    却未触发压缩）。超过阈值时：
      1. 找安全切割点（不拆开 AIMessage(tool_calls) + ToolMessage 对）
      2. 旧消息与已有 summary 增量合并 -> 新 summary
      3. 保留消息中的长工具输出 Prune 为占位符
      4. 重建消息列表：HumanMessage(摘要) + Pruned 近期消息
      5. 更新 state.summary

    工具输出 Prune **独立于压缩**：即使未达 token 阈值，只要裁剪历史工具输出的
    收益足够（保护最后 keep_recent 条消息），也在每次 model 调用前独立执行。

    state.summary 随 checkpoint 持久化，每个 thread 独立隔离。
    """

    SUMMARY_HEADER = "【历史对话摘要（上文因过长已被自动压缩）】\n"

    def __init__(
        self,
        model: Any | None = None,
        config: CompactionConfig | None = None,
        on_compaction: Callable[[str, int, int, int, float], None] | None = None,
        *,
        model_resolver: Callable[[Any], Any] | None = None,
    ) -> None:
        """初始化压缩中间件

        Args:
            model: LLM 模型，用于生成摘要
            config: 压缩配置
            on_compaction: 压缩完成回调，签名 (trigger, messages_before, messages_after, summary_length, duration_ms)
                           用于将自动触发的压缩记录到 MetricsCollector
        """
        if model is None and model_resolver is None:
            raise ValueError("model 和 model_resolver 至少提供一个")
        self.model = model
        self._model_resolver = model_resolver
        self.config = config or CompactionConfig()
        self._on_compaction = on_compaction

    # ============ 自动触发（中间件接口） ============

    def before_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """同步版本：在 model 调用前独立 Prune 工具输出 + 按 token 阈值压缩"""
        messages = state["messages"]
        prune_update = self._maybe_prune(messages)
        if self._over_threshold(messages):
            compact = self._do_compact_sync(state, runtime)
            if compact is not None:
                return compact
        return prune_update

    async def abefore_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """异步版本：在 model 调用前独立 Prune 工具输出 + 按 token 阈值压缩"""
        messages = state["messages"]
        prune_update = self._maybe_prune(messages)
        if self._over_threshold(messages):
            compact = await self._do_compact_async(state, runtime)
            if compact is not None:
                return compact
        return prune_update

    def _over_threshold(self, messages: list[AnyMessage]) -> bool:
        """预估 token 是否达到触发压缩的阈值（阈值 <= 0 时关闭自动触发）。"""
        threshold = self.config.max_context_tokens
        return bool(threshold > 0 and estimate_messages_tokens(messages) >= threshold)

    def _maybe_prune(self, messages: list[AnyMessage]) -> dict[str, Any] | None:
        """独立 Prune：只裁剪历史工具输出（保护最后 keep_recent 条）。"""
        if len(messages) <= self.config.keep_recent:
            return None
        cutoff = len(messages) - self.config.keep_recent
        prefix = messages[:cutoff]
        keep = messages[cutoff:]
        saved = self._prune_saved_chars(prefix)
        if saved <= self.config.max_tool_output_chars * 10:
            return None  # 收益过小，避免频繁 checkpoint 写入
        pruned_prefix = self._prune_tool_outputs(prefix)
        if pruned_prefix == list(prefix):
            return None
        return {"messages": [_make_remove_all(), *pruned_prefix, *keep]}

    def _prune_saved_chars(self, messages: list[AnyMessage]) -> int:
        """估算 Prune 可释放的字符数（仅统计超过阈值的工具输出）。"""
        total = 0
        for msg in messages:
            if isinstance(msg, ToolMessage):
                content = str(msg.content)
                if len(content) > self.config.max_tool_output_chars:
                    total += len(content) - self.config.tool_prune_preview
        return total

    # ============ 手动触发（供 AgentCore 调用） ============

    async def arun_compaction(
        self,
        messages: list[AnyMessage],
        existing_summary: str = "",
        force: bool = False,
        model: Any | None = None,
    ) -> dict[str, Any] | None:
        """手动执行一次压缩，返回状态更新字典（或 None 表示无需压缩）。

        供 AgentCore.manually_compact() 调用，不依赖 LangGraph 中间件上下文。

        Args:
            messages: 当前 thread 的完整消息列表
            existing_summary: 当前已有的摘要文本
            force: 为 True 时跳过 max_context_tokens 阈值检查，允许在预估
                   token 未超阈值时强制压缩。仍受 _find_safe_cutoff 约束
                   （消息数 <= keep_recent 时无法安全切割，返回 None）。

        Returns:
            {"messages": [...], "summary": str} 或 None（消息不足或摘要失败时）
        """
        if not force and not self._over_threshold(messages):
            return None

        cutoff = self._find_safe_cutoff(messages)
        if cutoff <= 0:
            return None

        to_summarize = messages[:cutoff]
        to_keep = messages[cutoff:]

        new_summary = await self._aincremental_summary(existing_summary, to_summarize, model=model)
        if not new_summary:
            return None

        result, _ = self._build_compact_result(new_summary, to_keep)
        return result

    # ============ 核心压缩逻辑 ============

    def _do_compact_sync(self, state: dict[str, Any], runtime: Any = None) -> dict[str, Any] | None:
        """同步执行压缩"""
        _start = time.time()
        messages = list(state.get("messages", []))
        cutoff = self._find_safe_cutoff(messages)
        if cutoff <= 0:
            return None

        to_summarize = messages[:cutoff]
        to_keep = messages[cutoff:]

        existing_summary = state.get("summary", "") or ""
        new_summary = self._create_summary_sync(
            existing_summary, to_summarize, model=self._resolve_model(runtime)
        )
        if not new_summary:
            return None

        result, messages_after = self._build_compact_result(new_summary, to_keep)
        self._notify_compaction_metric(len(messages), messages_after, len(new_summary), _start)
        return result

    async def _do_compact_async(self, state: dict[str, Any], runtime: Any = None) -> dict[str, Any] | None:
        """异步执行压缩"""
        _start = time.time()
        messages = list(state.get("messages", []))
        cutoff = self._find_safe_cutoff(messages)
        if cutoff <= 0:
            return None

        to_summarize = messages[:cutoff]
        to_keep = messages[cutoff:]

        existing_summary = state.get("summary", "") or ""
        new_summary = await self._aincremental_summary(
            existing_summary, to_summarize, model=self._resolve_model(runtime)
        )
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
                    HumanMessage(content=self.SUMMARY_HEADER + new_summary),
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

    def _resolve_model(self, runtime: Any) -> Any | None:
        """解析当前 runtime 的模型，失败时回退构建期模型。"""
        if self._model_resolver is None:
            return self.model
        try:
            return self._model_resolver(runtime)
        except (ValueError, TypeError, OSError, RuntimeError) as error:
            logger.warning("压缩模型解析失败，沿用默认模型: %s", error)
            return self.model

    def _create_summary_sync(
        self, existing: str, messages: list[AnyMessage], model: Any | None = None
    ) -> str:
        """同步生成增量摘要"""
        formatted = get_buffer_string(messages, format="xml")
        prompt = self._build_summary_prompt(existing, formatted)
        selected_model = self.model if model is None else model
        if selected_model is None:
            return ""
        try:
            response = selected_model.invoke(prompt)
            return response.text.strip()
        except Exception as error:
            logger.warning("增量摘要生成失败，跳过压缩: %s", error, exc_info=True)
            return ""

    async def _aincremental_summary(
        self, existing: str, messages: list[AnyMessage], model: Any | None = None
    ) -> str:
        """异步生成增量摘要：已有摘要 + 新消息 -> 更新后的摘要

        如果已有摘要，则请求 LLM 将新内容合并到已有摘要中（增量更新）；
        如果没有已有摘要，则请求 LLM 生成首次摘要。

        摘要失败时返回空字符串，调用方据此决定不压缩（保留原消息）。
        """
        formatted = get_buffer_string(messages, format="xml")
        prompt = self._build_summary_prompt(existing, formatted)
        selected_model = self.model if model is None else model
        if selected_model is None:
            return ""
        try:
            response = await selected_model.ainvoke(prompt)
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
