"""Agent 运行时指标收集

追踪三类核心指标：
1. LLM 调用：次数、prompt/completion tokens、按 provider 分组
2. 工具执行：次数、耗时、按工具名分组（min/max/avg）
3. 压缩统计：触发次数、压缩前后消息数、摘要长度

设计：
- MetricsCollector 为单实例，挂在 AgentCore 上
- 线程安全（所有写操作加锁）
- 支持 reset() 和 to_dict() 便于 CLI 展示和序列化
- token 估算：当 LLM 不返回 usage_metadata 时，用字符数 / 4 粗估
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

# ── 数据结构 ────────────────────────────────────────────────────

@dataclass(slots=True)
class LLMCallMetric:
    """单次 LLM 调用指标"""
    provider: str = ""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    duration_ms: float = 0.0


@dataclass(slots=True)
class ToolCallMetric:
    """单次工具调用指标"""
    name: str = ""
    duration_ms: float = 0.0
    success: bool = True
    timed_out: bool = False


@dataclass(slots=True)
class CompactionMetric:
    """单次压缩指标"""
    trigger: str = ""  # "auto" | "manual"
    messages_before: int = 0
    messages_after: int = 0
    summary_length: int = 0
    duration_ms: float = 0.0


@dataclass(slots=True)
class ToolStats:
    """按工具名聚合的统计"""
    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0
    failures: int = 0
    timeouts: int = 0

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.count if self.count > 0 else 0.0


@dataclass(slots=True)
class LLMStats:
    """按 provider 聚合的 LLM 统计"""
    count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_ms: float = 0.0

    @property
    def avg_tokens(self) -> float:
        return self.total_tokens / self.count if self.count > 0 else 0.0


# ── 指标收集器 ──────────────────────────────────────────────────

class MetricsCollector:
    """线程安全的运行时指标收集器

    用法:
        collector = MetricsCollector()
        collector.record_llm_call(provider="zhipu", model="glm-4",
                                  prompt_tokens=100, completion_tokens=50)
        collector.record_tool_call("run_shell", duration_ms=320.5)
        collector.record_compaction("auto", messages_before=60, messages_after=22)

        stats = collector.get_summary()
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._llm_calls: list[LLMCallMetric] = []
        self._tool_calls: list[ToolCallMetric] = []
        self._compactions: list[CompactionMetric] = []
        self._session_start: float = time.time()
        self._turn_count: int = 0

    # ============ LLM 指标 ============

    def record_llm_call(
        self,
        provider: str = "",
        model: str = "",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        duration_ms: float = 0.0,
    ) -> None:
        """记录一次 LLM 调用"""
        if total_tokens == 0 and (prompt_tokens or completion_tokens):
            total_tokens = prompt_tokens + completion_tokens

        metric = LLMCallMetric(
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            duration_ms=duration_ms,
        )
        with self._lock:
            self._llm_calls.append(metric)

    def extract_and_record_llm_usage(
        self,
        message: Any,
        provider: str = "",
        model: str = "",
        duration_ms: float = 0.0,
    ) -> None:
        """从 LangChain AIMessage 中提取 token 用量并记录

        LangChain 的 AIMessage.response_metadata 中包含 usage_metadata 字段：
        {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}

        如果消息不带 usage_metadata，则用字符数粗估（/4）。
        """
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        # 尝试从 response_metadata 提取
        resp_meta = getattr(message, "response_metadata", None) or {}
        usage = resp_meta.get("usage_metadata") or resp_meta.get("token_usage") or {}

        if usage:
            # LangChain 新格式: usage_metadata
            prompt_tokens = usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)

        # 兜底：从 usage 字段提取（OpenAI 原始格式）
        if not total_tokens:
            usage_raw = resp_meta.get("usage", {})
            if usage_raw:
                prompt_tokens = usage_raw.get("prompt_tokens", 0)
                completion_tokens = usage_raw.get("completion_tokens", 0)
                total_tokens = usage_raw.get("total_tokens", 0)

        # 最终兜底：字符数粗估
        if not total_tokens:
            content = getattr(message, "content", "")
            if content:
                text = str(content) if not isinstance(content, str) else content
                completion_tokens = max(1, len(text) // 4)
                total_tokens = completion_tokens

        self.record_llm_call(
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            duration_ms=duration_ms,
        )

    # ============ 工具指标 ============

    def record_tool_call(
        self,
        name: str,
        duration_ms: float = 0.0,
        success: bool = True,
        timed_out: bool = False,
    ) -> None:
        """记录一次工具调用"""
        metric = ToolCallMetric(
            name=name,
            duration_ms=duration_ms,
            success=success,
            timed_out=timed_out,
        )
        with self._lock:
            self._tool_calls.append(metric)

    # ============ 压缩指标 ============

    def record_compaction(
        self,
        trigger: str,
        messages_before: int = 0,
        messages_after: int = 0,
        summary_length: int = 0,
        duration_ms: float = 0.0,
    ) -> None:
        """记录一次压缩"""
        metric = CompactionMetric(
            trigger=trigger,
            messages_before=messages_before,
            messages_after=messages_after,
            summary_length=summary_length,
            duration_ms=duration_ms,
        )
        with self._lock:
            self._compactions.append(metric)

    # ============ Turn 计数 ============

    def increment_turn(self) -> None:
        """记录一次 Agent turn（arun/achat 调用）"""
        with self._lock:
            self._turn_count += 1

    # ============ 汇总查询 ============

    def get_summary(self) -> dict[str, Any]:
        """获取所有指标的汇总字典"""
        with self._lock:
            # LLM 汇总（按 provider 分组）
            llm_by_provider: dict[str, LLMStats] = {}
            total_prompt = 0
            total_completion = 0
            total_tokens = 0
            llm_total_ms = 0.0
            for call in self._llm_calls:
                key = call.provider or "unknown"
                if key not in llm_by_provider:
                    llm_by_provider[key] = LLMStats()
                s = llm_by_provider[key]
                s.count += 1
                s.prompt_tokens += call.prompt_tokens
                s.completion_tokens += call.completion_tokens
                s.total_tokens += call.total_tokens
                s.total_ms += call.duration_ms
                total_prompt += call.prompt_tokens
                total_completion += call.completion_tokens
                total_tokens += call.total_tokens
                llm_total_ms += call.duration_ms

            # 工具汇总（按 name 分组）
            tool_by_name: dict[str, ToolStats] = {}
            tool_total_ms = 0.0
            for tc in self._tool_calls:
                key = tc.name or "unknown"
                if key not in tool_by_name:
                    tool_by_name[key] = ToolStats()
                s = tool_by_name[key]
                s.count += 1
                s.total_ms += tc.duration_ms
                s.min_ms = min(s.min_ms, tc.duration_ms)
                s.max_ms = max(s.max_ms, tc.duration_ms)
                if not tc.success:
                    s.failures += 1
                if tc.timed_out:
                    s.timeouts += 1
                tool_total_ms += tc.duration_ms

            # 压缩汇总
            compaction_count = len(self._compactions)
            compaction_total_before = sum(c.messages_before for c in self._compactions)
            compaction_total_after = sum(c.messages_after for c in self._compactions)
            compaction_total_ms = sum(c.duration_ms for c in self._compactions)

            session_duration = time.time() - self._session_start

            return {
                "session": {
                    "duration_seconds": round(session_duration, 1),
                    "turn_count": self._turn_count,
                },
                "llm": {
                    "total_calls": len(self._llm_calls),
                    "total_prompt_tokens": total_prompt,
                    "total_completion_tokens": total_completion,
                    "total_tokens": total_tokens,
                    "total_duration_ms": round(llm_total_ms, 1),
                    "by_provider": {
                        k: {
                            "count": v.count,
                            "prompt_tokens": v.prompt_tokens,
                            "completion_tokens": v.completion_tokens,
                            "total_tokens": v.total_tokens,
                            "avg_tokens": round(v.avg_tokens, 1),
                            "total_ms": round(v.total_ms, 1),
                        }
                        for k, v in llm_by_provider.items()
                    },
                },
                "tools": {
                    "total_calls": len(self._tool_calls),
                    "total_duration_ms": round(tool_total_ms, 1),
                    "by_name": {
                        k: {
                            "count": v.count,
                            "total_ms": round(v.total_ms, 1),
                            "min_ms": round(v.min_ms, 1) if v.min_ms != float("inf") else 0,
                            "max_ms": round(v.max_ms, 1),
                            "avg_ms": round(v.avg_ms, 1),
                            "failures": v.failures,
                            "timeouts": v.timeouts,
                        }
                        for k, v in tool_by_name.items()
                    },
                },
                "compaction": {
                    "total_count": compaction_count,
                    "total_messages_before": compaction_total_before,
                    "total_messages_after": compaction_total_after,
                    "total_duration_ms": round(compaction_total_ms, 1),
                    "messages_saved": compaction_total_before - compaction_total_after,
                },
            }

    def reset(self) -> None:
        """清空所有指标"""
        with self._lock:
            self._llm_calls.clear()
            self._tool_calls.clear()
            self._compactions.clear()
            self._session_start = time.time()
            self._turn_count = 0


__all__ = [
    "CompactionMetric",
    "LLMCallMetric",
    "LLMStats",
    "MetricsCollector",
    "ToolCallMetric",
    "ToolStats",
]
