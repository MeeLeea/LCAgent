"""MetricsCollector 指标收集测试

验证:
1. LLM 调用指标记录与汇总（tokens / provider 分组）
2. 工具调用指标记录与汇总（耗时 / 成功率 / 超时）
3. 压缩指标记录与汇总（触发次数 / 消息裁剪量）
4. extract_and_record_llm_usage 从 AIMessage 提取 token 用量
5. reset() 清空和 turn 计数
6. compaction 中间件 on_compaction 回调
7. CLI metrics:status / metrics:reset 命令
"""
from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from agent.metrics import (
    MetricsCollector,
    estimate_tokens,
)

# ════════════════════════════════════════════════════════════════════════
#  LLM 指标测试
# ════════════════════════════════════════════════════════════════════════


class TestLLMMetrics:
    """LLM 调用指标测试"""

    def test_record_single_llm_call(self):
        collector = MetricsCollector()
        collector.record_llm_call(
            provider="zhipu",
            model="glm-4",
            prompt_tokens=100,
            completion_tokens=50,
            duration_ms=500.0,
        )
        assert collector.llm_call_count == 1
        summary = collector.get_summary()
        assert summary["llm"]["total_calls"] == 1
        assert summary["llm"]["total_prompt_tokens"] == 100
        assert summary["llm"]["total_completion_tokens"] == 50
        assert summary["llm"]["total_tokens"] == 150

    def test_total_tokens_auto_calculated(self):
        """total_tokens=0 但 prompt+completion 有值时自动计算"""
        collector = MetricsCollector()
        collector.record_llm_call(
            provider="openai",
            prompt_tokens=200,
            completion_tokens=80,
            total_tokens=0,
        )
        summary = collector.get_summary()
        assert summary["llm"]["total_tokens"] == 280

    def test_multiple_providers_grouped(self):
        """多 provider 分组统计"""
        collector = MetricsCollector()
        collector.record_llm_call(provider="zhipu", prompt_tokens=100, completion_tokens=50)
        collector.record_llm_call(provider="zhipu", prompt_tokens=200, completion_tokens=100)
        collector.record_llm_call(provider="deepseek", prompt_tokens=150, completion_tokens=75)

        summary = collector.get_summary()
        assert len(summary["llm"]["by_provider"]) == 2

        zhipu = summary["llm"]["by_provider"]["zhipu"]
        assert zhipu["count"] == 2
        assert zhipu["total_tokens"] == 450  # (100+50) + (200+100)

        deepseek = summary["llm"]["by_provider"]["deepseek"]
        assert deepseek["count"] == 1
        assert deepseek["total_tokens"] == 225

    def test_unknown_provider_grouped(self):
        """provider 为空时归入 unknown"""
        collector = MetricsCollector()
        collector.record_llm_call(prompt_tokens=10, completion_tokens=5)
        summary = collector.get_summary()
        assert "unknown" in summary["llm"]["by_provider"]

    def test_duration_tracking(self):
        """耗时累加"""
        collector = MetricsCollector()
        collector.record_llm_call(provider="test", duration_ms=300.0)
        collector.record_llm_call(provider="test", duration_ms=700.0)
        summary = collector.get_summary()
        assert summary["llm"]["total_duration_ms"] == 1000.0


# ════════════════════════════════════════════════════════════════════════
#  extract_and_record_llm_usage 测试
# ════════════════════════════════════════════════════════════════════════


class TestExtractLLMUsage:
    """从 AIMessage 提取 token 用量测试"""

    def test_extract_from_usage_metadata(self):
        """LangChain 新格式：response_metadata.usage_metadata"""
        msg = AIMessage(content="回复内容")
        msg.response_metadata = {
            "usage_metadata": {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            }
        }
        collector = MetricsCollector()
        collector.extract_and_record_llm_usage(msg, provider="zhipu", model="glm-4")
        summary = collector.get_summary()
        assert summary["llm"]["total_prompt_tokens"] == 100
        assert summary["llm"]["total_completion_tokens"] == 50
        assert summary["llm"]["total_tokens"] == 150

    def test_extract_from_token_usage(self):
        """旧格式：response_metadata.token_usage"""
        msg = AIMessage(content="回复内容")
        msg.response_metadata = {
            "token_usage": {
                "prompt_tokens": 200,
                "completion_tokens": 100,
                "total_tokens": 300,
            }
        }
        collector = MetricsCollector()
        collector.extract_and_record_llm_usage(msg, provider="openai")
        summary = collector.get_summary()
        assert summary["llm"]["total_tokens"] == 300

    def test_extract_from_openai_usage(self):
        """OpenAI 原始格式：response_metadata.usage"""
        msg = AIMessage(content="回复内容")
        msg.response_metadata = {
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 60,
                "total_tokens": 180,
            }
        }
        collector = MetricsCollector()
        collector.extract_and_record_llm_usage(msg, provider="openai")
        summary = collector.get_summary()
        assert summary["llm"]["total_tokens"] == 180

    def test_fallback_char_estimation(self):
        """无 usage_metadata 时用字符数粗估"""
        content = "a" * 40  # 40 字符 -> 10 tokens
        msg = AIMessage(content=content)
        msg.response_metadata = {}

        collector = MetricsCollector()
        collector.extract_and_record_llm_usage(msg, provider="test")
        summary = collector.get_summary()
        assert summary["llm"]["total_completion_tokens"] == 10
        assert summary["llm"]["total_tokens"] == 10

    def test_empty_content_no_tokens(self):
        """空内容不记录 token"""
        msg = AIMessage(content="")
        msg.response_metadata = {}

        collector = MetricsCollector()
        collector.extract_and_record_llm_usage(msg, provider="test")
        summary = collector.get_summary()
        assert summary["llm"]["total_tokens"] == 0


# ════════════════════════════════════════════════════════════════════════
#  工具指标测试
# ════════════════════════════════════════════════════════════════════════


class TestToolMetrics:
    """工具调用指标测试"""

    def test_record_single_tool_call(self):
        collector = MetricsCollector()
        collector.record_tool_call(name="run_shell", duration_ms=320.5, success=True)
        assert collector.tool_call_count == 1
        summary = collector.get_summary()
        assert summary["tools"]["total_calls"] == 1
        assert summary["tools"]["by_name"]["run_shell"]["count"] == 1
        assert summary["tools"]["by_name"]["run_shell"]["avg_ms"] == 320.5

    def test_tool_aggregation_by_name(self):
        """同名工具聚合统计"""
        collector = MetricsCollector()
        collector.record_tool_call(name="search", duration_ms=100.0, success=True)
        collector.record_tool_call(name="search", duration_ms=300.0, success=True)
        collector.record_tool_call(name="search", duration_ms=200.0, success=False)

        summary = collector.get_summary()
        stats = summary["tools"]["by_name"]["search"]
        assert stats["count"] == 3
        assert stats["total_ms"] == 600.0
        assert stats["min_ms"] == 100.0
        assert stats["max_ms"] == 300.0
        assert stats["avg_ms"] == 200.0
        assert stats["failures"] == 1
        assert stats["timeouts"] == 0

    def test_tool_timeout_tracking(self):
        """超时标记"""
        collector = MetricsCollector()
        collector.record_tool_call(name="slow_tool", duration_ms=60000.0, success=False, timed_out=True)
        summary = collector.get_summary()
        stats = summary["tools"]["by_name"]["slow_tool"]
        assert stats["timeouts"] == 1
        assert stats["failures"] == 1

    def test_empty_tools_summary(self):
        """无工具调用时 by_name 为空"""
        collector = MetricsCollector()
        summary = collector.get_summary()
        assert summary["tools"]["total_calls"] == 0
        assert summary["tools"]["by_name"] == {}


# ════════════════════════════════════════════════════════════════════════
#  压缩指标测试
# ════════════════════════════════════════════════════════════════════════


class TestCompactionMetrics:
    """压缩指标测试"""

    def test_record_single_compaction(self):
        collector = MetricsCollector()
        collector.record_compaction(
            trigger="auto",
            messages_before=60,
            messages_after=22,
            summary_length=500,
            duration_ms=1500.0,
        )
        assert collector.compaction_count == 1
        summary = collector.get_summary()
        assert summary["compaction"]["total_count"] == 1
        assert summary["compaction"]["total_messages_before"] == 60
        assert summary["compaction"]["total_messages_after"] == 22
        assert summary["compaction"]["messages_saved"] == 38

    def test_multiple_compactions(self):
        """多次压缩累加"""
        collector = MetricsCollector()
        collector.record_compaction(trigger="auto", messages_before=60, messages_after=22)
        collector.record_compaction(trigger="manual", messages_before=50, messages_after=20)

        summary = collector.get_summary()
        assert summary["compaction"]["total_count"] == 2
        assert summary["compaction"]["total_messages_before"] == 110
        assert summary["compaction"]["total_messages_after"] == 42
        assert summary["compaction"]["messages_saved"] == 68

    def test_empty_compaction_summary(self):
        """无压缩记录"""
        collector = MetricsCollector()
        summary = collector.get_summary()
        assert summary["compaction"]["total_count"] == 0
        assert summary["compaction"]["messages_saved"] == 0


# ════════════════════════════════════════════════════════════════════════
#  Turn 计数 & Reset 测试
# ════════════════════════════════════════════════════════════════════════


class TestTurnCountAndReset:
    """Turn 计数和 Reset 测试"""

    def test_increment_turn(self):
        collector = MetricsCollector()
        collector.increment_turn()
        collector.increment_turn()
        collector.increment_turn()
        summary = collector.get_summary()
        assert summary["session"]["turn_count"] == 3

    def test_reset_clears_all(self):
        collector = MetricsCollector()
        collector.record_llm_call(provider="test", prompt_tokens=100, completion_tokens=50)
        collector.record_tool_call(name="search", duration_ms=100.0)
        collector.record_compaction(trigger="auto", messages_before=60, messages_after=22)
        collector.increment_turn()

        collector.reset()

        summary = collector.get_summary()
        assert summary["llm"]["total_calls"] == 0
        assert summary["tools"]["total_calls"] == 0
        assert summary["compaction"]["total_count"] == 0
        assert summary["session"]["turn_count"] == 0

    def test_session_duration_positive(self):
        """session_duration 为正数"""
        collector = MetricsCollector()
        summary = collector.get_summary()
        assert summary["session"]["duration_seconds"] >= 0.0


# ════════════════════════════════════════════════════════════════════════
#  线程安全测试
# ════════════════════════════════════════════════════════════════════════


class TestThreadSafety:
    """多线程并发写入测试"""

    def test_concurrent_llm_calls(self):
        """100 个线程并发记录 LLM 调用"""
        collector = MetricsCollector()

        def worker():
            for i in range(10):
                collector.record_llm_call(provider="test", prompt_tokens=10, completion_tokens=5)

        threads = [threading.Thread(target=worker) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert collector.llm_call_count == 1000
        summary = collector.get_summary()
        assert summary["llm"]["total_calls"] == 1000

    def test_concurrent_mixed_writes(self):
        """并发混合写入 LLM / 工具 / 压缩"""
        collector = MetricsCollector()

        def llm_worker():
            for _ in range(50):
                collector.record_llm_call(provider="test", prompt_tokens=10, completion_tokens=5)

        def tool_worker():
            for _ in range(50):
                collector.record_tool_call(name="search", duration_ms=100.0)

        def compact_worker():
            for _ in range(10):
                collector.record_compaction(trigger="auto", messages_before=60, messages_after=22)

        threads = [
            threading.Thread(target=llm_worker),
            threading.Thread(target=tool_worker),
            threading.Thread(target=compact_worker),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        summary = collector.get_summary()
        assert summary["llm"]["total_calls"] == 50
        assert summary["tools"]["total_calls"] == 50
        assert summary["compaction"]["total_count"] == 10


# ════════════════════════════════════════════════════════════════════════
#  estimate_tokens 测试
# ════════════════════════════════════════════════════════════════════════


class TestEstimateTokens:
    """token 估算测试"""

    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_short_string(self):
        assert estimate_tokens("abc") == 1

    def test_long_string(self):
        assert estimate_tokens("a" * 40) == 10

    def test_none_returns_zero(self):
        assert estimate_tokens(None) == 0


# ════════════════════════════════════════════════════════════════════════
#  Compaction 中间件 on_compaction 回调测试
# ════════════════════════════════════════════════════════════════════════


class TestCompactionCallback:
    """压缩中间件 on_compaction 回调测试"""

    def _build_messages(self, count: int) -> list:
        """构建 N 条消息用于触发压缩"""
        msgs = []
        for i in range(count):
            if i % 2 == 0:
                msgs.append(HumanMessage(content=f"用户消息-{i}"))
            else:
                msgs.append(AIMessage(content=f"AI回复-{i}"))
        return msgs

    def test_sync_callback_called(self):
        """同步压缩时回调被调用"""
        from agent.compaction import CompactionConfig, LCAgentCompactionMiddleware

        calls = []

        def on_compaction(trigger, before, after, summary_len, duration_ms):
            calls.append((trigger, before, after, summary_len, duration_ms))

        model = SimpleNamespace(
            invoke=lambda prompt: SimpleNamespace(text="摘要内容"),
            ainvoke=None,
        )
        mw = LCAgentCompactionMiddleware(
            model=model,
            config=CompactionConfig(max_messages=5, keep_recent=2),
            on_compaction=on_compaction,
        )
        state = {"messages": self._build_messages(10), "summary": ""}
        result = mw.before_model(state, runtime=None)

        assert result is not None
        assert len(calls) == 1
        assert calls[0][0] == "auto"
        assert calls[0][1] == 10  # messages_before
        assert calls[0][3] == 4   # summary_length (len("摘要内容"))

    def test_async_callback_called(self):
        """异步压缩时回调被调用"""
        import asyncio

        from agent.compaction import CompactionConfig, LCAgentCompactionMiddleware

        calls = []

        def on_compaction(trigger, before, after, summary_len, duration_ms):
            calls.append((trigger, before, after, summary_len, duration_ms))

        async def fake_ainvoke(prompt):
            return SimpleNamespace(text="异步摘要")

        model = SimpleNamespace(
            invoke=lambda prompt: SimpleNamespace(text="摘要内容"),
            ainvoke=fake_ainvoke,
        )
        mw = LCAgentCompactionMiddleware(
            model=model,
            config=CompactionConfig(max_messages=5, keep_recent=2),
            on_compaction=on_compaction,
        )
        state = {"messages": self._build_messages(10), "summary": ""}

        asyncio.run(mw.abefore_model(state, runtime=None))

        assert len(calls) == 1
        assert calls[0][0] == "auto"
        assert calls[0][1] == 10

    def test_no_callback_when_not_set(self):
        """未设置回调时不报错"""
        from agent.compaction import CompactionConfig, LCAgentCompactionMiddleware

        model = SimpleNamespace(
            invoke=lambda prompt: SimpleNamespace(text="摘要内容"),
            ainvoke=None,
        )
        mw = LCAgentCompactionMiddleware(
            model=model,
            config=CompactionConfig(max_messages=5, keep_recent=2),
        )
        state = {"messages": self._build_messages(10), "summary": ""}
        result = mw.before_model(state, runtime=None)

        assert result is not None  # 压缩正常执行，只是没有回调

    def test_callback_exception_swallowed(self):
        """回调抛异常时不影响压缩结果"""
        from agent.compaction import CompactionConfig, LCAgentCompactionMiddleware

        def bad_callback(trigger, before, after, summary_len, duration_ms):
            raise RuntimeError("callback error")

        model = SimpleNamespace(
            invoke=lambda prompt: SimpleNamespace(text="摘要内容"),
            ainvoke=None,
        )
        mw = LCAgentCompactionMiddleware(
            model=model,
            config=CompactionConfig(max_messages=5, keep_recent=2),
            on_compaction=bad_callback,
        )
        state = {"messages": self._build_messages(10), "summary": ""}
        result = mw.before_model(state, runtime=None)

        assert result is not None  # 压缩仍成功返回


# ════════════════════════════════════════════════════════════════════════
#  CLI metrics 命令测试
# ════════════════════════════════════════════════════════════════════════


@dataclass
class FakeAgentWithMetrics:
    """带 metrics 属性的 FakeAgent"""
    memory: Any = None
    llm: Any = None
    local_tools: list = field(default_factory=list)
    mcp_tools: list = field(default_factory=list)
    tools: list = field(default_factory=list)
    auto_match_skills: bool = True
    metrics: MetricsCollector = field(default_factory=MetricsCollector)

    def get_memory_summary(self) -> dict[str, Any]:
        return {"thread_id": "t-1"}

    def get_available_tools(self) -> list[str]:
        return []

    def compress_memory(self) -> dict[str, Any]:
        return {}
    def list_skills(self) -> list[dict[str, str]]:
        return []
    def cot(self, task: str) -> str:
        return ""


@dataclass
class FakeAgentNoMetrics:
    """不带 metrics 属性的 FakeAgent"""
    memory: Any = None
    llm: Any = None
    local_tools: list = field(default_factory=list)
    mcp_tools: list = field(default_factory=list)
    tools: list = field(default_factory=list)
    auto_match_skills: bool = True

    def get_memory_summary(self) -> dict[str, Any]:
        return {"thread_id": "t-1"}
    def get_available_tools(self) -> list[str]:
        return []
    def compress_memory(self) -> dict[str, Any]:
        return {}
    def list_skills(self) -> list[dict[str, str]]:
        return []
    def cot(self, task: str) -> str:
        return ""


@dataclass
class FakeSafetyBackend:
    config: dict[str, Any] = field(default_factory=lambda: {"mode": "blacklist"})
    def load_config(self) -> dict[str, Any]:
        return dict(self.config)
    def save_config(self, config: dict[str, Any]) -> bool:
        self.config = dict(config)
        return True


def _make_context(agent, printed: list[str]) -> Any:
    from cli.commands.types import CommandContext

    async def fake_run(agent, task: str) -> str:
        return ""

    return CommandContext(
        agent=agent,
        base_dir=".",
        config_file="config/llm_config.json",
        mcp_config_file="config/mcp_servers.json",
        print_fn=printed.append,
        input_fn=lambda prompt="": "y",
        select_menu=lambda *args, **kwargs: None,
        create_llm=lambda provider: None,
        list_providers=dict,
        run_structured_until_completion=fake_run,
        chat_until_completion=fake_run,
        safety_backend=FakeSafetyBackend(),
    )


class TestCLIMetricsCommand:
    """CLI metrics 命令测试"""

    def test_metrics_status_empty(self):
        """空指标时正常显示"""
        agent = FakeAgentWithMetrics()
        printed: list[str] = []
        ctx = _make_context(agent, printed)

        from cli.commands.metrics import show_metrics
        result = show_metrics(ctx)

        assert result.handled is True
        assert any("运行时指标" in line for line in printed)
        assert any("总调用次数" in line for line in printed)

    def test_metrics_status_with_data(self):
        """有数据时正确显示"""
        agent = FakeAgentWithMetrics()
        agent.metrics.record_llm_call(provider="zhipu", prompt_tokens=100, completion_tokens=50, duration_ms=300.0)
        agent.metrics.record_tool_call(name="search", duration_ms=200.0, success=True)
        agent.metrics.record_compaction(trigger="auto", messages_before=60, messages_after=22)
        agent.metrics.increment_turn()

        printed: list[str] = []
        ctx = _make_context(agent, printed)

        from cli.commands.metrics import show_metrics
        show_metrics(ctx)

        output = "\n".join(printed)
        assert "zhipu" in output
        assert "150" in output  # total_tokens
        assert "search" in output
        assert "压缩" in output or "压缩统计" in output

    def test_metrics_reset(self):
        """metrics:reset 清空指标"""
        agent = FakeAgentWithMetrics()
        agent.metrics.record_llm_call(provider="test", prompt_tokens=100, completion_tokens=50)
        agent.metrics.increment_turn()

        printed: list[str] = []
        ctx = _make_context(agent, printed)

        from cli.commands.metrics import metrics_command
        result = metrics_command(ctx, "metrics:reset")

        assert result.handled is True
        assert agent.metrics.llm_call_count == 0
        summary = agent.metrics.get_summary()
        assert summary["session"]["turn_count"] == 0
        assert any("重置" in line for line in printed)

    def test_metrics_no_metrics_attr(self):
        """Agent 不支持 metrics 时不崩溃"""
        agent = FakeAgentNoMetrics()
        printed: list[str] = []
        ctx = _make_context(agent, printed)

        from cli.commands.metrics import show_metrics
        result = show_metrics(ctx)

        assert result.handled is True
        assert any("不支持" in line for line in printed)

    def test_metrics_command_dispatch(self):
        """通过 dispatcher 路由 metrics 命令"""
        from cli.commands.dispatcher import dispatch_command

        agent = FakeAgentWithMetrics()
        agent.metrics.record_llm_call(provider="zhipu", prompt_tokens=100, completion_tokens=50)
        printed: list[str] = []
        ctx = _make_context(agent, printed)

        result = asyncio.run(dispatch_command(ctx, "metrics"))
        assert result.handled is True
        assert any("运行时指标" in line for line in printed)

    def test_metrics_status_dispatch(self):
        """通过 dispatcher 路由 metrics:status"""
        from cli.commands.dispatcher import dispatch_command

        agent = FakeAgentWithMetrics()
        printed: list[str] = []
        ctx = _make_context(agent, printed)

        result = asyncio.run(dispatch_command(ctx, "metrics:status"))
        assert result.handled is True

    def test_metrics_reset_dispatch(self):
        """通过 dispatcher 路由 metrics:reset"""
        from cli.commands.dispatcher import dispatch_command

        agent = FakeAgentWithMetrics()
        agent.metrics.record_llm_call(provider="test", prompt_tokens=10, completion_tokens=5)
        printed: list[str] = []
        ctx = _make_context(agent, printed)

        result = asyncio.run(dispatch_command(ctx, "metrics:reset"))
        assert result.handled is True
        assert agent.metrics.llm_call_count == 0
