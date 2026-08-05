"""结构化日志配置测试

验证:
1. TraceContext 正确设置/恢复 trace_id 和 thread_id
2. StructuredFormatter 输出包含 trace_id/thread_id 字段
3. setup_logging 幂等配置（不重复添加 handler）
4. 异步上下文中 trace_id 正确传递
5. contextvars 隔离（不同任务不串 trace_id）
"""
from __future__ import annotations

import asyncio
import io
import logging

import pytest

from agent.logging_config import (
    StructuredFormatter,
    TraceContext,
    generate_trace_id,
    get_thread_id,
    get_trace_id,
    set_trace_context,
    setup_logging,
)


# ════════════════════════════════════════════════════════════════
#  TraceContext 测试
# ════════════════════════════════════════════════════════════════


class TestTraceContext:
    """TraceContext 上下文管理器测试"""

    def test_set_trace_id_within_context(self):
        assert get_trace_id() == "-"

        with TraceContext(trace_id="abc123"):
            assert get_trace_id() == "abc123"

        # 退出后恢复
        assert get_trace_id() == "-"

    def test_set_thread_id_within_context(self):
        assert get_thread_id() == "-"

        with TraceContext(thread_id="thread-xyz"):
            assert get_thread_id() == "thread-xyz"

        assert get_thread_id() == "-"

    def test_set_both(self):
        with TraceContext(trace_id="t1", thread_id="th1"):
            assert get_trace_id() == "t1"
            assert get_thread_id() == "th1"

        assert get_trace_id() == "-"
        assert get_thread_id() == "-"

    def test_nested_contexts(self):
        with TraceContext(trace_id="outer"):
            assert get_trace_id() == "outer"
            with TraceContext(trace_id="inner"):
                assert get_trace_id() == "inner"
            # 退出 inner 后恢复到 outer
            assert get_trace_id() == "outer"

        assert get_trace_id() == "-"

    def test_auto_generate_trace_id(self):
        with TraceContext(auto_generate_trace=True) as ctx:
            tid = get_trace_id()
            assert tid != "-"
            assert len(tid) == 8
            assert ctx.trace_id == tid

    def test_no_args_does_nothing(self):
        """不传任何参数时，上下文不做任何修改"""
        with TraceContext():
            assert get_trace_id() == "-"
            assert get_thread_id() == "-"

    def test_set_trace_context_function(self):
        """set_trace_context 直接设置（不自动恢复）"""
        set_trace_context(trace_id="direct", thread_id="th-direct")
        assert get_trace_id() == "direct"
        assert get_thread_id() == "th-direct"
        # 清理（不污染其他测试）
        set_trace_context(trace_id="-", thread_id="-")

    def test_generate_trace_id_format(self):
        tid = generate_trace_id()
        assert len(tid) == 8
        assert all(c in "0123456789abcdef" for c in tid)

    def test_generate_trace_id_unique(self):
        ids = {generate_trace_id() for _ in range(100)}
        # 极大概率全部不同
        assert len(ids) >= 95


# ════════════════════════════════════════════════════════════════
#  StructuredFormatter 测试
# ════════════════════════════════════════════════════════════════


class TestStructuredFormatter:
    """结构化日志格式器测试"""

    def test_format_includes_trace_id(self):
        formatter = StructuredFormatter(
            fmt="%(trace_id)s %(message)s",
        )
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="hello", args=(), exc_info=None,
        )

        with TraceContext(trace_id="abc123"):
            output = formatter.format(record)

        assert "abc123" in output
        assert "hello" in output

    def test_format_includes_thread_id(self):
        formatter = StructuredFormatter(
            fmt="%(thread_id)s %(message)s",
        )
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="world", args=(), exc_info=None,
        )

        with TraceContext(thread_id="th-1"):
            output = formatter.format(record)

        assert "th-1" in output

    def test_format_default_dash_when_no_context(self):
        formatter = StructuredFormatter(
            fmt="trace=%(trace_id)s thread=%(thread_id)s",
        )
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="", args=(), exc_info=None,
        )

        output = formatter.format(record)
        assert "trace=-" in output
        assert "thread=-" in output

    def test_full_format_with_context(self):
        """完整格式包含时间戳、级别、logger 名、trace_id、thread_id、消息"""
        formatter = StructuredFormatter(
            fmt="%(asctime)s [%(levelname)-5s] [%(name)s] "
                "[trace:%(trace_id)s] [thread:%(thread_id)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        record = logging.LogRecord(
            name="agent.agent_core", level=logging.WARNING, pathname="", lineno=0,
            msg="MCP 连接失败", args=(), exc_info=None,
        )

        with TraceContext(trace_id="req-abc", thread_id="thread-1"):
            output = formatter.format(record)

        assert "[WARN" in output
        assert "[agent.agent_core]" in output
        assert "[trace:req-abc]" in output
        assert "[thread:thread-1]" in output
        assert "MCP 连接失败" in output


# ════════════════════════════════════════════════════════════════
#  setup_logging 测试
# ════════════════════════════════════════════════════════════════


class TestSetupLogging:
    """日志初始化配置测试"""

    def test_setup_adds_handler(self):
        root = logging.getLogger()
        original_handlers = root.handlers[:]

        setup_logging(level=logging.DEBUG)

        # 应该恰好有一个 handler（控制台）
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0], logging.StreamHandler)

        # 恢复
        root.handlers = original_handlers

    def test_setup_is_idempotent(self):
        """重复调用不会累积 handler"""
        root = logging.getLogger()

        setup_logging()
        setup_logging()
        setup_logging()

        assert len(root.handlers) == 1

    def test_setup_sets_level(self):
        root = logging.getLogger()
        original_level = root.level

        setup_logging(level=logging.DEBUG)
        assert root.level == logging.DEBUG

        setup_logging(level=logging.WARNING)
        assert root.level == logging.WARNING

        # 恢复
        root.setLevel(original_level)

    def test_log_message_captured_with_trace_context(self):
        """日志消息中包含 trace_id"""
        setup_logging(level=logging.INFO)

        # 用 StringIO 捕获日志输出
        root = logging.getLogger()
        stream = io.StringIO()
        root.handlers[0].stream = stream

        logger = logging.getLogger("test.module")

        with TraceContext(trace_id="test-trace-1", thread_id="test-thread-1"):
            logger.info("测试消息")

        output = stream.getvalue()
        assert "test-trace-1" in output
        assert "test-thread-1" in output
        assert "测试消息" in output


# ════════════════════════════════════════════════════════════════
#  异步上下文传递测试
# ════════════════════════════════════════════════════════════════


class TestAsyncTraceContext:
    """异步上下文中 trace_id 传递测试"""

    def test_trace_id_propagates_in_async(self):
        async def run():
            with TraceContext(trace_id="async-1"):
                await asyncio.sleep(0.001)
                return get_trace_id()

        result = asyncio.run(run())
        assert result == "async-1"

    def test_trace_id_isolated_between_concurrent_tasks(self):
        """并发的 asyncio 任务各自有独立的 trace_id"""
        async def task(trace_id: str) -> str:
            with TraceContext(trace_id=trace_id):
                await asyncio.sleep(0.01)
                return get_trace_id()

        async def main():
            results = await asyncio.gather(
                task("task-a"),
                task("task-b"),
                task("task-c"),
            )
            return results

        results = asyncio.run(main())
        assert set(results) == {"task-a", "task-b", "task-c"}

    def test_trace_id_reset_after_async_context(self):
        async def run():
            with TraceContext(trace_id="temp"):
                await asyncio.sleep(0)
            return get_trace_id()

        result = asyncio.run(run())
        assert result == "-"

    def test_trace_context_with_async_with(self):
        async def run():
            async with TraceContext(trace_id="aio-ctx", thread_id="aio-th"):
                return get_trace_id(), get_thread_id()

        tid, thid = asyncio.run(run())
        assert tid == "aio-ctx"
        assert thid == "aio-th"
