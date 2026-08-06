"""工具超时包装器测试

验证 wrap_tool_with_timeout 的核心功能：
- 正常执行的工具不受影响
- 超时的工具返回 JSON 错误而非抛异常
- 按工具名覆盖超时配置
- 排除列表中的工具不加超时
- 重复包装不会叠加
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import pytest

from tools.tool_wrapper import (
    DEFAULT_TIMEOUT,
    NO_TIMEOUT_TOOLS,
    TOOL_TIMEOUTS,
    wrap_tool_with_timeout,
    wrap_tools_with_timeout,
)


@dataclass
class FakeSyncTool:
    """模拟只有同步 _run 的工具（无原生 _arun）"""
    name: str
    description: str = "fake"
    _run_result: Any = "ok"
    _run_delay: float = 0.0
    _timeout_wrapped: bool = False

    def _run(self, *args, **kwargs):
        if self._run_delay > 0:
            import time
            time.sleep(self._run_delay)
        return self._run_result


@dataclass
class FakeAsyncTool(FakeSyncTool):
    """模拟有原生 _arun 的工具"""
    _arun_result: Any = "async_ok"
    _arun_delay: float = 0.0

    async def _arun(self, *args, **kwargs):
        if self._arun_delay > 0:
            await asyncio.sleep(self._arun_delay)
        return self._arun_result


class TestWrapToolWithTimeout:
    """单工具包装测试"""

    def test_normal_sync_tool_returns_result(self):
        # Given: 一个正常执行的同步工具（无原生 _arun）
        tool = FakeSyncTool(name="calc", _run_result=42)
        wrap_tool_with_timeout(tool, timeout=5.0)

        # When: 异步调用（包装器会通过 to_thread 调用 _run）
        result = asyncio.run(tool._arun("input"))

        # Then: 返回原始结果
        assert result == 42

    def test_normal_async_tool_returns_result(self):
        # Given: 一个有原生 _arun 的工具
        tool = FakeAsyncTool(name="async_tool", _arun_result="async_ok")
        wrap_tool_with_timeout(tool, timeout=5.0)

        result = asyncio.run(tool._arun("input"))

        assert result == "async_ok"

    def test_sync_tool_timeout_returns_error_json(self):
        # Given: 一个会卡住的同步工具（sleep 2秒，超时 0.1秒）
        tool = FakeSyncTool(name="slow_tool", _run_delay=2.0)
        wrap_tool_with_timeout(tool, timeout=0.1)

        # When: 异步调用
        result = asyncio.run(tool._arun("input"))

        # Then: 返回 JSON 错误，而非抛异常
        parsed = json.loads(result)
        assert parsed["error"] == "tool_timeout"
        assert parsed["tool"] == "slow_tool"
        assert parsed["timeout"] == 0.1

    def test_async_tool_timeout_returns_error_json(self):
        # Given: 一个会卡住的异步工具
        tool = FakeAsyncTool(name="slow_async", _arun_result="ok", _arun_delay=2.0)
        wrap_tool_with_timeout(tool, timeout=0.1)

        result = asyncio.run(tool._arun("input"))

        parsed = json.loads(result)
        assert parsed["error"] == "tool_timeout"
        assert parsed["tool"] == "slow_async"

    def test_tool_with_none_timeout_not_wrapped(self):
        # Given: 一个在 NO_TIMEOUT_TOOLS 排除列表中的工具
        tool = FakeSyncTool(name="excluded_tool", _run_result="ok")
        # 临时加入排除列表
        NO_TIMEOUT_TOOLS.add("excluded_tool")
        try:
            wrap_tool_with_timeout(tool)
        finally:
            NO_TIMEOUT_TOOLS.discard("excluded_tool")

        # Then: 工具未被包装（_timeout_wrapped 保持 False）
        assert tool._timeout_wrapped is False

    def test_double_wrap_is_idempotent(self):
        # Given: 已包装的工具
        tool = FakeSyncTool(name="calc", _run_result=42)
        wrap_tool_with_timeout(tool, timeout=5.0)
        assert tool._timeout_wrapped is True

        # When: 再次包装
        wrap_tool_with_timeout(tool, timeout=10.0)

        # Then: 不重复包装，仍能正常执行
        result = asyncio.run(tool._arun("input"))
        assert result == 42

    def test_non_timeout_exception_propagates(self):
        # Given: 一个会抛 ValueError 的工具
        class ErrorTool(FakeSyncTool):
            def _run(self, *args, **kwargs):
                raise ValueError("bad input")

        tool = ErrorTool(name="error_tool")
        wrap_tool_with_timeout(tool, timeout=5.0)

        # When/Then: 非 timeout 异常照常抛出
        with pytest.raises(ValueError, match="bad input"):
            asyncio.run(tool._arun("input"))


class TestToolTimeoutConfig:
    """超时配置测试"""

    def test_tool_timeouts_dict_overrides_default(self):
        assert "ask_human" in TOOL_TIMEOUTS
        assert TOOL_TIMEOUTS["ask_human"] == 600.0

    def test_no_timeout_tools_set_exists(self):
        assert isinstance(NO_TIMEOUT_TOOLS, set)

    def test_default_timeout_is_60(self):
        assert DEFAULT_TIMEOUT == 60.0


class TestWrapToolsWithTimeout:
    """批量包装测试"""

    def test_wrap_multiple_tools(self):
        # Given: 多个工具
        tools = [
            FakeSyncTool(name="tool_a", _run_result="a"),
            FakeSyncTool(name="tool_b", _run_result="b"),
        ]

        # When: 批量包装
        wrapped = wrap_tools_with_timeout(tools, default_timeout=5.0)

        # Then: 每个工具都被包装了
        assert len(wrapped) == 2
        assert all(t._timeout_wrapped for t in wrapped)

        # 执行正常
        result_a = asyncio.run(wrapped[0]._arun("x"))
        result_b = asyncio.run(wrapped[1]._arun("x"))
        assert result_a == "a"
        assert result_b == "b"

    def test_wrap_empty_list(self):
        wrapped = wrap_tools_with_timeout([], default_timeout=5.0)
        assert wrapped == []
