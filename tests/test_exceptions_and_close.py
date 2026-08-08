"""异常层次结构和 AgentCore.close() 生命周期测试

验证:
1. 异常继承关系正确，可被 LCAgentError 统一 catch
2. AgentStateError 携带正确的 detail 属性
3. AgentCore.aclose() 释放资源、幂等、关闭后抛 AgentStateError
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.exceptions import (
    AgentStateError,
    LCAgentError,
)

# ════════════════════════════════════════════════════════════════════════
#  异常层次结构测试
# ════════════════════════════════════════════════════════════════════════


class TestExceptionHierarchy:
    """异常继承关系测试"""

    def test_all_inherit_from_lcagent_error(self):
        for exc_class in [
            AgentStateError,
        ]:
            assert issubclass(exc_class, LCAgentError)

    def test_all_inherit_from_exception(self):
        for exc_class in [
            LCAgentError,
            AgentStateError,
        ]:
            assert issubclass(exc_class, Exception)

    def test_catch_all_with_lcagent_error(self):
        # Given: 各种子异常
        errors = [
            AgentStateError("closed"),
        ]

        # When/Then: 全部可被 LCAgentError catch
        for err in errors:
            try:
                raise err
            except LCAgentError as caught:
                assert caught is err

    def test_agent_state_error_detail(self):
        err = AgentStateError("already closed", detail="called from arun")
        assert err.detail == "called from arun"


# ════════════════════════════════════════════════════════════════════════
#  AgentCore.aclose() 生命周期测试
# ════════════════════════════════════════════════════════════════════════


def _make_minimal_core():
    """创建一个最小化的 AgentCore，不连接真实 MCP/LLM

    通过 object.__new__ 跳过 __init__，手动设置必要属性。
    execution_history / pending_interrupts 已迁移至 SessionStore（惰性创建）。
    """
    from agent.agent_core import AgentCore

    core = object.__new__(AgentCore)
    core.verbose = False
    core._closed = False
    core._state_lock = asyncio.Lock()

    # Mock MCP pool
    core._mcp_pool = MagicMock()
    core._mcp_pool.close = AsyncMock()

    # memory 已剥离至 MemoryContext / SessionManager，AgentCore.aclose 不再关闭 memory

    return core


class TestAgentCoreClose:
    """aclose() 生命周期测试"""

    def test_close_releases_mcp_pool(self):
        # Given: 一个未关闭的 core
        core = _make_minimal_core()

        # When: 关闭
        asyncio.run(core.aclose())

        # Then: MCP 连接池被关闭
        core._mcp_pool.close.assert_awaited_once()
        assert core._closed is True

    def test_close_releases_memory(self):
        core = _make_minimal_core()

        asyncio.run(core.aclose())

        # memory 关闭已迁移至 MemoryContext/SessionManager，AgentCore.aclose 只关 MCP pool
        core._mcp_pool.close.assert_awaited_once()

    def test_close_does_not_crash_without_instance_state(self):
        """aclose() 不再清理实例级 execution_history / pending_interrupts
        （已迁移至 SessionStore，随 Store GC 自动回收）。
        此测试验证 aclose 在无实例级状态时不崩溃。
        """
        core = _make_minimal_core()

        asyncio.run(core.aclose())

        assert core._closed is True

    def test_close_is_idempotent(self):
        # Given: 已关闭的 core
        core = _make_minimal_core()
        asyncio.run(core.aclose())
        assert core._closed is True

        # When: 再次关闭
        asyncio.run(core.aclose())

        # Then: 不重复调用 close（幂等）
        core._mcp_pool.close.assert_awaited_once()

    def test_arun_structured_raises_after_close(self):
        core = _make_minimal_core()
        asyncio.run(core.aclose())

        with pytest.raises(AgentStateError, match="已关闭"):
            asyncio.run(core.arun_structured("test"))

    def test_achat_structured_raises_after_close(self):
        core = _make_minimal_core()
        asyncio.run(core.aclose())

        with pytest.raises(AgentStateError, match="已关闭"):
            asyncio.run(core.achat_structured("test"))

    def test_areload_mcp_tools_raises_after_close(self):
        core = _make_minimal_core()
        asyncio.run(core.aclose())

        with pytest.raises(AgentStateError, match="已关闭"):
            asyncio.run(core.areload_mcp_tools())

    def test_areload_mcp_server_raises_after_close(self):
        core = _make_minimal_core()
        asyncio.run(core.aclose())

        with pytest.raises(AgentStateError, match="已关闭"):
            asyncio.run(core.areload_mcp_server("test"))

    def test_async_context_manager(self):
        # Given: 通过 async with 使用 core
        core = _make_minimal_core()

        async def use_and_exit():
            async with core:
                assert core._closed is False
            # 退出后自动关闭
            assert core._closed is True

        asyncio.run(use_and_exit())

        # MCP pool 被关闭（memory 由 MemoryContext/SessionManager 管理）
        core._mcp_pool.close.assert_awaited_once()

    def test_close_handles_mcp_error_gracefully(self):
        # Given: MCP close 抛异常
        core = _make_minimal_core()
        core._mcp_pool.close = AsyncMock(side_effect=RuntimeError("MCP boom"))

        # When: 关闭（不抛异常）
        asyncio.run(core.aclose())

        # Then: 仍然继续关闭（memory 由 MemoryContext/SessionManager 管理）
        core._mcp_pool.close.assert_awaited_once()
        assert core._closed is True

    def test_close_handles_memory_error_gracefully(self):
        # memory 关闭已迁移至 MemoryContext/SessionManager，AgentCore.aclose 只关 MCP pool。
        # 此测试验证 aclose 在无 memory 依赖时仍正常工作。
        core = _make_minimal_core()

        # When: 关闭
        asyncio.run(core.aclose())

        # Then: 仍然标记为已关闭
        assert core._closed is True
        core._mcp_pool.close.assert_awaited_once()
