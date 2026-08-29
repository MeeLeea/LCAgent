"""mcp_loader 按工具名筛选加载的单元测试。

不依赖真实 MCP server:通过 patch load_mcp_tools 返回伪造工具列表,
验证 aload_mcp_tools_by_name / load_mcp_tools_by_name_sync 的筛选逻辑、
空输入短路、异常回退等契约。

真实 MCP 集成测试见 tests/tools/test_mcp_filesystem.py(标记 slow)。
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, patch

import pytest


@dataclass
class FakeTool:
    """最小工具桩:只携带 name 属性,供按名筛选测试使用。

    不继承 StructuredTool/BaseTool,避免 pydantic args_schema 校验开销;
    aload_mcp_tools_by_name / load_mcp_tools_by_name_sync 只读 .name 字段,
    不调 .invoke(),FakeTool 完全满足测试契约。
    """

    name: str


def _make_fake_tool(name: str) -> FakeTool:
    """构造一个最小工具桩用于测试筛选。"""
    return FakeTool(name=name)


@pytest.fixture
def fake_tools():
    """三个伪造工具:write_file / read_file / list_directory。"""
    return [
        _make_fake_tool("write_file"),
        _make_fake_tool("read_file"),
        _make_fake_tool("list_directory"),
    ]


class TestAloadMcpToolsByName:
    """aload_mcp_tools_by_name 异步筛选逻辑。"""

    def test_filter_returns_only_requested_names(self, fake_tools):
        """命中的工具名被筛选返回,未声明的不返回。"""
        async def _run():
            with patch(
                "tools.mcp_loader.load_mcp_tools",
                new_callable=AsyncMock,
                return_value=fake_tools,
            ):
                from tools.mcp_loader import aload_mcp_tools_by_name

                result = await aload_mcp_tools_by_name(["write_file"])
                return result

        result = asyncio.run(_run())
        assert len(result) == 1
        assert result[0].name == "write_file"

    def test_filter_multiple_names(self, fake_tools):
        """同时筛选多个工具名,全部命中时按全量列表顺序返回。"""
        async def _run():
            with patch(
                "tools.mcp_loader.load_mcp_tools",
                new_callable=AsyncMock,
                return_value=fake_tools,
            ):
                from tools.mcp_loader import aload_mcp_tools_by_name

                return await aload_mcp_tools_by_name(
                    ["write_file", "read_file"]
                )

        result = asyncio.run(_run())
        names = [t.name for t in result]
        assert set(names) == {"write_file", "read_file"}

    def test_filter_no_match_returns_empty(self, fake_tools):
        """声明的工具名在全量加载结果中不存在时返回空列表,不抛异常。"""
        async def _run():
            with patch(
                "tools.mcp_loader.load_mcp_tools",
                new_callable=AsyncMock,
                return_value=fake_tools,
            ):
                from tools.mcp_loader import aload_mcp_tools_by_name

                return await aload_mcp_tools_by_name(["not_exist_tool"])

        result = asyncio.run(_run())
        assert result == []

    def test_empty_names_short_circuits(self):
        """传入空列表直接返回空列表,不触发 load_mcp_tools 调用。"""
        async def _run():
            with patch(
                "tools.mcp_loader.load_mcp_tools",
                new_callable=AsyncMock,
            ) as mock_load:
                from tools.mcp_loader import aload_mcp_tools_by_name

                result = await aload_mcp_tools_by_name([])
                assert not mock_load.called
                return result

        result = asyncio.run(_run())
        assert result == []

    def test_load_mcp_tools_failure_returns_empty(self):
        """load_mcp_tools 抛异常时,aload_mcp_tools_by_name 不吞异常向上传播。

        异常回退契约由同步包装层 load_mcp_tools_by_name_sync 负责;
        异步层应忠实传播错误,让调用方决定降级策略。
        """
        async def _run():
            with patch(
                "tools.mcp_loader.load_mcp_tools",
                new_callable=AsyncMock,
                side_effect=RuntimeError("mcp server down"),
            ):
                from tools.mcp_loader import aload_mcp_tools_by_name

                with pytest.raises(RuntimeError, match="mcp server down"):
                    await aload_mcp_tools_by_name(["write_file"])

        asyncio.run(_run())


class TestLoadMcpToolsByNameSync:
    """load_mcp_tools_by_name_sync 同步包装逻辑。"""

    def test_sync_returns_filtered_tools(self, fake_tools):
        """同步调用能正确从异步加载结果中按名筛选。"""
        async def _fake_async(names, config_file=None):
            return [t for t in fake_tools if t.name in set(names)]

        with patch(
            "tools.mcp_loader.aload_mcp_tools_by_name",
            new=_fake_async,
        ):
            from tools.mcp_loader import load_mcp_tools_by_name_sync

            result = load_mcp_tools_by_name_sync(["write_file"])

        assert len(result) == 1
        assert result[0].name == "write_file"

    def test_sync_empty_names_short_circuits(self):
        """空列表直接返回空,不触发 _run_async。"""
        from tools.mcp_loader import load_mcp_tools_by_name_sync

        result = load_mcp_tools_by_name_sync([])
        assert result == []

    def test_sync_exception_returns_empty_with_warning(self):
        """aload_mcp_tools_by_name 抛异常时,同步包装层捕获并返回空列表。"""
        async def _boom(names, config_file=None):
            raise RuntimeError("mcp server down")

        with patch(
            "tools.mcp_loader.aload_mcp_tools_by_name",
            new=_boom,
        ):
            from tools.mcp_loader import load_mcp_tools_by_name_sync

            result = load_mcp_tools_by_name_sync(["write_file"])

        assert result == []

    def test_sync_returns_empty_when_no_match(self, fake_tools):
        """异步层返回空(无命中)时,同步包装层原样透传空列表。"""
        async def _fake_async(names, config_file=None):
            return []  # 无命中

        with patch(
            "tools.mcp_loader.aload_mcp_tools_by_name",
            new=_fake_async,
        ):
            from tools.mcp_loader import load_mcp_tools_by_name_sync

            result = load_mcp_tools_by_name_sync(["not_exist"])
            assert result == []
