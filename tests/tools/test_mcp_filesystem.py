"""MCP-Filesystem 工作空间隔离集成测试。

验证 MCP-Filesystem server 加载的文件工具经 WorkspaceSecurityMiddleware
拦截后，实现 per-session workspace 隔离：
- 两会话绑定不同 workspace，文件操作互不可见
- 路径逃逸被中间件拦截
- cwd 对齐到 workspace

前置条件：config/mcp_servers.json 中 filesystem server 已配置 allowed-paths。
本测试启动真实 MCP-Filesystem 子进程，需要 npx 可用。

运行：
  pytest tests/tools/test_mcp_filesystem.py -v
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any

import pytest

# 标记为需要真实 MCP 进程的慢测试
pytestmark = pytest.mark.slow


@dataclass
class FakeRuntime:
    """模拟 ToolRuntime，只保留中间件用到的 config 字段。"""

    config: dict[str, Any]


@dataclass
class FakeToolCallRequest:
    """模拟 ToolCallRequest，使 override() 返回含真实 args 的对象。

    绕开 MagicMock 的链式属性返回 MagicMock 问题：
    override() 返回新的 FakeToolCallRequest，其 tool_call 指向真实 dict。
    结构对齐 langchain 1.3 ToolCallRequest：config 在 runtime.config 而非顶层。
    """

    tool_call: dict[str, Any]
    runtime: FakeRuntime

    def override(self, **kwargs: Any) -> FakeToolCallRequest:
        """返回新的 request，tool_call 用 kwargs 覆盖。"""
        new_tc = dict(self.tool_call)
        if "tool_call" in kwargs:
            new_tc = dict(kwargs["tool_call"])
        return FakeToolCallRequest(tool_call=new_tc, runtime=self.runtime)


def _make_request(
    tool_name: str, args: dict[str, Any], config: dict[str, Any], call_id: str = "call-1"
) -> FakeToolCallRequest:
    """构造 FakeToolCallRequest，config 挂在 runtime.config。"""
    return FakeToolCallRequest(
        tool_call={"name": tool_name, "args": args, "id": call_id},
        runtime=FakeRuntime(config=config),
    )


def _has_npx() -> bool:
    """检查 npx 是否可用。"""
    import shutil
    return shutil.which("npx") is not None


skip_no_npx = pytest.mark.skipif(
    not _has_npx(),
    reason="npx 不可用，跳过 MCP-Filesystem 集成测试",
)


@skip_no_npx
class TestMCPFilesystemWorkspaceIsolation:
    """MCP-Filesystem + 中间件的 per-session 隔离。"""

    @pytest.fixture
    def workspace_a(self, tmp_path):
        ws = tmp_path / "workspace_a"
        ws.mkdir()
        (ws / "secret_a.txt").write_text("secret_a_content", encoding="utf-8")
        return ws

    @pytest.fixture
    def workspace_b(self, tmp_path):
        ws = tmp_path / "workspace_b"
        ws.mkdir()
        (ws / "secret_b.txt").write_text("secret_b_content", encoding="utf-8")
        return ws

    @pytest.fixture
    def mcp_pool(self, tmp_path):
        """启动真实 MCP-Filesystem server（allowed-paths 设为 tmp_path 根）。"""
        import json

        from tools.mcp_pool import MCPPool

        config_file = tmp_path / "mcp_test.json"
        config = {
            "servers": {
                "filesystem": {
                    "command": "npx",
                    "args": [
                        "-y",
                        "@modelcontextprotocol/server-filesystem",
                        str(tmp_path),
                    ],
                    "enabled": True,
                }
            }
        }
        config_file.write_text(
            json.dumps(config, ensure_ascii=False), encoding="utf-8"
        )

        pool = MCPPool(str(config_file))

        async def setup():
            await pool.initialize()
            return pool

        pool = asyncio.run(setup())
        yield pool

        asyncio.run(pool.close())

    def test_mcp_filesystem_tools_loaded(self, mcp_pool):
        """MCP-Filesystem 加载后应有 read_file 等工具。"""
        tools = mcp_pool.get_all_tools()
        tool_names = {t.name for t in tools}
        assert "read_file" in tool_names
        assert "write_file" in tool_names
        assert "list_directory" in tool_names

    def test_middleware_resolves_relative_path_for_mcp(
        self, mcp_pool, workspace_a
    ):
        """中间件把相对路径解析为 workspace 内绝对路径后传给 MCP 工具。"""
        from agent.workspace_middleware import WorkspaceSecurityMiddleware

        mw = WorkspaceSecurityMiddleware()
        read_file_tool = next(
            t for t in mcp_pool.get_all_tools() if t.name == "read_file"
        )

        config = {
            "configurable": {
                "workspace_path": os.path.realpath(str(workspace_a)),
            }
        }

        request = _make_request(
            "read_file", {"path": "secret_a.txt"}, config, "call-1"
        )

        # 中间件拦截后调用真实 MCP 工具
        def handler(req):
            return read_file_tool.invoke(req.tool_call["args"], config)

        result = mw.wrap_tool_call(request, handler)

        # 成功路径：MCP 工具 invoke 返回 (content, artifact) 元组，
        # content 为字符串或 content block 列表，统一转字符串断言
        content = result[0] if isinstance(result, tuple) else result
        assert "secret_a_content" in str(content)

    def test_cross_session_isolation(
        self, mcp_pool, workspace_a, workspace_b
    ):
        """两会话绑定不同 workspace，A 无法读 B 的文件。"""
        from agent.workspace_middleware import WorkspaceSecurityMiddleware

        mw = WorkspaceSecurityMiddleware()
        read_file_tool = next(
            t for t in mcp_pool.get_all_tools() if t.name == "read_file"
        )

        config_a = {
            "configurable": {
                "workspace_path": os.path.realpath(str(workspace_a)),
            }
        }
        config_b = {
            "configurable": {
                "workspace_path": os.path.realpath(str(workspace_b)),
            }
        }

        # 会话 A 尝试读 B 的文件（相对路径逃逸）
        request_a = _make_request(
            "read_file",
            {"path": "../workspace_b/secret_b.txt"},
            config_a,
            "call-a",
        )

        def handler_a(req):
            return read_file_tool.invoke(req.tool_call["args"], config_a)

        result = mw.wrap_tool_call(request_a, handler_a)
        # 逃逸被中间件拦截，返回 ToolMessage（未调用 handler / MCP 工具）
        assert "逃逸" in result.content

        # 会话 B 正常读自己的文件
        request_b = _make_request(
            "read_file", {"path": "secret_b.txt"}, config_b, "call-b"
        )

        def handler_b(req):
            return read_file_tool.invoke(req.tool_call["args"], config_b)

        result_b = mw.wrap_tool_call(request_b, handler_b)
        # 成功路径：MCP 工具 invoke 返回 (content, artifact) 元组
        content_b = result_b[0] if isinstance(result_b, tuple) else result_b
        assert "secret_b_content" in str(content_b)
