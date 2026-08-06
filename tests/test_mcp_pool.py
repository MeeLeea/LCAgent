"""MCPPool 单元测试

测试 MCPServerConnection 和 MCPPool 的连接管理、健康检查、
per-server 隔离、工具聚合等核心功能。

所有测试通过注入假的 MultiServerMCPClient 模块，避免真实网络/进程依赖。
异步操作用 asyncio.run() 包裹（与项目现有测试模式一致）。
"""
from __future__ import annotations

import asyncio
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import pytest

from tools.mcp_pool import (
    MCPPool,
    MCPServerConnection,
    ServerStatus,
)

# ── Fake tool: 避免 BaseTool 的 pydantic 校验开销 ─────────────────────


@dataclass
class FakeToolObj:
    """轻量假工具，模拟 BaseTool 接口

    make_sync_compatible 会检查 getattr(tool, "coroutine", None)，
    FakeToolObj 没有该属性，getattr 返回 None，工具原样返回。
    """

    name: str
    description: str = "fake tool"


# ── Fake MultiServerMCPClient ──────────────────────────────────────────


class FakeMCPClient:
    """模拟 langchain_mcp_adapters.client.MultiServerMCPClient"""

    # 类级配置：控制每个 server 返回的工具列表和是否抛异常
    _tools_map: ClassVar[dict[str, list[Any]]] = {}
    _should_fail: ClassVar[set[str]] = set()

    def __init__(self, config: dict[str, Any]):
        self._config = config
        self._server_name = next(iter(config.keys())) if config else ""

    async def get_tools(self) -> list[Any]:
        if self._server_name in FakeMCPClient._should_fail:
            raise ConnectionError(f"fake error for {self._server_name}")
        return list(FakeMCPClient._tools_map.get(self._server_name, []))


@pytest.fixture
def fake_mcp(monkeypatch):
    """注入假的 langchain_mcp_adapters.client 模块

    在测试前后重置 FakeMCPClient 的类级状态。
    """
    FakeMCPClient._tools_map = {}
    FakeMCPClient._should_fail = set()

    fake_module = types.ModuleType("langchain_mcp_adapters.client")
    fake_module.MultiServerMCPClient = FakeMCPClient
    fake_parent = types.ModuleType("langchain_mcp_adapters")
    monkeypatch.setitem(sys.modules, "langchain_mcp_adapters", fake_parent)
    monkeypatch.setitem(sys.modules, "langchain_mcp_adapters.client", fake_module)
    return FakeMCPClient


def _write_config(path: Path, servers: dict[str, dict]) -> None:
    """写入 mcp_servers.json 配置文件"""
    path.write_text(json.dumps({"servers": servers}, ensure_ascii=False), encoding="utf-8")


# ════════════════════════════════════════════════════════════════════════
#  MCPServerConnection 测试
# ════════════════════════════════════════════════════════════════════════


class TestMCPServerConnection:
    """单连接封装测试"""

    def test_connect_success(self, fake_mcp):
        # Given: 一个配置好的 server 连接，假 client 返回 2 个工具
        fake_mcp._tools_map["test-server"] = [
            FakeToolObj(name="tool_a"),
            FakeToolObj(name="tool_b"),
        ]
        conn = MCPServerConnection("test-server", {
            "transport": "stdio",
            "command": "echo",
            "args": [],
            "enabled": True,
        })

        # When: 连接
        result = asyncio.run(conn.connect())

        # Then: 连接成功，状态和工具缓存正确
        assert result is True
        assert conn.status == ServerStatus.CONNECTED
        assert len(conn.tools) == 2
        assert conn.tool_names == ["tool_a", "tool_b"]

    def test_connect_failure_sets_error(self, fake_mcp):
        # Given: 假 client 对该 server 抛异常
        fake_mcp._should_fail.add("bad-server")
        conn = MCPServerConnection("bad-server", {
            "transport": "stdio",
            "command": "nonexistent",
            "args": [],
        })

        # When: 连接
        result = asyncio.run(conn.connect())

        # Then: 连接失败，状态为 ERROR，工具为空，错误信息记录
        assert result is False
        assert conn.status == ServerStatus.ERROR
        assert conn.tools == []
        assert "fake error" in conn.last_error

    def test_disconnect_clears_state(self, fake_mcp):
        # Given: 已连接的 server
        fake_mcp._tools_map["srv"] = [FakeToolObj(name="t1")]
        conn = MCPServerConnection("srv", {"transport": "stdio", "command": "echo"})

        async def setup_and_disconnect():
            await conn.connect()
            await conn.disconnect()

        asyncio.run(setup_and_disconnect())

        # Then: 状态变为 DISCONNECTED，工具清空
        assert conn.status == ServerStatus.DISCONNECTED
        assert conn.tools == []

    def test_reconnect_disconnects_then_connects(self, fake_mcp):
        # Given: 已连接的 server，reconnect 后工具列表变化
        fake_mcp._tools_map["srv"] = [FakeToolObj(name="old_tool")]
        conn = MCPServerConnection("srv", {"transport": "stdio", "command": "echo"})

        async def connect_then_reconnect():
            await conn.connect()
            assert conn.tool_names == ["old_tool"]
            # 更新假 client 返回的工具
            fake_mcp._tools_map["srv"] = [FakeToolObj(name="new_tool")]
            return await conn.reconnect()

        result = asyncio.run(connect_then_reconnect())

        # Then: 重连成功，工具列表已更新
        assert result is True
        assert conn.status == ServerStatus.CONNECTED
        assert conn.tool_names == ["new_tool"]

    def test_health_check_connected_with_tools(self, fake_mcp):
        # Given: 已连接且有工具的 server
        fake_mcp._tools_map["srv"] = [FakeToolObj(name="t1")]
        conn = MCPServerConnection("srv", {"transport": "stdio", "command": "echo"})

        async def setup_and_check():
            await conn.connect()
            return await conn.health_check()

        result = asyncio.run(setup_and_check())

        # Then: 返回 True
        assert result is True

    def test_health_check_disconnected_returns_false(self, fake_mcp):
        # Given: 未连接的 server
        conn = MCPServerConnection("srv", {"transport": "stdio", "command": "echo"})

        result = asyncio.run(conn.health_check())

        # Then: 返回 False
        assert result is False

    def test_health_check_connected_no_tools_returns_false(self, fake_mcp):
        # Given: 已连接但工具列表为空（异常状态）
        fake_mcp._tools_map["srv"] = []
        conn = MCPServerConnection("srv", {"transport": "stdio", "command": "echo"})

        async def setup_and_check():
            await conn.connect()
            return await conn.health_check()

        result = asyncio.run(setup_and_check())

        # Then: 返回 False（工具为空视为不健康）
        assert result is False

    def test_get_info_stdio(self):
        # Given: stdio 类型的 server
        conn = MCPServerConnection("srv", {
            "transport": "stdio",
            "command": "python",
            "args": ["-m", "server"],
            "enabled": True,
        })

        # When: 获取信息
        info = conn.get_info()

        # Then: 信息正确
        assert info.name == "srv"
        assert info.transport == "stdio"
        assert info.enabled is True
        assert "python" in info.detail
        assert "-m" in info.detail
        assert info.status == ServerStatus.DISCONNECTED

    def test_get_info_sse(self):
        # Given: sse 类型的 server
        conn = MCPServerConnection("srv", {
            "transport": "sse",
            "url": "http://localhost:8080/sse",
            "enabled": True,
        })

        info = conn.get_info()

        assert info.transport == "sse"
        assert info.detail == "http://localhost:8080/sse"

    def test_build_client_config_stdio(self):
        conn = MCPServerConnection("srv", {
            "transport": "stdio",
            "command": "node",
            "args": ["server.js"],
            "env": {"FOO": "bar"},
        })

        cfg = conn._build_client_config()

        assert cfg["transport"] == "stdio"
        assert cfg["command"] == "node"
        assert cfg["args"] == ["server.js"]
        assert cfg["env"] == {"FOO": "bar"}

    def test_build_client_config_replaces_python(self):
        conn = MCPServerConnection("srv", {
            "transport": "stdio",
            "command": "python",
            "args": ["-m", "mcp_server"],
        })

        cfg = conn._build_client_config()

        assert cfg["command"] == sys.executable

    def test_build_client_config_sse(self):
        conn = MCPServerConnection("srv", {
            "transport": "sse",
            "url": "http://localhost:9000",
        })

        cfg = conn._build_client_config()

        assert cfg["transport"] == "sse"
        assert cfg["url"] == "http://localhost:9000"


# ════════════════════════════════════════════════════════════════════════
#  MCPPool 测试
# ════════════════════════════════════════════════════════════════════════


class TestMCPPool:
    """连接池管理测试"""

    def test_initialize_empty_config(self, tmp_path: Path):
        # Given: 配置文件不存在
        pool = MCPPool(str(tmp_path / "nonexistent.json"))

        count = asyncio.run(pool.initialize())

        assert count == 0
        assert pool.get_all_tools() == []

    def test_initialize_connects_enabled_servers(self, fake_mcp, tmp_path: Path):
        # Given: 2 个启用的 server
        fake_mcp._tools_map["srv-a"] = [FakeToolObj(name="a1"), FakeToolObj(name="a2")]
        fake_mcp._tools_map["srv-b"] = [FakeToolObj(name="b1")]
        cfg_file = tmp_path / "mcp.json"
        _write_config(cfg_file, {
            "srv-a": {"transport": "stdio", "command": "echo", "enabled": True},
            "srv-b": {"transport": "stdio", "command": "echo", "enabled": True},
        })
        pool = MCPPool(str(cfg_file))

        count = asyncio.run(pool.initialize())

        assert count == 3
        tool_names = {t.name for t in pool.get_all_tools()}
        assert tool_names == {"a1", "a2", "b1"}

    def test_initialize_skips_disabled_servers(self, fake_mcp, tmp_path: Path):
        fake_mcp._tools_map["enabled-srv"] = [FakeToolObj(name="tool1")]
        fake_mcp._tools_map["disabled-srv"] = [FakeToolObj(name="tool2")]
        cfg_file = tmp_path / "mcp.json"
        _write_config(cfg_file, {
            "enabled-srv": {"transport": "stdio", "command": "echo", "enabled": True},
            "disabled-srv": {"transport": "stdio", "command": "echo", "enabled": False},
        })
        pool = MCPPool(str(cfg_file))

        count = asyncio.run(pool.initialize())

        assert count == 1
        tool_names = {t.name for t in pool.get_all_tools()}
        assert tool_names == {"tool1"}

    def test_initialize_single_failure_doesnt_block_others(self, fake_mcp, tmp_path: Path):
        # Given: srv-a 连接会失败，srv-b 正常
        fake_mcp._should_fail.add("srv-a")
        fake_mcp._tools_map["srv-b"] = [FakeToolObj(name="b1")]
        cfg_file = tmp_path / "mcp.json"
        _write_config(cfg_file, {
            "srv-a": {"transport": "stdio", "command": "bad", "enabled": True},
            "srv-b": {"transport": "stdio", "command": "echo", "enabled": True},
        })
        pool = MCPPool(str(cfg_file))

        count = asyncio.run(pool.initialize())

        # Then: srv-b 的工具正常加载，srv-a 失败但不影响 srv-b
        assert count == 1
        infos = {i.name: i for i in pool.get_server_infos()}
        assert infos["srv-a"].status == ServerStatus.ERROR
        assert infos["srv-b"].status == ServerStatus.CONNECTED

    def test_reload_server_success(self, fake_mcp, tmp_path: Path):
        # Given: 已初始化的 pool
        fake_mcp._tools_map["srv"] = [FakeToolObj(name="old")]
        cfg_file = tmp_path / "mcp.json"
        _write_config(cfg_file, {
            "srv": {"transport": "stdio", "command": "echo", "enabled": True},
        })
        pool = MCPPool(str(cfg_file))

        async def init_and_reload():
            await pool.initialize()
            assert pool.get_all_tools()[0].name == "old"
            # 更新假 client 返回的工具
            fake_mcp._tools_map["srv"] = [FakeToolObj(name="new")]
            return await pool.reload_server("srv")

        result = asyncio.run(init_and_reload())

        assert result is True
        assert pool.get_all_tools()[0].name == "new"

    def test_reload_server_removed_returns_false(self, fake_mcp, tmp_path: Path):
        fake_mcp._tools_map["srv"] = [FakeToolObj(name="t1")]
        cfg_file = tmp_path / "mcp.json"
        _write_config(cfg_file, {
            "srv": {"transport": "stdio", "command": "echo", "enabled": True},
        })
        pool = MCPPool(str(cfg_file))

        async def init_then_remove():
            await pool.initialize()
            assert len(pool.get_all_tools()) == 1
            # 从配置中删除该 server
            _write_config(cfg_file, {})
            return await pool.reload_server("srv")

        result = asyncio.run(init_then_remove())

        assert result is False
        assert pool.get_all_tools() == []

    def test_reload_server_disabled_returns_false(self, fake_mcp, tmp_path: Path):
        fake_mcp._tools_map["srv"] = [FakeToolObj(name="t1")]
        cfg_file = tmp_path / "mcp.json"
        _write_config(cfg_file, {
            "srv": {"transport": "stdio", "command": "echo", "enabled": True},
        })
        pool = MCPPool(str(cfg_file))

        async def init_then_disable():
            await pool.initialize()
            _write_config(cfg_file, {
                "srv": {"transport": "stdio", "command": "echo", "enabled": False},
            })
            return await pool.reload_server("srv")

        result = asyncio.run(init_then_disable())

        assert result is False

    def test_get_all_tools_only_from_connected(self, fake_mcp, tmp_path: Path):
        fake_mcp._should_fail.add("bad-srv")
        fake_mcp._tools_map["good-srv"] = [FakeToolObj(name="good_tool")]
        cfg_file = tmp_path / "mcp.json"
        _write_config(cfg_file, {
            "bad-srv": {"transport": "stdio", "command": "bad", "enabled": True},
            "good-srv": {"transport": "stdio", "command": "echo", "enabled": True},
        })
        pool = MCPPool(str(cfg_file))

        asyncio.run(pool.initialize())

        tools = pool.get_all_tools()
        assert len(tools) == 1
        assert tools[0].name == "good_tool"

    def test_get_server_infos(self, fake_mcp, tmp_path: Path):
        fake_mcp._should_fail.add("fail-srv")
        fake_mcp._tools_map["ok-srv"] = [FakeToolObj(name="t1")]
        cfg_file = tmp_path / "mcp.json"
        _write_config(cfg_file, {
            "fail-srv": {"transport": "stdio", "command": "bad", "enabled": True},
            "ok-srv": {"transport": "sse", "url": "http://x", "enabled": True},
        })
        pool = MCPPool(str(cfg_file))

        asyncio.run(pool.initialize())
        infos = {i.name: i for i in pool.get_server_infos()}

        assert infos["ok-srv"].status == ServerStatus.CONNECTED
        assert infos["ok-srv"].tool_count == 1
        assert infos["fail-srv"].status == ServerStatus.ERROR
        assert "fake error" in infos["fail-srv"].last_error

    def test_get_server_info_single(self, fake_mcp, tmp_path: Path):
        fake_mcp._tools_map["srv"] = [FakeToolObj(name="t1")]
        cfg_file = tmp_path / "mcp.json"
        _write_config(cfg_file, {
            "srv": {"transport": "stdio", "command": "echo", "enabled": True},
        })
        pool = MCPPool(str(cfg_file))

        asyncio.run(pool.initialize())

        info = pool.get_server_info("srv")
        missing = pool.get_server_info("nonexistent")

        assert info is not None
        assert info.name == "srv"
        assert missing is None

    def test_close_disconnects_all(self, fake_mcp, tmp_path: Path):
        fake_mcp._tools_map["srv-a"] = [FakeToolObj(name="a1")]
        fake_mcp._tools_map["srv-b"] = [FakeToolObj(name="b1")]
        cfg_file = tmp_path / "mcp.json"
        _write_config(cfg_file, {
            "srv-a": {"transport": "stdio", "command": "echo", "enabled": True},
            "srv-b": {"transport": "stdio", "command": "echo", "enabled": True},
        })
        pool = MCPPool(str(cfg_file))

        async def init_and_close():
            await pool.initialize()
            assert len(pool.get_all_tools()) == 2
            await pool.close()

        asyncio.run(init_and_close())

        assert pool.get_all_tools() == []
        assert pool.connections == {}

    def test_initialize_removes_stale_connections(self, fake_mcp, tmp_path: Path):
        fake_mcp._tools_map["srv-a"] = [FakeToolObj(name="a1")]
        fake_mcp._tools_map["srv-b"] = [FakeToolObj(name="b1")]
        cfg_file = tmp_path / "mcp.json"
        _write_config(cfg_file, {
            "srv-a": {"transport": "stdio", "command": "echo", "enabled": True},
            "srv-b": {"transport": "stdio", "command": "echo", "enabled": True},
        })
        pool = MCPPool(str(cfg_file))

        async def init_twice():
            await pool.initialize()
            assert len(pool.connections) == 2
            # 删除 srv-b
            _write_config(cfg_file, {
                "srv-a": {"transport": "stdio", "command": "echo", "enabled": True},
            })
            await pool.initialize()

        asyncio.run(init_twice())

        assert "srv-a" in pool.connections
        assert "srv-b" not in pool.connections

    def test_reload_all_reconnects_everything(self, fake_mcp, tmp_path: Path):
        fake_mcp._tools_map["srv"] = [FakeToolObj(name="old")]
        cfg_file = tmp_path / "mcp.json"
        _write_config(cfg_file, {
            "srv": {"transport": "stdio", "command": "echo", "enabled": True},
        })
        pool = MCPPool(str(cfg_file))

        async def init_and_reload_all():
            await pool.initialize()
            fake_mcp._tools_map["srv"] = [FakeToolObj(name="new1"), FakeToolObj(name="new2")]
            return await pool.reload_all()

        count = asyncio.run(init_and_reload_all())

        assert count == 2
        names = {t.name for t in pool.get_all_tools()}
        assert names == {"new1", "new2"}
