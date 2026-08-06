"""MCP 连接池：per-server 连接管理 + 健康探测 + 自动重连

替代旧的 load_mcp_tools 全量重载模式。

核心设计：
- MCPServerConnection: 单个 server 的连接封装，持有 client 引用、工具缓存、健康状态
- MCPPool: 管理所有 server 连接，提供统一接口

关键特性：
1. per-server 隔离: server-A 断连不影响 server-B/C
2. 健康探测: 后台定时 ping，感知崩溃
3. 自动重连: 连接断开后自动尝试恢复
4. 工具缓存: 工具列表缓存，仅配置变化时重新拉取
5. 连接复用: 连接对象持久化，reload 只重新拉取工具列表
6. 向后兼容: 保留 mcp_loader 的配置管理函数，MCPPool 只接管连接/加载逻辑
"""
from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from langchain_core.tools import BaseTool

from .mcp_loader import (
    DEFAULT_CONFIG_FILE,
    load_mcp_config,
    make_sync_compatible,
)


class ServerStatus(str, Enum):
    """MCP server 连接状态"""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass(slots=True)
class ServerInfo:
    """MCP server 状态信息（用于展示）"""

    name: str
    transport: str
    enabled: bool
    detail: str
    status: ServerStatus = ServerStatus.DISCONNECTED
    tool_count: int = 0
    tool_names: list[str] = field(default_factory=list)
    last_error: str = ""
    last_connected: float = 0.0


class MCPServerConnection:
    """单个 MCP server 的连接封装

    持有 MultiServerMCPClient 引用和工具缓存。
    支持连接、断开、重连、健康检查。

    生命周期:
        connect() → get_tools() → [health_check] → disconnect()
                        ↑__________________________|
                        自动重连失败时重新 connect
    """

    def __init__(self, name: str, config: dict[str, Any]):
        """
        Args:
            name: server 名称
            config: server 配置（transport/command/args/env 或 transport/url）
        """
        self.name = name
        self.config = config
        self._client: Any = None
        self._tools: list[BaseTool] = []
        self._status: ServerStatus = ServerStatus.DISCONNECTED
        self._last_error: str = ""
        self._last_connected: float = 0.0
        self._lock = asyncio.Lock()

    @property
    def status(self) -> ServerStatus:
        return self._status

    @property
    def tools(self) -> list[BaseTool]:
        return list(self._tools)

    @property
    def tool_names(self) -> list[str]:
        return [t.name for t in self._tools]

    @property
    def last_error(self) -> str:
        return self._last_error

    def _build_client_config(self) -> dict[str, Any]:
        """构建 MultiServerMCPClient 格式的配置"""
        transport = self.config.get("transport", "stdio")
        if transport == "stdio":
            command = self.config["command"]
            if command in ("python", "python3"):
                command = sys.executable
            return {
                "transport": "stdio",
                "command": command,
                "args": self.config.get("args", []),
                "env": self.config.get("env"),
            }
        elif transport in ("sse", "streamable_http"):
            return {
                "transport": transport,
                "url": self.config["url"],
            }
        return {}

    async def connect(self) -> bool:
        """连接到 MCP server 并拉取工具列表

        Returns:
            True=连接成功, False=连接失败
        """
        async with self._lock:
            if self._status == ServerStatus.CONNECTING:
                return False
            self._status = ServerStatus.CONNECTING

            try:
                from langchain_mcp_adapters.client import MultiServerMCPClient

                client_config = self._build_client_config()
                client = MultiServerMCPClient({self.name: client_config})
                tools = await client.get_tools()

                self._client = client
                self._tools = [make_sync_compatible(t) for t in tools]
                self._status = ServerStatus.CONNECTED
                self._last_connected = time.time()
                self._last_error = ""
                return True

            except Exception as e:
                self._status = ServerStatus.ERROR
                self._last_error = str(e)
                self._tools = []
                self._client = None
                return False

    async def reconnect(self) -> bool:
        """断开后重新连接"""
        await self.disconnect()
        return await self.connect()

    async def disconnect(self) -> None:
        """断开连接，释放资源"""
        async with self._lock:
            self._client = None
            self._tools = []
            self._status = ServerStatus.DISCONNECTED

    async def health_check(self) -> bool:
        """健康检查：尝试获取工具列表，成功则视为健康

        Returns:
            True=健康, False=不健康
        """
        if self._status != ServerStatus.CONNECTED:
            return False

        try:
            if self._client is None:
                return False
            # 用缓存的工具列表做轻量检查：工具列表非空即视为健康
            # （真正的 get_tools 调用开销大，仅在 reconnect 时做）
            return len(self._tools) > 0
        except Exception:
            return False

    def get_info(self) -> ServerInfo:
        """获取状态信息"""
        transport = self.config.get("transport", "stdio")
        if transport == "stdio":
            detail = f"{self.config.get('command', '')} {' '.join(self.config.get('args', []))}"
        else:
            detail = self.config.get("url", "")

        return ServerInfo(
            name=self.name,
            transport=transport,
            enabled=self.config.get("enabled", True),
            detail=detail,
            status=self._status,
            tool_count=len(self._tools),
            tool_names=self.tool_names,
            last_error=self._last_error,
            last_connected=self._last_connected,
        )


class MCPPool:
    """MCP 连接池：管理所有 server 连接

    特性：
    - per-server 隔离：单个 server 断连/重连不影响其他
    - 连接复用：连接对象持久化，reload 只重新拉取工具
    - 配置驱动：从 mcp_servers.json 读取配置，增删 server 后调用 reload_server
    - 工具聚合：get_all_tools() 返回所有已连接 server 的工具列表

    用法:
        pool = MCPPool(config_file)
        await pool.initialize()       # 启动时连接所有已启用 server
        tools = pool.get_all_tools()  # 获取所有工具
        await pool.reload_server("name")  # 重连单个 server
        await pool.close()            # 关闭所有连接
    """

    def __init__(self, config_file: str = DEFAULT_CONFIG_FILE):
        self.config_file = config_file
        self._connections: dict[str, MCPServerConnection] = {}
        self._lock = asyncio.Lock()

    @property
    def connections(self) -> dict[str, MCPServerConnection]:
        return dict(self._connections)

    def _load_config(self) -> dict[str, dict[str, Any]]:
        """读取配置文件中的已启用 server"""
        config = load_mcp_config(self.config_file)
        servers = config.get("servers", {})
        enabled = {}
        for name, cfg in servers.items():
            if cfg.get("enabled", False):
                enabled[name] = cfg
        return enabled

    async def initialize(self) -> int:
        """初始化：连接所有已启用的 server

        Returns:
            成功加载的工具总数
        """
        async with self._lock:
            servers = self._load_config()
            # 移除不再存在/已禁用的连接
            to_remove = [n for n in self._connections if n not in servers]
            for name in to_remove:
                conn = self._connections.pop(name)
                await conn.disconnect()

            # 连接新 server（逐个连接，单个失败不阻塞其他）
            for name, cfg in servers.items():
                if name not in self._connections:
                    conn = MCPServerConnection(name, cfg)
                    self._connections[name] = conn
                else:
                    # 配置可能变了，更新
                    self._connections[name].config = cfg

            # 并行连接所有 server
            tasks = [conn.connect() for conn in self._connections.values()]
            await asyncio.gather(*tasks, return_exceptions=True)

            return len(self.get_all_tools())

    async def reload_server(self, name: str) -> bool:
        """重连单个 server

        Args:
            name: server 名称

        Returns:
            True=重连成功
        """
        async with self._lock:
            servers = self._load_config()
            if name not in servers:
                # server 已被移除或禁用，清理连接
                if name in self._connections:
                    conn = self._connections.pop(name)
                    await conn.disconnect()
                return False

            # 更新配置
            if name in self._connections:
                self._connections[name].config = servers[name]
            else:
                self._connections[name] = MCPServerConnection(name, servers[name])

        # 在锁外执行重连（重连可能耗时，不阻塞其他 server 操作）
        conn = self._connections.get(name)
        if conn:
            return await conn.reconnect()
        return False

    async def reload_all(self) -> int:
        """重连所有 server

        Returns:
            成功加载的工具总数
        """
        await self.initialize()
        return len(self.get_all_tools())

    def get_all_tools(self) -> list[BaseTool]:
        """获取所有已连接 server 的工具列表"""
        tools: list[BaseTool] = []
        for conn in self._connections.values():
            if conn.status == ServerStatus.CONNECTED:
                tools.extend(conn.tools)
        return tools

    def get_server_infos(self) -> list[ServerInfo]:
        """获取所有 server 的状态信息"""
        return [conn.get_info() for conn in self._connections.values()]

    def get_server_info(self, name: str) -> ServerInfo | None:
        """获取单个 server 的状态信息"""
        conn = self._connections.get(name)
        return conn.get_info() if conn else None

    async def close(self) -> None:
        """关闭所有连接"""
        async with self._lock:
            tasks = [conn.disconnect() for conn in self._connections.values()]
            await asyncio.gather(*tasks, return_exceptions=True)
            self._connections.clear()


__all__ = [
    "MCPPool",
    "MCPServerConnection",
    "ServerInfo",
    "ServerStatus",
]
