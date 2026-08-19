"""MCP 工具加载 Mixin - AgentCore 的 MCP 连接与工具列表管理。

从 agent_core.py 抽离，职责：
- 全量加载 / 单 server 重连 MCP 工具
- 工具签名检测（变化时触发 executor 重建）
- 可用工具名称列表

依赖 AgentCore 实例属性：_state_lock / _tools_signature / _mcp_pool /
local_tools / mcp_tools / tools / agent_executor / verbose。
"""
from __future__ import annotations

import logging

from tools.mcp_pool import ServerStatus

logger = logging.getLogger(__name__)


class McpTools:
    """MCP 工具加载 Mixin（供 AgentCore 多继承使用，自身不初始化状态）"""

    async def areload_mcp_tools(self) -> int:
        """
        异步重新加载 MCP 工具（通过 MCPPool 全量重连）

        使用 _state_lock 保护 tools 和 agent_executor 的并发修改。
        仅当工具签名（工具名集合）变化时才重建 Graph，否则只更新系统提示词。

        对于单个 server 的重连，推荐使用 areload_mcp_server(name)。

        Returns:
            加载到的 MCP 工具数量
        """
        self._ensure_not_closed()
        async with self._state_lock:
            try:
                old_signature = getattr(self, "_tools_signature", frozenset())
                count = await self._async_load_mcp_tools()
                # 合并工具列表
                self.tools = list(self.local_tools) + list(self.mcp_tools)
                new_signature = frozenset(t.name for t in self.tools)

                if getattr(self, "agent_executor", None) is not None and new_signature != old_signature:
                    await self._arebuild_agent_executor()
                return count
            except Exception:
                logger.exception("MCP 重新加载失败")
                return 0

    async def areload_mcp_server(self, name: str) -> bool:
        """重连单个 MCP server（不影响其他 server）

        Args:
            name: server 名称

        Returns:
            True=重连成功
        """
        self._ensure_not_closed()
        async with self._state_lock:
            try:
                old_signature = getattr(self, "_tools_signature", frozenset())
                success = await self._mcp_pool.reload_server(name)
                if not success and self.verbose:
                    logger.warning("MCP %s: 重连失败或已移除", name)
                # 从池中获取最新工具列表
                self.mcp_tools = self._mcp_pool.get_all_tools()
                self.tools = list(self.local_tools) + list(self.mcp_tools)
                new_signature = frozenset(t.name for t in self.tools)

                if getattr(self, "agent_executor", None) is not None:
                    if new_signature != old_signature:
                        await self._arebuild_agent_executor()
                return success
            except Exception:
                logger.exception("MCP %s: 重连失败", name)
                return False

    async def _async_load_mcp_tools(self) -> int:
        """通过 MCPPool 初始化所有 MCP 连接"""
        tool_count = await self._mcp_pool.initialize()
        self.mcp_tools = self._mcp_pool.get_all_tools()
        if self.mcp_tools and self.verbose:
            # 按 server 分组展示
            for info in self._mcp_pool.get_server_infos():
                if info.status == ServerStatus.CONNECTED:
                    logger.info("MCP %s: %d 个工具 (%s)",
                                info.name, info.tool_count, ", ".join(info.tool_names))
                elif info.status == ServerStatus.ERROR:
                    logger.warning("MCP %s: 连接失败 - %s", info.name, info.last_error)
        elif self.verbose:
            logger.info("MCP 未加载到任何工具(可能配置为空或服务器未启用)")
        return tool_count

    def get_available_tools(self) -> list[str]:
        """获取可用工具名称列表"""
        return [t.name for t in self.tools]