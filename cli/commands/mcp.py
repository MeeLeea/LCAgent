"""MCP management commands."""

from __future__ import annotations

import asyncio

from .types import CommandContext, CommandOutcome, HANDLED


def show_mcp(context: CommandContext) -> CommandOutcome:
    backend = _backend(context)
    if backend is None:
        return HANDLED
    servers = backend.list_configured_servers(context.mcp_config_file)
    context.print("\nMCP Servers:")
    context.print("-" * 60)
    if not servers:
        context.print("  (空) 使用 'mcp:add' 添加")

    # 获取连接池状态
    pool = getattr(context.agent, "_mcp_pool", None)
    pool_infos = {info.name: info for info in pool.get_server_infos()} if pool else {}

    for server in servers:
        name = server["name"]
        enabled = server["enabled"]
        status_str = "✓启用" if enabled else "✗禁用"

        # 连接状态
        if name in pool_infos:
            info = pool_infos[name]
            if info.status.value == "connected":
                conn_str = f" [已连接 {info.tool_count}工具]"
            elif info.status.value == "error":
                conn_str = f" [错误: {info.last_error[:40]}]"
            else:
                conn_str = f" [{info.status.value}]"
        else:
            conn_str = ""

        context.print(f"  [{status_str}] {name} ({server['transport']}){conn_str}")
        context.print(f"           {server['detail']}")
    context.print("-" * 60)
    context.print(f"已加载 MCP 工具数: {len(context.agent.mcp_tools)}")
    if context.agent.mcp_tools:
        context.print(f"工具: {', '.join(tool.name for tool in context.agent.mcp_tools)}")
    context.print("\n命令: mcp:reload [name] | mcp:add | mcp:remove | mcp:toggle")
    return HANDLED


def reload_mcp(context: CommandContext, user_input: str = "") -> CommandOutcome:
    """重新加载 MCP 工具

    用法:
        mcp:reload          - 全量重连所有 server
        mcp:reload <name>   - 只重连指定 server
    """
    parts = user_input.split(None, 1) if user_input else []
    server_name = parts[1].strip() if len(parts) > 1 else None

    if server_name:
        context.print(f"\n重连 MCP Server: {server_name}...")
        success = asyncio.run(context.agent.areload_mcp_server(server_name))
        if success:
            context.print(f"完成: {server_name} 已重连")
        else:
            context.print(f"失败: {server_name} 重连失败或已移除")
    else:
        context.print("\n重新加载所有 MCP 工具...")
        count = asyncio.run(context.agent.areload_mcp_tools())
        context.print(f"完成: 已加载 {count} 个 MCP 工具")
    return HANDLED


def add_mcp(context: CommandContext, user_input: str) -> CommandOutcome:
    backend = _backend(context)
    if backend is None:
        return HANDLED
    parts = user_input.split(None, 2)
    if len(parts) < 3:
        context.print("\n用法:")
        context.print("  stdio: mcp:add <name> <command> <arg1> [arg2...]")
        context.print("  sse:   mcp:add <name> sse:<url>")
        context.print("  http:  mcp:add <name> http:<url>")
        return HANDLED
    name = parts[1]
    rest = parts[2].strip()
    try:
        if rest.startswith("sse:"):
            backend.add_server(name=name, transport="sse", url=rest[4:], config_file=context.mcp_config_file)
            context.print(f"\n已添加 sse MCP Server: {name}")
        elif rest.startswith("http:"):
            backend.add_server(
                name=name,
                transport="streamable_http",
                url=rest[5:],
                config_file=context.mcp_config_file,
            )
            context.print(f"\n已添加 http MCP Server: {name}")
        else:
            tokens = rest.split()
            if not tokens:
                context.print("错误: 缺少 command")
                return HANDLED
            backend.add_server(
                name=name,
                transport="stdio",
                command=tokens[0],
                args=tokens[1:],
                config_file=context.mcp_config_file,
            )
            context.print(f"\n已添加 stdio MCP Server: {name}")
        # 只重连新添加的 server，不影响其他
        success = asyncio.run(context.agent.areload_mcp_server(name))
        if success:
            context.print(f"已连接: {name}")
        else:
            context.print(f"连接失败: {name}（配置已保存，可稍后 mcp:reload {name}）")
    except (OSError, RuntimeError, ValueError) as error:
        context.print(f"\n添加失败: {error}")
    return HANDLED


def remove_mcp(context: CommandContext, user_input: str) -> CommandOutcome:
    backend = _backend(context)
    if backend is None:
        return HANDLED
    parts = user_input.split(None, 1)
    if len(parts) < 2:
        context.print("用法: mcp:remove <name>")
        return HANDLED
    name = parts[1].strip()
    if backend.remove_server(name, context.mcp_config_file):
        context.print(f"\n已删除 MCP Server: {name}")
        # 重连以移除该 server 的工具（其他 server 不受影响）
        asyncio.run(context.agent.areload_mcp_server(name))
        context.print(f"已移除 {name} 的工具")
    else:
        context.print(f"\n未找到: {name}")
    return HANDLED


def toggle_mcp(context: CommandContext, user_input: str) -> CommandOutcome:
    backend = _backend(context)
    if backend is None:
        return HANDLED
    parts = user_input.split()
    if len(parts) < 3:
        context.print("用法: mcp:toggle <name> <on|off>")
        return HANDLED
    name = parts[1]
    enabled = parts[2].lower() in ("on", "true", "1", "启用")
    if backend.toggle_server(name, enabled, context.mcp_config_file):
        state = "启用" if enabled else "禁用"
        context.print(f"\n已{state} MCP Server: {name}")
        # 重连该 server（禁用时移除其工具，启用时加载其工具）
        asyncio.run(context.agent.areload_mcp_server(name))
        context.print(f"已更新 {name} 的工具")
    else:
        context.print(f"\n未找到: {name}")
    return HANDLED


def _backend(context: CommandContext):
    if context.mcp_backend is not None:
        return context.mcp_backend
    # 延迟导入可避免仅使用普通对话时提前加载 MCP 依赖，也方便测试注入替身。
    from tools import mcp_loader

    context.mcp_backend = mcp_loader
    return context.mcp_backend
