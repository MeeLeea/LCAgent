"""
MCP 工具加载器 - 从 MCP Server 动态加载 LangChain 工具
基于 langchain-mcp-adapters 0.3.x
"""
import asyncio
import json
import logging
import os
import sys
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

logger = logging.getLogger(__name__)

# 默认配置文件路径
DEFAULT_CONFIG_FILE = "config/mcp_servers.json"


def load_mcp_config(config_file: str = DEFAULT_CONFIG_FILE) -> dict[str, Any]:
    """
    读取 MCP servers 配置文件

    Args:
        config_file: 配置文件路径

    Returns:
        配置字典 {"servers": {...}}
    """
    if not os.path.exists(config_file):
        return {"servers": {}}
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.error("MCP 配置文件读取失败: %s", e)
        return {"servers": {}}


def save_mcp_config(config: dict[str, Any], config_file: str = DEFAULT_CONFIG_FILE):
    """保存 MCP servers 配置"""
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def _run_async(coro):
    """
    在同步上下文中运行异步协程

    如果当前已有事件循环在运行，则新建一个线程运行；
    否则直接用 asyncio.run。
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 已在事件循环中，需另起线程避免冲突
            import threading
            result = {}
            def runner():
                new_loop = asyncio.new_event_loop()
                try:
                    result["value"] = new_loop.run_until_complete(coro)
                finally:
                    new_loop.close()
            t = threading.Thread(target=runner)
            t.start()
            t.join()
            return result.get("value")
    except RuntimeError:
        pass
    return asyncio.run(coro)


def make_sync_compatible(tool: BaseTool) -> BaseTool:
    """
    给只支持异步调用的 MCP 工具包一层同步调用支持

    langchain-mcp-adapters 返回的 StructuredTool 只有 coroutine 形式的 func，
    直接被 LangGraph 同步调用会报错 "StructuredTool does not support sync invocation"。
    本函数把原工具封装成同步+异步双兼容的工具。
    """
    # 获取原工具的异步调用函数
    original_coroutine = getattr(tool, "coroutine", None)
    if original_coroutine is None:
        # 已支持同步，无需包装
        return tool

    async def _async_runner(**kwargs):
        return await original_coroutine(**kwargs)

    def _sync_runner(**kwargs):
        return _run_async(original_coroutine(**kwargs))

    # 重建工具，同时提供 sync func 和 async coroutine
    new_tool = StructuredTool(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        func=_sync_runner,
        coroutine=_async_runner,
    )
    return new_tool


async def load_mcp_tools(config_file: str = DEFAULT_CONFIG_FILE) -> list[BaseTool]:
    """
    从所有已启用的 MCP Server 异步加载 LangChain 工具

    逐个服务器加载，单个失败不影响其他服务器。

    Args:
        config_file: MCP 配置文件路径

    Returns:
        LangChain BaseTool 列表（已包装为同步兼容）
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient

    config = load_mcp_config(config_file)
    servers = config.get("servers", {})

    # 过滤出启用的服务器，并转换为客户端格式
    enabled_servers = {}
    for name, cfg in servers.items():
        if not cfg.get("enabled", False):
            continue
        transport = cfg.get("transport", "stdio")
        if transport == "stdio":
            command = cfg["command"]
            # 用当前解释器(venv)替换 python/python3，避免子进程误用系统解释器
            if command in ("python", "python3"):
                command = sys.executable
            enabled_servers[name] = {
                "transport": "stdio",
                "command": command,
                "args": cfg.get("args", []),
                "env": cfg.get("env"),
            }
        elif transport in ("sse", "streamable_http"):
            enabled_servers[name] = {
                "transport": transport,
                "url": cfg["url"],
            }

    if not enabled_servers:
        return []

    # 逐个服务器加载，避免一个失败导致全部失败
    all_tools: list[BaseTool] = []
    for name, server_cfg in enabled_servers.items():
        try:
            # 单服务器客户端
            client = MultiServerMCPClient({name: server_cfg})
            tools = await client.get_tools()
            if tools:
                logger.info("MCP %s: 加载了 %d 个工具", name, len(tools))
                all_tools.extend(tools)
        except Exception:
            logger.exception("MCP %s: 加载失败", name)

    # 包装为同步兼容
    sync_tools = [make_sync_compatible(t) for t in all_tools]
    return sync_tools


async def aload_mcp_tools_by_name(
    names: list[str],
    config_file: str = DEFAULT_CONFIG_FILE,
) -> list[BaseTool]:
    """按工具名从已启用的 MCP server 中筛选并加载工具。

    全量加载所有启用 server 的工具后按名字过滤,返回命中工具的列表。
    命中数为 0 时返回空列表(不抛异常,调用方据此降级为纯文本模式)。
    单个 server 加载失败由 load_mcp_tools 内部兜底(异常 + 继续),不影响其他 server。

    Args:
        names: 期望加载的工具名列表(如 ["write_file"])
        config_file: MCP 配置文件路径

    Returns:
        命中的 BaseTool 列表;无命中或 MCP 未配置时为空
    """
    if not names:
        return []
    wanted = set(names)
    all_tools = await load_mcp_tools(config_file)
    return [t for t in all_tools if t.name in wanted]


def load_mcp_tools_by_name_sync(
    names: list[str],
    config_file: str = DEFAULT_CONFIG_FILE,
) -> list[BaseTool]:
    """aload_mcp_tools_by_name 的同步包装。

    复用 mcp_loader._run_async 在同步上下文(如 build_workflow 装配期)拉起异步
    加载;当前线程已在事件循环中时,_run_async 会另起线程执行,避免嵌套循环报错。

    Args:
        names: 期望加载的工具名列表
        config_file: MCP 配置文件路径

    Returns:
        命中的 BaseTool 列表;加载失败或无命中时为空(静默降级,记 WARNING)
    """
    if not names:
        return []
    try:
        result = _run_async(aload_mcp_tools_by_name(names, config_file))
        return list(result) if result else []
    except Exception as e:
        logger.warning(
            "MCP 工具按名同步加载失败(names=%s),将降级为纯文本模式: %s", names, e
        )
        return []


def load_all_mcp_tools_sync(
    config_file: str = DEFAULT_CONFIG_FILE,
) -> list[BaseTool]:
    """同步加载所有已启用 MCP server 的全部工具(不过滤)。

    供 register_agent(mcp_all=True) 在 build_workflow 装配期调用:
    一次性连接所有 enabled MCP server、拉取全部工具、按名去重后返回。

    失败语义与 load_mcp_tools_by_name_sync 一致(静默降级):
      - 单个 server 加载失败由 load_mcp_tools 内部兜底(异常 + 继续),
        不影响其他 server,返回部分成功列表
      - 所有 server 全部失败或配置缺失时返回空列表
      - _run_async 自身异常时 WARNING + 返回空列表,不向上抛
        (调用方 _build 据此降级角色为纯文本模式)

    命名冲突处理:多个 server 提供同名工具时,按 server 遍历顺序
    (json 配置中的 enabled_servers 字典序)保留先到者,丢弃后到者并记 WARNING。
    这与 aload_mcp_tools_by_name 的既有行为(直接 extend 不去重)略有差异——
    全量加载场景冲突面更大,故在此收敛。

    Args:
        config_file: MCP 配置文件路径

    Returns:
        去重后的 BaseTool 列表;加载失败或无 enabled server 时为空
    """
    try:
        result = _run_async(load_mcp_tools(config_file))
        tools = list(result) if result else []
    except Exception as e:
        logger.warning(
            "MCP 全量同步加载失败,将降级为纯文本模式: %s", e
        )
        return []

    # 按 tool.name 去重,保留先到者(enabled_servers 字典遍历序可预测)
    seen: set[str] = set()
    deduped: list[BaseTool] = []
    for tool in tools:
        if tool.name in seen:
            logger.warning(
                "MCP 工具名冲突: '%s' 已存在,丢弃后到的同名工具", tool.name
            )
            continue
        seen.add(tool.name)
        deduped.append(tool)
    return deduped


def list_configured_servers(config_file: str = DEFAULT_CONFIG_FILE) -> list[dict[str, Any]]:
    """
    列出配置文件中所有 MCP 服务器

    Returns:
        [{"name":..., "transport":..., "enabled":..., "detail":...}]
    """
    config = load_mcp_config(config_file)
    servers = config.get("servers", {})
    result = []
    for name, cfg in servers.items():
        transport = cfg.get("transport", "stdio")
        if transport == "stdio":
            detail = f"{cfg.get('command','')} {' '.join(cfg.get('args',[]))}"
        else:
            detail = cfg.get("url", "")
        result.append({
            "name": name,
            "transport": transport,
            "enabled": cfg.get("enabled", False),
            "detail": detail
        })
    return result


def add_server(
    name: str,
    transport: str,
    command: str | None = None,
    args: list[str] | None = None,
    url: str | None = None,
    env: dict[str, str] | None = None,
    enabled: bool = True,
    config_file: str = DEFAULT_CONFIG_FILE
):
    """
    添加一个 MCP Server 到配置文件

    Args:
        name: 服务器名称
        transport: 传输类型 (stdio / sse / streamable_http)
        command: stdio 模式下的启动命令
        args: stdio 模式下的命令参数
        url: sse/http 模式下的服务器 URL
        env: 环境变量
        enabled: 是否启用
    """
    config = load_mcp_config(config_file)
    if "servers" not in config:
        config["servers"] = {}

    entry = {"transport": transport, "enabled": enabled}
    if transport == "stdio":
        if not command:
            raise ValueError("stdio 传输必须提供 command")
        entry["command"] = command
        entry["args"] = args or []
        if env:
            entry["env"] = env
    elif transport in ("sse", "streamable_http"):
        if not url:
            raise ValueError(f"{transport} 传输必须提供 url")
        entry["url"] = url
    else:
        raise ValueError(f"不支持的传输类型: {transport}")

    config["servers"][name] = entry
    save_mcp_config(config, config_file)


def remove_server(name: str, config_file: str = DEFAULT_CONFIG_FILE) -> bool:
    """从配置中删除服务器，返回是否删除成功"""
    config = load_mcp_config(config_file)
    if name in config.get("servers", {}):
        del config["servers"][name]
        save_mcp_config(config, config_file)
        return True
    return False


def toggle_server(name: str, enabled: bool, config_file: str = DEFAULT_CONFIG_FILE) -> bool:
    """启用/禁用服务器"""
    config = load_mcp_config(config_file)
    if name in config.get("servers", {}):
        config["servers"][name]["enabled"] = enabled
        save_mcp_config(config, config_file)
        return True
    return False
