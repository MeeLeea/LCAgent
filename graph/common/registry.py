"""工作流注册表与构建入口

两个注册表:
  - WORKFLOWS:       工作流名称 → 规格字典(由 register_workflow 写入)
  - AGENT_REGISTRY:  角色名 → {agent_class, config_file, tools}(由 @register_agent 装饰器填充)

工作流注册方式(唯一入口 register_workflow):
  - 内置工作流: 模块 import 时自注册(见 graph/simple.py 与 graph/rtl_graph.py 末尾)
  - 动态工作流: 任意运行时代码调用 register_workflow

工作流规格字段:
  - builder:     构建函数 build_xxx(agents: dict) -> 编译好的 StateGraph
  - runner:      自定义异步运行器,缺失时回退到 graph.simple.arun_simple_workflow
  - roles:       该工作流依赖的角色列表,缺失时构建全部已注册角色
  - description: 工作流描述,用于 CLI 列表展示
"""
from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import TypeVar

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

# 项目根目录(基于本文件位置计算)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ──────────────────────────────────────────────
# 工作流注册表
# ──────────────────────────────────────────────
WORKFLOWS: dict[str, dict] = {}

# ──────────────────────────────────────────────
# Agent 注册表
# ──────────────────────────────────────────────
AGENT_REGISTRY: dict[str, dict] = {}

T = TypeVar("T")


def register_agent(
    name: str,
    config_file: str,
    tools: list[BaseTool] | None = None,
    mcp_tools: list[str] | None = None,
    mcp_all: bool = False,
) -> Callable[[type[T]], type[T]]:
    """将 Agent 类注册到全局 AGENT_REGISTRY,供 build_workflow 统一构建。"""

    def decorator(cls: type[T]) -> type[T]:
        AGENT_REGISTRY[name] = {
            "agent_class": cls,
            "config_file": config_file,
            "tools": tools,
            "mcp_tools": mcp_tools,
            "mcp_all": mcp_all,
        }
        return cls

    return decorator


def register_workflow(
    name: str,
    builder: Callable,
    runner: Callable | None = None,
    roles: list[str] | None = None,
    description: str = "",
) -> None:
    """动态注册工作流(工作流唯一注册入口)。"""
    WORKFLOWS[name] = {
        "builder": builder,
        "runner": runner,
        "roles": roles,
        "description": description,
    }


def _get_workflow_spec(name: str) -> dict:
    """解析工作流注册项,统一返回字典格式。"""
    entry = WORKFLOWS[name]
    return {
        "builder": entry["builder"],
        "runner": entry.get("runner"),
        "roles": entry.get("roles"),
        "description": entry.get("description", ""),
    }


def get_workflow_runner(name: str) -> Callable | None:
    """获取工作流的异步运行器函数,缺失返回 None。"""
    return _get_workflow_spec(name)["runner"]


def list_workflows() -> list[tuple[str, str]]:
    """列出所有已注册工作流。"""
    result = []
    for name in WORKFLOWS:
        spec = _get_workflow_spec(name)
        result.append((name, spec["description"]))
    return result


def build_workflow(name: str, checkpointer=None) -> tuple[object, dict[str, object]]:
    """构建指定名称的工作流。

    Args:
        name: 工作流名称(如 "simple"/"rtl_graph")
        checkpointer: LangGraph checkpointer 实例

    Returns:
        (graph, agents) 元组

    Raises:
        KeyError: 工作流名称不存在,或所需角色未注册
    """
    if name not in WORKFLOWS:
        available = ", ".join(WORKFLOWS.keys())
        logger.warning("未知工作流: %s（可用: %s）", name, available)
        raise KeyError(f"未知工作流: {name}。可用工作流: {available}")

    spec = _get_workflow_spec(name)

    from team import build_team_agent

    def _build(role: str) -> object:
        if role not in AGENT_REGISTRY:
            available = ", ".join(AGENT_REGISTRY.keys()) or "(空)"
            logger.warning("未注册的角色: %s（已注册: %s）", role, available)
            raise KeyError(f"未注册的角色: {role}。已注册角色: {available}")
        role_spec = AGENT_REGISTRY[role]
        tools = list(role_spec["tools"]) if role_spec["tools"] else []
        mcp_names = role_spec.get("mcp_tools")
        mcp_all = role_spec.get("mcp_all", False)
        if mcp_all and mcp_names:
            logger.warning(
                "角色 %s: mcp_all=True 与 mcp_tools=%s 同时声明, "
                "mcp_all 优先,mcp_tools 被忽略",
                role, mcp_names,
            )
        if mcp_all:
            from tools.mcp_loader import load_all_mcp_tools_sync

            mcp_loaded = load_all_mcp_tools_sync()
            if mcp_loaded:
                tools.extend(mcp_loaded)
                logger.info(
                    "角色 %s: 加载 %d 个 MCP 工具(全部): %s",
                    role, len(mcp_loaded), [t.name for t in mcp_loaded],
                )
            else:
                logger.warning(
                    "角色 %s: mcp_all 加载失败或无 enabled MCP server,降级为纯文本模式",
                    role,
                )
        elif mcp_names:
            from tools.mcp_loader import load_mcp_tools_by_name_sync

            mcp_loaded = load_mcp_tools_by_name_sync(mcp_names)
            if mcp_loaded:
                tools.extend(mcp_loaded)
                logger.info(
                    "角色 %s: 加载 %d 个 MCP 工具: %s",
                    role, len(mcp_loaded), [t.name for t in mcp_loaded],
                )
            else:
                logger.warning(
                    "角色 %s: 声明的 MCP 工具 %s 加载失败或未配置,降级为纯文本模式",
                    role, mcp_names,
                )
        return build_team_agent(
            role_spec["agent_class"],
            role_spec["config_file"],
            BASE_DIR,
            tools=tools or None,
            checkpointer=checkpointer,
        )

    required_roles = spec["roles"]
    if required_roles:
        roles_to_build = [r for r in required_roles if r in AGENT_REGISTRY]
        missing = [r for r in required_roles if r not in AGENT_REGISTRY]
        if missing:
            available = ", ".join(AGENT_REGISTRY.keys()) or "(空)"
            logger.warning("工作流 '%s' 缺少角色: %s（已注册: %s）", name, missing, available)
            raise KeyError(f"工作流 '{name}' 缺少角色: {missing}。已注册角色: {available}")
    else:
        roles_to_build = list(AGENT_REGISTRY.keys())

    agents = {role: _build(role) for role in roles_to_build}

    graph = spec["builder"](agents, checkpointer=checkpointer)

    logger.info("工作流构建成功: %s（角色: %s）", name, ", ".join(sorted(roles_to_build)))
    return graph, agents


def _load_builtin_workflows() -> None:
    """加载内置工作流模块，触发其模块自注册。

    自动扫描 graph/ 目录下所有 .py 文件（排除 __init__.py 和 common/ 包），
    import 即触发各模块内 register_workflow 调用完成注册。
    """
    import importlib
    import pkgutil

    _NON_WORKFLOW_MODULES = frozenset({"__init__"})

    import graph as _graph_pkg

    for _finder, _name, _is_pkg in pkgutil.iter_modules(_graph_pkg.__path__):
        if _is_pkg or _name in _NON_WORKFLOW_MODULES:
            continue
        importlib.import_module(f"graph.{_name}")


async def arun_workflow_by_name(
    workflow_name: str,
    task: str,
    checkpointer=None,
    thread_id: str | None = None,
    workspace_path: str | None = None,
    on_node_start: Callable | None = None,
    on_node_end: Callable | None = None,
    memory=None,
    memory_thread_id: str | None = None,
    is_run_mode: bool = False,
) -> dict:
    """按名称构建并异步运行工作流(不依赖 CLI 上下文)。

    供 scheduler/executor 等非 CLI 场景调用。
    """
    graph, _agents = build_workflow(workflow_name, checkpointer=checkpointer)

    runner = get_workflow_runner(workflow_name)
    if runner is None:
        from graph.simple import arun_simple_workflow as runner

    return await runner(
        graph,
        task,
        raw_context="",
        thread_id=thread_id,
        workspace_path=workspace_path,
        on_node_start=on_node_start,
        on_node_end=on_node_end,
        memory=memory,
        memory_thread_id=memory_thread_id,
        is_run_mode=is_run_mode,
    )
