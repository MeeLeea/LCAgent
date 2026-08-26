"""
工作流注册表与构建入口

两个注册表:
  - WORKFLOWS:       工作流名称 → 规格字典(由 register_workflow 写入)
  - AGENT_REGISTRY:  角色名 → {agent_class, config_file, tools}(由 @register_agent 装饰器填充)

工作流注册方式(唯一入口 register_workflow):
  - 内置工作流: 模块 import 时自注册(见 graph/simple.py 与 graph/pipline.py 末尾)
  - 动态工作流: 任意运行时代码调用 register_workflow,适合插件式/条件式工作流

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
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ──────────────────────────────────────────────
# 工作流注册表
# ──────────────────────────────────────────────
# name -> {"builder", "runner", "roles", "description"}
# 由 register_workflow 写入;内置工作流在文件底部 _load_builtin_workflows 中加载
WORKFLOWS: dict[str, dict] = {}

# ──────────────────────────────────────────────
# Agent 注册表
# ──────────────────────────────────────────────
# name -> {agent_class, config_file, tools}
# 由 team/*/*.py 中的 @register_agent 装饰器在模块加载时填充
AGENT_REGISTRY: dict[str, dict] = {}

T = TypeVar("T")


def register_agent(
    name: str,
    config_file: str,
    tools: list[BaseTool] | None = None,
    mcp_tools: list[str] | None = None,
) -> Callable[[type[T]], type[T]]:
    """
    将 Agent 类注册到全局 AGENT_REGISTRY,供 build_workflow 统一构建

    Args:
        name: 角色名(如 "manager"/"spec_analyst")
        config_file: agent_config.json 路径(相对项目根)
        tools: 该角色的本地工具列表(纯文本角色传 None)。这些工具在装饰器
            执行时即确定(模块加载期),与 mcp_tools 互补
        mcp_tools: 该角色依赖的 MCP 工具名列表(如 ["write_file"])。这些工具
            在 build_workflow 装配期由 mcp_loader.load_mcp_tools_by_name_sync
            同步拉取(遍历已启用 MCP server 按名筛选),失败时静默降级为空列表
            (角色退化为纯文本模式)。声明工具名而非 server 名,解耦 server 配置

    Returns:
        装饰器函数,原样返回被装饰的类
    """

    def decorator(cls: type[T]) -> type[T]:
        AGENT_REGISTRY[name] = {
            "agent_class": cls,
            "config_file": config_file,
            "tools": tools,
            "mcp_tools": mcp_tools,
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
    """
    动态注册工作流(工作流唯一注册入口)

    允许在运行时添加新工作流,无需修改 WORKFLOWS 字典源码。

    Args:
        name: 工作流名称
        builder: 工作流构建函数,签名 build_xxx(agents: dict) -> StateGraph
        runner: 工作流异步运行函数,签名 run_xxx(graph, task, ...) -> dict
            缺失时回退到 graph.simple.arun_simple_workflow
        roles: 该工作流依赖的角色名列表,为 None 时构建全部已注册角色
        description: 工作流描述(用于 CLI 列表展示)
    """
    WORKFLOWS[name] = {
        "builder": builder,
        "runner": runner,
        "roles": roles,
        "description": description,
    }


# ──────────────────────────────────────────────
# 工作流规格解析
# ──────────────────────────────────────────────
def _get_workflow_spec(name: str) -> dict:
    """
    解析工作流注册项,统一返回字典格式

    Returns:
        {"builder": fn, "runner": fn|None, "roles": list|None, "description": str}
    """
    entry = WORKFLOWS[name]
    return {
        "builder": entry["builder"],
        "runner": entry.get("runner"),
        "roles": entry.get("roles"),
        "description": entry.get("description", ""),
    }


def get_workflow_runner(name: str) -> Callable | None:
    """获取工作流的异步运行器函数,缺失返回 None(由调用方回退到 arun_simple_workflow)"""
    return _get_workflow_spec(name)["runner"]


def list_workflows() -> list[tuple[str, str]]:
    """
    列出所有已注册工作流

    Returns:
        [(name, description), ...] 列表
    """
    result = []
    for name in WORKFLOWS:
        spec = _get_workflow_spec(name)
        result.append((name, spec["description"]))
    return result


# ──────────────────────────────────────────────
# 构建入口
# ──────────────────────────────────────────────
def build_workflow(name: str, checkpointer=None) -> tuple[object, dict[str, object]]:
    """
    构建指定名称的工作流(静默返回,不打印)

    新格式工作流通过 roles 声明依赖角色,仅构建所需角色;
    未声明 roles 的构建全部已注册角色。

    Args:
        name: 工作流名称(如 "simple"/"pipline")
        checkpointer: LangGraph checkpointer 实例。传入时图编译带持久化，
            工作流状态按 thread_id 保存/恢复；为 None 时无持久化。

    Returns:
        (graph, agents) 元组:
            - graph: 编译好的 LangGraph StateGraph
            - agents: 包含各已注册角色 Agent 实例的字典

    Raises:
        KeyError: 工作流名称不存在,或所需角色未注册
    """
    if name not in WORKFLOWS:
        available = ", ".join(WORKFLOWS.keys())
        logger.warning("未知工作流: %s（可用: %s）", name, available)
        raise KeyError(f"未知工作流: {name}。可用工作流: {available}")

    spec = _get_workflow_spec(name)

    # 函数内延迟导入 team,触发各 agent 模块的 @register_agent 装饰器执行
    from team import build_team_agent

    def _build(role: str) -> object:
        if role not in AGENT_REGISTRY:
            available = ", ".join(AGENT_REGISTRY.keys()) or "(空)"
            logger.warning("未注册的角色: %s（已注册: %s）", role, available)
            raise KeyError(f"未注册的角色: {role}。已注册角色: {available}")
        role_spec = AGENT_REGISTRY[role]
        # 同步拉取声明的 MCP 工具并合并到本地 tools
        # 失败时 mcp_loader 静默返回空列表,角色降级为纯文本模式(若本地 tools 也为空)
        tools = list(role_spec["tools"]) if role_spec["tools"] else []
        mcp_names = role_spec.get("mcp_tools")
        if mcp_names:
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

    # 按工作流声明的 roles 构建;未声明则构建全部已注册角色
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

    # 调用工作流构建器(统一接收 agents 字典 + checkpointer)
    graph = spec["builder"](agents, checkpointer=checkpointer)

    logger.info("工作流构建成功: %s（角色: %s）", name, ", ".join(sorted(roles_to_build)))
    return graph, agents


# ──────────────────────────────────────────────
# 内置工作流加载
# ──────────────────────────────────────────────
def _load_builtin_workflows() -> None:
    """加载内置工作流模块，触发其模块自注册（延迟导入避免循环依赖）。

    自动扫描 graph/ 目录下所有 .py 文件（排除 __init__.py 和基础设施文件），
    import 即触发各模块内 register_workflow 调用完成注册。

    systemc_cmodel 等依赖 team 包中 Agent 的模块，
    在 build_workflow 时才延迟导入 team，此处的 import 仅触发工作流注册。

    新增工作流只需在 graph/ 下新建 .py 文件，底部调用 register_workflow() 即可，
    无需修改本文件。
    """
    import importlib
    import pkgutil

    # 基础设施文件列表（非工作流模块，跳过）
    _NON_WORKFLOW_MODULES = frozenset({"common", "registry", "__init__"})

    import graph as _graph_pkg

    for _finder, _name, _is_pkg in pkgutil.iter_modules(_graph_pkg.__path__):
        if _is_pkg or _name in _NON_WORKFLOW_MODULES:
            continue
        importlib.import_module(f"graph.{_name}")


_load_builtin_workflows()


# ──────────────────────────────────────────────
# 独立执行入口(供 scheduler 等非 CLI 场景使用)
# ──────────────────────────────────────────────
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
    """
    按名称构建并异步运行工作流(不依赖 CLI 上下文)

    供 scheduler/executor 等非 CLI 场景调用:只需工作流名称和任务文本,
    内部完成构建 → 异步运行 → 返回结果字典。

    Args:
        workflow_name: 工作流名称(如 "simple"/"systemc_cmodel")
        task: 用户任务文本
        checkpointer: LangGraph checkpointer 实例。传入时图编译带持久化；
            为 None 时无持久化（scheduler 场景默认无持久化）。
        thread_id: 会话线程 ID。为 None 时自动生成；传入显式值时配合
            checkpointer 可实现状态持久化。
        workspace_path: 会话绑定的工作空间绝对路径。为 None 时工作流内
            Worker 工具调用不做 workspace 隔离（scheduler 场景默认无绑定）。
        on_node_start: 节点开始回调(可选,接收节点名)
        on_node_end: 节点结束回调(可选,接收节点名)
        memory: MemoryManager 实例（长期记忆召回与结果沉淀）；None 禁用
        memory_thread_id: 长期记忆使用的会话线程 ID
        is_run_mode: 是否运行模式（决定 DONE 事件是否标记为重要记忆）

    Returns:
        工作流结果字典(含 "final_answer" 键)

    Raises:
        KeyError: 工作流不存在或角色未注册
        Exception: 工作流执行中的异常
    """
    graph, _agents = build_workflow(workflow_name, checkpointer=checkpointer)

    # 获取工作流专用运行器,缺失时回退到 arun_simple_workflow
    runner = get_workflow_runner(workflow_name)
    if runner is None:
        from graph.simple import arun_simple_workflow as runner

    return await runner(
        graph,
        task,
        raw_context="",  # scheduler 场景无会话记忆
        thread_id=thread_id,
        workspace_path=workspace_path,
        on_node_start=on_node_start,
        on_node_end=on_node_end,
        memory=memory,
        memory_thread_id=memory_thread_id,
        is_run_mode=is_run_mode,
    )
