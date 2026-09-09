"""graph.common 包 — 工作流共享组件（节点跟踪、运行器、注册表等）。

从原 graph/common.py 和 graph/registry.py 拆分重组而成。
所有公共符号从 __init__.py 再导出，保持 ``from graph.common import X`` 向后兼容。

子模块:
    node_tracking      — NodeTrackingHandler 回调处理器
    interrupt_forward   — run_team_turn_with_interrupt 中断转发
    node_factory        — create_llm_node 节点工厂
    workflow_runner     — arun_compiled_workflow 通用运行器
    compaction_utils    — _build_compaction_middleware 压缩中间件构造
    node_spec           — NodeSpec / register_nodes 声明式注册
    registry            — WORKFLOWS / AGENT_REGISTRY / build_workflow 注册表
"""
from __future__ import annotations

# 节点跟踪
from graph.common.node_tracking import (
    NodeCallback,
    NodeTrackingHandler,
    _extract_node_output,
)

# 中断转发
from graph.common.interrupt_forward import run_team_turn_with_interrupt

# 节点工厂
from graph.common.node_factory import create_llm_node

# 通用运行器
from graph.common.workflow_runner import (
    _aget_previous_workflow_summary,
    arun_compiled_workflow,
)

# 压缩中间件构造工具
from graph.common.compaction_utils import _build_compaction_middleware

# 声明式节点规格与注册
from graph.common.node_spec import NodeSpec, register_nodes

# 注册表
from graph.common.registry import (
    AGENT_REGISTRY,
    BASE_DIR,
    WORKFLOWS,
    arun_workflow_by_name,
    build_workflow,
    get_workflow_runner,
    list_workflows,
    register_agent,
    register_workflow,
)

# 加载内置工作流模块（触发自注册）— 放在所有子模块导入完成后调用，
# 避免 graph.simple 等模块 import graph.common 时因 __init__ 未完成而循环导入
from graph.common.registry import _load_builtin_workflows

_load_builtin_workflows()

__all__ = [
    "AGENT_REGISTRY",
    "BASE_DIR",
    "NodeCallback",
    "NodeSpec",
    "NodeTrackingHandler",
    "WORKFLOWS",
    "_build_compaction_middleware",
    "_extract_node_output",
    "_aget_previous_workflow_summary",
    "arun_compiled_workflow",
    "arun_workflow_by_name",
    "build_workflow",
    "create_llm_node",
    "get_workflow_runner",
    "list_workflows",
    "register_agent",
    "register_nodes",
    "register_workflow",
    "run_team_turn_with_interrupt",
]
