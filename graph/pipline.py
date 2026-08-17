"""
监督者模式工作流（别名模块）- Manager 拆解 → Worker 执行 → Terminator 汇总

本模块是 graph/simple.py 的别名/兼容层：pipline 与 simple 在历史上是
逐字重复的同一套工作流定义（状态、节点、图结构完全一致），现已去重为
对 graph/simple 符号的惰性 re-export，并保留 pipline 专属的
build_pipline_workflow / arun_pipline_workflow 名称与 "pipline" 注册名，
确保既有调用方（CLI、scheduler、测试、前端 workflow 名）零改动。

循环导入说明：
    graph.registry 初始化时（文件底部 _load_builtin_workflows）会扫描并
    import 本模块，而本模块的符号来自 graph.simple；若 simple 正处于
    部分初始化状态（它 import registry 后未执行完），顶层直接 import
    simple 会触发 ImportError。因此：
    - 节点函数/WorkflowState 通过 PEP 562 模块级 __getattr__ 惰性导出，
      首次属性访问时才 import graph.simple（彼时必然已初始化完毕）；
    - build/arun 以惰性委托 wrapper 提供（register_workflow 需要真实
      可调用对象，执行时才解析 simple）。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from graph.registry import register_workflow

if TYPE_CHECKING:
    # 惰性 re-export 符号的类型占位:仅供静态检查(F822)解析,运行时经
    # 模块级 __getattr__ 延迟导入,避免 registry 初始化扫描期的循环导入。
    from graph.simple import (
        WorkflowState,
        manager_plan_node,
        summarize_context,
        terminator_final_node,
        worker_exec_node,
    )

# simple 的惰性导入符号集:属性首次访问时 graph.simple 已完全初始化
_LAZY_SYMBOLS = frozenset(
    {
        "WorkflowState",
        "summarize_context",
        "manager_plan_node",
        "worker_exec_node",
        "terminator_final_node",
    }
)


def __getattr__(name: str):
    """PEP 562 模块级惰性属性:延迟 re-export graph.simple 的符号。"""
    if name in _LAZY_SYMBOLS:
        from graph import simple as _simple

        return getattr(_simple, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def build_pipline_workflow(*args, **kwargs):
    """构建监督者模式工作流（惰性委托 graph.simple.build_simple_workflow）。

    保留 pipline 专属名称以兼容既有调用方;实际图构建逻辑与 simple 完全一致。
    """
    from graph import simple as _simple

    return _simple.build_simple_workflow(*args, **kwargs)


async def arun_pipline_workflow(*args, **kwargs) -> dict:
    """运行监督者工作流（惰性委托 graph.simple.arun_simple_workflow）。"""
    from graph import simple as _simple

    return await _simple.arun_simple_workflow(*args, **kwargs)


__all__ = [
    "WorkflowState",
    "arun_pipline_workflow",
    "build_pipline_workflow",
    "manager_plan_node",
    "summarize_context",
    "terminator_final_node",
    "worker_exec_node",
]


# 注: import 置于模块顶部、调用置于文件末尾——register_workflow 与 WORKFLOWS
# 在 graph.registry 文件前部定义,先于本模块被 import 时执行,循环导入安全。
register_workflow(
    "pipline",
    builder=build_pipline_workflow,
    runner=arun_pipline_workflow,
    roles=["manager", "worker", "terminator"],
    description="监督者模式工作流(Manager 拆解→Worker 执行→Terminator 汇总)",
)
