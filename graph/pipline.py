"""
监督者模式工作流 - Manager 拆解 → Worker 执行 → Terminator 汇总

与 graph/simple.py 结构相同(独立保留状态/节点/图定义),通过
register_workflow 在模块被 import 时自注册(方式二)。
"""
from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from graph.common import NodeCallback, arun_compiled_workflow
from graph.registry import register_workflow


# 1. 定义工作流状态
class WorkflowState(TypedDict):
    """监督者工作流状态"""
    task: str             # 用户原始任务
    raw_context: str      # 原始记忆文本(当前会话+长期记忆,仅 summarize 节点消费)
    context_summary: str  # Manager 提炼后的上下文摘要(注入 plan/final 节点)
    plan: str             # Manager 拆解的执行计划
    worker_result: str    # Worker 执行结果
    final_answer: str     # Terminator 最终答案


# 2. 节点函数(提示词模板由各角色 TeamAgent 懒加载,节点需要时调用 get_template)
def summarize_context(state: WorkflowState, manager) -> WorkflowState:
    """Manager 提炼记忆上下文,生成分发给下游节点的上下文摘要"""
    result = manager.summarize_context(state.get("raw_context", ""))
    return {"context_summary": result}


def manager_plan_node(state: WorkflowState, manager) -> WorkflowState:
    """Manager 拆解任务,生成执行计划(结合记忆上下文摘要)"""
    task = state["task"]
    summary = state.get("context_summary", "")
    template = manager.get_template("manager_plan")
    prompt = manager.render_template(template, task=task, context_summary=summary)
    result = manager.invoke(prompt)
    return {"plan": result}


def worker_exec_node(state: WorkflowState, worker) -> WorkflowState:
    """Worker 执行计划中的子任务"""
    plan = state["plan"]
    template = worker.get_template("worker_exec")
    prompt = worker.render_template(template, plan=plan)
    result = worker.invoke(prompt)
    return {"worker_result": result}


def terminator_final_node(state: WorkflowState, terminator) -> WorkflowState:
    """Terminator 汇总结果并返回最终答案(结合记忆上下文摘要)"""
    task = state["task"]
    plan = state["plan"]
    worker_result = state["worker_result"]
    summary = state.get("context_summary", "")
    template = terminator.get_template("terminator_final")

    prompt = terminator.render_template(
        template,
        task=task,
        plan=plan,
        worker_result=worker_result,
        context_summary=summary,
    )
    result = terminator.invoke(prompt)
    return {"final_answer": result}


# 3. 构建工作流图
def build_pipline_workflow(agents: dict, checkpointer=None) -> StateGraph:
    """
    构建监督者模式工作流

    Args:
        agents: 角色字典,需包含 manager/worker/terminator 三个键,
            分别对应管理者/执行者/终结者 Agent 实例
        checkpointer: LangGraph checkpointer 实例。传入时图编译带持久化，
            工作流状态按 thread_id 保存/恢复；为 None 时无持久化（测试/临时运行）。

    Returns:
        编译好的 LangGraph StateGraph
    """
    manager = agents["manager"]
    worker = agents["worker"]
    terminator = agents["terminator"]

    builder = StateGraph(WorkflowState)

    # 添加节点(使用 lambda 绑定 agent 实例;提示词模板由节点内懒加载)
    builder.add_node("summarize", lambda state: summarize_context(state, manager))
    builder.add_node("manager_plan", lambda state: manager_plan_node(state, manager))
    builder.add_node("worker_exec", lambda state: worker_exec_node(state, worker))
    builder.add_node("terminator_final", lambda state: terminator_final_node(state, terminator))

    # 添加边: START → summarize → manager_plan → worker_exec → terminator_final → END
    builder.add_edge(START, "summarize")
    builder.add_edge("summarize", "manager_plan")
    builder.add_edge("manager_plan", "worker_exec")
    builder.add_edge("worker_exec", "terminator_final")
    builder.add_edge("terminator_final", END)

    return builder.compile(checkpointer=checkpointer)


# 4. 运行工作流
async def arun_pipline_workflow(
    graph: StateGraph,
    task: str,
    raw_context: str = "",
    thread_id: str | None = None,
    on_node_start: NodeCallback | None = None,
    on_node_end: NodeCallback | None = None,
    on_node_error: NodeCallback | None = None,
) -> dict:
    """
    运行监督者工作流（异步）

    Args:
        graph: 编译好的工作流图
        task: 用户任务
        raw_context: 原始记忆文本(当前会话+长期记忆),为空则不注入记忆
        thread_id: 会话线程 ID。为 None 时自动生成；传入显式值时配合
            checkpointer 编译的图可实现状态持久化。
        on_node_start: 节点开始回调,接收节点名(用于运行进度跟踪)
        on_node_end: 节点结束回调,接收节点名
        on_node_error: 节点异常回调,接收节点名

    Returns:
        包含 final_answer 的结果字典
    """
    return await arun_compiled_workflow(
        graph,
        task,
        state_fields={"plan": "", "worker_result": "", "final_answer": ""},
        raw_context=raw_context,
        thread_id=thread_id,
        on_node_start=on_node_start,
        on_node_end=on_node_end,
        on_node_error=on_node_error,
    )


# 注: import 置于模块顶部、调用置于文件末尾——register_workflow 与 WORKFLOWS
# 在 graph.registry 文件前部定义,先于本模块被 import 时执行,循环导入安全。
register_workflow(
    "pipline",
    builder=build_pipline_workflow,
    runner=arun_pipline_workflow,
    roles=["manager", "worker", "terminator"],
    description="监督者模式工作流(Manager 拆解→Worker 执行→Terminator 汇总)",
)
