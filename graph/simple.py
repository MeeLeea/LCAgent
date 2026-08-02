"""
监督者模式工作流 - Manager 拆解 → Worker 执行 → Terminator 汇总
"""
from __future__ import annotations

import uuid
from typing import TypedDict

from langgraph.graph import END, START, StateGraph


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
def build_simple_workflow(agents: dict) -> StateGraph:
    """
    构建监督者模式工作流
    
    Args:
        agents: 角色字典,需包含 manager/worker/terminator 三个键,
            分别对应管理者/执行者/终结者 Agent 实例
        
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
    
    return builder.compile()


# 4. 运行工作流
def run_simple_workflow(graph: StateGraph, task: str, raw_context: str = "") -> dict:
    """
    运行监督者工作流
    
    Args:
        graph: 编译好的工作流图
        task: 用户任务
        raw_context: 原始记忆文本(当前会话+长期记忆),为空则不注入记忆
        
    Returns:
        包含 final_answer 的结果字典
    """
    # 生成独立 thread_id 确保不同运行间状态隔离
    thread_id = f"workflow-{uuid.uuid4().hex[:8]}"
    
    initial_state = {
        "task": task,
        "raw_context": raw_context,
        "context_summary": "",
        "plan": "",
        "worker_result": "",
        "final_answer": ""
    }
    
    result = graph.invoke(initial_state, config={"configurable": {"thread_id": thread_id}})
    return result


# 5. 可视化工具
def graph_display(graph: StateGraph, output_file: str = "workflow.png") -> None:
    """
    导出工作流图为 PNG
    
    Args:
        graph: 工作流图
        output_file: 输出文件路径
    """
    try:
        png_data = graph.get_graph().draw_mermaid_png()
        with open(output_file, "wb") as f:
            f.write(png_data)
        print(f"工作流图已导出到: {output_file}")
    except Exception as e:
        print(f"导出工作流图失败: {e}")
