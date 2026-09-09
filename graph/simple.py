"""
监督者模式工作流 - Manager 拆解 → Worker 执行 → Terminator 汇总

节点进度跟踪与通用运行器已提取至 graph/common/ 包，本文件仅保留
工作流状态定义、节点函数与图构建逻辑。

标准 LLM 节点通过 ``create_llm_node`` 工厂生成，消除样板代码；
summarize_context 节点因短路逻辑与模板拼接方式特殊，保留手写实现。

节点执行链路说明：
    节点函数在自身渲染 prompt（get_template + render_template + 技能注入）后，
    调 ``run_team_turn_with_interrupt(agent, prompt, config)``（见 graph/common/）。
    helper 内部经 ``TeamAgent.arun_structured`` 流式执行 LLM（token 增量经
    config["callbacks"] 流出到外层事件流）；工具内 ``interrupt()`` 时透传给
    外层 graph 的 checkpointer，由外层 resume 恢复。

workspace 隔离说明：
    worker_exec 节点接收 LangGraph 注入的 config（含 configurable.workspace_path），
    透传给 Worker 的工作流 → self.invoke，使 Worker 工具调用受
    WorkspaceSecurityMW 约束（见 graph/common.arun_compiled_workflow）。

会话化说明：
    WorkflowState 含 ``messages``（add_messages reducer）与 ``summary`` 字段，
    每个节点把自身产出追加为 AIMessage；消息通道压缩统一由
    LCAgentCompactionMiddleware.before_model 中间件在模型调用前触发。
"""
from __future__ import annotations

from typing import Annotated, Optional, TypedDict

from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from graph.common import (
    NodeCallback,
    NodeSpec,
    arun_compiled_workflow,
    create_llm_node,
    register_nodes,
    register_workflow,
    run_team_turn_with_interrupt,
)
from skmng.injector import SkillInjector
from team.base import TeamAgent


# 1. 定义工作流状态
class WorkflowState(TypedDict, total=False):
    """监督者工作流状态"""

    task: str             # 用户原始任务
    raw_context: str      # 原始记忆文本(当前会话+长期记忆,仅 summarize 节点消费)
    context_summary: str  # Manager 提炼后的上下文摘要(注入 plan/final 节点)
    plan: str             # Manager 拆解的执行计划
    worker_result: str    # Worker 执行结果
    final_answer: str     # Terminator 最终答案
    # 会话化消息通道:各节点产出追加为 AIMessage,经 add_messages reducer 累积;
    # 超阈值时由 compaction 中间件压缩(摘要进 summary,旧消息清空)
    messages: Annotated[list[AnyMessage], add_messages]
    summary: str          # 历史消息摘要(compaction 产物,随 checkpoint 持久化)


# 2. 节点函数(提示词模板由各角色 TeamAgent 懒加载,节点需要时调用 get_template)
async def summarize_context(
    state: WorkflowState,
    agent: TeamAgent,
    injector=None,
    config: Optional[RunnableConfig] = None,  # noqa: UP045 - 见 worker_exec_node 注释
) -> WorkflowState:
    """Manager 提炼记忆上下文,生成分发给下游节点的上下文摘要

    raw_context 为空时短路返回空串,跳过 LLM 调用(对照原
    ManagerAgent.asummarize_context 的短路语义)。非空时把
    ``summarize_context`` 模板内容拼到 prompt 前部作为指令(原实现经
    _astream_messages 的 system 消息语义,helper 单 prompt 通道下合并为用户消息),
    调 ``run_team_turn_with_interrupt`` 流式执行。

    config 透传(含 callbacks):使 summarize 的 LLM token 增量可流出到外层事件流。
    """
    raw = state.get("raw_context", "")
    # 与原 asummarize_context 一致:raw 为空时短路返回空串(不调 helper)
    if not raw:
        return {"context_summary": "", "messages": [AIMessage(content="")]}
    # summarize 节点不注入技能块(原 asummarize_context 也不调 injector)
    prompt = f"{agent.get_template('summarize_context')}\n\n{raw}"
    result = await run_team_turn_with_interrupt(agent, prompt, config)
    return {"context_summary": result, "messages": [AIMessage(content=result)]}


manager_plan_node = create_llm_node(
    template_name="manager_plan",
    output_field="plan",
    template_vars_fn=lambda s: {"task": s["task"], "context_summary": s.get("context_summary", "")},
    match_text_fn=lambda s: s["task"],
)


worker_exec_node = create_llm_node(
    template_name="worker_exec",
    output_field="worker_result",
    template_vars_fn=lambda s: {"plan": s["plan"]},
    match_text_fn=lambda s: s["plan"],
)


terminator_final_node = create_llm_node(
    template_name="terminator_final",
    output_field="final_answer",
    template_vars_fn=lambda s: {
        "task": s["task"],
        "plan": s["plan"],
        "worker_result": s["worker_result"],
        "context_summary": s.get("context_summary", ""),
    },
    match_text_fn=lambda s: s["task"],
)


# 3. 构建工作流图
def build_simple_workflow(
    agents: dict,
    checkpointer=None,
    skills_dir: str | None = None,
    auto_match_skills: bool = True,
) -> StateGraph:
    """构建监督者模式工作流

    Args:
        agents: 角色字典,需包含 manager/worker/terminator 三个键
        checkpointer: LangGraph checkpointer 实例
        skills_dir: 技能目录路径,为 None 时使用默认目录(.agents/skills)
        auto_match_skills: 是否在节点渲染 prompt 时按任务自动匹配注入技能

    Returns:
        编译好的 LangGraph StateGraph
    """
    injector = SkillInjector(
        skills_dir=skills_dir,
        auto_match=auto_match_skills,
    )

    builder = StateGraph(WorkflowState)

    register_nodes(
        builder,
        agents,
        injector,
        [
            NodeSpec("summarize", summarize_context, role="manager"),
            NodeSpec("manager_plan", manager_plan_node, role="manager"),
            NodeSpec("worker_exec", worker_exec_node, role="worker"),
            NodeSpec("terminator_final", terminator_final_node, role="terminator"),
        ],
    )

    # 添加边: START → summarize → manager_plan → worker_exec → terminator_final → END
    builder.add_edge(START, "summarize")
    builder.add_edge("summarize", "manager_plan")
    builder.add_edge("manager_plan", "worker_exec")
    builder.add_edge("worker_exec", "terminator_final")
    builder.add_edge("terminator_final", END)

    return builder.compile(checkpointer=checkpointer)


# 4. 运行工作流
async def arun_simple_workflow(
    graph: StateGraph,
    task: str,
    raw_context: str = "",
    thread_id: str | None = None,
    workspace_path: str | None = None,
    on_node_start: NodeCallback | None = None,
    on_node_end: NodeCallback | None = None,
    on_node_error: NodeCallback | None = None,
    max_history_chars: int = 6000,
    memory=None,
    memory_thread_id: str | None = None,
    is_run_mode: bool = False,
) -> dict:
    """
    运行监督者工作流（异步）

    Args:
        graph: 编译好的工作流图
        task: 用户任务
        raw_context: 原始记忆文本(当前会话+长期记忆),为空则不注入记忆
        thread_id: 会话线程 ID。为 None 时自动生成；传入显式值时配合
            checkpointer 编译的图可实现状态持久化。
        workspace_path: 会话绑定的工作空间绝对路径。为 None 时工作流内
            Worker 工具调用不做 workspace 隔离（兼容旧场景）。
        on_node_start: 节点开始回调,接收节点名(用于运行进度跟踪)
        on_node_end: 节点结束回调,接收节点名
        on_node_error: 节点异常回调,接收节点名
        max_history_chars: 跨轮次记忆摘要最大字符数(超长截断)
        memory: MemoryManager 实例（长期记忆召回与结果沉淀）；None 禁用
        memory_thread_id: 长期记忆使用的会话线程 ID
        is_run_mode: 是否运行模式（决定 DONE 事件是否标记为重要记忆）

    Returns:
        包含 final_answer 的结果字典
    """
    return await arun_compiled_workflow(
        graph,
        task,
        state_fields={"plan": "", "worker_result": "", "final_answer": ""},
        raw_context=raw_context,
        thread_id=thread_id,
        workspace_path=workspace_path,
        on_node_start=on_node_start,
        on_node_end=on_node_end,
        on_node_error=on_node_error,
        max_history_chars=max_history_chars,
        memory=memory,
        memory_thread_id=memory_thread_id,
        is_run_mode=is_run_mode,
    )


# 注: import 置于模块顶部、调用置于文件末尾——register_workflow 与 WORKFLOWS
# 在 graph.common 包前部定义,先于本模块被 import 时执行,循环导入安全。
register_workflow(
    "simple",
    builder=build_simple_workflow,
    runner=arun_simple_workflow,
    roles=["manager", "worker", "terminator"],
    description="监督者模式工作流(Manager 拆解→Worker 执行→Terminator 汇总)",
)
