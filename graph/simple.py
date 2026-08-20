"""
监督者模式工作流 - Manager 拆解 → Worker 执行 → Terminator 汇总

节点进度跟踪与通用运行器已提取至 graph/common.py，本文件仅保留
工作流状态定义、节点函数与图构建逻辑。

异步化说明：
    节点函数全部为 async，直接 await 团队 Agent 的异步业务方法
    （aplan_task/aexecute_task/afinalize/asummarize_context，内部经 TeamAgent.astream
    流式执行 LLM，token 增量经 config["callbacks"] 流出到外层事件流）；技能注入
    通过 SkillInjector 在节点渲染 prompt 时追加技能指引块。

workspace 隔离说明：
    worker_exec 节点接收 LangGraph 注入的 config（含 configurable.workspace_path），
    透传给 Worker.execute_task → self.invoke，使 Worker 工具调用受
    WorkspaceSecurityMW 约束（见 graph/common.arun_compiled_workflow）。

会话化说明：
    WorkflowState 含 ``messages``（add_messages reducer）与 ``summary`` 字段，
    每个节点把自身产出追加为 AIMessage；build 时可选注入 compaction 中间件，
    消息通道超阈值（默认 50）时节点级增量压缩（见 graph/common._compaction_wrapper），
    为 workflow 提供与 AgentCore 会话链路同构的长期消息流。
"""
from __future__ import annotations

from functools import partial
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agent.compaction import CompactionConfig
from graph.common import (
    NodeCallback,
    _build_compaction_middleware,
    arun_compiled_workflow,
    wrap_node_with_compaction,
)
from graph.registry import register_workflow
from skmng.injector import SkillInjector


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
    manager,
    injector=None,
    config: Optional[RunnableConfig] = None,  # noqa: UP045 - 见 worker_exec_node 注释
) -> WorkflowState:
    """Manager 提炼记忆上下文,生成分发给下游节点的上下文摘要

    raw_context 为空时仍调用 manager.asummarize_context（其内部短路返回空串），
    保持调用链可观测性（测试依赖 summarize 总是被调用的行为）。

    config 透传(含 callbacks):使 summarize 的 LLM token 增量可流出到外层事件流。
    """
    raw = state.get("raw_context", "")
    result = await manager.asummarize_context(raw, config)
    return {"context_summary": result, "messages": [AIMessage(content=result)]}


async def manager_plan_node(
    state: WorkflowState,
    manager,
    injector=None,
    config: Optional[RunnableConfig] = None,  # noqa: UP045 - 见 worker_exec_node 注释
) -> WorkflowState:
    """Manager 拆解任务,生成执行计划(结合记忆上下文摘要)"""
    task = state["task"]
    summary = state.get("context_summary", "")
    result = await manager.aplan_task(task, summary, injector, config)
    return {"plan": result, "messages": [AIMessage(content=result)]}


async def worker_exec_node(
    state: WorkflowState,
    worker,
    injector=None,
    config: Optional[RunnableConfig] = None,  # noqa: UP045 - 见下文:须用 Optional 写法,LangGraph 注解判定仅接受该字符串形态
) -> WorkflowState:
    """Worker 执行计划中的子任务。

    config 由 LangGraph 按节点签名以关键字注入（含 configurable.workspace_path），
    透传给 Worker.aexecute_task 使工具调用受 workspace 隔离约束、LLM token 增量
    经 callbacks 流出到外层事件流。

    注:必须用 Optional[RunnableConfig] 而非 RunnableConfig | None——模块启用
    ``from __future__ import annotations`` 后注解为字符串,仅
    'Optional[RunnableConfig]'/'RunnableConfig' 在 LangGraph 判定中被接受,
    'RunnableConfig | None' 字符串不匹配会导致 config 静默不注入。
    """
    plan = state["plan"]
    result = await worker.aexecute_task(plan, injector, config)
    return {"worker_result": result, "messages": [AIMessage(content=result)]}


async def terminator_final_node(
    state: WorkflowState,
    terminator,
    injector=None,
    config: Optional[RunnableConfig] = None,  # noqa: UP045 - 见 worker_exec_node 注释
) -> WorkflowState:
    """Terminator 汇总结果并返回最终答案(结合记忆上下文摘要)"""
    task = state["task"]
    plan = state["plan"]
    worker_result = state["worker_result"]
    summary = state.get("context_summary", "")
    result = await terminator.afinalize(task, plan, worker_result, summary, injector, config)
    return {"final_answer": result, "messages": [AIMessage(content=result)]}


# 3. 构建工作流图
def build_simple_workflow(
    agents: dict,
    checkpointer=None,
    skills_dir: str | None = None,
    auto_match_skills: bool = True,
    compaction_config: CompactionConfig | None = None,
) -> StateGraph:
    """
    构建监督者模式工作流

    Args:
        agents: 角色字典,需包含 manager/worker/terminator 三个键,
            分别对应管理者/执行者/终结者 Agent 实例
        checkpointer: LangGraph checkpointer 实例。传入时图编译带持久化，
            工作流状态按 thread_id 保存/恢复；为 None 时无持久化（测试/临时运行）。
        skills_dir: 技能目录路径,为 None 时使用默认目录(.agents/skills)
        auto_match_skills: 是否在节点渲染 prompt 时按任务自动匹配注入技能
        compaction_config: 消息通道压缩配置。为 None 时使用默认配置
            （阈值 50）；agent 无 llm 时自动禁用压缩（如测试 Fake）。

    Returns:
        编译好的 LangGraph StateGraph
    """
    manager = agents["manager"]
    worker = agents["worker"]
    terminator = agents["terminator"]

    # 技能注入器:节点渲染 prompt 时追加匹配的技能指引块
    injector = SkillInjector(
        skills_dir=skills_dir,
        auto_match=auto_match_skills,
    )

    # compaction 中间件:消息通道超阈值时节点级增量压缩(agent 无 llm 时禁用)
    compaction_mw = _build_compaction_middleware(manager, compaction_config)

    builder = StateGraph(WorkflowState)

    # 添加节点(使用 partial 绑定 agent 实例;提示词模板由节点内懒加载)
    # 注意:必须用 functools.partial 而非 lambda —— partial 保留 async 函数
    # 的 coroutine 特征(LangGraph 据此判定节点为异步并 await),lambda 会返回
    # 未 await 的 coroutine 导致 InvalidUpdateError
    builder.add_node("summarize", wrap_node_with_compaction(partial(summarize_context, manager=manager, injector=injector), compaction_mw))
    builder.add_node("manager_plan", wrap_node_with_compaction(partial(manager_plan_node, manager=manager, injector=injector), compaction_mw))
    builder.add_node("worker_exec", wrap_node_with_compaction(partial(worker_exec_node, worker=worker, injector=injector), compaction_mw))
    builder.add_node("terminator_final", wrap_node_with_compaction(partial(terminator_final_node, terminator=terminator, injector=injector), compaction_mw))

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
# 在 graph.registry 文件前部定义,先于本模块被 import 时执行,循环导入安全。
register_workflow(
    "simple",
    builder=build_simple_workflow,
    runner=arun_simple_workflow,
    roles=["manager", "worker", "terminator"],
    description="监督者模式工作流(Manager 拆解→Worker 执行→Terminator 汇总)",
)
