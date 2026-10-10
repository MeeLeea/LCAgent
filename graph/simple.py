"""
监督者模式工作流 - Manager 拆解 → Worker 执行 → Terminator 汇总

节点进度跟踪与通用运行器已提取至 graph/common/，本文件仅保留
工作流状态定义、LLM 节点定义与图构建逻辑。

节点执行链路说明：
    节点函数在自身渲染 prompt（get_template + render_template + 技能注入）后，
    调 ``run_team_turn_with_interrupt(agent, prompt, config)``（见 team/interrupt.py）。
    helper 内部经 ``TeamAgent.arun_structured`` 流式执行 LLM（token 增量经
    config["callbacks"] 流出到外层事件流）；工具内 ``interrupt()`` 时透传给
    外层 graph 的 checkpointer，由外层 resume 恢复（对照 plan team-checkpointer-interrupt）。
    summarize 节点实现已下沉至 ``ManagerAgent.summarize_context_node``
    (team/manager/manager.py,与 rtl_graph 工作流共用),经 register_nodes 的
    bound-method 分支注册;其余节点为本文件模块级的 create_llm_node 产物。

workspace 隔离说明：
    worker_exec 节点接收 LangGraph 注入的 config（含 configurable.workspace_path），
    透传给 Worker 的工作流 → self.invoke，使 Worker 工具调用受
    WorkspaceSecurityMW 约束（见 graph/common.arun_compiled_workflow）。

会话化说明：
    State schema 由 ``compose_state_schema(agents)`` 动态组合（底座含 task /
    raw_context / messages(add_messages reducer) / active_skills;各角色经
    ``state_fields`` 声明自身产出通道,如 Manager 的 context_summary/plan）。
    每个节点把自身产出追加为 AIMessage; ``messages`` 的压缩统一由
    LCAgentCompactionMiddleware.before_model 中间件在模型调用前自动触发
    （见 agent/compaction.py），节点级压缩已移除。底座不含工作流级
    ``summary`` 通道。
"""
from __future__ import annotations

from collections.abc import Sequence

from langgraph.graph import END, START, StateGraph

from graph.common import (
    NodeCallback,
    NodeSpec,
    arun_compiled_workflow,
    compose_state_schema,
    register_nodes,
    register_workflow,
)
from skmng.injector import SkillInjector
from team import ManagerAgent, TerminatorAgent, WorkerAgent

# 2. 节点函数(提示词模板由各角色 TeamAgent 懒加载,节点需要时调用 get_template)
# 全部节点实现归属角色类:
#   summarize        → ManagerAgent.summarize_context_node 实例方法(与 rtl_graph 共用)
#   manager_plan     → ManagerAgent.plan_node 类属性(create_llm_node 闭包)
#   worker_exec      → WorkerAgent.exec_node 类属性
#   terminator_final → TerminatorAgent.final_node 类属性


# 3. 构建工作流图
def build_simple_workflow(
    agents: dict,
    checkpointer=None,
    skills_dir: str | None = None,
    auto_match_skills: bool = True,
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

    Returns:
        编译好的 LangGraph StateGraph
    """
    manager = agents["manager"]

    # 技能注入器:节点渲染 prompt 时追加匹配的技能指引块
    injector = SkillInjector(
        skills_dir=skills_dir,
        auto_match=auto_match_skills,
    )

    builder = StateGraph(compose_state_schema(agents))

    # 添加节点(声明式 NodeSpec 表:partial 绑定 + add_node 两步合一)
    # 注意:register_nodes 内部用 functools.partial 绑定 agent 实例;提示词模板由节点内懒加载
    # partial 保留 async 函数的 coroutine 特征(LangGraph 据此判定节点为异步并 await),
    # lambda 会返回未 await 的 coroutine 导致 InvalidUpdateError
    register_nodes(
        builder,
        agents,
        injector,
        [
            NodeSpec("summarize", manager.summarize_context_node, role="manager"),
            NodeSpec("manager_plan", ManagerAgent.plan_node, role="manager"),
            NodeSpec("worker_exec", WorkerAgent.exec_node, role="worker"),
            NodeSpec("terminator_final", TerminatorAgent.final_node, role="terminator"),
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
    active_skills: Sequence[str] = (),
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
        active_skills: 显式注入的手动加载技能名(仅非空时写入初始状态,
            不覆盖 checkpoint 已持久化的值)

    Returns:
        包含 final_answer 的结果字典
    """
    return await arun_compiled_workflow(
        graph,
        task,
        initial_fields={"context_summary": "", "plan": "", "worker_result": "", "final_answer": ""},
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
        active_skills=active_skills,
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
