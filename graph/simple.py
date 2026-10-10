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

from langgraph.graph import END, START, StateGraph

from graph.common import (
    NodeSpec,
    register_nodes,
    register_workflow,
)
from graph.common.builder_utils import build_graph_scaffold
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

    # 基础 setup(技能注入器 + State schema 组合)收敛为公共构图辅助;
    # simple 无需工作流级额外通道,不传 extra_channels 等价于 compose_state_schema(agents)
    builder, injector = build_graph_scaffold(
        agents,
        skills_dir=skills_dir,
        auto_match_skills=auto_match_skills,
    )

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


# 4. 工作流注册
# 注: import 置于模块顶部、调用置于文件末尾——register_workflow 与 WORKFLOWS
# 在 graph.registry 文件前部定义,先于本模块被 import 时执行,循环导入安全。
# 运行调度统一经 graph.common.arun_compiled_workflow(arun_workflow_by_name 路由),
# 不再单独定义 arun_*_workflow 透传运行器;本工作流特有的初始状态字段
# 经 initial_fields 声明(见 arun_compiled_workflow 的 initial_fields 参数)。
register_workflow(
    "simple",
    builder=build_simple_workflow,
    roles=["manager", "worker", "terminator"],
    description="监督者模式工作流(Manager 拆解→Worker 执行→Terminator 汇总)",
    initial_fields={
        "context_summary": "",
        "plan": "",
        "worker_result": "",
        "final_answer": "",
    },
)
