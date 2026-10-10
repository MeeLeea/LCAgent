"""
RTL 芯片设计流水线工作流 - Manager 提炼 → Architect 架构 → Designer 设计 ↔ Verification 多轮验证 → Designer 交付

本文件只保留**图编排**(NodeSpec 注册表 + 边 + 条件路由 + 构建/运行入口);
全部节点实现归属各角色类(team/ 目录):

    - manager(team/manager/):              summarize_context_node 实例方法,提炼记忆上下文摘要
    - architect(team/architect/):          plan/design/analyze/review/spec_node 五个类属性
                                           (create_llm_node 闭包),依次产出执行计划 → 方案设计 →
                                           PPA 权衡分析 → 多维评审 → 规格文档
    - rtl_designer(team/rtl_designer/):    spec_node / verilog_node 类属性(规格梳理+Filelist、RTL 编码)
                                           + designer_file_check_node / designer_output_node 实例方法
    - rtl_verification(team/rtl_verification/): plan_node / check_node 类属性(验证计划、Testbench/验证)
                                           + sim_exec_check_node 实例方法(Vivado 仿真执行)

节点两种形态(register_nodes 自动识别,见 graph/common/node_spec.py):
    - 实例方法(bound method): ``async def node(self, state, injector=None, config=None)``,
      NodeSpec 经 ``agents[role].xxx_node`` 取引用,只补绑 injector;
    - 类属性闭包(create_llm_node 产物): NodeSpec 经 ``ArchiAgent.plan_node`` 等
      **类属性引用**取未绑定函数(切勿经实例引用——描述符协议会把 self 绑到
      state 参数位),经经典分支补绑 agent=/injector=。

多轮交互说明：
    designer_verilog → designer_file_check → verification_check → sim_exec_check 构成迭代环,
    由条件路由 route_after_file_check / route_after_sim_check 判定:
      designer_file_check 校验本轮产出 RTL 文件存在且非空,缺失则回 designer_verilog 重做;
      sim_exec_check 实际执行 Vivado 仿真并检查覆盖率(syn_filelist ⊆ sim_filelist 且报告已生成),
      仿真+覆盖率通过 → 经 designer_output 交付后终止(END);失败且轮次未达 max_rounds → 携带上轮验证报告反馈回到
      designer_verilog 重新设计;轮次达 max_rounds 上限 → 强制经 designer_output 交付后终止(END,防止死循环)。

节点执行链路说明：
    节点函数在自身渲染 prompt(get_template + render_template + 技能注入)后,
    调 ``run_team_turn_with_interrupt(agent, prompt, config)``(见 team/interrupt.py)。
    helper 内部经 ``TeamAgent.arun_structured`` 流式执行 LLM(token 增量经
    config["callbacks"] 流出到外层事件流);工具内 ``interrupt()`` 时透传给
    外层 graph 的 checkpointer,由外层 resume 恢复(对照 plan team-checkpointer-interrupt)。

注册说明：
    team/__init__.py 未导入 rtl_designer / rtl_verification 模块,本模块顶部
    以 from-import 引入三个角色类(NodeSpec 需引用 ArchiAgent/DesignerAgent/
    VerificationAgent 的类属性节点),同时触发其 @register_agent 装饰器完成注册;
    该 import 位于 register_workflow 调用之前;循环导入安全由 graph/common/__init__.py 的导入顺序保证(所有子模块导入完成后才调用 _load_builtin_workflows)。
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

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

# 角色类的 from-import 同时触发其模块顶部 @register_agent 装饰器完成注册
# (team/__init__.py 未导入 rtl_designer / rtl_verification;NodeSpec 亦引用类属性)
from team.architect.Architect import ArchiAgent
from team.rtl_designer.rtl_designer import DesignerAgent
from team.rtl_verification.rtl_verification import VerificationAgent

# 2. 节点实现归属说明见模块 docstring(全部节点方法/类属性在 team/ 各角色类中),
#    本文件仅保留编排:条件路由(§3) + 图构建(§4) + 运行入口(§5)。


# 3. 验证结论解析与条件路由
def verification_passed(report: str) -> bool:
    """从验证报告中解析验证结论:PASS 返回 True,FAIL/未明确返回 False。

    优先匹配报告中的"验证结论: PASS/FAIL"标记行;无显式标记时回退全文扫描:
    含 PASS 且不含 FAIL 视为通过(容错 LLM 未严格按格式输出的场景)。
    """
    if not report:
        return False
    upper = report.upper()
    for line in report.splitlines():
        stripped = line.strip().upper()
        if "验证结论" in stripped or stripped.startswith("结论"):
            return "PASS" in stripped and "FAIL" not in stripped
    return "PASS" in upper and "FAIL" not in upper


def route_after_file_check(state: dict[str, Any]) -> str:
    """designer_file_check 后的条件路由:文件校验通过 → verification_check;
    失败且未达轮次上限 → 回 designer_verilog 重做;达上限 → designer_output 交付。"""
    if state.get("file_check_passed"):
        return "verification_check"
    if state.get("round", 0) >= state.get("max_rounds", 3):
        return "designer_output"
    return "designer_verilog"


def route_after_sim_check(state: dict[str, Any]) -> str:
    """sim_exec_check 后的条件路由:仿真+覆盖率通过 → designer_output 交付;
    失败且未达轮次上限 → 回 designer_verilog 重做;达上限 → designer_output 交付。"""
    if state.get("sim_check_passed"):
        return "designer_output"
    if state.get("round", 0) >= state.get("max_rounds", 3):
        return "designer_output"
    return "designer_verilog"


# 4. 构建工作流图
def build_rtl_graph_workflow(
    agents: dict,
    checkpointer=None,
    skills_dir: str | None = None,
    auto_match_skills: bool = True,
    max_rounds: int = 3,
) -> StateGraph:
    """
    构建 RTL 芯片设计流水线工作流

    Args:
        agents: 角色字典,需包含 manager/architect/rtl_designer/rtl_verification 四个键,
            分别对应上下文提炼者/架构师/RTL 设计师/验证工程师 Agent 实例
        checkpointer: LangGraph checkpointer 实例。传入时图编译带持久化,
            工作流状态按 thread_id 保存/恢复;为 None 时无持久化(测试/临时运行)。
        skills_dir: 技能目录路径,为 None 时使用默认目录(.agents/skills)
        auto_match_skills: 是否在节点渲染 prompt 时按任务自动匹配注入技能
        max_rounds: 设计-验证多轮迭代最大轮次,超限强制进入交付节点

    Returns:
        编译好的 LangGraph StateGraph
    """
    manager = agents["manager"]
    rtl_designer = agents["rtl_designer"]
    rtl_verification = agents["rtl_verification"]

    # 技能注入器:节点渲染 prompt 时追加匹配的技能指引块
    injector = SkillInjector(
        skills_dir=skills_dir,
        auto_match=auto_match_skills,
    )

    builder = StateGraph(
        compose_state_schema(
            agents,
            extra_channels={
                "round": int,
                "max_rounds": int,
                "final_answer": str,
                "file_check_passed": bool,
                "sim_check_passed": bool,
                "sim_status": str,
                "sim_log_path": str,
            },
        )
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
            NodeSpec("architect_plan", ArchiAgent.plan_node, role="architect"),
            NodeSpec("architect_design", ArchiAgent.design_node, role="architect"),
            NodeSpec("architect_analyze", ArchiAgent.analyze_node, role="architect"),
            NodeSpec("architect_review", ArchiAgent.review_node, role="architect"),
            NodeSpec("architect_spec", ArchiAgent.spec_node, role="architect"),
            NodeSpec("designer_spec", DesignerAgent.spec_node, role="rtl_designer"),
            NodeSpec("verification_plan", VerificationAgent.plan_node, role="rtl_verification"),
            NodeSpec("designer_verilog", DesignerAgent.verilog_node, role="rtl_designer"),
            NodeSpec("verification_check", VerificationAgent.check_node, role="rtl_verification"),
            NodeSpec("designer_file_check", rtl_designer.designer_file_check_node, role="rtl_designer"),
            NodeSpec("sim_exec_check", rtl_verification.sim_exec_check_node, role="rtl_verification"),
            NodeSpec("designer_output", rtl_designer.designer_output_node, role="rtl_designer"),
        ],
    )

    # 添加边: START → summarize → architect 五阶段 → designer_spec → verification_plan
    #        → designer_verilog → verification_check →(条件) designer_output 交付 或回 designer_verilog
    #        designer_output → END(两条终态路径均经交付节点后终止)
    builder.add_edge(START, "summarize")
    builder.add_edge("summarize", "architect_plan")
    builder.add_edge("architect_plan", "architect_design")
    builder.add_edge("architect_design", "architect_analyze")
    builder.add_edge("architect_analyze", "architect_review")
    builder.add_edge("architect_review", "architect_spec")
    builder.add_edge("architect_spec", "designer_spec")
    builder.add_edge("designer_spec", "verification_plan")
    builder.add_edge("verification_plan", "designer_verilog")
    builder.add_edge("designer_verilog", "designer_file_check")
    builder.add_conditional_edges(
        "designer_file_check",
        route_after_file_check,
        {
            "verification_check": "verification_check",
            "designer_output": "designer_output",
            "designer_verilog": "designer_verilog",
        },
    )
    builder.add_edge("verification_check", "sim_exec_check")
    builder.add_conditional_edges(
        "sim_exec_check",
        route_after_sim_check,
        {
            "designer_output": "designer_output",
            "designer_verilog": "designer_verilog",
        },
    )
    builder.add_edge("designer_output", END)

    return builder.compile(checkpointer=checkpointer)


# 5. 运行工作流
async def arun_rtl_graph_workflow(
    graph: StateGraph,
    task: str,
    raw_context: str = "",
    thread_id: str | None = None,
    workspace_path: str | None = None,
    on_node_start: NodeCallback | None = None,
    on_node_end: NodeCallback | None = None,
    on_node_error: NodeCallback | None = None,
    max_history_chars: int = 6000,
    max_rounds: int = 3,
    memory=None,
    memory_thread_id: str | None = None,
    is_run_mode: bool = False,
    active_skills: Sequence[str] = (),
) -> dict:
    """
    运行 RTL 芯片设计流水线工作流(异步)

    Args:
        graph: 编译好的工作流图
        task: 用户任务
        raw_context: 原始记忆文本(当前会话+长期记忆),为空则不注入记忆
        thread_id: 会话线程 ID。为 None 时自动生成;传入显式值时配合
            checkpointer 编译的图可实现状态持久化。
        workspace_path: 会话绑定的工作空间绝对路径(本工作流纯文本角色,仅透传兼容)。
        on_node_start: 节点开始回调,接收节点名(用于运行进度跟踪)
        on_node_end: 节点结束回调,接收节点名
        on_node_error: 节点异常回调,接收节点名
        max_history_chars: 跨轮次记忆摘要最大字符数(超长截断)
        max_rounds: 设计-验证多轮迭代最大轮次,超限强制进入交付节点
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
        initial_fields={
            "context_summary": "",
            "arch_plan": "",
            "arch_design": "",
            "arch_analysis": "",
            "arch_review": "",
            "arch_spec": "",
            "design_spec": "",
            "verification_plan": "",
            "rtl_code": "",
            "verification_report": "",
            "round": 0,
            "max_rounds": max_rounds,
            "final_answer": "",
            "output_files": [],
            "file_check_passed": False,
            "sim_check_passed": False,
        },
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
    "rtl_graph",
    builder=build_rtl_graph_workflow,
    runner=arun_rtl_graph_workflow,
    roles=["manager", "architect", "rtl_designer", "rtl_verification"],
    description="RTL 芯片设计流水线(Manager 提炼→Architect 架构→Designer 设计↔Verification 多轮验证→Designer 交付)",
)
