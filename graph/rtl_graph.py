"""
RTL 芯片设计流水线工作流 - Manager 提炼 → Architect 架构 → Designer 设计 ↔ Verification 多轮验证 → Designer 交付

角色分工：
    - manager(team.manager):        负责 summarize_context,提炼记忆上下文摘要
    - architect(team.architect):    负责 plan_task / design_task / analyze_task / review_task / spec_task,
                                    依次产出架构执行计划 → 架构方案设计 → PPA 权衡分析 → 多维评审 → 规格文档
    - rtl_designer(team.rtl_designer): 负责 spec_design_task(规格梳理+Filelist) 与 verilog_design_task(RTL 编码)
    - rtl_verification(team.rtl_verification): 负责 spec_design_task(验证计划) 与 verilog_design_task(Testbench/验证)

多轮交互说明：
    designer_verilog → verification_check 构成迭代环,由条件路由 route_after_verification 判定:
    验证报告含"验证结论: PASS"标记 → 进入 designer_output 交付;
    验证未通过且轮次未达 max_rounds → 携带上轮验证报告反馈回到 designer_verilog 重新设计;
    轮次达 max_rounds 上限 → 强制进入 designer_output 交付(防止死循环)。

异步化说明：
    节点函数全部为 async,通过 asyncio.to_thread 执行团队 Agent 的同步方法
    (不阻塞事件循环);技能注入通过 SkillInjector 在节点渲染 prompt 时
    追加技能指引块(TeamAgent 零改动)。

注册说明：
    team/__init__.py 未导入 rtl_designer / rtl_verification 模块,本模块顶部
    显式 import 两个角色模块,触发其 @register_agent 装饰器执行完成注册;
    该 import 位于 register_workflow 调用之前,与 graph.registry 无循环导入。
"""
from __future__ import annotations

from functools import partial
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

# 显式导入 RTL 角色模块,触发 @register_agent 装饰器注册(team/__init__.py 未导入)
import team.rtl_designer.rtl_designer
import team.rtl_verification.rtl_verification  # noqa: F401
from agent.compaction import CompactionConfig
from graph.common import (
    NodeCallback,
    _build_compaction_middleware,
    arun_compiled_workflow,
    wrap_node_with_compaction,
)
from graph.registry import register_workflow
from skmng.injector import SkillInjector
from team.base import TeamAgent


# 1. 定义工作流状态
class RTLGraphState(TypedDict, total=False):
    """RTL 芯片设计流水线工作流状态"""

    task: str               # 用户原始任务
    raw_context: str        # 原始记忆文本(仅 summarize 节点消费)
    context_summary: str    # Manager 提炼后的上下文摘要
    arch_plan: str          # Architect 架构执行计划
    arch_design: str        # Architect 架构方案设计
    arch_analysis: str      # Architect PPA 权衡分析
    arch_review: str        # Architect 多维评审意见
    arch_spec: str          # Architect 规格文档
    design_spec: str        # Designer 规格梳理与 Filelist
    verification_plan: str  # Verification 验证计划
    rtl_code: str           # Designer 输出的 RTL 源码
    verification_report: str  # Verification 验证报告(含验证结论标记)
    round: int              # 当前设计-验证迭代轮次
    max_rounds: int         # 最大迭代轮次(超限强制交付)
    final_answer: str       # Designer 最终交付内容
    # 会话化消息通道:各节点产出追加为 AIMessage,经 add_messages reducer 累积;
    # 超阈值时由 compaction 中间件压缩(摘要进 summary,旧消息清空)
    messages: Annotated[list[AnyMessage], add_messages]
    summary: str            # 历史消息摘要(compaction 产物,随 checkpoint 持久化)


# 2. 节点函数(提示词模板由各角色 TeamAgent 懒加载,节点需要时调用 get_template)
async def summarize_context(
    state: RTLGraphState,
    manager: TeamAgent,
    injector=None,
    config: Optional[RunnableConfig] = None,  # noqa: UP045 - 与 designer_verilog_node 一致
) -> RTLGraphState:
    """Manager 提炼记忆上下文,生成分发给下游节点的上下文摘要"""
    raw = state.get("raw_context", "")
    result = await manager.asummarize_context(raw, config)
    return {"context_summary": result, "messages": [AIMessage(content=result)]}


async def architect_plan_node(
    state: RTLGraphState,
    architect: TeamAgent,
    injector=None,
    config: Optional[RunnableConfig] = None,  # noqa: UP045 - 与 designer_verilog_node 一致
) -> RTLGraphState:
    """Architect 制定架构执行计划(结合记忆上下文摘要)"""
    task = state["task"]
    summary = state.get("context_summary", "")
    result = await architect.aplan_task(task, summary, injector, config)
    return {"arch_plan": result, "messages": [AIMessage(content=result)]}


async def architect_design_node(
    state: RTLGraphState,
    architect: TeamAgent,
    injector=None,
    config: Optional[RunnableConfig] = None,  # noqa: UP045 - 与 designer_verilog_node 一致
) -> RTLGraphState:
    """Architect 输出架构方案设计(结合执行计划)"""
    task = state["task"]
    plan = state.get("arch_plan", "")
    summary = state.get("context_summary", "")
    prompt_task = f"{task}\n\n【架构执行计划】\n{plan}" if plan else task
    result = await architect.adesign_task(prompt_task, summary, injector, config)
    return {"arch_design": result, "messages": [AIMessage(content=result)]}


async def architect_analyze_node(
    state: RTLGraphState,
    architect: TeamAgent,
    injector=None,
    config: Optional[RunnableConfig] = None,  # noqa: UP045 - 与 designer_verilog_node 一致
) -> RTLGraphState:
    """Architect 进行 PPA 权衡分析与瓶颈风险识别"""
    task = state["task"]
    design = state.get("arch_design", "")
    summary = state.get("context_summary", "")
    prompt_task = f"{task}\n\n【架构方案设计】\n{design}" if design else task
    result = await architect.aanalyze_task(prompt_task, summary, injector, config)
    return {"arch_analysis": result, "messages": [AIMessage(content=result)]}


async def architect_review_node(
    state: RTLGraphState,
    architect: TeamAgent,
    injector=None,
    config: Optional[RunnableConfig] = None,  # noqa: UP045 - 与 designer_verilog_node 一致
) -> RTLGraphState:
    """Architect 从架构/RTL/后端/软件/验证多维度评审方案"""
    task = state["task"]
    design = state.get("arch_design", "")
    analysis = state.get("arch_analysis", "")
    summary = state.get("context_summary", "")
    parts = [task]
    if design:
        parts.append(f"【架构方案设计】\n{design}")
    if analysis:
        parts.append(f"【权衡分析】\n{analysis}")
    prompt_task = "\n\n".join(parts)
    result = await architect.areview_task(prompt_task, summary, injector, config)
    return {"arch_review": result, "messages": [AIMessage(content=result)]}


async def architect_spec_node(
    state: RTLGraphState,
    architect: TeamAgent,
    injector=None,
    config: Optional[RunnableConfig] = None,  # noqa: UP045 - 与 designer_verilog_node 一致
) -> RTLGraphState:
    """Architect 整理可交付 RTL 开发的规格文档"""
    task = state["task"]
    design = state.get("arch_design", "")
    review = state.get("arch_review", "")
    summary = state.get("context_summary", "")
    parts = [task]
    if design:
        parts.append(f"【架构方案设计】\n{design}")
    if review:
        parts.append(f"【评审意见】\n{review}")
    prompt_task = "\n\n".join(parts)
    result = await architect.aspec_task(prompt_task, summary, injector, config)
    return {"arch_spec": result, "messages": [AIMessage(content=result)]}


async def designer_spec_node(
    state: RTLGraphState,
    designer: TeamAgent,
    injector=None,
    config: Optional[RunnableConfig] = None,  # noqa: UP045 - 与 designer_verilog_node 一致
) -> RTLGraphState:
    """Designer 基于架构规格完成 RTL 设计前的规格梳理与 Filelist 规划"""
    task = state["task"]
    arch_spec = state.get("arch_spec", "")
    prompt_task = f"{task}\n\n【架构规格文档】\n{arch_spec}" if arch_spec else task
    result = await designer.aspec_design_task(prompt_task, injector, config)
    return {"design_spec": result, "messages": [AIMessage(content=result)]}


async def verification_plan_node(
    state: RTLGraphState,
    verifier: TeamAgent,
    injector=None,
    config: Optional[RunnableConfig] = None,  # noqa: UP045 - 与 designer_verilog_node 一致
) -> RTLGraphState:
    """Verification 基于架构规格与设计规格制定验证计划"""
    task = state["task"]
    arch_spec = state.get("arch_spec", "")
    design_spec = state.get("design_spec", "")
    parts = [task]
    if arch_spec:
        parts.append(f"【架构规格文档】\n{arch_spec}")
    if design_spec:
        parts.append(f"【设计规格与Filelist】\n{design_spec}")
    prompt_task = "\n\n".join(parts)
    result = await verifier.aspec_design_task(prompt_task, injector, config)
    return {"verification_plan": result, "messages": [AIMessage(content=result)]}


async def designer_verilog_node(
    state: RTLGraphState,
    designer: TeamAgent,
    injector=None,
    config: Optional[RunnableConfig] = None,  # noqa: UP045 - 与 simple.py 一致:LangGraph 注解判定要求该字符串形态
) -> RTLGraphState:
    """Designer 输出可综合 RTL 源码(多轮迭代时携带上轮验证反馈)。

    round 计数:每次进入本节点轮次 +1,供条件路由判断是否达 max_rounds 上限。
    config 透传(含 callbacks):使 Designer LLM token 增量可流出到外层事件流。
    """
    task = state["task"]
    design_spec = state.get("design_spec", "")
    vplan = state.get("verification_plan", "")
    report = state.get("verification_report", "")
    round_n = state.get("round", 0)
    parts = [task]
    if design_spec:
        parts.append(f"【设计规格与Filelist】\n{design_spec}")
    if vplan:
        parts.append(f"【验证计划】\n{vplan}")
    if round_n > 0 and report:
        parts.append(f"【第 {round_n} 轮验证报告反馈(请据此修正 RTL)】\n{report}")
    prompt_task = "\n\n".join(parts)
    result = await designer.averilog_design_task(prompt_task, injector, config)
    return {"rtl_code": result, "round": round_n + 1, "messages": [AIMessage(content=result)]}


async def verification_check_node(
    state: RTLGraphState,
    verifier: TeamAgent,
    injector=None,
    config: Optional[RunnableConfig] = None,  # noqa: UP045 - 与 designer_verilog_node 一致
) -> RTLGraphState:
    """Verification 对 RTL 输出 Testbench/验证报告,并强制输出验证结论标记。

    提示词末尾要求输出"验证结论: PASS / FAIL"行,供 route_after_verification
    解析判定是否进入下一轮迭代。
    """
    task = state["task"]
    design_spec = state.get("design_spec", "")
    vplan = state.get("verification_plan", "")
    rtl = state.get("rtl_code", "")
    parts = [task]
    if design_spec:
        parts.append(f"【设计规格与Filelist】\n{design_spec}")
    if vplan:
        parts.append(f"【验证计划】\n{vplan}")
    if rtl:
        parts.append(f"【待验证 RTL 源码】\n{rtl}")
    parts.append(
        "最后必须单独输出一行验证结论,格式严格为: 验证结论: PASS(表示 RTL 无需修改) "
        "或 验证结论: FAIL(表示需修改,并在报告中给出具体修改建议)。"
    )
    prompt_task = "\n\n".join(parts)
    result = await verifier.averilog_design_task(prompt_task, injector, config)
    return {"verification_report": result, "messages": [AIMessage(content=result)]}


async def designer_output_node(
    state: RTLGraphState,
    designer: TeamAgent,
    injector=None,
    config: Optional[RunnableConfig] = None,  # noqa: UP045 - 与 designer_verilog_node 一致
) -> RTLGraphState:
    """Designer 基于最终 RTL 与验证报告整理交付文件,输出最终答案。"""
    task = state["task"]
    rtl = state.get("rtl_code", "")
    report = state.get("verification_report", "")
    design_spec = state.get("design_spec", "")
    parts = [task]
    if design_spec:
        parts.append(f"【设计规格与Filelist】\n{design_spec}")
    if rtl:
        parts.append(f"【最终 RTL 源码】\n{rtl}")
    if report:
        parts.append(f"【验证报告】\n{report}")
    parts.append("请整理以上内容,输出最终交付文件清单与完整 RTL 源码(作为最终交付物)。")
    prompt_task = "\n\n".join(parts)
    result = await designer.averilog_design_task(prompt_task, injector, config)
    return {"final_answer": result, "messages": [AIMessage(content=result)]}


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


def route_after_verification(state: RTLGraphState) -> str:
    """verification_check 后的条件路由:通过或达轮次上限 → designer_output;否则回 designer_verilog。"""
    round_n = state.get("round", 0)
    max_rounds = state.get("max_rounds", 3)
    if round_n >= max_rounds or verification_passed(state.get("verification_report", "")):
        return "designer_output"
    return "designer_verilog"


# 4. 构建工作流图
def build_rtl_graph_workflow(
    agents: dict,
    checkpointer=None,
    skills_dir: str | None = None,
    auto_match_skills: bool = True,
    max_rounds: int = 3,
    compaction_config: CompactionConfig | None = None,
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
        compaction_config: 消息通道压缩配置。为 None 时使用默认配置
            （阈值 50）；agent 无 llm 时自动禁用压缩（如测试 Fake）。

    Returns:
        编译好的 LangGraph StateGraph
    """
    manager = agents["manager"]
    architect = agents["architect"]
    designer = agents["rtl_designer"]
    verifier = agents["rtl_verification"]

    # 技能注入器:节点渲染 prompt 时追加匹配的技能指引块
    injector = SkillInjector(
        skills_dir=skills_dir,
        auto_match=auto_match_skills,
    )

    # compaction 中间件:消息通道超阈值时节点级增量压缩(agent 无 llm 时禁用)
    compaction_mw = _build_compaction_middleware(manager, compaction_config)

    builder = StateGraph(RTLGraphState)

    # 添加节点(使用 partial 绑定 agent 实例;提示词模板由节点内懒加载)
    # 注意:必须用 functools.partial 而非 lambda —— partial 保留 async 函数
    # 的 coroutine 特征(LangGraph 据此判定节点为异步并 await),lambda 会返回
    # 未 await 的 coroutine 导致 InvalidUpdateError
    builder.add_node("summarize", wrap_node_with_compaction(partial(summarize_context, manager=manager, injector=injector), compaction_mw))
    builder.add_node("architect_plan", wrap_node_with_compaction(partial(architect_plan_node, architect=architect, injector=injector), compaction_mw))
    builder.add_node("architect_design", wrap_node_with_compaction(partial(architect_design_node, architect=architect, injector=injector), compaction_mw))
    builder.add_node("architect_analyze", wrap_node_with_compaction(partial(architect_analyze_node, architect=architect, injector=injector), compaction_mw))
    builder.add_node("architect_review", wrap_node_with_compaction(partial(architect_review_node, architect=architect, injector=injector), compaction_mw))
    builder.add_node("architect_spec", wrap_node_with_compaction(partial(architect_spec_node, architect=architect, injector=injector), compaction_mw))
    builder.add_node("designer_spec", wrap_node_with_compaction(partial(designer_spec_node, designer=designer, injector=injector), compaction_mw))
    builder.add_node("verification_plan", wrap_node_with_compaction(partial(verification_plan_node, verifier=verifier, injector=injector), compaction_mw))
    builder.add_node("designer_verilog", wrap_node_with_compaction(partial(designer_verilog_node, designer=designer, injector=injector), compaction_mw))
    builder.add_node("verification_check", wrap_node_with_compaction(partial(verification_check_node, verifier=verifier, injector=injector), compaction_mw))
    builder.add_node("designer_output", wrap_node_with_compaction(partial(designer_output_node, designer=designer, injector=injector), compaction_mw))

    # 添加边: START → summarize → architect 五阶段 → designer_spec → verification_plan
    #        → designer_verilog → verification_check →(条件) designer_output → END
    builder.add_edge(START, "summarize")
    builder.add_edge("summarize", "architect_plan")
    builder.add_edge("architect_plan", "architect_design")
    builder.add_edge("architect_design", "architect_analyze")
    builder.add_edge("architect_analyze", "architect_review")
    builder.add_edge("architect_review", "architect_spec")
    builder.add_edge("architect_spec", "designer_spec")
    builder.add_edge("designer_spec", "verification_plan")
    builder.add_edge("verification_plan", "designer_verilog")
    builder.add_edge("designer_verilog", "verification_check")

    # 多轮交互条件路由:验证通过或达轮次上限 → designer_output;否则回 designer_verilog
    builder.add_conditional_edges(
        "verification_check",
        route_after_verification,
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

    Returns:
        包含 final_answer 的结果字典
    """
    return await arun_compiled_workflow(
        graph,
        task,
        state_fields={
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
