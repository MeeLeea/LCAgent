"""
RTL 芯片设计流水线工作流 - Manager 提炼 → Architect 架构 → Designer 设计 ↔ Verification 多轮验证 → Designer 交付

角色分工：
    - manager(team.manager):        负责 summarize_context,提炼记忆上下文摘要
    - architect(team.architect):    负责 plan_task / design_task / analyze_task / review_task / spec_task,
                                    依次产出架构执行计划 → 架构方案设计 → PPA 权衡分析 → 多维评审 → 规格文档
    - rtl_designer(team.rtl_designer): 负责 spec_design_task(规格梳理+Filelist) 与 verilog_design_task(RTL 编码)
    - rtl_verification(team.rtl_verification): 负责 spec_design_task(验证计划) 与 verilog_design_task(Testbench/验证)

多轮交互说明：
    designer_verilog → designer_file_check → verification_check → sim_exec_check 构成迭代环,
    由条件路由 route_after_file_check / route_after_sim_check 判定:
      designer_file_check 校验本轮产出 RTL 文件存在且非空,缺失则回 designer_verilog 重做;
      sim_exec_check 实际执行 Vivado 仿真并检查覆盖率(syn_filelist ⊆ sim_filelist 且报告已生成),
      仿真+覆盖率通过 → 直接终止(END);失败且轮次未达 max_rounds → 携带上轮验证报告反馈回到
      designer_verilog 重新设计;轮次达 max_rounds 上限 → 强制终止(END,防止死循环)。

节点执行链路说明：
    节点函数在自身渲染 prompt(get_template + render_template + 技能注入)后,
    调 ``run_team_turn_with_interrupt(agent, prompt, config)``(见 graph/common.py)。
    helper 内部经 ``TeamAgent.arun_structured`` 流式执行 LLM(token 增量经
    config["callbacks"] 流出到外层事件流);工具内 ``interrupt()`` 时透传给
    外层 graph 的 checkpointer,由外层 resume 恢复(对照 plan team-checkpointer-interrupt)。

注册说明：
    team/__init__.py 未导入 rtl_designer / rtl_verification 模块,本模块顶部
    显式 import 两个角色模块,触发其 @register_agent 装饰器执行完成注册;
    该 import 位于 register_workflow 调用之前,与 graph.registry 无循环导入。
"""
from __future__ import annotations

import asyncio
import os
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
    NodeSpec,
    _build_compaction_middleware,
    arun_compiled_workflow,
    register_nodes,
    run_team_turn_with_interrupt,
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
    # 校验状态(新增 check 节点使用)
    output_files: list[str]   # designer_verilog 解析 syn_filelist.f 得到的待交付 RTL 文件相对路径列表
    file_check_passed: bool   # designer_file_check 文件存在/非空校验结果
    sim_check_passed: bool    # sim_exec_check 仿真+覆盖率校验结果
    sim_status: str           # sim_exec_check 结论(PASS/FAIL/ERROR)
    sim_log_path: str         # sim_exec_check 写出的完整 Vivado 日志路径


# 2. 节点函数(提示词模板由各角色 TeamAgent 懒加载,节点需要时调用 get_template)

# ---- 通用工具:filelist 解析与覆盖率包含检查(供 check 节点复用) ----
def _parse_filelist(workspace: Optional[str], rel_path: str) -> list[str]:  # noqa: UP045
    """解析 filelist(.f),返回其中的源文件相对路径列表。

    跳过空行、`#`/`//` 注释、`+incdir+`/`-f`/`+` 开头的指令行;仅保留
    以 .v/.sv/.vhd/.vhdl 结尾的 token(取行内首个 token,忽略行内注释与多余参数)。
    """
    abs_p = rel_path if os.path.isabs(rel_path) else os.path.join(workspace, rel_path) if workspace else rel_path
    if not os.path.exists(abs_p):
        return []
    files: list[str] = []
    with open(abs_p, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith(("#", "//")):
                continue
            if line.startswith(("+", "-")):
                continue
            token = line.split()[0]
            if token.endswith((".v", ".sv", ".vhd", ".vhdl")):
                files.append(token)
    return files


def _coverage_inclusion_ok(workspace: Optional[str]) -> bool:  # noqa: UP045
    """覆盖率兜底:scripts/syn_filelist.f 的全部 src 文件必须被 scripts/sim_filelist.f 包含。

    使用集合包含判定(对 filelist 语法稳定,比解析 LLM 生成的 tcl 可靠)。
    """
    syn = _parse_filelist(workspace, "scripts/syn_filelist.f")
    sim = _parse_filelist(workspace, "scripts/sim_filelist.f")
    if not syn:
        return False
    syn_set = {os.path.normpath(p) for p in syn}
    sim_set = {os.path.normpath(p) for p in sim}
    return syn_set.issubset(sim_set)


def _write_text_file(path: str, text: str) -> None:
    """同步写文本文件(供 asyncio.to_thread 在事件循环外调用,避免阻塞)。"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


async def summarize_context(
    state: RTLGraphState,
    agent: TeamAgent,
    injector=None,
    config: Optional[RunnableConfig] = None,  # noqa: UP045 - 与 designer_verilog_node 一致
) -> RTLGraphState:
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


async def architect_plan_node(
    state: RTLGraphState,
    agent: TeamAgent,
    injector=None,
    config: Optional[RunnableConfig] = None,  # noqa: UP045 - 与 designer_verilog_node 一致
) -> RTLGraphState:
    """Architect 制定架构执行计划(结合记忆上下文摘要)

    节点内渲染 ``architect_plan`` 模板 + 注入技能块,然后调
    ``run_team_turn_with_interrupt`` 流式执行;技能注入 match 文本为
    ``task``(对照原 ArchiAgent.aplan_task)。
    """
    task = state["task"]
    summary = state.get("context_summary", "")
    prompt = agent.render_template(
        agent.get_template("architect_plan"), task=task, context_summary=summary
    )
    if injector is not None:
        prompt = injector.inject_into_prompt(prompt, task)
    result = await run_team_turn_with_interrupt(agent, prompt, config)
    return {"arch_plan": result, "messages": [AIMessage(content=result)]}


async def architect_design_node(
    state: RTLGraphState,
    agent: TeamAgent,
    injector=None,
    config: Optional[RunnableConfig] = None,  # noqa: UP045 - 与 designer_verilog_node 一致
) -> RTLGraphState:
    """Architect 输出架构方案设计(结合执行计划)

    节点内拼装 task + 架构执行计划为 ``prompt_task``,渲染
    ``architect_design`` 模板 + 注入技能块,然后调
    ``run_team_turn_with_interrupt`` 流式执行;技能注入 match 文本为
    ``prompt_task``(对照原 ArchiAgent.adesign_task,任务文本含上游计划)。
    """
    task = state["task"]
    plan = state.get("arch_plan", "")
    summary = state.get("context_summary", "")
    prompt_task = f"{task}\n\n【架构执行计划】\n{plan}" if plan else task
    prompt = agent.render_template(
        agent.get_template("architect_design"),
        task=prompt_task,
        context_summary=summary,
    )
    if injector is not None:
        prompt = injector.inject_into_prompt(prompt, prompt_task)
    result = await run_team_turn_with_interrupt(agent, prompt, config)
    return {"arch_design": result, "messages": [AIMessage(content=result)]}


async def architect_analyze_node(
    state: RTLGraphState,
    agent: TeamAgent,
    injector=None,
    config: Optional[RunnableConfig] = None,  # noqa: UP045 - 与 designer_verilog_node 一致
) -> RTLGraphState:
    """Architect 进行 PPA 权衡分析与瓶颈风险识别

    节点内拼装 task + 架构方案设计为 ``prompt_task``,渲染
    ``architect_analyze`` 模板 + 注入技能块,然后调
    ``run_team_turn_with_interrupt`` 流式执行;技能注入 match 文本为
    ``prompt_task``(对照原 ArchiAgent.aanalyze_task)。
    """
    task = state["task"]
    design = state.get("arch_design", "")
    summary = state.get("context_summary", "")
    prompt_task = f"{task}\n\n【架构方案设计】\n{design}" if design else task
    prompt = agent.render_template(
        agent.get_template("architect_analyze"),
        task=prompt_task,
        context_summary=summary,
    )
    if injector is not None:
        prompt = injector.inject_into_prompt(prompt, prompt_task)
    result = await run_team_turn_with_interrupt(agent, prompt, config)
    return {"arch_analysis": result, "messages": [AIMessage(content=result)]}


async def architect_review_node(
    state: RTLGraphState,
    agent: TeamAgent,
    injector=None,
    config: Optional[RunnableConfig] = None,  # noqa: UP045 - 与 designer_verilog_node 一致
) -> RTLGraphState:
    """Architect 从架构/RTL/后端/软件/验证多维度评审方案

    节点内拼装 task + 架构方案设计 + 权衡分析为 ``prompt_task``,渲染
    ``architect_review`` 模板 + 注入技能块,然后调
    ``run_team_turn_with_interrupt`` 流式执行;技能注入 match 文本为
    ``prompt_task``(对照原 ArchiAgent.areview_task)。
    """
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
    prompt = agent.render_template(
        agent.get_template("architect_review"),
        task=prompt_task,
        context_summary=summary,
    )
    if injector is not None:
        prompt = injector.inject_into_prompt(prompt, prompt_task)
    result = await run_team_turn_with_interrupt(agent, prompt, config)
    return {"arch_review": result, "messages": [AIMessage(content=result)]}


async def architect_spec_node(
    state: RTLGraphState,
    agent: TeamAgent,
    injector=None,
    config: Optional[RunnableConfig] = None,  # noqa: UP045 - 与 designer_verilog_node 一致
) -> RTLGraphState:
    """Architect 整理可交付 RTL 开发的规格文档

    节点内拼装 task + 架构方案设计 + 评审意见为 ``prompt_task``,渲染
    ``architect_spec`` 模板 + 注入技能块,然后调
    ``run_team_turn_with_interrupt`` 流式执行;技能注入 match 文本为
    ``prompt_task``(对照原 ArchiAgent.aspec_task)。
    """
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
    prompt = agent.render_template(
        agent.get_template("architect_spec"),
        task=prompt_task,
        context_summary=summary,
    )
    if injector is not None:
        prompt = injector.inject_into_prompt(prompt, prompt_task)
    result = await run_team_turn_with_interrupt(agent, prompt, config)
    return {"arch_spec": result, "messages": [AIMessage(content=result)]}


async def designer_spec_node(
    state: RTLGraphState,
    agent: TeamAgent,
    injector=None,
    config: Optional[RunnableConfig] = None,  # noqa: UP045 - 与 designer_verilog_node 一致
) -> RTLGraphState:
    """Designer 基于架构规格完成 RTL 设计前的规格梳理与 Filelist 规划

    节点内拼装 task + 架构规格文档为 ``prompt_task``,渲染
    ``spec_design`` 模板 + 注入技能块,然后调
    ``run_team_turn_with_interrupt`` 流式执行;技能注入 match 文本为
    ``prompt_task``(对照原 DesignerAgent.aspec_design_task)。
    """
    task = state["task"]
    arch_spec = state.get("arch_spec", "")
    prompt_task = f"{task}\n\n【架构规格文档】\n{arch_spec}" if arch_spec else task
    prompt = agent.render_template(agent.get_template("spec_design"), task=prompt_task)
    if injector is not None:
        prompt = injector.inject_into_prompt(prompt, prompt_task)
    result = await run_team_turn_with_interrupt(agent, prompt, config)
    return {"design_spec": result, "messages": [AIMessage(content=result)]}


async def verification_plan_node(
    state: RTLGraphState,
    agent: TeamAgent,
    injector=None,
    config: Optional[RunnableConfig] = None,  # noqa: UP045 - 与 designer_verilog_node 一致
) -> RTLGraphState:
    """Verification 基于架构规格与设计规格制定验证计划

    节点内拼装 task + 架构规格 + 设计规格为 ``prompt_task``,渲染
    ``spec_design`` 模板 + 注入技能块,然后调
    ``run_team_turn_with_interrupt`` 流式执行;技能注入 match 文本为
    ``prompt_task``(对照原 VerificationAgent.aspec_design_task)。
    """
    task = state["task"]
    arch_spec = state.get("arch_spec", "")
    design_spec = state.get("design_spec", "")
    parts = [task]
    if arch_spec:
        parts.append(f"【架构规格文档】\n{arch_spec}")
    if design_spec:
        parts.append(f"【设计规格与Filelist】\n{design_spec}")
    prompt_task = "\n\n".join(parts)
    prompt = agent.render_template(agent.get_template("spec_design"), task=prompt_task)
    if injector is not None:
        prompt = injector.inject_into_prompt(prompt, prompt_task)
    result = await run_team_turn_with_interrupt(agent, prompt, config)
    return {"verification_plan": result, "messages": [AIMessage(content=result)]}


async def designer_verilog_node(
    state: RTLGraphState,
    agent: TeamAgent,
    injector=None,
    config: Optional[RunnableConfig] = None,  # noqa: UP045 - 与 simple.py 一致:LangGraph 注解判定要求该字符串形态
) -> RTLGraphState:
    """Designer 输出可综合 RTL 源码(多轮迭代时携带上轮验证反馈)。

    节点内拼装 task + 设计规格 + 验证计划 + 上轮反馈为 ``prompt_task``,
    渲染 ``verilog_design`` 模板 + 注入技能块,然后调
    ``run_team_turn_with_interrupt`` 流式执行;技能注入 match 文本为
    ``prompt_task``(对照原 DesignerAgent.averilog_design_task)。

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
    prompt = agent.render_template(agent.get_template("verilog_design"), task=prompt_task)
    if injector is not None:
        prompt = injector.inject_into_prompt(prompt, prompt_task)
    result = await run_team_turn_with_interrupt(agent, prompt, config)
    # 解析 designer 产出的 syn_filelist.f,得到本轮待交付 RTL 文件清单(相对路径),
    # 供下游 designer_file_check 校验存在/非空,以及 sim_exec_check 做覆盖率包含检查。
    configurable = ((config or {}).get("configurable", {}) if config else {})
    workspace = configurable.get("workspace_path")
    output_files = _parse_filelist(workspace, "scripts/syn_filelist.f")
    return {
        "rtl_code": result,
        "round": round_n + 1,
        "output_files": output_files,
        "messages": [AIMessage(content=result)],
    }


async def verification_check_node(
    state: RTLGraphState,
    agent: TeamAgent,
    injector=None,
    config: Optional[RunnableConfig] = None,  # noqa: UP045 - 与 designer_verilog_node 一致
) -> RTLGraphState:
    """Verification 对 RTL 输出 Testbench/验证报告,并强制输出验证结论标记。

    节点内拼装 task + 设计规格 + 验证计划 + 待验证 RTL 为 ``prompt_task``,
    渲染 ``verilog_design`` 模板 + 注入技能块,然后调
    ``run_team_turn_with_interrupt`` 流式执行;技能注入 match 文本为
    ``prompt_task``(对照原 VerificationAgent.averilog_design_task)。

    提示词末尾要求输出"验证结论: PASS / FAIL"行,供下一轮 designer_verilog 携带反馈;
    但迭代环的真正路由由 sim_exec_check(实际执行仿真+覆盖率兜底)判定,而非 LLM 结论本身。
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
    prompt = agent.render_template(agent.get_template("verilog_design"), task=prompt_task)
    if injector is not None:
        # 排除 vivado-2025.2 合成/实现流技能:它用 add_files 目录 glob,与 Xsim
        # 基于 sim_filelist.f 的仿真编译冲突,会带偏 start.tcl 生成。
        prompt = injector.inject_into_prompt(prompt, prompt_task, exclude_skills=("vivado-2025.2",))
    result = await run_team_turn_with_interrupt(agent, prompt, config)
    return {"verification_report": result, "messages": [AIMessage(content=result)]}


# ---- 校验节点(代码型,不调用 LLM) ----
async def designer_file_check_node(
    state: RTLGraphState,
    agent: TeamAgent,
    injector=None,
    config: Optional[RunnableConfig] = None,  # noqa: UP045 - 与 designer_verilog_node 一致
) -> RTLGraphState:
    """designer_verilog 之后:校验 output_files 中每个 RTL 文件存在且非空。

    文件清单由 designer_verilog_node 解析 scripts/syn_filelist.f 得到。
    缺失或空文件 → file_check_passed=False;全部通过 → True。失败信息写入
    messages 供下一轮 designer_verilog 消费(提示补齐缺失文件)。
    """
    configurable = ((config or {}).get("configurable", {}) if config else {})
    workspace = configurable.get("workspace_path")
    files = state.get("output_files", [])
    missing, empty = [], []
    for rel in files:
        abs_p = rel if os.path.isabs(rel) else os.path.join(workspace, rel) if workspace else rel
        if not os.path.exists(abs_p):
            missing.append(rel)
        elif os.path.getsize(abs_p) == 0:
            empty.append(rel)
    passed = not (missing or empty)
    if passed:
        detail = "文件检查通过: 全部 RTL 文件存在且非空"
    else:
        detail = f"文件检查失败: 缺失 {missing or '无'}, 空文件 {empty or '无'}"
    return {"file_check_passed": passed, "messages": [AIMessage(content=detail)]}


async def sim_exec_check_node(
    state: RTLGraphState,
    agent: TeamAgent,
    injector=None,
    config: Optional[RunnableConfig] = None,  # noqa: UP045 - 与 designer_verilog_node 一致
) -> RTLGraphState:
    """verification_check 之后:实际执行 scripts/start.tcl 并做覆盖率兜底。

    1. 用 asyncio.create_subprocess_exec 调用 Vivado batch 跑 start.tcl(不阻塞事件循环);
    2. 退出码 0 视为仿真运行成功;
    3. 覆盖率兜底(编译 inclusion):scripts/syn_filelist.f 的全部 src 文件必须被
       scripts/sim_filelist.f 包含,且覆盖率报告 ./reports/coverage_report.txt 已生成;
    4. 完整日志写入 workspace/logs/sim_exec.log,state 仅存路径与结论摘要(避免
       海量 stdout 撑大 checkpoint)。
    """
    configurable = ((config or {}).get("configurable", {}) if config else {})
    workspace = configurable.get("workspace_path")
    if not workspace:
        return {
            "sim_check_passed": False,
            "sim_status": "ERROR",
            "sim_log_path": "",
            "messages": [AIMessage(content="sim_exec_check: 缺少 workspace_path,无法执行")],
        }
    start_tcl = os.path.join(workspace, "scripts", "start.tcl")
    if not os.path.exists(start_tcl):
        return {
            "sim_check_passed": False,
            "sim_status": "ERROR",
            "sim_log_path": "",
            "messages": [AIMessage(content="sim_exec_check: 未找到 scripts/start.tcl")],
        }
    # Vivado 可执行文件:环境变量 VIVADO_BIN 指定绝对路径,缺省回退 vivado(依赖 PATH)
    vivado_bin = os.environ.get("VIVADO_BIN") or "vivado"
    log_dir = os.path.join(workspace, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "sim_exec.log")
    try:
        proc = await asyncio.create_subprocess_exec(
            vivado_bin, "-mode", "batch", "-source", start_tcl,
            cwd=workspace,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        log_text = stdout.decode("utf-8", errors="replace") if stdout else ""
        await asyncio.to_thread(_write_text_file, log_path, log_text)
        exit_code = proc.returncode
    except FileNotFoundError as exc:
        return {
            "sim_check_passed": False,
            "sim_status": "ERROR",
            "sim_log_path": "",
            "messages": [AIMessage(content=f"sim_exec_check: 未找到 Vivado 可执行文件({vivado_bin}): {exc}")],
        }
    except Exception as exc:  # 捕获具体异常,禁止裸 except
        return {
            "sim_check_passed": False,
            "sim_status": "ERROR",
            "sim_log_path": "",
            "messages": [AIMessage(content=f"sim_exec_check: 执行 Vivado 失败: {exc}")],
        }
    # 判定:退出码 0 + 覆盖率报告已生成 + syn_filelist ⊆ sim_filelist
    coverage_report = os.path.join(workspace, "reports", "coverage_report.txt")
    coverage_report_ok = os.path.exists(coverage_report)
    inclusion_ok = _coverage_inclusion_ok(workspace)
    passed = (exit_code == 0) and coverage_report_ok and inclusion_ok
    status = "PASS" if passed else "FAIL"
    detail = (
        f"仿真退出码={exit_code}, 覆盖率报告={'已生成' if coverage_report_ok else '缺失'}, "
        f"src 包含检查={'通过' if inclusion_ok else '未通过'}, 结论={status}"
    )
    return {
        "sim_check_passed": bool(passed),
        "sim_status": status,
        "sim_log_path": log_path,
        "messages": [AIMessage(content=detail)],
    }


async def designer_output_node(
    state: RTLGraphState,
    agent: TeamAgent,
    injector=None,
    config: Optional[RunnableConfig] = None,  # noqa: UP045 - 与 designer_verilog_node 一致
) -> RTLGraphState:
    """Designer 基于最终 RTL 与验证报告整理交付文件,输出最终答案。

    节点内拼装 task + 设计规格 + 最终 RTL + 验证报告为 ``prompt_task``,
    渲染 ``verilog_design`` 模板 + 注入技能块,然后调
    ``run_team_turn_with_interrupt`` 流式执行;技能注入 match 文本为
    ``prompt_task``(对照原 DesignerAgent.averilog_design_task)。
    """
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
    prompt = agent.render_template(agent.get_template("verilog_design"), task=prompt_task)
    if injector is not None:
        prompt = injector.inject_into_prompt(prompt, prompt_task)
    result = await run_team_turn_with_interrupt(agent, prompt, config)
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


def route_after_file_check(state: RTLGraphState) -> str:
    """designer_file_check 后的条件路由:文件校验通过 → verification_check;
    失败且未达轮次上限 → 回 designer_verilog 重做;达上限 → END(防死循环)。"""
    if state.get("file_check_passed"):
        return "verification_check"
    if state.get("round", 0) >= state.get("max_rounds", 3):
        return END
    return "designer_verilog"


def route_after_sim_check(state: RTLGraphState) -> str:
    """sim_exec_check 后的条件路由:仿真+覆盖率通过 → END;
    失败且未达轮次上限 → 回 designer_verilog 重做;达上限 → END(防死循环)。"""
    if state.get("sim_check_passed"):
        return END
    if state.get("round", 0) >= state.get("max_rounds", 3):
        return END
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

    # 技能注入器:节点渲染 prompt 时追加匹配的技能指引块
    injector = SkillInjector(
        skills_dir=skills_dir,
        auto_match=auto_match_skills,
    )

    # compaction 中间件:消息通道超阈值时节点级增量压缩(agent 无 llm 时禁用)
    compaction_mw = _build_compaction_middleware(manager, compaction_config)

    builder = StateGraph(RTLGraphState)

    # 添加节点(声明式 NodeSpec 表:partial 绑定 + compaction 包装 + add_node 三步合一)
    # 注意:register_nodes 内部用 functools.partial 绑定 agent 实例;提示词模板由节点内懒加载
    # partial 保留 async 函数的 coroutine 特征(LangGraph 据此判定节点为异步并 await),
    # lambda 会返回未 await 的 coroutine 导致 InvalidUpdateError
    register_nodes(
        builder,
        agents,
        injector,
        compaction_mw,
        [
            NodeSpec("summarize", summarize_context, role="manager"),
            NodeSpec("architect_plan", architect_plan_node, role="architect"),
            NodeSpec("architect_design", architect_design_node, role="architect"),
            NodeSpec("architect_analyze", architect_analyze_node, role="architect"),
            NodeSpec("architect_review", architect_review_node, role="architect"),
            NodeSpec("architect_spec", architect_spec_node, role="architect"),
            NodeSpec("designer_spec", designer_spec_node, role="rtl_designer"),
            NodeSpec("verification_plan", verification_plan_node, role="rtl_verification"),
            NodeSpec("designer_verilog", designer_verilog_node, role="rtl_designer"),
            NodeSpec("verification_check", verification_check_node, role="rtl_verification"),
            NodeSpec("designer_file_check", designer_file_check_node, role="rtl_designer"),
            NodeSpec("sim_exec_check", sim_exec_check_node, role="rtl_verification"),
        ],
    )

    # 添加边: START → summarize → architect 五阶段 → designer_spec → verification_plan
    #        → designer_verilog → verification_check →(条件) END 或回 designer_verilog
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
            "designer_verilog": "designer_verilog",
        },
    )
    builder.add_edge("verification_check", "sim_exec_check")
    builder.add_conditional_edges(
        "sim_exec_check",
        route_after_sim_check,
        {
            END: END,
            "designer_verilog": "designer_verilog",
        },
    )

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
