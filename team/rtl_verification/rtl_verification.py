"""
Verification Agent - 数字芯片 RTL 验证工程师,负责验证需求梳理、验证计划、Testbench/UVM 开发与 Vivado Xsim 仿真
"""
from __future__ import annotations

import asyncio
import os
from typing import Any, ClassVar, Optional

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from graph.common import register_agent
from team.base import TeamAgent
from team.checker.checker import coverage_inclusion_ok, write_text_file
from team.node_factory import create_llm_node
from tools.human_confirmation import request_user_confirmation


# ---- 上游产物拼接辅助:承载节点原有的 parts 逻辑,供模板变量与技能匹配文本复用 ----
def _verification_plan_task(s: dict[str, Any]) -> str:
    task = s["task"]
    arch_spec = s.get("arch_spec", "")
    design_spec = s.get("design_spec", "")
    parts = [task]
    if arch_spec:
        parts.append(f"【架构规格文档】\n{arch_spec}")
    if design_spec:
        parts.append(f"【设计规格与Filelist】\n{design_spec}")
    return "\n\n".join(parts)


def _verification_check_task(s: dict[str, Any]) -> str:
    task = s["task"]
    design_spec = s.get("design_spec", "")
    vplan = s.get("verification_plan", "")
    rtl = s.get("rtl_code", "")
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
    return "\n\n".join(parts)


@register_agent(
    "rtl_verification",
    tools=[request_user_confirmation],
    mcp_tools=["write_file","edit_file","list_directory","read_file","delete_file","create_directory","delete_directory"],
)
class VerificationAgent(TeamAgent):
    """验证师 Agent。工作流提示词见 AGENT.md 的 `## workflow:*` 小节,LLM 参数见 team_agents.json。"""

    # 验证环境始终使用 Vivado Xsim 仿真,与任务关键词无关,故作为角色级固定技能
    # 无条件注入(经 create_llm_node 读取 agent.fixed_skills 传入 injector)。
    # check_node 例外:该节点经 exclude_skills=("vivado-2025.2",)
    # 显式 opt-out(其 add_files 目录通配与 Xsim sim_filelist.f 流程冲突)。
    fixed_skills: ClassVar[list[str]] = ["vivado-2025.2"]

    # 本角色产出的 workflow state 通道（供 graph 动态组合 State schema）
    state_fields: ClassVar[dict[str, type]] = {
        "verification_plan": str,
        "verification_report": str,
    }

    # ---- 工作流 LLM 节点(create_llm_node 闭包作为类属性) ----
    # graph/rtl_graph.py 的 NodeSpec 以类属性引用(VerificationAgent.plan_node 等)
    # 取未绑定函数;切勿经实例引用(描述符协议会把 self 绑到 state 参数位)。
    # base_prompts="" 使 prompt 与历史手写版逐字一致(不注入 load_agent_rules)。
    plan_node = create_llm_node(
        template_name="spec_design",
        output_field="verification_plan",
        template_vars_fn=lambda s: {"task": _verification_plan_task(s)},
        match_text_fn=_verification_plan_task,
        base_prompts="",
    )

    check_node = create_llm_node(
        template_name="verilog_design",
        output_field="verification_report",
        template_vars_fn=lambda s: {"task": _verification_check_task(s)},
        match_text_fn=_verification_check_task,
        exclude_skills=("vivado-2025.2",),
        base_prompts="",
    )

    # ---- 工作流节点方法(由 graph/rtl_graph.py 经 register_nodes bound-method 分支注册) ----

    async def sim_exec_check_node(
        self,
        state: dict[str, Any],
        injector=None,
        config: Optional[RunnableConfig] = None,  # noqa: UP045 - LangGraph 注解判定仅接受该字符串形态
    ) -> dict[str, Any]:
        """verification_check 之后:实际执行 scripts/start.tcl 并做覆盖率兜底。

        1. 用 asyncio.create_subprocess_exec 调用 Vivado batch 跑 start.tcl(不阻塞事件循环);
        2. 退出码 0 视为仿真运行成功;
        3. 覆盖率兜底(编译 inclusion):scripts/syn_filelist.f 的全部 src 文件必须被
           scripts/sim_filelist.f 包含,且覆盖率报告 ./reports/coverage_report.txt 已生成;
        4. 完整日志写入 workspace/logs/sim_exec.log,state 仅存路径与结论摘要(避免
           海量 stdout 撑大 checkpoint)。

        纯代码型节点(不调 LLM/模板),self 仅承载方法归属,不读取实例状态。
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
        vivado_bin = os.environ.get("VIVADO_BIN") or "D:\\AMDTools\\2025.2\\Vivado\\bin\\vivado.bat"
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
            await asyncio.to_thread(write_text_file, log_path, log_text)
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
        inclusion_ok = coverage_inclusion_ok(workspace)
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
