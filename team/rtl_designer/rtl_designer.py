"""
Designer Agent - 数字芯片 RTL 设计工程师,负责规格梳理、模块拆分、Filelist 生成与可综合 RTL 编码
"""
from __future__ import annotations

import os
from typing import Any, ClassVar, Optional

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from graph.common import register_agent
from team.base import TeamAgent
from team.checker.checker import parse_filelist
from team.node_factory import create_llm_node
from tools.human_confirmation import request_user_confirmation


# ---- 上游产物拼接辅助:承载节点原有的 parts 逻辑,供模板变量与技能匹配文本复用 ----
def _designer_spec_task(s: dict[str, Any]) -> str:
    task = s["task"]
    arch_spec = s.get("arch_spec", "")
    return f"{task}\n\n【架构规格文档】\n{arch_spec}" if arch_spec else task


def _designer_verilog_task(s: dict[str, Any]) -> str:
    """Designer RTL 编码节点的任务文本:task + 设计规格 + 验证计划 + 上轮验证反馈。

    与手写节点逐字一致(拼装逻辑承载原 parts 逻辑,供模板变量与技能匹配文本复用):
    仅当 ``round > 0`` 且存在 ``verification_report`` 时追加反馈小节。
    """
    task = s["task"]
    design_spec = s.get("design_spec", "")
    vplan = s.get("verification_plan", "")
    report = s.get("verification_report", "")
    round_n = s.get("round", 0)
    parts = [task]
    if design_spec:
        parts.append(f"【设计规格与Filelist】\n{design_spec}")
    if vplan:
        parts.append(f"【验证计划】\n{vplan}")
    if round_n > 0 and report:
        parts.append(f"【第 {round_n} 轮验证报告反馈(请据此修正 RTL)】\n{report}")
    return "\n\n".join(parts)


def _designer_verilog_extra(
    state: dict[str, Any],
    result: str,
    config: Optional[RunnableConfig] = None,  # noqa: UP045 - 与 node_fn 注解形态一致
) -> dict[str, Any]:
    """Designer RTL 编码节点的额外返回字段:round 递增 + output_files 解析。

    解析 designer 产出的 ``scripts/syn_filelist.f``,得到本轮待交付 RTL 文件
    清单(相对路径),供下游 designer_file_check 校验存在/非空,以及
    sim_exec_check 做覆盖率包含检查。workspace 取自
    ``config.configurable.workspace_path``,缺失时优雅降级为空列表。
    """
    workspace = ((config or {}).get("configurable", {}) or {}).get("workspace_path")
    return {
        "round": state.get("round", 0) + 1,
        "output_files": parse_filelist(workspace, "scripts/syn_filelist.f"),
    }


@register_agent(
    "rtl_designer",
    tools=[request_user_confirmation],
    mcp_tools=[
        "write_file",
        "edit_file",
        "list_directory",
        "read_file",
        "delete_file",
        "create_directory",
        "delete_directory"
    ],
)
class DesignerAgent(TeamAgent):
    """设计师 Agent。工作流提示词见 AGENT.md 的 `## workflow:*` 小节,LLM 参数见 team_agents.json。"""

    # 本角色产出的 workflow state 通道（供 graph 动态组合 State schema）
    state_fields: ClassVar[dict[str, type]] = {
        "design_spec": str,
        "rtl_code": str,
        "output_files": list[str],
    }

    # ---- 工作流 LLM 节点(create_llm_node 闭包作为类属性) ----
    # graph/rtl_graph.py 的 NodeSpec 以类属性引用(DesignerAgent.spec_node 等)
    # 取未绑定函数;切勿经实例引用(描述符协议会把 self 绑到 state 参数位)。
    # base_prompts="" 使 prompt 与历史手写版逐字一致(不注入 load_agent_rules)。
    spec_node = create_llm_node(
        template_name="spec_design",
        output_field="design_spec",
        template_vars_fn=lambda s: {"task": _designer_spec_task(s)},
        match_text_fn=_designer_spec_task,
        base_prompts="",
    )

    verilog_node = create_llm_node(
        template_name="verilog_design",
        output_field="rtl_code",
        template_vars_fn=lambda s: {"task": _designer_verilog_task(s)},
        match_text_fn=_designer_verilog_task,
        extra_return_fn=_designer_verilog_extra,
        base_prompts="",
    )

    # ---- 工作流节点方法(由 graph/rtl_graph.py 经 register_nodes bound-method 分支注册) ----

    async def designer_file_check_node(
        self,
        state: dict[str, Any],
        injector=None,
        config: Optional[RunnableConfig] = None,  # noqa: UP045 - LangGraph 注解判定仅接受该字符串形态
    ) -> dict[str, Any]:
        """designer_verilog 之后:校验 output_files 中每个 RTL 文件存在且非空。

        文件清单由 designer_verilog 节点解析 scripts/syn_filelist.f 得到。
        缺失或空文件 → file_check_passed=False;全部通过 → True。失败信息写入
        messages 供下一轮 designer_verilog 消费(提示补齐缺失文件)。

        纯代码型节点(不调 LLM/模板),self 仅承载方法归属,不读取实例状态。
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

    async def designer_output_node(
        self,
        state: dict[str, Any],
        injector=None,
        config: Optional[RunnableConfig] = None,  # noqa: UP045 - LangGraph 注解判定仅接受该字符串形态
    ) -> dict[str, Any]:
        """Designer 交付节点:机械拼装最终交付物(零 LLM 调用)。

        本节点不调用任何 LLM、不渲染模板、不做网络/磁盘 IO,仅做确定性字符串拼装:
          - 交付文件清单:取 ``state["output_files"]``(非空时),逐行 ``- <path>``
          - 正文小节:【任务】【设计规格与Filelist】【最终 RTL 源码】【验证报告】
          - 任一字段为空则整段省略(不输出空标题)
        返回 ``{"final_answer": <拼装文本>, "messages": [AIMessage(content=<拼装文本>)]}``;
        ``final_answer`` 恒非空(全部字段为空时至少回退到 ``state["task"]``)。

        纯代码型节点,self 仅承载方法归属,不读取实例状态(零 LLM 契约由
        tests/graph/test_rtl_output_wiring.py 守护)。
        """
        parts: list[str] = []

        files = state.get("output_files") or []
        if files:
            parts.append("【交付文件清单】")
            parts.extend(f"- {p}" for p in files)

        task = state.get("task", "")
        if task:
            parts.append(f"【任务】\n{task}")
        design_spec = state.get("design_spec", "")
        if design_spec:
            parts.append(f"【设计规格与Filelist】\n{design_spec}")
        rtl_code = state.get("rtl_code", "")
        if rtl_code:
            parts.append(f"【最终 RTL 源码】\n{rtl_code}")
        verification_report = state.get("verification_report", "")
        if verification_report:
            parts.append(f"【验证报告】\n{verification_report}")

        assembled = "\n\n".join(parts)
        if not assembled.strip():
            # 全部字段为空时仍须返回非空交付文本,至少包含任务描述(禁止返回空串)
            assembled = f"【任务】\n{task}" if task else "【任务】\n(无任务描述)"
        return {"final_answer": assembled, "messages": [AIMessage(content=assembled)]}
