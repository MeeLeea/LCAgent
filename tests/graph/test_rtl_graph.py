"""
测试 RTL 芯片设计流水线工作流 (graph/rtl_graph.py)

覆盖:
- 各角色节点函数(manager 提炼 / architect 五阶段 / designer 规格+编码 / verifier 验证)
- 验证结论解析与条件路由(多轮交互判定)
- 图构建(节点/边/条件边)
- 多轮交互:一次通过 / FAIL→PASS 迭代 / 达 max_rounds 强制交付
- 工作流注册与 RTL 角色注册
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field

from langgraph.graph import END

from agent.turn_types import AgentTurnResult
from graph.rtl_graph import (
    RTLGraphState,
    _coverage_inclusion_ok,
    _parse_filelist,
    architect_analyze_node,
    architect_design_node,
    architect_plan_node,
    architect_review_node,
    architect_spec_node,
    arun_rtl_graph_workflow,
    build_rtl_graph_workflow,
    designer_file_check_node,
    designer_output_node,
    designer_spec_node,
    designer_verilog_node,
    route_after_file_check,
    route_after_sim_check,
    sim_exec_check_node,
    summarize_context,
    verification_check_node,
    verification_passed,
    verification_plan_node,
)
from team.base import TeamAgent

# 默认模板(与各角色类的 default_templates 一致,供 FakeRTLAgent.get_template 兜底)
DEFAULT_TEMPLATES: dict[str, str] = {
    "summarize_context": "你是一个工作流上下文提炼助手。",
    "architect_plan": "请为以下芯片架构设计任务制定详细的执行计划:\n\n{task}\n\n",
    "architect_design": "请为以下任务完成芯片架构方案设计:\n\n{task}\n\n",
    "architect_analyze": "请对以下架构方案进行权衡分析(性能-面积-功耗PPA、风险与瓶颈):\n\n{task}\n\n",
    "architect_review": "请从架构、RTL实现、后端物理、软件驱动、验证多维度评审以下方案:\n\n{task}\n\n",
    "architect_spec": "请将以下架构方案整理为可直接交付RTL开发的规格文档:\n\n{task}\n\n",
    "spec_design": "请根据以下任务完成 RTL 设计前的规格梳理与工程规划:\n\n{task}\n\n",
    "verilog_design": "请根据以下任务与上下文输出可综合的 SystemVerilog RTL 源码:\n\n{task}\n\n",
}


@dataclass
class FakeRTLAgent:
    """模拟 TeamAgent,不联网;节点改调 run_team_turn_with_interrupt 后,
    统一经 ``arun_structured`` 入口记录调用并返回 ``AgentTurnResult.completed``。

    sequence 为 verification_check 节点的按序响应(耗尽后回退 response),
    供多轮迭代测试设置不同轮次的返回值(如 verifier 的 FAIL→PASS 序列);
    通过 prompt 含"待验证 RTL 源码"特征识别 verification_check 节点
    (其他节点回退 response,与原 verilog_sequence 语义一致)。
    """

    # verification_check 节点 prompt 的特征字符串(节点模板固定,稳定可识别)
    _CHECK_PROMPT_MARKER: str = "待验证 RTL 源码"

    name: str = "test-agent"
    response: str = "fake response"
    sequence: list[str] = field(default_factory=list)  # verification_check 专用序列
    calls: list[tuple[str, str]] = field(default_factory=list)

    def _is_check_node(self, prompt: str) -> bool:
        """识别是否为 verification_check 节点的 prompt(含"待验证 RTL 源码"特征)"""
        return self._CHECK_PROMPT_MARKER in prompt

    def _next(self, prompt: str) -> str:
        if self._is_check_node(prompt) and self.sequence:
            return self.sequence.pop(0)
        return self.response

    def get_template(self, name: str) -> str:
        """懒加载模板:复用 TeamAgent 静态解析(无 prompt_file 时回退默认模板)"""
        return DEFAULT_TEMPLATES.get(name, "")

    def render_template(self, template: str, **kwargs) -> str:
        """占位符替换(复用 TeamAgent 静态方法,与生产路径一致)"""
        return TeamAgent.render_template(template, **kwargs)

    def inject_into_prompt(self, prompt: str, task: str, active_names=()) -> str:
        """技能注入占位:测试不验证技能块,原样返回(prompt 已含渲染内容)"""
        return prompt

    async def arun_structured(self, task: str, config=None) -> AgentTurnResult:
        """节点经 run_team_turn_with_interrupt 调用的唯一入口;
        记录 (arun_structured, task) 并返回 completed(self._next(task))"""
        self.calls.append(("arun_structured", task))
        return AgentTurnResult.completed(self._next(task))


def build_fake_agents(
    manager_response: str = "摘要",
    arch_response: str = "架构输出",
    designer_response: str = "RTL 代码",
    verifier_sequence: list[str] | None = None,
) -> dict:
    """构建四个角色的 FakeRTLAgent 字典"""
    return {
        "manager": FakeRTLAgent(name="manager", response=manager_response),
        "architect": FakeRTLAgent(name="architect", response=arch_response),
        "rtl_designer": FakeRTLAgent(name="rtl_designer", response=designer_response),
        "rtl_verification": FakeRTLAgent(
            name="rtl_verification",
            response=verifier_sequence[-1] if verifier_sequence else "验证结论: PASS",
            sequence=list(verifier_sequence or []),
        ),
    }


def initial_state(**overrides) -> RTLGraphState:
    """构造完整初始状态(带默认值)"""
    state: RTLGraphState = {
        "task": "设计一个 UART 模块",
        "raw_context": "",
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
        "max_rounds": 3,
        "final_answer": "",
    }
    state.update(overrides)
    return state


# ==================== 节点函数测试 ====================

def test_summarize_context_node():
    """Manager 提炼记忆上下文摘要"""
    manager = FakeRTLAgent(name="manager", response="摘要: 用户偏好中文")
    result = asyncio.run(summarize_context(initial_state(raw_context="历史对话"), manager))
    assert result["context_summary"] == "摘要: 用户偏好中文"
    # 节点改调 run_team_turn_with_interrupt → arun_structured,prompt 含模板+raw
    assert manager.calls[0][0] == "arun_structured"
    assert "历史对话" in manager.calls[0][1]


def test_architect_plan_node():
    """Architect 制定执行计划(注入上下文摘要)"""
    architect = FakeRTLAgent(name="architect", response="计划")
    state = initial_state(context_summary="摘要S")
    result = asyncio.run(architect_plan_node(state, architect))
    assert result["arch_plan"] == "计划"
    assert architect.calls[0][0] == "arun_structured"
    assert "设计一个 UART 模块" in architect.calls[0][1]


def test_architect_design_node_injects_plan():
    """Architect 设计节点把执行计划拼入任务"""
    architect = FakeRTLAgent(name="architect", response="方案")
    state = initial_state(arch_plan="计划P")
    result = asyncio.run(architect_design_node(state, architect))
    assert result["arch_design"] == "方案"
    prompt = architect.calls[0][1]
    assert "计划P" in prompt


def test_architect_analyze_node_injects_design():
    """Architect 分析节点把方案设计拼入任务"""
    architect = FakeRTLAgent(name="architect", response="分析")
    state = initial_state(arch_design="方案D")
    result = asyncio.run(architect_analyze_node(state, architect))
    assert result["arch_analysis"] == "分析"
    assert "方案D" in architect.calls[0][1]


def test_architect_review_node_injects_design_and_analysis():
    """Architect 评审节点拼入方案设计与分析"""
    architect = FakeRTLAgent(name="architect", response="评审")
    state = initial_state(arch_design="方案D", arch_analysis="分析A")
    result = asyncio.run(architect_review_node(state, architect))
    assert result["arch_review"] == "评审"
    prompt = architect.calls[0][1]
    assert "方案D" in prompt
    assert "分析A" in prompt


def test_architect_spec_node_injects_design_and_review():
    """Architect 规格节点拼入方案设计与评审"""
    architect = FakeRTLAgent(name="architect", response="规格")
    state = initial_state(arch_design="方案D", arch_review="评审R")
    result = asyncio.run(architect_spec_node(state, architect))
    assert result["arch_spec"] == "规格"
    prompt = architect.calls[0][1]
    assert "方案D" in prompt
    assert "评审R" in prompt


def test_designer_spec_node_injects_arch_spec():
    """Designer 规格节点拼入架构规格"""
    designer = FakeRTLAgent(name="rtl_designer", response="设计规格")
    state = initial_state(arch_spec="规格S")
    result = asyncio.run(designer_spec_node(state, designer))
    assert result["design_spec"] == "设计规格"
    assert "规格S" in designer.calls[0][1]


def test_verification_plan_node_injects_specs():
    """Verification 计划节点拼入架构规格与设计规格"""
    verifier = FakeRTLAgent(name="rtl_verification", response="验证计划")
    state = initial_state(arch_spec="规格S", design_spec="设计规格D")
    result = asyncio.run(verification_plan_node(state, verifier))
    assert result["verification_plan"] == "验证计划"
    prompt = verifier.calls[0][1]
    assert "规格S" in prompt
    assert "设计规格D" in prompt


def test_designer_verilog_node_increments_round():
    """Designer 编码节点输出 RTL 并递增轮次"""
    designer = FakeRTLAgent(name="rtl_designer", response="module uart;")
    state = initial_state(design_spec="规格D", verification_plan="计划V")
    result = asyncio.run(designer_verilog_node(state, designer))
    assert result["rtl_code"] == "module uart;"
    assert result["round"] == 1
    assert "规格D" in designer.calls[0][1]


def test_designer_verilog_node_injects_feedback_on_later_rounds():
    """Designer 第二轮编码时拼入上一轮验证反馈"""
    designer = FakeRTLAgent(name="rtl_designer", response="module uart;")
    state = initial_state(
        design_spec="规格D",
        verification_plan="计划V",
        verification_report="验证结论: FAIL\n波特率错误",
        round=1,
    )
    asyncio.run(designer_verilog_node(state, designer))
    prompt = designer.calls[0][1]
    assert "验证结论: FAIL" in prompt
    assert "波特率错误" in prompt
    assert "第 1 轮" in prompt


def test_verification_check_node_requires_verdict_marker():
    """Verification 检查节点输出验证报告,并强制要求结论标记"""
    verifier = FakeRTLAgent(name="rtl_verification", response="验证结论: PASS\n无问题")
    state = initial_state(design_spec="规格D", rtl_code="module uart;")
    result = asyncio.run(verification_check_node(state, verifier))
    assert result["verification_report"] == "验证结论: PASS\n无问题"
    prompt = verifier.calls[0][1]
    assert "module uart;" in prompt
    assert "验证结论: PASS" in prompt  # 强制结论行要求出现在提示词中


def test_designer_output_node_produces_final_answer():
    """Designer 交付节点输出最终交付物"""
    designer = FakeRTLAgent(name="rtl_designer", response="最终交付: filelist + rtl")
    state = initial_state(design_spec="规格D", rtl_code="module uart;", verification_report="验证结论: PASS")
    result = asyncio.run(designer_output_node(state, designer))
    assert result["final_answer"] == "最终交付: filelist + rtl"
    prompt = designer.calls[0][1]
    assert "module uart;" in prompt
    assert "验证结论: PASS" in prompt


# ==================== 验证结论解析与条件路由 ====================

def test_verification_passed_marker():
    """显式结论标记:PASS 通过,FAIL 不通过"""
    assert verification_passed("...\n验证结论: PASS\n...")
    assert not verification_passed("...\n验证结论: FAIL\n...")
    # 无标记时全文含 PASS 且不含 FAIL → 通过
    assert verification_passed("所有测试用例均 PASS")
    # 含 FAIL 且含 PASS → 不通过
    assert not verification_passed("用例1 PASS,用例2 FAIL")


def test_verification_passed_empty():
    """空报告视为未通过"""
    assert not verification_passed("")
    assert not verification_passed(None)


# ==================== 图构建测试 ====================

def test_build_rtl_graph_workflow_nodes():
    """构建 RTL 流水线工作流图,验证全部节点存在"""
    agents = build_fake_agents()
    graph = build_rtl_graph_workflow(agents)
    assert graph is not None
    node_names = [n.id for n in graph.get_graph().nodes.values()]
    expected = [
        "summarize",
        "architect_plan",
        "architect_design",
        "architect_analyze",
        "architect_review",
        "architect_spec",
        "designer_spec",
        "verification_plan",
        "designer_verilog",
        "designer_file_check",
        "verification_check",
        "sim_exec_check",
    ]
    for node in expected:
        assert node in node_names


def test_build_rtl_graph_workflow_edges():
    """验证线性边连接正确"""
    agents = build_fake_agents()
    graph = build_rtl_graph_workflow(agents)
    edges = {(e.source, e.target) for e in graph.get_graph().edges}
    assert ("__start__", "summarize") in edges
    assert ("summarize", "architect_plan") in edges
    assert ("architect_plan", "architect_design") in edges
    assert ("architect_design", "architect_analyze") in edges
    assert ("architect_analyze", "architect_review") in edges
    assert ("architect_review", "architect_spec") in edges
    assert ("architect_spec", "designer_spec") in edges
    assert ("designer_spec", "verification_plan") in edges
    assert ("verification_plan", "designer_verilog") in edges
    assert ("designer_verilog", "designer_file_check") in edges
    assert ("designer_file_check", "verification_check") in edges
    assert ("verification_check", "sim_exec_check") in edges


def test_build_rtl_graph_workflow_conditional_edges():
    """验证多轮交互条件边存在(验证检查 → 交付/回环)"""
    agents = build_fake_agents()
    graph = build_rtl_graph_workflow(agents)
    edges = {(e.source, e.target) for e in graph.get_graph().edges}
    assert ("designer_file_check", "verification_check") in edges
    assert ("designer_file_check", "designer_verilog") in edges
    assert ("sim_exec_check", "__end__") in edges
    assert ("sim_exec_check", "designer_verilog") in edges


def test_build_rtl_graph_workflow_max_rounds_param():
    """max_rounds 参数传入不影响编译"""
    agents = build_fake_agents()
    graph = build_rtl_graph_workflow(agents, max_rounds=5)
    assert graph is not None

# ==================== 多轮交互运行测试 ====================


class _FakeProc:
    """模拟 asyncio 子进程:communicate 返回 (stdout, stderr),returncode 可控。"""

    def __init__(self, rc: int):
        self.returncode = rc

    async def communicate(self):
        return b"", b""


def _make_workspace(tmp_path) -> str:
    """构造最小可运行工作空间(含 filelist / start.tcl / 覆盖率报告)。"""
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "src").mkdir()
    (ws / "src" / "top.sv").write_text("module top; endmodule\n")
    (ws / "scripts").mkdir()
    (ws / "scripts" / "syn_filelist.f").write_text("src/top.sv\n")
    (ws / "scripts" / "sim_filelist.f").write_text("src/top.sv\n")
    (ws / "scripts" / "start.tcl").write_text("# mock start.tcl\n")
    (ws / "reports").mkdir()
    (ws / "reports" / "coverage_report.txt").write_text("coverage data\n")
    return str(ws)


def _patch_vivado(monkeypatch, rc_seq):
    """monkeypatch asyncio.create_subprocess_exec:按 rc_seq 顺序返回退出码。"""
    calls = {"n": 0}
    seq = list(rc_seq)

    async def fake(*args, **kwargs):
        i = calls["n"]
        calls["n"] += 1
        rc = seq[i] if i < len(seq) else seq[-1]
        return _FakeProc(rc)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake)


def test_run_rtl_graph_pass_once(tmp_path, monkeypatch):
    """验证一次通过:designer_verilog 只执行一次,仿真驱动路由直接交付。"""
    manager = FakeRTLAgent(name="manager", response="摘要: 用户偏好中文")
    architect = FakeRTLAgent(name="architect", response="架构: 单核 UART")
    designer = FakeRTLAgent(name="rtl_designer", response="module uart;")
    verifier = FakeRTLAgent(
        name="rtl_verification",
        response="验证结论: PASS\nRTL 无问题",
    )
    agents = {
        "manager": manager,
        "architect": architect,
        "rtl_designer": designer,
        "rtl_verification": verifier,
    }
    graph = build_rtl_graph_workflow(agents)
    ws = _make_workspace(tmp_path)
    _patch_vivado(monkeypatch, [0])

    result = asyncio.run(arun_rtl_graph_workflow(graph, "设计一个 UART 模块", workspace_path=ws))

    assert result["round"] == 1
    assert result["arch_plan"] == "架构: 单核 UART"
    assert result["arch_design"] == "架构: 单核 UART"
    assert result["arch_analysis"] == "架构: 单核 UART"
    assert result["arch_review"] == "架构: 单核 UART"
    assert result["arch_spec"] == "架构: 单核 UART"
    assert result["design_spec"] == "module uart;"
    assert result["verification_plan"] == "验证结论: PASS\nRTL 无问题"
    assert result["verification_report"] == "验证结论: PASS\nRTL 无问题"
    # 仿真通过 → sim_check_passed 为 True(路由据此进入 END)
    assert result["sim_check_passed"] is True

    designer_calls = [c[0] for c in designer.calls]
    assert designer_calls == ["arun_structured", "arun_structured"]
    verifier_calls = [c[0] for c in verifier.calls]
    assert verifier_calls == ["arun_structured", "arun_structured"]


def test_run_rtl_graph_multi_round_iteration(tmp_path, monkeypatch):
    """验证仿真失败→成功多轮迭代:sim 第一次退出码 1,第二次 0,designer_verilog 执行两次。"""
    manager = FakeRTLAgent(name="manager", response="摘要")
    architect = FakeRTLAgent(name="architect", response="架构方案")
    designer = FakeRTLAgent(name="rtl_designer", response="module uart;")
    verifier = FakeRTLAgent(
        name="rtl_verification",
        sequence=[
            "验证结论: FAIL\n波特率配置错误",
            "验证结论: PASS\n修改后正确",
        ],
    )
    agents = {
        "manager": manager,
        "architect": architect,
        "rtl_designer": designer,
        "rtl_verification": verifier,
    }
    graph = build_rtl_graph_workflow(agents, max_rounds=3)
    ws = _make_workspace(tmp_path)
    # 仿真第一次失败(退出码 1),第二次成功(退出码 0)→ 路由据此迭代两轮
    _patch_vivado(monkeypatch, [1, 0])

    result = asyncio.run(arun_rtl_graph_workflow(graph, "设计 UART", workspace_path=ws))

    assert result["round"] == 2
    # designer 的 verilog 类节点(designer_verilog)被调 2 次;
    # 用 verilog_design 模板特征过滤(含"可综合...RTL 源码")
    designer_prompts = [
        c[1] for c in designer.calls
        if c[0] == "arun_structured" and "可综合" in c[1]
    ]
    assert len(designer_prompts) == 2
    # 首轮设计无反馈,第二轮设计携带首轮 FAIL 反馈(路由由仿真而非 LLM 报告驱动)
    assert "验证结论: FAIL" not in designer_prompts[0]
    assert "验证结论: FAIL" in designer_prompts[1]
    # 仿真最终通过
    assert result["sim_check_passed"] is True


def test_run_rtl_graph_forced_output_at_max_rounds(tmp_path, monkeypatch):
    """验证仿真一直失败达 max_rounds 上限强制交付(不无限循环)。"""
    manager = FakeRTLAgent(name="manager", response="摘要")
    architect = FakeRTLAgent(name="architect", response="架构方案")
    designer = FakeRTLAgent(name="rtl_designer", response="module uart;")
    verifier = FakeRTLAgent(
        name="rtl_verification",
        response="验证结论: FAIL\n始终有问题",
    )
    agents = {
        "manager": manager,
        "architect": architect,
        "rtl_designer": designer,
        "rtl_verification": verifier,
    }
    graph = build_rtl_graph_workflow(agents, max_rounds=3)
    ws = _make_workspace(tmp_path)
    # 仿真三次均失败(退出码 1)→ 达 max_rounds 后强制交付
    _patch_vivado(monkeypatch, [1, 1, 1])

    result = asyncio.run(arun_rtl_graph_workflow(graph, "设计 UART", workspace_path=ws))

    # 3 次迭代后强制交付,round 达上限
    assert result["round"] == 3
    assert result["sim_check_passed"] is False
    # verifier 的 verification_check 被调 3 次(prompt 含"待验证 RTL 源码")
    verifier_check_calls = [
        c for c in verifier.calls
        if c[0] == "arun_structured" and "待验证 RTL 源码" in c[1]
    ]
    assert len(verifier_check_calls) == 3


def test_run_rtl_graph_with_context_memory(tmp_path, monkeypatch):
    """带记忆上下文运行:manager 提炼后注入 architect_plan;仿真通过一次交付。"""
    manager = FakeRTLAgent(name="manager", response="摘要: 用户偏好 SystemVerilog")
    architect = FakeRTLAgent(name="architect", response="架构方案")
    designer = FakeRTLAgent(name="rtl_designer", response="module uart;")
    verifier = FakeRTLAgent(name="rtl_verification", response="验证结论: PASS")
    agents = {
        "manager": manager,
        "architect": architect,
        "rtl_designer": designer,
        "rtl_verification": verifier,
    }
    graph = build_rtl_graph_workflow(agents)
    ws = _make_workspace(tmp_path)
    _patch_vivado(monkeypatch, [0])

    result = asyncio.run(
        arun_rtl_graph_workflow(
            graph,
            "设计 UART",
            raw_context="之前聊过偏好 SystemVerilog",
            workspace_path=ws,
        )
    )

    assert result["context_summary"] == "摘要: 用户偏好 SystemVerilog"
    # manager 调用:summarize 节点改调 arun_structured,prompt 含模板+raw
    assert manager.calls[0][0] == "arun_structured"
    assert "之前聊过偏好 SystemVerilog" in manager.calls[0][1]
    # architect_plan 收到上下文摘要(经 arun_structured)
    assert architect.calls[0][0] == "arun_structured"
    # 仿真通过一次交付
    assert result["sim_check_passed"] is True


# ==================== 注册测试 ====================

def test_rtl_graph_registered():
    """rtl_graph 工作流已注册且 roles 正确"""
    from graph.registry import WORKFLOWS

    assert "rtl_graph" in WORKFLOWS
    spec = WORKFLOWS["rtl_graph"]
    assert callable(spec["builder"])
    assert spec["runner"] is not None
    assert spec["roles"] == ["manager", "architect", "rtl_designer", "rtl_verification"]


def test_rtl_agents_registered():
    """rtl_designer / rtl_verification 角色已注册(由 rtl_graph 顶部导入触发)"""
    from graph.registry import AGENT_REGISTRY
    from team.base import TeamAgent

    for role in ("rtl_designer", "rtl_verification"):
        assert role in AGENT_REGISTRY
        assert issubclass(AGENT_REGISTRY[role]["agent_class"], TeamAgent)


def test_build_workflow_rtl_graph_via_registry():
    """通过 registry.build_workflow 构建 rtl_graph(验证角色构建链路)"""
    from graph.registry import build_workflow

    graph, agents = build_workflow("rtl_graph", checkpointer=None)
    assert graph is not None
    for role in ("manager", "architect", "rtl_designer", "rtl_verification"):
        assert role in agents


# ==================== 新增校验节点 / 辅助函数单元测试 ====================


def test_parse_filelist(tmp_path):
    """_parse_filelist 解析 .f,跳过注释/指令行,返回 .v/.sv token。"""
    fl = tmp_path / "syn.f"
    fl.write_text(
        "# comment\n"
        "src/top.sv\n"
        "// line comment\n"
        "+incdir+./inc\n"
        "-f other.f\n"
        "src/util.v\n"
        "src/pkg.vhd  # trailing\n"
    )
    out = _parse_filelist(str(tmp_path), "syn.f")
    assert out == ["src/top.sv", "src/util.v", "src/pkg.vhd"]


def test_parse_filelist_missing(tmp_path):
    """filelist 不存在时返回空列表。"""
    assert _parse_filelist(str(tmp_path), "nope.f") == []


def test_coverage_inclusion_ok(tmp_path):
    """_coverage_inclusion_ok:syn ⊆ sim 时为 True,否则 False。"""
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "syn_filelist.f").write_text("src/a.sv\n")
    (tmp_path / "scripts" / "sim_filelist.f").write_text("src/a.sv\nsrc/b.sv\n")
    assert _coverage_inclusion_ok(str(tmp_path)) is True
    (tmp_path / "scripts" / "sim_filelist.f").write_text("src/b.sv\n")
    assert _coverage_inclusion_ok(str(tmp_path)) is False


def test_coverage_inclusion_ok_no_syn(tmp_path):
    """syn_filelist.f 为空/不存在时返回 False。"""
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "sim_filelist.f").write_text("src/a.sv\n")
    assert _coverage_inclusion_ok(str(tmp_path)) is False


def test_route_after_file_check():
    """designer_file_check 路由:通过→verification_check;失败未达上限→回环;达上限→END。"""
    assert route_after_file_check({"file_check_passed": True, "round": 1, "max_rounds": 3}) == "verification_check"
    assert route_after_file_check({"file_check_passed": False, "round": 1, "max_rounds": 3}) == "designer_verilog"
    assert route_after_file_check({"file_check_passed": False, "round": 3, "max_rounds": 3}) == END


def test_route_after_sim_check():
    """sim_exec_check 路由:通过→END;失败未达上限→回环;达上限→END。"""
    assert route_after_sim_check({"sim_check_passed": True, "round": 1, "max_rounds": 3}) == END
    assert route_after_sim_check({"sim_check_passed": False, "round": 1, "max_rounds": 3}) == "designer_verilog"
    assert route_after_sim_check({"sim_check_passed": False, "round": 3, "max_rounds": 3}) == END


async def _run_file_check(state, workspace):
    agent = FakeRTLAgent()
    config = {"configurable": {"workspace_path": workspace}} if workspace else None
    return await designer_file_check_node(state, agent, config=config)


def test_designer_file_check_node_pass(tmp_path):
    """output_files 中文件均存在且非空 → file_check_passed True。"""
    (tmp_path / "src").mkdir()
    f = tmp_path / "src" / "top.sv"
    f.write_text("module top; endmodule\n")
    state = {"output_files": ["src/top.sv"]}
    result = asyncio.run(_run_file_check(state, str(tmp_path)))
    assert result["file_check_passed"] is True


def test_designer_file_check_node_missing(tmp_path):
    """output_files 含不存在文件 → file_check_passed False。"""
    state = {"output_files": ["src/missing.sv"]}
    result = asyncio.run(_run_file_check(state, str(tmp_path)))
    assert result["file_check_passed"] is False


def test_designer_file_check_node_empty_list(tmp_path):
    """output_files 为空 → 视为通过(无文件可校验)。"""
    state = {"output_files": []}
    result = asyncio.run(_run_file_check(state, str(tmp_path)))
    assert result["file_check_passed"] is True


def test_designer_file_check_node_no_workspace(tmp_path):
    """无 workspace 时,相对路径按 cwd 解析;绝对路径按绝对路径校验。"""
    # 绝对路径且文件存在 → 通过
    f = tmp_path / "abs.sv"
    f.write_text("module x; endmodule\n")
    state = {"output_files": [str(f)]}
    result = asyncio.run(_run_file_check(state, None))
    assert result["file_check_passed"] is True
    # 相对路径(cwd 下不存在) → 失败
    state2 = {"output_files": ["src/nope.sv"]}
    result2 = asyncio.run(_run_file_check(state2, None))
    assert result2["file_check_passed"] is False


async def _run_sim_check(workspace, monkeypatch, rc=0):
    agent = FakeRTLAgent()
    config = {"configurable": {"workspace_path": workspace}} if workspace else None

    async def fake(*args, **kwargs):
        return _FakeProc(rc)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake)
    return await sim_exec_check_node(dict(), agent, config=config)


def test_sim_exec_check_node_pass(tmp_path, monkeypatch):
    """start.tcl 存在 + vivado 退出码 0 + 覆盖率报告存在 + 包含检查通过 → 通过。"""
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "start.tcl").write_text("# tcl\n")
    (tmp_path / "scripts" / "syn_filelist.f").write_text("src/top.sv\n")
    (tmp_path / "scripts" / "sim_filelist.f").write_text("src/top.sv\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "top.sv").write_text("module top; endmodule\n")
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "coverage_report.txt").write_text("cov\n")
    result = asyncio.run(_run_sim_check(str(tmp_path), monkeypatch, rc=0))
    assert result["sim_check_passed"] is True
    assert result["sim_status"] == "PASS"


def test_sim_exec_check_node_fail_rc(tmp_path, monkeypatch):
    """vivado 退出码非 0 → 失败。"""
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "start.tcl").write_text("# tcl\n")
    (tmp_path / "scripts" / "syn_filelist.f").write_text("src/top.sv\n")
    (tmp_path / "scripts" / "sim_filelist.f").write_text("src/top.sv\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "top.sv").write_text("module top; endmodule\n")
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "coverage_report.txt").write_text("cov\n")
    result = asyncio.run(_run_sim_check(str(tmp_path), monkeypatch, rc=1))
    assert result["sim_check_passed"] is False
    assert result["sim_status"] == "FAIL"


def test_sim_exec_check_node_no_start_tcl(tmp_path, monkeypatch):
    """缺少 start.tcl → 失败(不调用 vivado)。"""
    (tmp_path / "scripts").mkdir()
    result = asyncio.run(_run_sim_check(str(tmp_path), monkeypatch, rc=0))
    assert result["sim_check_passed"] is False
    assert result["sim_status"] == "ERROR"


def test_sim_exec_check_node_no_workspace(monkeypatch):
    """无 workspace → 失败(ERROR),不执行 vivado。"""
    result = asyncio.run(_run_sim_check(None, monkeypatch, rc=0))
    assert result["sim_check_passed"] is False
    assert result["sim_status"] == "ERROR"


def test_sim_exec_check_node_missing_report(tmp_path, monkeypatch):
    """覆盖率报告缺失 → 失败。"""
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "start.tcl").write_text("# tcl\n")
    (tmp_path / "scripts" / "syn_filelist.f").write_text("src/top.sv\n")
    (tmp_path / "scripts" / "sim_filelist.f").write_text("src/top.sv\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "top.sv").write_text("module top; endmodule\n")
    # 不创建 reports/coverage_report.txt
    result = asyncio.run(_run_sim_check(str(tmp_path), monkeypatch, rc=0))
    assert result["sim_check_passed"] is False