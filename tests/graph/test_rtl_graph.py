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
from dataclasses import dataclass, field

from langgraph.graph import END

from agent.turn_types import AgentTurnResult
from graph.rtl_graph import (
    RTLGraphState,
    architect_analyze_node,
    architect_design_node,
    architect_plan_node,
    architect_review_node,
    architect_spec_node,
    arun_rtl_graph_workflow,
    build_rtl_graph_workflow,
    designer_output_node,
    designer_spec_node,
    designer_verilog_node,
    route_after_verification,
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


def test_route_after_verification_passed():
    """验证通过 → END"""
    state = initial_state(verification_report="验证结论: PASS", round=1)
    assert route_after_verification(state) == END


def test_route_after_verification_fail_continues():
    """验证未通过且未达上限 → 回 designer_verilog"""
    state = initial_state(verification_report="验证结论: FAIL", round=1)
    assert route_after_verification(state) == "designer_verilog"


def test_route_after_verification_max_rounds():
    """达轮次上限即使未通过也强制终止(END)"""
    state = initial_state(verification_report="验证结论: FAIL", round=3, max_rounds=3)
    assert route_after_verification(state) == END


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
        "verification_check",
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
    assert ("designer_verilog", "verification_check") in edges


def test_build_rtl_graph_workflow_conditional_edges():
    """验证多轮交互条件边存在(验证检查 → 交付/回环)"""
    agents = build_fake_agents()
    graph = build_rtl_graph_workflow(agents)
    edges = {(e.source, e.target) for e in graph.get_graph().edges}
    assert ("verification_check", "__end__") in edges
    assert ("verification_check", "designer_verilog") in edges


def test_build_rtl_graph_workflow_max_rounds_param():
    """max_rounds 参数传入不影响编译"""
    agents = build_fake_agents()
    graph = build_rtl_graph_workflow(agents, max_rounds=5)
    assert graph is not None


# ==================== 多轮交互运行测试 ====================

def test_run_rtl_graph_pass_once():
    """验证一次通过:designer_verilog 只执行一次,直接交付"""
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

    result = asyncio.run(arun_rtl_graph_workflow(graph, "设计一个 UART 模块"))

    assert result["round"] == 1
    assert result["arch_plan"] == "架构: 单核 UART"
    assert result["arch_design"] == "架构: 单核 UART"
    assert result["arch_analysis"] == "架构: 单核 UART"
    assert result["arch_review"] == "架构: 单核 UART"
    assert result["arch_spec"] == "架构: 单核 UART"
    assert result["design_spec"] == "module uart;"
    assert result["verification_plan"] == "验证结论: PASS\nRTL 无问题"
    assert result["verification_report"] == "验证结论: PASS\nRTL 无问题"

    # designer 调用:spec_design + verilog_design(编码)(交付节点已移除,验证通过直接 END)
    # 节点改调 run_team_turn_with_interrupt 后,统一经 arun_structured 入口
    designer_calls = [c[0] for c in designer.calls]
    assert designer_calls == ["arun_structured", "arun_structured"]
    # verifier 调用:spec_design(计划) + verilog_design(检查)
    verifier_calls = [c[0] for c in verifier.calls]
    assert verifier_calls == ["arun_structured", "arun_structured"]


def test_run_rtl_graph_multi_round_iteration():
    """验证 FAIL→PASS 多轮迭代:designer_verilog 执行两次,第二轮携带反馈"""
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

    result = asyncio.run(arun_rtl_graph_workflow(graph, "设计 UART"))

    assert result["round"] == 2
    # designer 的 verilog 类节点(designer_verilog)被调 2 次:
    # 2 轮迭代设计;交付节点已移除,验证通过直接 END。用 verilog_design 模板特征过滤(含"可综合...RTL 源码")
    designer_prompts = [
        c[1] for c in designer.calls
        if c[0] == "arun_structured" and "可综合" in c[1]
    ]
    assert len(designer_prompts) == 2
    # 首轮设计无反馈,第二轮设计携带 FAIL 反馈
    assert "验证结论: FAIL" not in designer_prompts[0]
    assert "验证结论: FAIL" in designer_prompts[1]


def test_run_rtl_graph_forced_output_at_max_rounds():
    """验证一直 FAIL 时达 max_rounds 上限强制交付(不无限循环)"""
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

    result = asyncio.run(arun_rtl_graph_workflow(graph, "设计 UART"))

    # 3 次迭代后强制交付,round 达上限
    assert result["round"] == 3
    # verifier 调用:1 次 verification_plan + 3 次 verification_check(共 4 次 arun_structured)
    # verification_check 的 prompt 含"待验证 RTL 源码",据此过滤 3 次
    verifier_check_calls = [
        c for c in verifier.calls
        if c[0] == "arun_structured" and "待验证 RTL 源码" in c[1]
    ]
    assert len(verifier_check_calls) == 3


def test_run_rtl_graph_with_context_memory():
    """带记忆上下文运行:manager 提炼后注入 architect_plan"""
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

    result = asyncio.run(
        arun_rtl_graph_workflow(graph, "设计 UART", raw_context="之前聊过偏好 SystemVerilog")
    )

    assert result["context_summary"] == "摘要: 用户偏好 SystemVerilog"
    # manager 调用:summarize 节点改调 arun_structured,prompt 含模板+raw
    assert manager.calls[0][0] == "arun_structured"
    assert "之前聊过偏好 SystemVerilog" in manager.calls[0][1]
    # architect_plan 收到上下文摘要(经 arun_structured)
    assert architect.calls[0][0] == "arun_structured"


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