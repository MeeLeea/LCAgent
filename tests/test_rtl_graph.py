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


@dataclass
class FakeRTLAgent:
    """模拟 TeamAgent,不联网;按方法名记录调用,支持顺序返回序列"""

    name: str = "test-agent"
    response: str = "fake response"
    sequence: list[str] = field(default_factory=list)  # 按序弹出,耗尽后回退 response
    verilog_sequence: list[str] = field(default_factory=list)  # verilog_design_task 专用序列
    calls: list[tuple[str, str]] = field(default_factory=list)

    def _next(self, method: str) -> str:
        seq = self.verilog_sequence if method == "verilog_design_task" else self.sequence
        if seq:
            return seq.pop(0)
        return self.response

    def _record(self, method: str, prompt: str) -> str:
        self.calls.append((method, prompt))
        return self._next(method)

    # ---- manager ----
    def summarize_context(self, memory_text: str) -> str:
        return self._record("summarize_context", memory_text)

    # ---- architect ----
    def plan_task(self, task: str, context_summary: str = "", injector=None) -> str:
        return self._record("plan_task", task)

    def design_task(self, task: str, context_summary: str = "", injector=None) -> str:
        return self._record("design_task", task)

    def analyze_task(self, task: str, context_summary: str = "", injector=None) -> str:
        return self._record("analyze_task", task)

    def review_task(self, task: str, context_summary: str = "", injector=None) -> str:
        return self._record("review_task", task)

    def spec_task(self, task: str, context_summary: str = "", injector=None) -> str:
        return self._record("spec_task", task)

    # ---- rtl_designer / rtl_verification ----
    def spec_design_task(self, task: str, injector=None) -> str:
        return self._record("spec_design_task", task)

    def verilog_design_task(self, task: str, injector=None) -> str:
        return self._record("verilog_design_task", task)

    # ---- async 对应方法(节点改调 await agent.a*_task 后的兼容) ----
    # calls 记录基础名(不带 a 前缀),保持既有断言兼容
    async def asummarize_context(self, memory_text: str, config=None) -> str:
        return self._record("summarize_context", memory_text)

    async def aplan_task(self, task: str, context_summary: str = "", injector=None, config=None) -> str:
        return self._record("plan_task", task)

    async def adesign_task(self, task: str, context_summary: str = "", injector=None, config=None) -> str:
        return self._record("design_task", task)

    async def aanalyze_task(self, task: str, context_summary: str = "", injector=None, config=None) -> str:
        return self._record("analyze_task", task)

    async def areview_task(self, task: str, context_summary: str = "", injector=None, config=None) -> str:
        return self._record("review_task", task)

    async def aspec_task(self, task: str, context_summary: str = "", injector=None, config=None) -> str:
        return self._record("spec_task", task)

    async def aspec_design_task(self, task: str, injector=None, config=None) -> str:
        return self._record("spec_design_task", task)

    async def averilog_design_task(self, task: str, injector=None, config=None) -> str:
        return self._record("verilog_design_task", task)


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
            verilog_sequence=list(verifier_sequence or []),
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
    assert manager.calls == [("summarize_context", "历史对话")]


def test_architect_plan_node():
    """Architect 制定执行计划(注入上下文摘要)"""
    architect = FakeRTLAgent(name="architect", response="计划")
    state = initial_state(context_summary="摘要S")
    result = asyncio.run(architect_plan_node(state, architect))
    assert result["arch_plan"] == "计划"
    assert architect.calls[0][0] == "plan_task"
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
    """验证通过 → designer_output"""
    state = initial_state(verification_report="验证结论: PASS", round=1)
    assert route_after_verification(state) == "designer_output"


def test_route_after_verification_fail_continues():
    """验证未通过且未达上限 → 回 designer_verilog"""
    state = initial_state(verification_report="验证结论: FAIL", round=1)
    assert route_after_verification(state) == "designer_verilog"


def test_route_after_verification_max_rounds():
    """达轮次上限即使未通过也强制交付"""
    state = initial_state(verification_report="验证结论: FAIL", round=3, max_rounds=3)
    assert route_after_verification(state) == "designer_output"


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
        "designer_output",
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
    assert ("designer_output", "__end__") in edges


def test_build_rtl_graph_workflow_conditional_edges():
    """验证多轮交互条件边存在(验证检查 → 交付/回环)"""
    agents = build_fake_agents()
    graph = build_rtl_graph_workflow(agents)
    edges = {(e.source, e.target) for e in graph.get_graph().edges}
    assert ("verification_check", "designer_output") in edges
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

    assert result["final_answer"] == "module uart;"
    assert result["round"] == 1
    assert result["arch_plan"] == "架构: 单核 UART"
    assert result["arch_design"] == "架构: 单核 UART"
    assert result["arch_analysis"] == "架构: 单核 UART"
    assert result["arch_review"] == "架构: 单核 UART"
    assert result["arch_spec"] == "架构: 单核 UART"
    assert result["design_spec"] == "module uart;"
    assert result["verification_plan"] == "验证结论: PASS\nRTL 无问题"
    assert result["verification_report"] == "验证结论: PASS\nRTL 无问题"

    # designer 调用:spec_design + verilog_design(编码) + verilog_design(交付)
    designer_calls = [c[0] for c in designer.calls]
    assert designer_calls == ["spec_design_task", "verilog_design_task", "verilog_design_task"]
    # verifier 调用:spec_design(计划) + verilog_design(检查)
    verifier_calls = [c[0] for c in verifier.calls]
    assert verifier_calls == ["spec_design_task", "verilog_design_task"]


def test_run_rtl_graph_multi_round_iteration():
    """验证 FAIL→PASS 多轮迭代:designer_verilog 执行两次,第二轮携带反馈"""
    manager = FakeRTLAgent(name="manager", response="摘要")
    architect = FakeRTLAgent(name="architect", response="架构方案")
    designer = FakeRTLAgent(name="rtl_designer", response="module uart;")
    verifier = FakeRTLAgent(
        name="rtl_verification",
        verilog_sequence=[
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
    assert result["final_answer"] == "module uart;"
    # designer 的 verilog_design_task 被调用 3 次:两轮迭代设计 + 最终交付
    designer_prompts = [c[1] for c in designer.calls if c[0] == "verilog_design_task"]
    assert len(designer_prompts) == 3
    # 首轮设计无反馈,第二轮设计携带 FAIL 反馈,交付轮拼入最终 RTL 与验证报告
    assert "验证结论: FAIL" not in designer_prompts[0]
    assert "验证结论: FAIL" in designer_prompts[1]
    assert "波特率配置错误" in designer_prompts[1]
    assert "验证结论: PASS" in designer_prompts[2]


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
    assert result["final_answer"] == "module uart;"
    verifier_design_calls = [c for c in verifier.calls if c[0] == "verilog_design_task"]
    assert len(verifier_design_calls) == 3


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
    # manager 调用:summarize + (无 plan_task,architect 承担计划)
    assert manager.calls == [("summarize_context", "之前聊过偏好 SystemVerilog")]
    # architect_plan 收到上下文摘要
    assert architect.calls[0][0] == "plan_task"


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