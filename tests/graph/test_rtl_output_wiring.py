"""
交付接线回归测试 (graph/rtl_graph.py 的 designer_output 接线)

覆盖:
- 图节点集合含 designer_output,且 designer_output → END 边存在
- route_after_sim_check 的两条终态(通过 / 达上限)均返回 designer_output
- route_after_file_check 的达上限终态返回 designer_output
- designer_output_node 交付非空 + 零 LLM 契约(方案 B:fake agent 的 LLM 方法零调用)
- state 字段全空时仍返回含 task 的非空文本

注意:本文件不触发 sim_exec_check_node(它会 spawn Vivado 子进程)。
"""
from __future__ import annotations

import asyncio
import types
from typing import ClassVar

from langchain_core.messages import AIMessage

from graph.rtl_graph import (
    build_rtl_graph_workflow,
    route_after_file_check,
    route_after_sim_check,
)
from team.rtl_designer.rtl_designer import DesignerAgent


class _FakeGraphAgent:
    """仅用于图构建的最小 fake Agent(无 llm,构建期不调用任何方法)。

    ``__getattr__`` 对 ``*_node`` 属性返回一个 bound 空实现:节点实现下沉到
    角色类后,build_rtl_graph_workflow 的 NodeSpec 经 ``agents[role].xxx_node``
    在构建期取方法引用(register_nodes 的 bound-method 分支只补绑 injector),
    本 fake 借此对未来新增节点免疫,无需逐节点补桩。
    """

    name = "fake"
    tools: ClassVar[list] = []
    system_prompt = ""
    llm = None

    def get_template(self, name: str) -> str:
        return ""

    def render_template(self, template: str, **kwargs) -> str:
        return template

    def __getattr__(self, name: str):
        if name.endswith("_node"):
            async def _stub(state, injector=None, config=None):
                return {}

            return types.MethodType(_stub, self)
        raise AttributeError(name)


def _fake_agents() -> dict:
    return {
        role: _FakeGraphAgent()
        for role in ("manager", "architect", "rtl_designer", "rtl_verification")
    }


class _RecordingAgent:
    """记录 LLM/模板方法调用的 fake Agent,证明 designer_output_node 零 LLM。

    任一被禁止的方法一旦被调用即记入 ``calls``(并抛错),用例据此断言列表为空。
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    def _record(self, name: str):
        self.calls.append(name)
        raise AssertionError(f"designer_output_node 不得调用 {name}")

    async def arun_structured(self, *args, **kwargs):
        self._record("arun_structured")

    async def ainvoke(self, *args, **kwargs):
        self._record("ainvoke")

    async def astream(self, *args, **kwargs):
        self._record("astream")

    def render_template(self, *args, **kwargs):
        self._record("render_template")

    def get_template(self, *args, **kwargs):
        self._record("get_template")


# ==================== 用例 (a):节点与边 ====================

def test_designer_output_node_and_end_edge_present():
    """build_rtl_graph_workflow 的节点集合含 designer_output,且存在 → END 边。"""
    graph = build_rtl_graph_workflow(_fake_agents())
    node_ids = {n.id for n in graph.get_graph().nodes.values()}
    assert "designer_output" in node_ids

    edges = {(e.source, e.target) for e in graph.get_graph().edges}
    assert ("designer_output", "__end__") in edges


# ==================== 用例 (b)(c):条件路由终态 ====================

def test_route_after_sim_check_terminal_paths_deliver():
    """sim_exec_check 两条终态(通过 / 达上限)均路由到 designer_output 交付。"""
    assert route_after_sim_check({"sim_check_passed": True, "round": 1, "max_rounds": 3}) == "designer_output"
    assert route_after_sim_check({"sim_check_passed": False, "round": 3, "max_rounds": 3}) == "designer_output"
    # 未达上限仍回环重做
    assert route_after_sim_check({"sim_check_passed": False, "round": 1, "max_rounds": 3}) == "designer_verilog"


def test_route_after_file_check_max_rounds_delivers():
    """designer_file_check 达上限终态路由到 designer_output 交付。"""
    assert route_after_file_check({"file_check_passed": False, "round": 3, "max_rounds": 3}) == "designer_output"
    # 通过 / 未达上限语义保持不变
    assert route_after_file_check({"file_check_passed": True, "round": 1, "max_rounds": 3}) == "verification_check"
    assert route_after_file_check({"file_check_passed": False, "round": 1, "max_rounds": 3}) == "designer_verilog"


def _run_designer_output(state: dict, agent: _RecordingAgent) -> dict:
    """把 DesignerAgent 的真实交付节点方法绑到 _RecordingAgent 上执行。

    节点实现已下沉为 DesignerAgent.designer_output_node(bound-method 形态);
    MethodType 绑定后任一被禁用的 self.* LLM/模板方法若被调用,_RecordingAgent
    会记录并抛错,零 LLM 契约的回归保护保持不变。
    """
    bound = types.MethodType(DesignerAgent.designer_output_node, agent)
    return asyncio.run(bound(state))


# ==================== 用例 (d):非空交付 + 零 LLM 契约 ====================

def test_designer_output_non_empty_and_zero_llm():
    """designer_output_node 交付非空且不调用任何 LLM/模板方法。"""
    agent = _RecordingAgent()
    state = {
        "task": "设计 UART 模块",
        "design_spec": "规格D",
        "rtl_code": "module uart; endmodule",
        "verification_report": "验证结论: PASS",
        "output_files": ["src/uart.sv"],
    }

    result = _run_designer_output(state, agent)

    assert result["final_answer"]
    assert isinstance(result["final_answer"], str)
    assert result["messages"]
    assert isinstance(result["messages"][0], AIMessage)
    assert result["messages"][0].content == result["final_answer"]
    # 交付内容含各非空小节与文件清单
    assert "【任务】" in result["final_answer"]
    assert "module uart; endmodule" in result["final_answer"]
    assert "验证结论: PASS" in result["final_answer"]
    assert "- src/uart.sv" in result["final_answer"]
    # 零 LLM 契约:fake agent 的 LLM/模板方法调用列表为空
    assert agent.calls == []


def test_designer_output_all_empty_still_non_empty_with_task():
    """state 字段全空时仍返回含 task 的非空文本(禁止返回空串)。"""
    agent = _RecordingAgent()

    # 仅 task 非空:仍须输出含 task 的非空文本
    result = _run_designer_output({"task": "边界任务X"}, agent)
    assert result["final_answer"]
    assert "边界任务X" in result["final_answer"]
    assert result["messages"][0].content == result["final_answer"]

    # 完全空 state(含 task 为空的极端):仍须非空
    result_empty = _run_designer_output({}, agent)
    assert result_empty["final_answer"]
    assert agent.calls == []
