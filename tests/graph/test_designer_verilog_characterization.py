"""designer_verilog_node 行为保持(characterization)测试。

本文件在 **未改动** 的 ``graph/rtl_graph.py`` 上即应通过,用于钉住
``designer_verilog_node`` 的现有可观测行为,使后续"改用 create_llm_node
工厂"的重构不能静默漂移。覆盖:

1. prompt 内容:task + 【设计规格与Filelist】 + 【验证计划】;
   仅当 ``round > 0 and verification_report`` 时追加
   【第 N 轮验证报告反馈(请据此修正 RTL)】;且不得出现「重要规则」前缀。
2. ``round == state["round"] + 1``。
3. ``output_files``:从 ``config.configurable.workspace_path`` 下
   ``scripts/syn_filelist.f`` 解析,仅保留 .v/.sv/.vhd/.vhdl token。
4. ``messages == [AIMessage(content=result)]``。
5. ``rtl_code == result``。
6. 技能注入:injector 以 ``(prompt, prompt_task)`` 被调用
   (prompt_task 为未套渲染模板外壳的拼装文本)。

另断言缺失 workspace 时优雅降级(``output_files == []``)。

全部离线:FakeDesignerAgent 不联网,RecordingInjector 仅记录调用。
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage

from agent.turn_types import AgentTurnResult
from graph.rtl_graph import RTLGraphState, designer_verilog_node
from team.base import TeamAgent

# 与 team/rtl_designer/AGENT.md 的 ## workflow:verilog_design 默认模板一致
DEFAULT_TEMPLATES: dict[str, str] = {
    "verilog_design": "请根据以下任务与上下文输出可综合的 SystemVerilog RTL 源码:\n\n{task}\n\n",
}


@dataclass
class FakeDesignerAgent:
    """模拟 TeamAgent,不联网;记录 run_team_turn_with_interrupt 传入的 prompt。"""

    name: str = "rtl_designer"
    response: str = "module uart; endmodule"
    calls: list[tuple[str, str]] = field(default_factory=list)

    def get_template(self, name: str) -> str:
        return DEFAULT_TEMPLATES.get(name, "")

    def render_template(self, template: str, **kwargs) -> str:
        return TeamAgent.render_template(template, **kwargs)

    async def arun_structured(self, task: str, config=None) -> AgentTurnResult:
        self.calls.append(("arun_structured", task))
        return AgentTurnResult.completed(self.response)


@dataclass
class RecordingInjector:
    """记录 inject_into_prompt 的入参,原样返回 prompt(不实际注入技能块)。

    同时兼容两种调用形态:手写节点的 ``(prompt, task)`` 与工厂的
    ``(prompt, task, exclude_skills=...)``(经 ``**kwargs`` 吸收)。
    """

    calls: list[tuple[str, str]] = field(default_factory=list)

    def inject_into_prompt(self, prompt: str, task: str, **kwargs) -> str:
        self.calls.append((prompt, task))
        return prompt


def _initial_state(**overrides) -> RTLGraphState:
    """构造完整初始状态(带默认值),与 tests/graph/test_rtl_graph.py 一致。"""
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


def _expected_prompt_task(state: RTLGraphState) -> str:
    """按节点现有拼装逻辑复算 prompt_task(独立于生产实现,避免同源)。"""
    task = state["task"]
    parts = [task]
    if state.get("design_spec"):
        parts.append(f"【设计规格与Filelist】\n{state['design_spec']}")
    if state.get("verification_plan"):
        parts.append(f"【验证计划】\n{state['verification_plan']}")
    if state.get("round", 0) > 0 and state.get("verification_report"):
        parts.append(
            f"【第 {state['round']} 轮验证报告反馈(请据此修正 RTL)】\n"
            f"{state['verification_report']}"
        )
    return "\n\n".join(parts)


def _expected_prompt(state: RTLGraphState) -> str:
    """渲染模板外壳后的完整 prompt。"""
    return (
        DEFAULT_TEMPLATES["verilog_design"].replace(
            "{task}", _expected_prompt_task(state)
        )
    )


# ==================== 1. prompt 内容 ====================

def test_prompt_contains_task_spec_and_plan_without_feedback_at_round_zero():
    """round==0:prompt 含 task/设计规格/验证计划,不含反馈小节与「重要规则」前缀。"""
    designer = FakeDesignerAgent()
    state = _initial_state(design_spec="规格D", verification_plan="计划V")
    asyncio.run(designer_verilog_node(state, designer))
    prompt = designer.calls[0][1]

    assert prompt == _expected_prompt(state)
    assert "设计一个 UART 模块" in prompt
    assert "【设计规格与Filelist】\n规格D" in prompt
    assert "【验证计划】\n计划V" in prompt
    assert "验证报告反馈" not in prompt
    # 手写节点不调 load_agent_rules,故无「重要规则」前缀(字节级无漂移的守护)
    assert "重要规则" not in prompt


def test_prompt_appends_feedback_when_round_positive_and_report_present():
    """round>0 且 report 非空:追加【第 N 轮验证报告反馈(请据此修正 RTL)】。"""
    designer = FakeDesignerAgent()
    state = _initial_state(
        design_spec="规格D",
        verification_plan="计划V",
        verification_report="验证结论: FAIL\n波特率错误",
        round=1,
    )
    asyncio.run(designer_verilog_node(state, designer))
    prompt = designer.calls[0][1]

    assert prompt == _expected_prompt(state)
    assert "【第 1 轮验证报告反馈(请据此修正 RTL)】\n验证结论: FAIL\n波特率错误" in prompt
    assert "重要规则" not in prompt


def test_prompt_omits_feedback_when_round_positive_but_report_empty():
    """round>0 但 report 为空:不追加反馈小节。"""
    designer = FakeDesignerAgent()
    state = _initial_state(design_spec="规格D", round=2, verification_report="")
    asyncio.run(designer_verilog_node(state, designer))
    prompt = designer.calls[0][1]

    assert prompt == _expected_prompt(state)
    assert "验证报告反馈" not in prompt


def test_prompt_omits_feedback_at_round_zero_even_with_report():
    """round==0 但 report 非空:仍不追加反馈小节(需 round>0)。"""
    designer = FakeDesignerAgent()
    state = _initial_state(
        design_spec="规格D", round=0, verification_report="验证结论: FAIL"
    )
    asyncio.run(designer_verilog_node(state, designer))
    prompt = designer.calls[0][1]

    assert prompt == _expected_prompt(state)
    assert "验证报告反馈" not in prompt


def test_prompt_omits_spec_and_plan_sections_when_empty():
    """design_spec/verification_plan 为空:对应小节整段省略。"""
    designer = FakeDesignerAgent()
    state = _initial_state(design_spec="", verification_plan="")
    asyncio.run(designer_verilog_node(state, designer))
    prompt = designer.calls[0][1]

    assert prompt == _expected_prompt(state)
    assert "【设计规格与Filelist】" not in prompt
    assert "【验证计划】" not in prompt


# ==================== 2. round 递增 ====================

def test_round_is_state_round_plus_one():
    """round == state["round"] + 1。"""
    designer = FakeDesignerAgent()
    result = asyncio.run(designer_verilog_node(_initial_state(round=0), designer))
    assert result["round"] == 1

    designer2 = FakeDesignerAgent()
    result2 = asyncio.run(designer_verilog_node(_initial_state(round=2), designer2))
    assert result2["round"] == 3


# ==================== 3. output_files 解析 ====================

def test_output_files_parsed_from_workspace_filelist(tmp_path):
    """从 workspace 下 scripts/syn_filelist.f 解析,仅保留源文件 token。"""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "syn_filelist.f").write_text(
        "# comment\n"
        "src/top.sv\n"
        "// line comment\n"
        "+incdir+./inc\n"
        "-f other.f\n"
        "\n"
        "src/util.v\n"
        "src/pkg.vhd  # trailing\n"
        "src/legacy.vhdl\n"
        "README.md\n"
    )
    designer = FakeDesignerAgent()
    state = _initial_state(design_spec="规格D", verification_plan="计划V")
    result = asyncio.run(
        designer_verilog_node(
            state, designer, config={"configurable": {"workspace_path": str(tmp_path)}}
        )
    )
    assert result["output_files"] == [
        "src/top.sv",
        "src/util.v",
        "src/pkg.vhd",
        "src/legacy.vhdl",
    ]


def test_output_files_empty_when_config_is_none():
    """config 为 None:output_files 优雅降级为空列表。"""
    designer = FakeDesignerAgent()
    result = asyncio.run(designer_verilog_node(_initial_state(), designer))
    assert result["output_files"] == []


def test_output_files_empty_when_no_workspace_path():
    """config 无 workspace_path:output_files 优雅降级为空列表。"""
    designer = FakeDesignerAgent()
    result = asyncio.run(
        designer_verilog_node(
            _initial_state(), designer, config={"configurable": {}}
        )
    )
    assert result["output_files"] == []


def test_output_files_empty_when_filelist_missing(tmp_path):
    """workspace 存在但 filelist 缺失:output_files 为空列表。"""
    designer = FakeDesignerAgent()
    result = asyncio.run(
        designer_verilog_node(
            _initial_state(),
            designer,
            config={"configurable": {"workspace_path": str(tmp_path)}},
        )
    )
    assert result["output_files"] == []


# ==================== 4/5. messages 与 rtl_code ====================

def test_messages_and_rtl_code_match_result():
    """messages == [AIMessage(content=result)],rtl_code == result。"""
    designer = FakeDesignerAgent(response="module top; endmodule")
    result = asyncio.run(designer_verilog_node(_initial_state(), designer))
    assert result["rtl_code"] == "module top; endmodule"
    assert result["messages"] == [AIMessage(content="module top; endmodule")]


# ==================== 6. 技能注入 ====================

def test_injector_called_with_prompt_and_assembled_prompt_task():
    """injector.inject_into_prompt(prompt, prompt_task),prompt_task 不含模板外壳。"""
    designer = FakeDesignerAgent()
    injector = RecordingInjector()
    state = _initial_state(design_spec="规格D", verification_plan="计划V")
    asyncio.run(designer_verilog_node(state, designer, injector))

    assert len(injector.calls) == 1
    injected_prompt, injected_task = injector.calls[0]
    # 第二个参数为拼装文本本身(无模板外壳)
    assert injected_task == _expected_prompt_task(state)
    assert injected_task.startswith("设计一个 UART 模块")
    assert "SystemVerilog RTL 源码" not in injected_task
    # 第一个参数为渲染后的完整 prompt(模板外壳 + 拼装文本)
    assert injected_prompt == _expected_prompt(state)
