"""request_user_confirmation 工具单元测试

验证:
1. 工具调用触发 langgraph interrupt, value kind="user_confirmation"
2. resume 后 interrupt() 返回值经 json.dumps 成 str, LLM 下轮可消费
3. build_interrupt_event 对 user_confirmation kind 正确渲染多问题 prompt + 带 item_id 的 choices
4. resume payload {answers:{item_id:{choice_id}}} 正确透传

测试风格对齐 tests/agent/test_human_input.py: 真实 StateGraph + MemorySaver + Command(resume=...)
"""
from __future__ import annotations

import json
from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from llm.message_utils import build_interrupt_event
from tools.human_confirmation import request_user_confirmation

# ============ 工具函数级: interrupt 触发与 resume 返回 ============

class _State(TypedDict, total=False):
    """测试用工作流状态。"""

    answer: str
    final: str


def _build_confirmation_graph() -> CompiledStateGraph:
    """构建单节点图: 节点内调 request_user_confirmation 工具, 返回 JSON + final。"""

    def confirm_node(state: _State) -> dict[str, Any]:
        items = [
            {
                "id": "fpga_path",
                "question": "FPGA 端是否允许基8 Booth 降级为基4?",
                "choices": [
                    {"id": "dsp_hard", "label": "用 DSP 硬核"},
                    {"id": "base4_logic", "label": "纯逻辑基4"},
                ],
            },
            {
                "id": "clk_strategy",
                "question": "系统级采用同步复位还是异步复位?",
                "choices": [
                    {"id": "sync", "label": "同步复位"},
                    {"id": "async", "label": "异步复位"},
                ],
            },
        ]
        # 工具内部 __call__ 调 _request_user_confirmation, 触发 interrupt
        answer_json = request_user_confirmation(items)
        return {"answer": answer_json, "final": f"done:{answer_json[:20]}"}

    builder = StateGraph(_State)
    builder.add_node("confirm", confirm_node)
    builder.add_edge(START, "confirm")
    builder.add_edge("confirm", END)
    return builder.compile(checkpointer=MemorySaver())


def test_interrupt_value_kind_is_user_confirmation():
    """工具调用触发 interrupt, value.kind == 'user_confirmation', items 结构正确。"""
    graph = _build_confirmation_graph()
    config = {"configurable": {"thread_id": "thread-confirm-1"}}

    result = graph.invoke({}, config)

    assert "__interrupt__" in result
    interrupt = result["__interrupt__"][0]
    value = interrupt.value
    assert value["kind"] == "user_confirmation"
    assert len(value["items"]) == 2
    assert value["items"][0]["id"] == "fpga_path"
    assert value["items"][1]["id"] == "clk_strategy"


def test_resume_payload_returns_json_string_with_answers():
    """resume 后 interrupt() 返回 payload, 工具 json.dumps 成 str, 含 answers 映射。"""
    graph = _build_confirmation_graph()
    config = {"configurable": {"thread_id": "thread-confirm-2"}}

    # 首次 invoke 触发 interrupt
    graph.invoke({}, config)

    # resume: 用户为 fpga_path 选 dsp_hard, 为 clk_strategy 选 sync
    resume_payload = {
        "answers": {
            "fpga_path": {"choice_id": "dsp_hard"},
            "clk_strategy": {"choice_id": "sync"},
        }
    }
    completed = graph.invoke(Command(resume=resume_payload), config)

    # 工具返回值(answer 字段)是 JSON 字符串
    answer_json = completed["answer"]
    assert isinstance(answer_json, str)
    parsed = json.loads(answer_json)
    assert parsed["answers"]["fpga_path"]["choice_id"] == "dsp_hard"
    assert parsed["answers"]["clk_strategy"]["choice_id"] == "sync"


def test_resume_with_non_dict_payload_falls_back_to_empty_answers():
    """resume payload 非 dict 时, 工具兜底为 {"answers": {}}, 不让 LLM 读非法结构。"""
    graph = _build_confirmation_graph()
    config = {"configurable": {"thread_id": "thread-confirm-3"}}

    graph.invoke({}, config)
    # 异常 resume: 裸字符串而非 dict
    completed = graph.invoke(Command(resume="invalid_string"), config)

    parsed = json.loads(completed["answer"])
    assert parsed == {"answers": {}}


# ============ build_interrupt_event 级: 前端 SSE 渲染 ============

def test_build_interrupt_event_user_confirmation_renders_multi_question_prompt():
    """user_confirmation kind: prompt 多段拼接带 [item_id] 前缀, choices 扁平带 item_id。"""
    value = {
        "kind": "user_confirmation",
        "items": [
            {
                "id": "fpga_path",
                "question": "FPGA Booth 降级?",
                "choices": [
                    {"id": "dsp_hard", "label": "用 DSP 硬核"},
                    {"id": "base4_logic", "label": "纯逻辑基4"},
                ],
            },
            {
                "id": "clk_strategy",
                "question": "复位策略?",
                "choices": [
                    {"id": "sync", "label": "同步复位"},
                ],
            },
        ],
    }
    event = build_interrupt_event(value)

    assert event["type"] == "interrupt"
    assert "[fpga_path] FPGA Booth 降级?" in event["prompt"]
    assert "[clk_strategy] 复位策略?" in event["prompt"]
    # choices 扁平聚合, 每项带 item_id
    assert len(event["choices"]) == 3
    assert event["choices"][0] == {
        "item_id": "fpga_path",
        "id": "dsp_hard",
        "label": "用 DSP 硬核",
    }
    assert event["choices"][2]["item_id"] == "clk_strategy"
    # items 结构化分组列表: 每项 {id, question, choices: [{id, label}]}
    assert len(event["items"]) == 2
    assert event["items"][0]["id"] == "fpga_path"
    assert event["items"][0]["question"] == "FPGA Booth 降级?"
    assert event["items"][0]["choices"][0] == {"id": "dsp_hard", "label": "用 DSP 硬核"}
    assert event["items"][1]["id"] == "clk_strategy"
    assert event["items"][1]["choices"][0] == {"id": "sync", "label": "同步复位"}


def test_build_interrupt_event_user_confirmation_empty_items_falls_back():
    """items 为空时: prompt 兜底文案, choices 为空列表。"""
    event = build_interrupt_event({"kind": "user_confirmation", "items": []})
    assert event["type"] == "interrupt"
    assert event["prompt"] == "需要用户确认多个架构决策点"
    assert event["choices"] == []
    # items 为空时 make_interrupt_dict 仍输出空列表（非 None）
    assert event.get("items") == []


def test_build_interrupt_event_user_confirmation_skips_malformed_items():
    """item/choice 字段缺失时跳过, 不让前端拿到残缺结构。"""
    value = {
        "kind": "user_confirmation",
        "items": [
            {"id": "ok", "question": "正常问题?", "choices": [{"id": "y", "label": "是"}]},
            {"id": "no_question"},  # 缺 question, 跳过
            {"question": "无 id?", "choices": []},  # 缺 id, 跳过
            {"id": "no_choices", "question": "无选项?"},  # choices 缺失, 跳过
        ],
    }
    event = build_interrupt_event(value)

    # 只有第一个完整 item 的 prompt 段
    assert "[ok] 正常问题?" in event["prompt"]
    assert "无 id" not in event["prompt"]
    assert len(event["choices"]) == 1
    assert event["choices"][0]["item_id"] == "ok"
    # items 分组列表同样跳过残缺 item, 只保留完整项
    assert len(event["items"]) == 1
    assert event["items"][0]["id"] == "ok"
    assert event["items"][0]["choices"] == [{"id": "y", "label": "是"}]


# ============ 与 ask_human 隔离: kind 不串扰 ============

def test_build_interrupt_event_human_choice_unchanged():
    """user_confirmation 扩展不影响现有 human_choice kind 的渲染。"""
    event = build_interrupt_event(
        {"kind": "human_choice", "prompt": "选一个", "choices": [{"id": "a", "label": "A"}]}
    )
    assert event == {
        "type": "interrupt",
        "prompt": "选一个",
        "choices": [{"id": "a", "label": "A"}],
    }
    # human_choice 的 choices 项无 item_id 字段
    assert "item_id" not in event["choices"][0]


def test_build_interrupt_event_dangerous_command_unchanged():
    """user_confirmation 扩展不影响现有 dangerous_command kind 的渲染。"""
    event = build_interrupt_event(
        {
            "kind": "dangerous_command",
            "prompt": "危险命令确认",
            "choices": [
                {"id": "approve", "label": "确认执行"},
                {"id": "deny", "label": "拒绝执行"},
            ],
        }
    )
    assert event["type"] == "interrupt"
    assert event["prompt"] == "危险命令确认"
    assert len(event["choices"]) == 2
    # dangerous_command 的 choices 项无 item_id 字段
    for choice in event["choices"]:
        assert "item_id" not in choice


# ============ 工具元信息 ============

def test_tool_name_and_description():
    """工具 name/description 符合 LLM 工具选择所需的可读性。"""
    assert request_user_confirmation.name == "request_user_confirmation"
    assert "架构决策点" in request_user_confirmation.description
    # description 含 resume 返回 JSON 结构说明, 引导 LLM 下轮正确消费
    assert "answers" in request_user_confirmation.description
