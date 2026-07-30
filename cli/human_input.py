"""LangGraph human-input tool and CLI pause/resume orchestration."""

from typing import Any
from typing_extensions import TypedDict

import langgraph.types
from langchain_core.tools import StructuredTool

from .cli_menu import select_menu


class Choice(TypedDict):
    """A selectable option shown to the human operator."""

    id: str
    label: str


def _ask_human(prompt: str, choices: list[Choice]) -> Any:
    """暂停图执行，并在恢复后原样返回人工输入负载。"""
    return langgraph.types.interrupt(
        {
            "kind": "human_choice",
            "prompt": prompt,
            "choices": choices,
        }
    )


class CallableStructuredTool(StructuredTool):
    """同时支持 LangChain 调用和图节点直接调用的结构化工具。"""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)


ask_human = CallableStructuredTool.from_function(
    func=_ask_human,
    name="ask_human",
    description="Pause execution and ask the human to choose from structured options.",
)


def complete_human_input_turn(turn, render_interrupt, read_resume):
    """循环处理人工中断，直到 LangGraph 返回最终结果。"""
    while getattr(turn, "status", None) == "interrupted":
        for interrupt in turn.interrupts:
            render_interrupt(interrupt)
        turn = read_resume(turn.interrupts)
    return turn.output or ""


def _resume_payload_for_interrupts(interrupts, read_resume):
    # 单中断直接传值；并行中断必须按 interrupt id 映射，避免答案错配。
    if len(interrupts) == 1:
        return read_resume(interrupts[0])
    return {interrupt.id: read_resume(interrupt) for interrupt in interrupts}


def run_human_input_loop(agent, message, render_interrupt, read_resume):
    """启动对话并持续恢复人工中断，直到本轮完成。"""
    turn = agent.chat_structured(message)
    return complete_human_input_turn(
        turn,
        render_interrupt,
        lambda interrupts: agent.resume_structured(
            _resume_payload_for_interrupts(interrupts, read_resume)
        ),
    )


def render_human_interrupt(interrupt):
    """Render the human interrupt prompt before collecting the CLI response."""
    value = interrupt.value
    if isinstance(value, dict) and value.get("kind") == "human_choice":
        prompt = str(value.get("prompt") or "需要人工输入")
        print()
        for line in prompt.split("\n"):
            print(line.strip())


def read_human_resume(interrupt):
    """从 CLI 收集 LangGraph 人工中断所需的恢复负载。"""
    value = interrupt.value
    # 非结构化中断回退到自由文本，结构化选择则保持 choice_id 协议。
    if not isinstance(value, dict) or value.get("kind") != "human_choice":
        return {"text": input("请输入: ").strip()}

    prompt = str(value.get("prompt") or "请选择")
    choices = value.get("choices") or []
    if not choices:
        return {"text": input(f"{prompt}: ").strip()}

    options = [(str(choice["label"]), choice["id"]) for choice in choices]
    selected = select_menu(prompt, options)
    if selected is None:
        return {"cancelled": True}
    return {"choice_id": selected}


def run_structured_until_completion(agent, message):
    """Run an Agent task and finish any human-input interruptions."""
    turn = agent.run_structured(message)
    return complete_human_input_turn(
        turn,
        render_human_interrupt,
        lambda interrupts: agent.resume_structured(
            _resume_payload_for_interrupts(interrupts, read_human_resume)
        ),
    )


def chat_until_completion(agent, message):
    """Chat through the structured path so human interrupts can be resumed."""
    return run_human_input_loop(
        agent,
        message,
        render_human_interrupt,
        read_human_resume,
    )
