import importlib
import os
import sys
from types import SimpleNamespace
from typing import TypedDict

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Interrupt


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def test_ask_human_invokes_interrupt_with_structured_choice_payload(monkeypatch):
    # Given: LangGraph interrupt is replaced by a narrow offline fake.
    captured = []

    def fake_interrupt(payload):
        captured.append(payload)
        return {"choice_id": "approve"}

    import langgraph.types

    monkeypatch.setattr(langgraph.types, "interrupt", fake_interrupt)
    human_input = importlib.import_module("utils.human_input")

    # When: the human-input tool asks for a structured choice.
    result = human_input.ask_human(
        prompt="Deploy this change?",
        choices=[
            {"id": "approve", "label": "Approve"},
            {"id": "reject", "label": "Reject"},
        ],
    )

    # Then: the exact structured payload is sent through interrupt.
    assert result == {"choice_id": "approve"}
    assert captured == [
        {
            "kind": "human_choice",
            "prompt": "Deploy this change?",
            "choices": [
                {"id": "approve", "label": "Approve"},
                {"id": "reject", "label": "Reject"},
            ],
        }
    ]


def test_agent_turn_result_distinguishes_completed_and_interrupted():
    # Given: one completed turn and one interrupted turn.
    from agent.agent_core import AgentTurnResult

    interrupt = Interrupt(
        value={"kind": "human_choice", "prompt": "Pick one", "choices": []},
        id="interrupt-1",
    )

    # When: factory constructors create the typed results.
    completed = AgentTurnResult.completed(output="Finished")
    interrupted = AgentTurnResult.interrupted(interrupts=[interrupt])

    # Then: callers can branch without inspecting raw LangGraph dictionaries.
    assert completed.status == "completed"
    assert completed.output == "Finished"
    assert completed.interrupts == []
    assert completed.is_completed is True
    assert completed.is_interrupted is False

    assert interrupted.status == "interrupted"
    assert interrupted.output is None
    assert interrupted.interrupts == [interrupt]
    assert interrupted.is_completed is False
    assert interrupted.is_interrupted is True


def test_agent_core_parses_langgraph_interrupt_result():
    # Given: a raw LangGraph result containing __interrupt__.
    from agent.agent_core import AgentCore

    core = object.__new__(AgentCore)
    interrupt = Interrupt(
        value={
            "kind": "human_choice",
            "prompt": "Continue?",
            "choices": [{"id": "yes", "label": "Yes"}],
        },
        id="interrupt-1",
    )

    # When: AgentCore parses the raw invoke result.
    turn = core._parse_turn_result({"__interrupt__": [interrupt], "messages": []})

    # Then: the interrupt is exposed as an interrupted AgentTurnResult.
    assert turn.status == "interrupted"
    assert turn.output is None
    assert turn.interrupts == [interrupt]


def test_agent_core_parses_completed_result_from_last_ai_message():
    # Given: a raw LangGraph result with normal AI messages and no interrupt.
    from agent.agent_core import AgentCore

    core = object.__new__(AgentCore)
    result = {
        "messages": [
            AIMessage(content="draft"),
            AIMessage(content="final answer"),
        ]
    }

    # When: AgentCore parses the raw invoke result.
    turn = core._parse_turn_result(result)

    # Then: the final AI content is exposed as a completed AgentTurnResult.
    assert turn.status == "completed"
    assert turn.output == "final answer"
    assert turn.interrupts == []


def test_resume_structured_invokes_command_resume_with_same_thread_config():
    # Given: an AgentCore with a fake executor and AgentMemory-style config.
    from agent.agent_core import AgentCore

    calls = []

    class FakeExecutor:
        def invoke(self, command, config):
            calls.append((command, config))
            return {"messages": [AIMessage(content="resumed")]} 

    core = object.__new__(AgentCore)
    core.agent_executor = FakeExecutor()
    core.memory = SimpleNamespace(get_config=lambda: {"configurable": {"thread_id": "thread-123"}})
    core.max_iterations = 25

    # When: a structured resume payload is sent.
    turn = core.resume_structured({"choice_id": "approve"})

    # Then: Command(resume=...) uses the same thread config plus recursion limit.
    assert turn.status == "completed"
    assert turn.output == "resumed"
    assert len(calls) == 1
    command, config = calls[0]
    assert isinstance(command, Command)
    assert command.resume == {"choice_id": "approve"}
    assert config == {"configurable": {"thread_id": "thread-123"}, "recursion_limit": 25}


def test_cli_helper_renders_and_resumes_until_completion():
    # Given: an agent that interrupts twice before completing.
    from utils import human_input

    interrupts = [
        Interrupt(value={"kind": "human_choice", "prompt": "First?", "choices": []}, id="i-1"),
        Interrupt(value={"kind": "human_choice", "prompt": "Second?", "choices": []}, id="i-2"),
    ]
    rendered = []
    resume_requests = []

    class FakeAgent:
        def __init__(self):
            self.resume_count = 0

        def chat_structured(self, message):
            assert message == "start"
            return SimpleNamespace(status="interrupted", output=None, interrupts=[interrupts[0]])

        def resume_structured(self, payload):
            resume_requests.append(payload)
            self.resume_count += 1
            if self.resume_count == 1:
                return SimpleNamespace(status="interrupted", output=None, interrupts=[interrupts[1]])
            return SimpleNamespace(status="completed", output="done", interrupts=[])

    def render_interrupt(interrupt):
        rendered.append(interrupt.value["prompt"])

    def read_resume(interrupt):
        return {"choice_id": interrupt.id}

    # When: the CLI helper drives the turn.
    output = human_input.run_human_input_loop(
        agent=FakeAgent(),
        message="start",
        render_interrupt=render_interrupt,
        read_resume=read_resume,
    )

    # Then: it renders every interrupt and resumes until the final output.
    assert output == "done"
    assert rendered == ["First?", "Second?"]
    assert resume_requests == [{"choice_id": "i-1"}, {"choice_id": "i-2"}]


def test_real_state_graph_ask_human_interrupts_and_resumes_on_same_thread():
    # Given: a real offline LangGraph with MemorySaver and the production ask_human tool.
    from utils.human_input import ask_human

    class State(TypedDict, total=False):
        answer: dict
        final: str

    def ask_node(state):
        answer = ask_human(
            prompt="confirm",
            choices=[{"id": "approve", "label": "Approve"}],
        )
        return {"answer": answer, "final": f"done:{answer['choice_id']}"}

    builder = StateGraph(State)
    builder.add_node("ask", ask_node)
    builder.add_edge(START, "ask")
    builder.add_edge("ask", END)
    graph = builder.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "thread-real-hitl"}}

    # When: the graph interrupts and is resumed on the same thread.
    interrupted = graph.invoke({}, config)
    interrupt = interrupted["__interrupt__"][0]
    completed = graph.invoke(Command(resume={"choice_id": "approve"}), config)

    # Then: the real runtime returns the resume payload through the node result.
    assert interrupt.value["kind"] == "human_choice"
    assert completed == {"answer": {"choice_id": "approve"}, "final": "done:approve"}


def test_real_state_graph_two_sequential_interrupts_require_two_resumes():
    # Given: a real graph node that asks twice in sequence.
    from utils.human_input import ask_human

    class State(TypedDict, total=False):
        first: dict
        second: dict
        final: str

    def ask_twice(state):
        first = ask_human(
            prompt="first",
            choices=[{"id": "one", "label": "One"}],
        )
        second = ask_human(
            prompt="second",
            choices=[{"id": "two", "label": "Two"}],
        )
        return {
            "first": first,
            "second": second,
            "final": f"{first['choice_id']}:{second['choice_id']}",
        }

    builder = StateGraph(State)
    builder.add_node("ask_twice", ask_twice)
    builder.add_edge(START, "ask_twice")
    builder.add_edge("ask_twice", END)
    graph = builder.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "thread-sequential-hitl"}}

    # When: only the first interrupt is resumed.
    first_interrupt = graph.invoke({}, config)
    second_interrupt = graph.invoke(Command(resume={"choice_id": "one"}), config)

    # Then: the graph is still interrupted and only completes after the second resume.
    assert "__interrupt__" in first_interrupt
    assert "__interrupt__" in second_interrupt
    completed = graph.invoke(Command(resume={"choice_id": "two"}), config)
    assert completed["final"] == "one:two"


def test_cli_helper_collects_simultaneous_interrupts_before_one_resume():
    # Given: one interrupted turn with multiple simultaneous interrupts.
    from utils import human_input

    interrupts = [
        Interrupt(value={"kind": "human_choice", "choices": []}, id="interrupt-a"),
        Interrupt(value={"kind": "human_choice", "choices": []}, id="interrupt-b"),
    ]
    rendered = []
    resume_payloads = []

    class FakeAgent:
        def chat_structured(self, message):
            assert message == "start"
            return SimpleNamespace(status="interrupted", output=None, interrupts=interrupts)

        def resume_structured(self, payload):
            resume_payloads.append(payload)
            return SimpleNamespace(status="completed", output="done", interrupts=[])

    def render_interrupt(interrupt):
        rendered.append(interrupt.id)

    def read_resume(interrupt):
        return {"choice_id": f"answer-{interrupt.id}"}

    # When: the CLI helper handles the turn.
    output = human_input.run_human_input_loop(
        agent=FakeAgent(),
        message="start",
        render_interrupt=render_interrupt,
        read_resume=read_resume,
    )

    # Then: all answers are collected and resumed with one interrupt-id keyed payload.
    assert output == "done"
    assert rendered == ["interrupt-a", "interrupt-b"]
    assert resume_payloads == [
        {
            "interrupt-a": {"choice_id": "answer-interrupt-a"},
            "interrupt-b": {"choice_id": "answer-interrupt-b"},
        }
    ]


def test_resume_after_switching_thread_is_rejected():
    # Given: AgentCore has an interrupted turn for one thread.
    from agent.agent_core import AgentCore

    class FakeMemory:
        def __init__(self):
            self.thread_id = "thread-before"

        def get_config(self):
            return {"configurable": {"thread_id": self.thread_id}}

        def add(self, role, content, metadata=None):
            raise AssertionError("memory should not be written while interrupted")

    class FakeExecutor:
        def __init__(self):
            self.calls = 0

        def invoke(self, value, config):
            self.calls += 1
            if self.calls == 1:
                return {"__interrupt__": [Interrupt(value={"kind": "human_choice"}, id="i-1")]}
            return {"messages": [AIMessage(content="should not resume")]} 

    core = object.__new__(AgentCore)
    core.memory = FakeMemory()
    core.agent_executor = FakeExecutor()
    core.max_iterations = 25
    core.verbose = False
    core.execution_history = []
    core._maybe_compact = lambda: None
    core._compute_skill_block = lambda task: ""
    core._create_agent_executor = lambda skill_block="": core.agent_executor

    # When: the run interrupts and the current thread is switched before resume.
    turn = core.run_structured("needs approval")
    core.memory.thread_id = "thread-after"

    # Then: resume is rejected instead of resuming against the wrong thread.
    assert turn.status == "interrupted"
    with pytest.raises(ValueError, match="thread"):
        core.resume_structured({"choice_id": "approve"})


def test_interrupted_run_records_final_important_assistant_memory_after_resume():
    # Given: an interrupted AgentCore run later completes by resume.
    from agent.agent_core import AgentCore

    recorded_memory = []

    class FakeMemory:
        def get_config(self):
            return {"configurable": {"thread_id": "thread-memory"}}

        def add(self, role, content, metadata=None):
            recorded_memory.append((role, content, metadata))

    class FakeExecutor:
        def __init__(self):
            self.calls = 0

        def invoke(self, value, config):
            self.calls += 1
            if self.calls == 1:
                return {"__interrupt__": [Interrupt(value={"kind": "human_choice"}, id="i-1")]}
            return {"messages": [AIMessage(content="approved result")]}

    core = object.__new__(AgentCore)
    core.memory = FakeMemory()
    core.agent_executor = FakeExecutor()
    core.max_iterations = 25
    core.verbose = False
    core.execution_history = []
    core._maybe_compact = lambda: None
    core._compute_skill_block = lambda task: ""
    core._create_agent_executor = lambda skill_block="": core.agent_executor

    # When: run interrupts and then completes after resume.
    interrupted = core.run_structured("needs approval")
    completed = core.resume_structured({"choice_id": "approve"})

    # Then: the final assistant output is saved as important memory after completion.
    assert interrupted.status == "interrupted"
    assert completed.status == "completed"
    assert ("assistant", "approved result", {"important": True}) in recorded_memory


def test_legacy_run_and_chat_raise_on_interrupt_instead_of_empty_string():
    # Given: legacy wrappers receive an interrupted structured turn.
    from agent.agent_core import AgentCore

    interrupted = SimpleNamespace(status="interrupted", output=None, interrupts=[Interrupt(value={}, id="i-1")])
    core = object.__new__(AgentCore)
    core.run_structured = lambda task: interrupted
    core.chat_structured = lambda message: interrupted

    # When / Then: callers get an explicit error instead of an empty string.
    with pytest.raises(RuntimeError, match="interrupt"):
        core.run("needs approval")
    with pytest.raises(RuntimeError, match="interrupt"):
        core.chat("needs approval")
