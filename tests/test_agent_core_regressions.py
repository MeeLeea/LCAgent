from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from tools.terminal_tools import UserRejectedCommandError


def test_compaction_retains_recent_messages_in_new_thread_state():
    # Given: the current thread has more messages than the compaction threshold.
    from agent.agent_core import AgentCore
    from agent.memory import AgentMemory

    messages = [HumanMessage(content=f"message-{idx}") for idx in range(5)]
    summarized_batches = []
    state_updates = []

    class FakeMemory:
        def __init__(self):
            self.thread_id = "thread-before"

        def get_messages(self):
            return messages

        def new_thread(self):
            self.thread_id = "thread-after"
            return self.thread_id

        def get_config(self):
            return {"configurable": {"thread_id": self.thread_id}}

        maybe_compact = AgentMemory.maybe_compact

    class FakeExecutor:
        def update_state(self, config, values):
            state_updates.append((config, values))

    core = object.__new__(AgentCore)
    core.memory = FakeMemory()
    core.max_context_messages = 3
    core.context_trim_keep = 2
    core.compaction_summary = ""
    core.verbose = False
    core.agent_executor = FakeExecutor()
    core._compute_skill_block = lambda task: ""
    core._create_agent_executor = lambda skill_block="": core.agent_executor
    core.llm = SimpleNamespace(
        chat=lambda messages: summarized_batches.append(messages) or "summary"
    )

    # When: compaction runs.
    core._compact_if_needed()

    # Then: only older messages are summarized and retained messages seed the new thread.
    assert len(summarized_batches) == 1
    assert core.memory.thread_id == "thread-after"
    assert core.compaction_summary == "summary"
    assert state_updates == [
        (
            {"configurable": {"thread_id": "thread-after"}},
            {"messages": messages[-2:]},
        )
    ]


def test_compaction_does_not_change_thread_when_summary_fails():
    # Given: compaction is needed but summarization returns no summary.
    from agent.agent_core import AgentCore
    from agent.memory import AgentMemory

    messages = [HumanMessage(content=f"message-{idx}") for idx in range(4)]
    new_thread_calls = []
    state_updates = []

    class FakeMemory:
        def __init__(self):
            self.thread_id = "thread-before"

        def get_messages(self):
            return messages

        def new_thread(self):
            new_thread_calls.append(True)
            self.thread_id = "thread-after"
            return self.thread_id

        maybe_compact = AgentMemory.maybe_compact

    class FakeExecutor:
        def update_state(self, config, values):
            state_updates.append((config, values))

    core = object.__new__(AgentCore)
    core.memory = FakeMemory()
    core.max_context_messages = 3
    core.context_trim_keep = 2
    core.compaction_summary = "existing"
    core.verbose = False
    core.agent_executor = FakeExecutor()
    core._compute_skill_block = lambda task: ""
    core._create_agent_executor = lambda skill_block="": core.agent_executor
    core.llm = SimpleNamespace(chat=lambda messages: "")

    # When: compaction runs.
    core._compact_if_needed()

    # Then: failed summary leaves the active thread and state untouched.
    assert core.memory.thread_id == "thread-before"
    assert core.compaction_summary == "existing"
    assert new_thread_calls == []
    assert state_updates == []


def test_record_tool_steps_deduplicates_full_history_and_maps_observations():
    # Given: LangGraph invoke returns full message history across turns.
    from agent.agent_core import AgentCore

    first_input = HumanMessage(content="first")
    second_input = HumanMessage(content="second")
    first_history = [
        first_input,
        AIMessage(content="", tool_calls=[{"name": "search", "args": {"q": "old"}, "id": "call-old"}]),
        ToolMessage(content="old result", tool_call_id="call-old"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "read", "args": {"path": "a"}, "id": "call-a"},
                {"name": "write", "args": {"path": "b"}, "id": "call-b"},
            ],
        ),
        ToolMessage(content="read result", tool_call_id="call-a"),
        ToolMessage(content="write result", tool_call_id="call-b"),
    ]
    second_history = [
        *first_history,
        second_input,
        AIMessage(content="", tool_calls=[{"name": "calc", "args": {"x": 1}, "id": "call-new"}]),
        ToolMessage(content="new result", tool_call_id="call-new"),
    ]

    core = object.__new__(AgentCore)
    core.verbose = False
    core.execution_history = []

    # When: two full histories are recorded.
    core._record_tool_steps(first_history, first_input)
    core._record_tool_steps(second_history, second_input)

    # Then: every tool_call id appears once and observations attach to the matching call.
    assert core.execution_history == [
        {"step": 1, "tool": "search", "input": {"q": "old"}, "observation": "old result"},
        {"step": 2, "tool": "read", "input": {"path": "a"}, "observation": "read result"},
        {"step": 3, "tool": "write", "input": {"path": "b"}, "observation": "write result"},
        {"step": 4, "tool": "calc", "input": {"x": 1}, "observation": "new result"},
    ]


def test_clear_history_resets_tool_call_dedupe_state():
    # Given: a tool call has already been recorded.
    from agent.agent_core import AgentCore

    input_msg = HumanMessage(content="run")
    history = [
        input_msg,
        AIMessage(content="", tool_calls=[{"name": "search", "args": {"q": "x"}, "id": "call-1"}]),
        ToolMessage(content="result", tool_call_id="call-1"),
    ]
    core = object.__new__(AgentCore)
    core.verbose = False
    core.execution_history = []

    # When: history is cleared and the same call id is observed again later.
    core._record_tool_steps(history, input_msg)
    core.clear_history()
    core._record_tool_steps(history, input_msg)

    # Then: clear_history also clears dedupe state, so the call can be recorded again.
    assert core.execution_history == [
        {"step": 1, "tool": "search", "input": {"q": "x"}, "observation": "result"}
    ]


def test_run_structured_stops_after_user_rejects_command(monkeypatch):
    # Given: 图执行期间终端工具收到用户拒绝信号。
    from agent.agent_core import AgentCore

    class FakeMemory:
        def get_config(self):
            return {"configurable": {"thread_id": "thread-rejected"}}

        def add(self, role, content, metadata=None):
            raise AssertionError("取消的任务不应写入长期记忆")

    from tools.terminal_tools import run_shell

    monkeypatch.setattr("tools.terminal_tools.confirm", lambda prompt: False)

    class RejectingExecutor:
        def __init__(self):
            self.calls = 0

        def invoke(self, value, config):
            self.calls += 1
            return run_shell.invoke({"command": "python cleanup.py"})

        def get_state(self, config):
            return SimpleNamespace(values={"messages": []})

    executor = RejectingExecutor()
    core = object.__new__(AgentCore)
    core.memory = FakeMemory()
    core.max_iterations = 25
    core.verbose = False
    core.execution_history = []
    core._compact_if_needed = lambda: None
    core._compute_skill_block = lambda task: ""
    core._create_agent_executor = lambda skill_block="": executor

    # When: 当前任务尝试执行被拒绝的命令。
    turn = core.run_structured("commit changes")

    # Then: 当前 turn 立即取消，不会把错误交回模型形成下一轮重试。
    assert executor.calls == 1
    assert turn.status == "cancelled"
    assert turn.output == "用户已拒绝执行危险命令，当前任务已取消。"


def test_resume_structured_stops_after_user_rejects_command():
    # Given: 中断恢复期间（如飞书 deny 选择）用户拒绝危险命令。
    from agent.agent_core import AgentCore

    class FakeMemory:
        def get_config(self):
            return {"configurable": {"thread_id": "thread-resume-rejected"}}

        def add(self, role, content, metadata=None):
            raise AssertionError("取消的任务不应写入长期记忆")

    class RejectingExecutor:
        def invoke(self, value, config):
            raise UserRejectedCommandError("python cleanup.py")

        def get_state(self, config):
            return SimpleNamespace(values={"messages": []})

    executor = RejectingExecutor()
    core = object.__new__(AgentCore)
    core.memory = FakeMemory()
    core.max_iterations = 25
    core.verbose = False
    core.execution_history = []
    core.agent_executor = executor
    core._pending_interrupt_thread_id = "thread-resume-rejected"
    core._pending_interrupt_mode = "chat"

    # When: 用户以 deny 恢复中断。
    turn = core.resume_structured({"choice_id": "deny"})

    # Then: 本轮取消并清理中断状态，而不是抛异常给调用方。
    assert turn.status == "cancelled"
    assert turn.output == "用户已拒绝执行危险命令，当前任务已取消。"
    assert core._pending_interrupt_thread_id is None
    assert core._pending_interrupt_mode is None


def test_run_structured_repairs_checkpoint_after_user_rejects_command():
    # Given: 工具拒绝前，LangGraph 已把包含 tool_calls 的 AIMessage 写入 checkpoint。
    from agent.agent_core import AgentCore

    pending = AIMessage(
        content="",
        tool_calls=[
            {"name": "run_shell", "args": {"command": "python cleanup.py"}, "id": "call-rejected"},
            {"name": "read_file", "args": {"file_path": "README.md"}, "id": "call-finished"},
        ],
    )
    finished = ToolMessage(content="read", name="read_file", tool_call_id="call-finished")
    state = SimpleNamespace(values={"messages": [pending, finished]})
    updates = []

    class FakeMemory:
        def get_config(self):
            return {"configurable": {"thread_id": "thread-rejected"}}

        def add(self, role, content, metadata=None):
            raise AssertionError("取消的任务不应写入长期记忆")

    class RejectingExecutor:
        def invoke(self, value, config):
            raise UserRejectedCommandError("python cleanup.py")

        def get_state(self, config):
            return state

        def update_state(self, config, values, as_node=None):
            updates.append((config, values, as_node))

    executor = RejectingExecutor()
    core = object.__new__(AgentCore)
    core.memory = FakeMemory()
    core.max_iterations = 25
    core.verbose = False
    core.execution_history = []
    core._compact_if_needed = lambda: None
    core._compute_skill_block = lambda task: ""
    core._create_agent_executor = lambda skill_block="": executor

    # When: 拒绝信号终止当前 turn。
    turn = core.run_structured("commit changes")

    # Then: 已完成结果被保留，缺失的调用得到匹配的错误 ToolMessage。
    assert turn.status == "cancelled"
    assert len(updates) == 1
    config, values, as_node = updates[0]
    assert config == {"configurable": {"thread_id": "thread-rejected"}, "recursion_limit": 25}
    assert as_node == "tools"
    repaired = values["messages"]
    assert finished in repaired
    rejected = [message for message in repaired if message.tool_call_id == "call-rejected"]
    assert len(rejected) == 1
    assert rejected[0].status == "error"


# ============ _check_and_raise_if_interrupted ============


def test_check_and_raise_if_interrupted_raises_only_for_interrupted_status():
    # Given: 三种 turn 状态分别对应中断/正常完成/取消。
    from agent.agent_core import AgentCore, AgentTurnResult

    core = object.__new__(AgentCore)

    # When/Then: 只有 interrupted 状态触发异常，且异常信息包含 resume 指引。
    with pytest.raises(RuntimeError, match="resume_structured"):
        core._check_and_raise_if_interrupted(AgentTurnResult.interrupted([]))

    assert core._check_and_raise_if_interrupted(AgentTurnResult.completed("ok")) is None
    assert core._check_and_raise_if_interrupted(AgentTurnResult.cancelled("no")) is None


def test_run_and_chat_share_the_same_interrupt_error(monkeypatch, capsys):
    # Given: run_structured 与 chat_structured 都返回 interrupted turn。
    from agent.agent_core import AgentCore, AgentTurnResult

    interrupted = AgentTurnResult.interrupted([])
    core = object.__new__(AgentCore)
    core.run_structured = lambda task: interrupted
    core.chat_structured = lambda message: interrupted
    core._fallback_chat = lambda message: pytest.fail("中断不应降级到 fallback")

    # When: 两个公开入口分别执行。
    with pytest.raises(RuntimeError) as run_error:
        core.run("task")
    with pytest.raises(RuntimeError) as chat_error:
        core.chat("hello")

    # Then: 统一的检查方法让两条路径抛出完全相同的错误信息。
    assert str(run_error.value) == str(chat_error.value)
    assert "resume with resume_structured()" in str(run_error.value)
    capsys.readouterr()


# ============ run() / chat() 异常处理 ============


def test_run_returns_error_message_for_runtime_and_generic_failures(capsys):
    # Given: run_structured 分别抛出非中断 RuntimeError 和普通 Exception。
    from agent.agent_core import AgentCore

    core = object.__new__(AgentCore)

    def _raise(exc):
        core.run_structured = lambda task: (_ for _ in ()).throw(exc)
        return core.run("task")

    # When: 两类异常先后发生。
    runtime_result = _raise(RuntimeError("boom"))
    generic_result = _raise(ValueError("bad value"))

    # Then: 两类异常都被转换成同一格式的错误文本而不是向外抛出。
    assert runtime_result == "任务执行失败: boom"
    assert generic_result == "任务执行失败: bad value"
    assert "错误: 任务执行失败: boom" in capsys.readouterr().out


def test_run_reraises_runtime_error_mentioning_interrupt(capsys):
    # Given: run_structured 抛出内部包含 interrupt 字样的 RuntimeError。
    from agent.agent_core import AgentCore

    core = object.__new__(AgentCore)
    core.run_structured = lambda task: (_ for _ in ()).throw(
        RuntimeError("Agent turn interrupted; resume with resume_structured().")
    )

    # When/Then: 中断异常必须原样抛出，不能被降级成错误字符串。
    with pytest.raises(RuntimeError, match="interrupted"):
        core.run("task")
    capsys.readouterr()


def test_chat_falls_back_for_non_interrupt_failures():
    # Given: chat_structured 抛出非中断的 RuntimeError 与普通 Exception。
    from agent.agent_core import AgentCore

    core = object.__new__(AgentCore)
    core._fallback_chat = lambda message: f"fallback:{message}"

    def _raise(exc):
        core.chat_structured = lambda message: (_ for _ in ()).throw(exc)
        return core.chat("hello")

    # When: 两类异常先后发生。
    runtime_result = _raise(RuntimeError("model unavailable"))
    generic_result = _raise(ValueError("bad payload"))

    # Then: 两类异常都降级为纯 LLM 对话。
    assert runtime_result == "fallback:hello"
    assert generic_result == "fallback:hello"


def test_chat_reraises_langgraph_interrupt_exceptions():
    # Given: LangGraph 抛出 GraphInterrupt 这类按类名识别的中断异常。
    from agent.agent_core import AgentCore

    class GraphInterrupt(Exception):
        pass

    core = object.__new__(AgentCore)
    core._fallback_chat = lambda message: pytest.fail("中断不应降级到 fallback")
    core.chat_structured = lambda message: (_ for _ in ()).throw(GraphInterrupt("paused"))

    # When/Then: 中断异常原样抛出。
    with pytest.raises(GraphInterrupt):
        core.chat("hello")


# ============ _parse_turn_result ============


def test_parse_turn_result_prefers_interrupts_over_messages():
    # Given: 结果里同时存在 __interrupt__ 和可用的 AIMessage。
    from agent.agent_core import AgentCore

    core = object.__new__(AgentCore)

    # When: 解析该结果。
    turn = core._parse_turn_result(
        {"__interrupt__": ["pause"], "messages": [AIMessage(content="ignored")]}
    )

    # Then: 中断优先，输出不被填充。
    assert turn.status == "interrupted"
    assert turn.interrupts == ["pause"]
    assert turn.output is None


def test_parse_turn_result_returns_last_ai_message_with_content():
    # Given: 消息历史里有多条 AIMessage，末尾还有空内容的 AIMessage 和工具消息。
    from agent.agent_core import AgentCore

    core = object.__new__(AgentCore)
    messages = [
        SystemMessage(content="system"),
        HumanMessage(content="question"),
        AIMessage(content="first answer"),
        AIMessage(content="final answer"),
        AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "call-1"}]),
        ToolMessage(content="tool output", tool_call_id="call-1"),
    ]

    # When: 解析该结果。
    turn = core._parse_turn_result({"messages": messages})

    # Then: 取最后一条有内容的 AIMessage，忽略空内容与非 AI 消息。
    assert turn.status == "completed"
    assert turn.output == "final answer"


@pytest.mark.parametrize(
    "result",
    [
        {},
        {"messages": []},
        {"__interrupt__": [], "messages": []},
        {"messages": [HumanMessage(content="only human")]},
        {"messages": [AIMessage(content="")]},
    ],
    ids=["no-keys", "empty-messages", "empty-interrupt", "no-ai-message", "blank-ai-content"],
)
def test_parse_turn_result_completes_with_empty_output_when_no_answer(result):
    # Given: 结果中没有可用的 AI 回答，且 __interrupt__ 为空或缺失。
    from agent.agent_core import AgentCore

    core = object.__new__(AgentCore)

    # When: 解析该结果。
    turn = core._parse_turn_result(result)

    # Then: 视为 completed 且输出为空字符串，避免 None 泄漏给调用方。
    assert turn.status == "completed"
    assert turn.output == ""


def test_parse_turn_result_stringifies_non_string_content():
    # Given: AIMessage 的 content 是结构化的内容块列表。
    from agent.agent_core import AgentCore

    core = object.__new__(AgentCore)
    blocks = [{"type": "text", "text": "hello"}]

    # When: 解析该结果。
    turn = core._parse_turn_result({"messages": [AIMessage(content=blocks)]})

    # Then: 输出被转成字符串，保证返回类型稳定。
    assert turn.status == "completed"
    assert turn.output == str(blocks)
