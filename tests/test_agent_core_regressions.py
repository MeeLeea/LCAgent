from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from tools.terminal_tools import UserRejectedCommandError


def test_compaction_retains_recent_messages_in_new_thread_state():
    # Given: the current thread has more messages than the compaction threshold.
    from agent.agent_core import AgentCore

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
    core._summarize_messages = lambda batch: summarized_batches.append(batch) or "summary"

    # When: compaction runs.
    core._maybe_compact()

    # Then: only older messages are summarized and retained messages seed the new thread.
    assert summarized_batches == [messages[:3]]
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
    core._summarize_messages = lambda batch: ""

    # When: compaction runs.
    core._maybe_compact()

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
    core._maybe_compact = lambda: None
    core._compute_skill_block = lambda task: ""
    core._create_agent_executor = lambda skill_block="": executor

    # When: 当前任务尝试执行被拒绝的命令。
    turn = core.run_structured("commit changes")

    # Then: 当前 turn 立即取消，不会把错误交回模型形成下一轮重试。
    assert executor.calls == 1
    assert turn.status == "cancelled"
    assert turn.output == "用户已拒绝执行危险命令，当前任务已取消。"


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
    core._maybe_compact = lambda: None
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
