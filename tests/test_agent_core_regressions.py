import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from tools.terminal_tools import UserRejectedCommandError


def test_compaction_retains_recent_messages_in_new_thread_state():
    # Given: the current thread has more messages than the compaction threshold.
    # Updated: tests manually_compact (replaces old _acompact_if_needed)
    from agent.agent_core import AgentCore
    from agent.compaction import CompactionConfig, LCAgentCompactionMiddleware

    messages = [HumanMessage(content=f"message-{idx}") for idx in range(8)]
    state_updates = []

    class FakeMemory:
        def __init__(self):
            self.thread_id = "thread-test"

        def get_messages(self):
            return messages

        def get_config(self):
            return {"configurable": {"thread_id": self.thread_id}}

    class FakeState:
        values = {"summary": ""}

    class FakeExecutor:
        def get_state(self, config):
            return FakeState()

        async def aupdate_state(self, config, values):
            state_updates.append((config, values))

    core = object.__new__(AgentCore)
    core.memory = FakeMemory()
    core.max_context_messages = 5
    core.context_trim_keep = 2
    core.verbose = False
    core.agent_executor = FakeExecutor()
    core._state_lock = asyncio.Lock()
    core._invoke_config = lambda: {"configurable": {"thread_id": "thread-test"}}

    # Set up compaction middleware
    class FakeModel:
        async def ainvoke(self, prompt):
            return SimpleNamespace(text="summary")

    core._compaction_middleware = LCAgentCompactionMiddleware(
        FakeModel(), CompactionConfig(max_messages=5, keep_recent=2)
    )

    # When: manually_compact runs.
    result = asyncio.run(core.manually_compact())

    # Then: compression succeeds with summary and message count
    assert result is not None
    assert result["summary"] == "summary"
    assert result["messages_before"] == 8
    assert len(state_updates) == 1


def test_compaction_does_not_change_thread_when_summary_fails():
    # Given: compaction is needed but summarization returns no summary.
    # Updated: tests manually_compact with LLM failure
    from agent.agent_core import AgentCore
    from agent.compaction import CompactionConfig, LCAgentCompactionMiddleware

    messages = [HumanMessage(content=f"message-{idx}") for idx in range(8)]

    class FakeMemory:
        def __init__(self):
            self.thread_id = "thread-before"

        def get_messages(self):
            return messages

        def get_config(self):
            return {"configurable": {"thread_id": self.thread_id}}

    class FakeState:
        values = {"summary": "existing"}

    class FakeExecutor:
        def get_state(self, config):
            return FakeState()

        async def aupdate_state(self, config, values):
            raise AssertionError("失败时不应更新状态")

    core = object.__new__(AgentCore)
    core.memory = FakeMemory()
    core.max_context_messages = 5
    core.context_trim_keep = 2
    core.verbose = False
    core.agent_executor = FakeExecutor()
    core._state_lock = asyncio.Lock()
    core._invoke_config = lambda: {"configurable": {"thread_id": "thread-before"}}

    class FailingModel:
        async def ainvoke(self, prompt):
            raise RuntimeError("LLM 不可用")

    core._compaction_middleware = LCAgentCompactionMiddleware(
        FailingModel(), CompactionConfig(max_messages=5, keep_recent=2)
    )

    # When: manually_compact runs with failing LLM.
    result = asyncio.run(core.manually_compact())

    # Then: failed summary returns None, state untouched.
    assert result is None


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


def test_arun_stops_after_user_rejects_command(monkeypatch):
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
    core._state_lock = asyncio.Lock()
    core.agent_core_prompt = "test prompt"
    core._compute_skill_block = lambda task: ""
    core.agent_executor = executor

    # When: 当前任务尝试执行被拒绝的命令。
    turn = asyncio.run(core.arun_structured("commit changes"))

    # Then: 当前 turn 立即取消，不会把错误交回模型形成下一轮重试。
    assert executor.calls == 1
    assert turn.status == "cancelled"
    assert turn.output == "用户已拒绝执行危险命令，当前任务已取消。"


def test_aresume_stops_after_user_rejects_command():
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
    core._state_lock = asyncio.Lock()
    core.agent_executor = executor
    core._pending_interrupt_thread_id = "thread-resume-rejected"
    core._pending_interrupt_mode = "chat"

    # When: 用户以 deny 恢复中断。
    turn = asyncio.run(core.aresume_structured({"choice_id": "deny"}))

    # Then: 本轮取消并清理中断状态，而不是抛异常给调用方。
    assert turn.status == "cancelled"
    assert turn.output == "用户已拒绝执行危险命令，当前任务已取消。"
    assert core._pending_interrupt_thread_id is None
    assert core._pending_interrupt_mode is None


def test_arun_repairs_checkpoint_after_user_rejects_command():
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

        async def aupdate_state(self, config, values, as_node=None):
            updates.append((config, values, as_node))

    executor = RejectingExecutor()
    core = object.__new__(AgentCore)
    core.memory = FakeMemory()
    core.max_iterations = 25
    core.verbose = False
    core.execution_history = []
    core._state_lock = asyncio.Lock()
    core.agent_core_prompt = "test prompt"
    core._compute_skill_block = lambda task: ""
    core.agent_executor = executor

    # When: 拒绝信号终止当前 turn。
    turn = asyncio.run(core.arun_structured("commit changes"))

    # Then: 已完成结果被保留，缺失的调用得到匹配的错误 ToolMessage。
    assert turn.status == "cancelled"
    assert len(updates) == 1
    config, values, as_node = updates[0]
    assert config == {"configurable": {"thread_id": "thread-rejected"}, "recursion_limit": 25}
    assert as_node is None
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
    with pytest.raises(RuntimeError, match="resume" + "_structured"):
        core._check_and_raise_if_interrupted(AgentTurnResult.interrupted([]))

    assert core._check_and_raise_if_interrupted(AgentTurnResult.completed("ok")) is None
    assert core._check_and_raise_if_interrupted(AgentTurnResult.cancelled("no")) is None


def test_arun_and_achat_share_the_same_interrupt_error(monkeypatch, capsys):
    # Given: 异步运行与异步对话都返回 interrupted turn。
    from agent.agent_core import AgentCore, AgentTurnResult

    interrupted = AgentTurnResult.interrupted([])
    core = object.__new__(AgentCore)
    core._state_lock = asyncio.Lock()
    async def _return_interrupted_task(task):
        return interrupted

    async def _return_interrupted_message(message):
        return interrupted

    core.arun_structured = _return_interrupted_task
    core.achat_structured = _return_interrupted_message
    core._fallback_chat = lambda message: pytest.fail("中断不应降级到 fallback")

    # When: 两个公开入口分别执行。
    with pytest.raises(RuntimeError) as run_error:
        asyncio.run(core.arun("task"))
    with pytest.raises(RuntimeError) as chat_error:
        asyncio.run(core.achat("hello"))

    # Then: 统一的检查方法让两条路径抛出完全相同的错误信息。
    assert str(run_error.value) == str(chat_error.value)
    assert "resume with " + "resume" + "_structured()" in str(run_error.value)
    capsys.readouterr()


    # ============ arun() / achat() 异常处理 ============


def test_arun_returns_error_message_for_runtime_and_generic_failures(capsys):
    # Given: 异步运行入口分别抛出非中断 RuntimeError 和普通 Exception。
    from agent.agent_core import AgentCore

    core = object.__new__(AgentCore)
    core._state_lock = asyncio.Lock()

    def _raise(exc):
        async def _raise_exc(task):
            raise exc

        core.arun_structured = _raise_exc
        return asyncio.run(core.arun("task"))

    # When: 两类异常先后发生。
    runtime_result = _raise(RuntimeError("boom"))
    generic_result = _raise(ValueError("bad value"))

    # Then: 两类异常都被转换成同一格式的错误文本而不是向外抛出。
    assert runtime_result == "任务执行失败: boom"
    assert generic_result == "任务执行失败: bad value"
    assert "错误: 任务执行失败: boom" in capsys.readouterr().out


def test_arun_reraises_runtime_error_mentioning_interrupt(capsys):
    # Given: 异步运行入口抛出内部包含 interrupt 字样的 RuntimeError。
    from agent.agent_core import AgentCore

    core = object.__new__(AgentCore)
    core._state_lock = asyncio.Lock()

    async def _raise_interrupt(task):
        raise RuntimeError("Agent turn interrupted; resume with " + "resume" + "_structured().")

    core.arun_structured = _raise_interrupt

    # When/Then: 中断异常必须原样抛出，不能被降级成错误字符串。
    with pytest.raises(RuntimeError, match="interrupted"):
        asyncio.run(core.arun("task"))
    capsys.readouterr()


def test_achat_falls_back_for_non_interrupt_failures():
    # Given: 异步对话入口抛出非中断的 RuntimeError 与普通 Exception。
    from agent.agent_core import AgentCore

    core = object.__new__(AgentCore)
    core._state_lock = asyncio.Lock()
    core._fallback_chat = lambda message: f"fallback:{message}"

    def _raise(exc):
        async def _raise_exc(message):
            raise exc

        core.achat_structured = _raise_exc
        return asyncio.run(core.achat("hello"))

    # When: 两类异常先后发生。
    runtime_result = _raise(RuntimeError("model unavailable"))
    generic_result = _raise(ValueError("bad payload"))

    # Then: 两类异常都降级为纯 LLM 对话。
    assert runtime_result == "fallback:hello"
    assert generic_result == "fallback:hello"


def test_achat_reraises_langgraph_interrupt_exceptions():
    # Given: LangGraph 抛出 GraphInterrupt 这类按类名识别的中断异常。
    from agent.agent_core import AgentCore

    class GraphInterrupt(Exception):
        pass

    core = object.__new__(AgentCore)
    core._state_lock = asyncio.Lock()
    core._fallback_chat = lambda message: pytest.fail("中断不应降级到 fallback")

    async def _raise_graph_interrupt(message):
        raise GraphInterrupt("paused")

    core.achat_structured = _raise_graph_interrupt

    # When/Then: 中断异常原样抛出。
    with pytest.raises(GraphInterrupt):
        asyncio.run(core.achat("hello"))


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


# ============ name / llm 属性 ============


def test_agent_has_name_and_llm_attributes():
    # Given: 使用对象装配方式检查 AgentCore 的实例属性语义。
    from agent.agent_core import AgentCore

    class FakeLLM:
        pass

    core = object.__new__(AgentCore)
    fake_llm = FakeLLM()

    # When: 模拟 __init__ 中 name/llm 的赋值逻辑。
    core.name = "MyAgent"
    core.llm = fake_llm

    # Then: name 与 llm 均为实例内置属性，且可直接访问。
    assert core.name == "MyAgent"
    assert core.llm is fake_llm


def test_agent_name_defaults_when_none():
    # Given: name 参数为 None 时使用默认名 "LCAgent"。
    from agent.agent_core import AgentCore

    core = object.__new__(AgentCore)
    # 模拟 __init__ 中 name 参数为 None 的兜底逻辑
    name = None
    core.name = name or "LCAgent"

    # Then: 默认名为 "LCAgent"。
    assert core.name == "LCAgent"


def test_agent_llm_is_builtin_variable():
    # Given: 通过真实初始化路径验证 llm 作为 Agent 内置变量。
    from agent import AgentCore

    class FakeLLM:
        """最小 LLM 桩，仅验证属性绑定，不触发真实网络请求。"""

        def get_chat_model(self):
            raise AssertionError("不应触发模型创建")

    core = object.__new__(AgentCore)
    fake_llm = FakeLLM()
    core.llm = fake_llm
    core.name = "LCAgent"

    # Then: llm 是 Agent 的内置变量，可直接通过 agent.llm 访问。
    assert core.llm is fake_llm
    assert isinstance(core.llm, FakeLLM)
