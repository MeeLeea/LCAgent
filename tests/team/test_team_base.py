"""TeamAgent 基类测试 - 覆盖 system_prompt 自动解析与工作流模板缓存

重点验证「只需 prompt_file」的初始化语义:
- system_prompt 为空时自动从 AGENT.md 解析(剥离 ## workflow:* 小节)
- 显式传入的 system_prompt 优先
- __init__ 解析出的模板缓存被 get_template 复用,不重复读文件
- 孤儿 tool_call 修复(_arepair_orphan_tool_calls):拒绝/异常残留的
  tool_calls 无 ToolMessage 配对时补齐取消结果,防下次开跑 OpenAI 400
"""
import asyncio
from types import SimpleNamespace
from typing import ClassVar

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from team.base import TeamAgent
from tools.terminal_tools import UserRejectedCommandError


class _FakeLLM:
    """替换真实 LLMClient,避免测试依赖 API 密钥与网络"""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def get_chat_model(self):
        return None

    def chat(self, messages) -> str:
        return ""


class _FallbackAgent(TeamAgent):
    """带默认工作流模板的子类,验证缺失小节时回退"""

    default_templates: ClassVar[dict[str, str]] = {"manager_plan": "默认计划模板"}


def _monkeypatch_llm(monkeypatch) -> None:
    """把 team.base 中的 LLMClient 替换为假实现"""
    monkeypatch.setattr("team.base.LLMClient", _FakeLLM)


def test_auto_parse_system_prompt_from_prompt_file(tmp_path, monkeypatch):
    """只传 prompt_file 时,system_prompt 自动解析且剥离 workflow 小节"""
    _monkeypatch_llm(monkeypatch)

    agent_md = tmp_path / "AGENT.md"
    agent_md.write_text(
        "# 角色提示词\n"
        "你是测试经理。\n\n"
        "## 重要规则\n"
        "1. 规则A\n\n"
        "## workflow:manager_plan\n"
        "请制定计划:{task}\n",
        encoding="utf-8",
    )

    agent = TeamAgent(name="test", prompt_file=str(agent_md))

    assert "你是测试经理" in agent.system_prompt
    assert "## 重要规则" in agent.system_prompt
    # 模板小节不应混入系统提示词
    assert "workflow:" not in agent.system_prompt
    assert "请制定计划" not in agent.system_prompt
    # 模板可正常读取
    assert agent.get_template("manager_plan") == "请制定计划:{task}"


def test_explicit_system_prompt_wins(tmp_path, monkeypatch):
    """显式传入 system_prompt 时,不覆盖为 prompt_file 内容"""
    _monkeypatch_llm(monkeypatch)

    agent_md = tmp_path / "AGENT.md"
    agent_md.write_text("# 文件提示词\n", encoding="utf-8")

    agent = TeamAgent(
        name="test",
        system_prompt="显式提示词",
        prompt_file=str(agent_md),
    )

    assert agent.system_prompt == "显式提示词"


def test_missing_prompt_file_falls_back(tmp_path, monkeypatch):
    """prompt_file 不存在时 system_prompt 为空,get_template 回退默认模板"""
    _monkeypatch_llm(monkeypatch)

    agent = _FallbackAgent(
        name="test",
        prompt_file=str(tmp_path / "not_exists.md"),
    )

    assert agent.system_prompt == ""
    assert agent.get_template("manager_plan") == "默认计划模板"


def test_template_cache_reuses_init_parse(tmp_path, monkeypatch):
    """__init__ 已解析模板时,get_template 复用缓存,不重复读文件"""
    _monkeypatch_llm(monkeypatch)

    agent_md = tmp_path / "AGENT.md"
    agent_md.write_text(
        "# 提示词\n\n"
        "## workflow:manager_plan\n"
        "计划:{task}\n",
        encoding="utf-8",
    )

    agent = TeamAgent(name="test", prompt_file=str(agent_md))
    # 预热:确认模板已解析
    assert agent.get_template("manager_plan") == "计划:{task}"

    # 删除源文件后模板仍可读取 → 证明走的是 __init__ 建立的缓存而非重新读盘
    agent_md.unlink()
    assert agent.get_template("manager_plan") == "计划:{task}"


# ==================== 孤儿 tool_call 修复(_arepair_orphan_tool_calls) ====================


class _FakeExecutor:
    """模拟 agent_executor:记录方法调用顺序与 aupdate_state 参数"""

    def __init__(self, state=None, events=None, error=None):
        self.state = state  # StateSnapshot 或 Exception(aget_state 时抛出)
        self.events = events or []
        self.error = error  # astream_events 时抛出的异常
        self.updates: list[dict] = []
        self.call_order: list[str] = []
        self.seen_configs: list[dict] = []

    async def aget_state(self, config):
        self.call_order.append("aget_state")
        self.seen_configs.append(config)
        if isinstance(self.state, Exception):
            raise self.state
        return self.state

    async def aupdate_state(self, config, values):
        self.call_order.append("aupdate_state")
        self.updates.append(values)
        # 模拟 add_messages reducer:追加修复消息,使后续 aget_state 可见
        if self.state is not None and not isinstance(self.state, Exception):
            self.state.values["messages"] = [
                *self.state.values["messages"],
                *values.get("messages", []),
            ]

    async def astream_events(self, input, config=None, version=None):
        self.call_order.append("astream_events")
        if self.error is not None:
            raise self.error
        for ev in self.events:
            yield ev


def _make_repair_agent(executor: _FakeExecutor) -> TeamAgent:
    """构造仅含修复所需属性的最小 TeamAgent(绕过 __init__)"""
    agent = object.__new__(TeamAgent)
    agent.name = "test"
    agent.verbose = False
    agent.max_iterations = 25  # _build_run_config 需要
    agent.agent_executor = executor
    agent._checkpointer = object()  # 非 None 即启用修复路径
    return agent


def _state_with(messages, interrupts=None):
    """构造 aget_state 返回的 StateSnapshot 替身"""
    tasks = [SimpleNamespace(interrupts=interrupts or [])]
    return SimpleNamespace(values={"messages": messages}, tasks=tasks)


def test_repair_orphan_tool_calls_appends_cancel_result():
    """孤儿 tool_call 无 ToolMessage 配对时,补齐 status=error 的取消结果"""
    orphan = AIMessage(
        content="",
        tool_calls=[{"name": "run_python", "args": {"code": "x"}, "id": "run_python_16"}],
    )
    executor = _FakeExecutor(state=_state_with([HumanMessage("任务"), orphan]))
    agent = _make_repair_agent(executor)

    asyncio.run(agent._arepair_orphan_tool_calls({"configurable": {"thread_id": "t"}}))

    assert len(executor.updates) == 1
    repairs = executor.updates[0]["messages"]
    assert len(repairs) == 1
    assert repairs[0].tool_call_id == "run_python_16"
    assert repairs[0].name == "run_python"
    assert repairs[0].status == "error"
    assert "未完成" in repairs[0].content


def test_repair_skips_answered_tool_calls():
    """已有 ToolMessage 配对的 tool_call 不重复补写"""
    ai = AIMessage(
        content="",
        tool_calls=[{"name": "run_python", "args": {}, "id": "call_1"}],
    )
    answered = ToolMessage(content="ok", name="run_python", tool_call_id="call_1")
    executor = _FakeExecutor(state=_state_with([HumanMessage("任务"), ai, answered]))
    agent = _make_repair_agent(executor)

    asyncio.run(agent._arepair_orphan_tool_calls({}))

    assert executor.updates == []


def test_repair_skips_pending_interrupts():
    """有 pending interrupts 时跳过修复(合法暂停,等 resume 补结果)"""
    orphan = AIMessage(
        content="",
        tool_calls=[{"name": "run_python", "args": {}, "id": "call_1"}],
    )
    executor = _FakeExecutor(
        state=_state_with(
            [HumanMessage("任务"), orphan],
            interrupts=[SimpleNamespace(id="intr_1")],
        )
    )
    agent = _make_repair_agent(executor)

    asyncio.run(agent._arepair_orphan_tool_calls({}))

    assert executor.updates == []


def test_repair_silent_on_aget_state_error():
    """aget_state 抛异常时静默返回,不影响本次执行"""
    executor = _FakeExecutor(state=RuntimeError("checkpoint 不可达"))
    agent = _make_repair_agent(executor)

    # 不抛异常即通过
    asyncio.run(agent._arepair_orphan_tool_calls({}))
    assert executor.updates == []


def test_repair_noop_without_checkpointer():
    """未启用 checkpointer 时直接返回(无 checkpoint 可修)"""
    executor = _FakeExecutor(state=_state_with([]))
    agent = _make_repair_agent(executor)
    agent._checkpointer = None

    asyncio.run(agent._arepair_orphan_tool_calls({}))

    assert executor.call_order == []


def _collect_chunks(agent: TeamAgent, task: str, config):
    """消费 _astream_with_tools 的全部输出(异常向上传播)"""

    async def run():
        out = []
        async for chunk in agent._astream_with_tools(task, config):
            out.append(chunk)
        return out

    return asyncio.run(run())


def test_astream_with_tools_repairs_before_run():
    """开跑前防御性修复:历史残留孤儿 tool_call 在 astream_events 之前被补齐"""
    orphan = AIMessage(
        content="",
        tool_calls=[{"name": "run_python", "args": {}, "id": "run_python_16"}],
    )
    executor = _FakeExecutor(state=_state_with([HumanMessage("旧任务"), orphan]))
    agent = _make_repair_agent(executor)

    chunks = _collect_chunks(agent, "新任务", {"configurable": {"thread_id": "tid"}})

    # 修复先于执行:顺序为 aget_state → aupdate_state → astream_events
    assert executor.call_order == ["aget_state", "aupdate_state", "astream_events"]
    assert chunks == []
    # 内层隔离 thread_id 正确传入
    assert executor.seen_configs[0]["configurable"]["thread_id"] == "team:test:tid"


def test_astream_with_tools_repairs_on_user_rejection():
    """用户拒绝危险命令(UserRejectedCommandError)时当次修复后再 re-raise"""
    orphan = AIMessage(
        content="",
        tool_calls=[{"name": "run_python", "args": {}, "id": "run_python_16"}],
    )
    executor = _FakeExecutor(
        state=_state_with([HumanMessage("任务"), orphan]),
        error=UserRejectedCommandError("rm -rf /"),
    )
    agent = _make_repair_agent(executor)

    try:
        _collect_chunks(agent, "任务", {"configurable": {"thread_id": "tid"}})
        raise AssertionError("应向上传播 UserRejectedCommandError")
    except UserRejectedCommandError:
        pass

    # 开跑前修复(1 次 update)+ 拒绝时再次检查(孤儿已补齐,无新 update)
    assert executor.call_order == [
        "aget_state",
        "aupdate_state",
        "astream_events",
        "aget_state",
    ]
    assert len(executor.updates) == 1
