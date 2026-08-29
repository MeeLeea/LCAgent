"""TeamAgent 中断/恢复语义测试 - 对齐 agent/ 模块的 HITL 能力

覆盖:
- checkpointer 注入:_checkpointer 与 agent_executor 编译时带 checkpointer
- 工具内 interrupt() → arun_structured 返回 interrupted
- 纯文本模式 → arun_structured 返回 completed(不检查 interrupt)
- interrupt → resume(approve) → completed
- interrupt → resume(deny) → cancelled(UserRejectedCommandError 语义)
- 多轮 interrupt → 多次 resume → completed

参考:
- tests/agent/test_human_input.py 的 _FakeExecutor + aget_state 模式(L126-313)
- tests/team/test_team_metrics.py 的 _FakeLLM/_FakeExecutor/_monkeypatch 模式
- team/base.py:arun_structured/aresume_structured/_aget_pending_interrupts 签名
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import ClassVar

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, Interrupt

from agent.turn_types import AgentTurnResult
from team.base import TeamAgent
from tools.terminal_tools import UserRejectedCommandError


@tool
def _echo(message: str) -> str:
    """回显输入文本"""
    return message


class _FakeModel:
    """假 chat model:可配置产出 chunk 序列"""

    def __init__(self, chunks=None) -> None:
        self._chunks = chunks or []

    async def astream(self, messages, config=None):
        for chunk in self._chunks:
            yield chunk


class _FakeLLM:
    """替换 team.base.LLMClient,避免测试依赖 API 密钥与网络"""

    instances: ClassVar[list[_FakeLLM]] = []

    def __init__(self, *args, **kwargs) -> None:
        self.provider = "test"
        self.model = "fake-model"
        self._model = _FakeModel()
        _FakeLLM.instances.append(self)

    def get_chat_model(self):
        return self._model

    def chat(self, messages) -> str:
        return ""


def _stream_event(content: str) -> dict:
    """构造 on_chat_model_stream 事件(文本 token 增量)"""
    return {
        "event": "on_chat_model_stream",
        "data": {"chunk": AIMessageChunk(content=content)},
        "name": "FakeModel",
    }


def _interrupt_value(kind: str = "human_choice") -> dict:
    """构造 interrupt value(对齐 cli/human_input.ask_human 的 payload 形态)"""
    return {
        "kind": kind,
        "prompt": "确认执行?",
        "choices": [{"id": "approve", "label": "确认"}, {"id": "deny", "label": "拒绝"}],
    }


def _make_interrupt(interrupt_id: str = "i-1", kind: str = "human_choice") -> Interrupt:
    """构造单个 Interrupt 对象"""
    return Interrupt(value=_interrupt_value(kind), id=interrupt_id)


def _make_state(interrupt_ids: list[str] | None = None) -> SimpleNamespace:
    """构造 aget_state 返回值:tasks[].interrupts 列表

    Args:
        interrupt_ids: 非空时构造对应 id 的 pending interrupts;空列表/None 返回空 tasks

    Returns:
        类似 LangGraph StateSnapshot 的 SimpleNamespace(tasks 字段为列表)
    """
    if not interrupt_ids:
        return SimpleNamespace(tasks=[])
    tasks = [
        SimpleNamespace(interrupts=[_make_interrupt(tid)]) for tid in interrupt_ids
    ]
    return SimpleNamespace(tasks=tasks)


class _InterruptExecutor:
    """带状态的假 agent_executor,支持 interrupt/resume 全流程

    状态机:
    - _pending_ids 非空:aget_state 返回含 interrupts 的 state(模拟工具内 interrupt())
    - astream_events:正常产出 token 事件(流式文本不中断)
    - ainvoke(resume_command):按 _resume_behavior 决定结果
        - "complete":清空 pending,返回 {messages:[AIMessage(...)]}(恢复后完成)
        - "deny":raise UserRejectedCommandError(用户拒绝危险命令)
        - "interrupt_again":切换到下一组 pending(再次中断)
    """

    def __init__(
        self,
        stream_chunks: list[str] | None = None,
        initial_pending_ids: list[str] | None = None,
        resume_behavior: str = "complete",
        next_pending_ids: list[str] | None = None,
    ) -> None:
        self._stream_chunks = stream_chunks or ["正在执行"]
        self._pending_ids: list[str] = list(initial_pending_ids or [])
        self._resume_behavior = resume_behavior
        self._next_pending_ids: list[str] | None = next_pending_ids
        self.ainvoke_calls: list[tuple] = []
        self.aget_state_calls: list[dict] = []

    async def astream_events(self, messages, config=None, version="v2"):
        """产出 token 流式事件(不模拟 interrupt,由 aget_state 返回 pending)"""
        for chunk in self._stream_chunks:
            yield _stream_event(chunk)

    async def aget_state(self, config):
        """返回当前 pending interrupts 对应的 state"""
        self.aget_state_calls.append(config)
        return _make_state(list(self._pending_ids))

    async def ainvoke(self, command, config):
        """处理 resume:按 _resume_behavior 决定完成/拒绝/再次中断"""
        self.ainvoke_calls.append((command, config))
        if self._resume_behavior == "deny":
            raise UserRejectedCommandError("用户拒绝执行危险命令")
        if self._resume_behavior == "interrupt_again":
            # 切换到下一组 pending(多轮 interrupt 场景)
            if self._next_pending_ids is not None:
                self._pending_ids = list(self._next_pending_ids)
                self._next_pending_ids = None
            return {"__interrupt__": [_make_interrupt(i) for i in self._pending_ids]}
        # 默认 complete:清空 pending,返回最终消息
        self._pending_ids = []
        return {"messages": [AIMessage(content="已恢复执行")]}


def _set_fake_model(agent, chunks: list[str] | None = None) -> _FakeLLM:
    """把最新 _FakeLLM 的模型替换为指定 chunk 序列"""
    llm = _FakeLLM.instances[-1]
    llm._model = _FakeModel(chunks=[AIMessageChunk(content=c) for c in (chunks or [])])
    return llm


def _monkeypatch_llm(monkeypatch) -> None:
    monkeypatch.setattr("team.base.LLMClient", _FakeLLM)


def _monkeypatch_executor(monkeypatch, executor) -> None:
    monkeypatch.setattr("langchain.agents.create_agent", lambda **kw: executor)


def _monkeypatch_executor_capturing(monkeypatch, capture: dict) -> None:
    """捕获 create_agent 的 kwargs(用于验证 checkpointer 注入)"""

    def _fake_create_agent(**kwargs):
        capture["kwargs"] = kwargs
        # 返回一个占位 executor,后续由测试自行替换
        return _InterruptExecutor()

    monkeypatch.setattr("langchain.agents.create_agent", _fake_create_agent)


# ── checkpointer 注入 ─────────────────────────────────────────────


def test_team_agent_checkpointer_injection(monkeypatch):
    """构造带 checkpointer + tools 的 TeamAgent:验证 _checkpointer 已存,
    且 create_agent 编译时收到同一 checkpointer(agent_executor 带它,
    使 interrupt/resume 在工具模式可用)

    Given:checkpointer=MemorySaver() + tools=[_echo]
    When:构造 TeamAgent(触发 _create_tool_agent → create_agent)
    Then:self._checkpointer is MemorySaver;create_agent 收到同 checkpointer
    """
    _monkeypatch_llm(monkeypatch)
    capture: dict = {}
    _monkeypatch_executor_capturing(monkeypatch, capture)

    checkpointer = MemorySaver()
    agent = TeamAgent(name="t", tools=[_echo], system_prompt="sys", checkpointer=checkpointer)

    # _checkpointer 属性保留原值
    assert agent._checkpointer is checkpointer
    # create_agent 编译时收到同一 checkpointer(使 agent_executor 带 checkpoint 能力)
    assert capture["kwargs"].get("checkpointer") is checkpointer
    # agent_executor 已创建(工具模式)
    assert agent.agent_executor is not None


# ── arun_structured 返回 interrupted ──────────────────────────────


def test_arun_structured_returns_interrupted(monkeypatch):
    """工具内 interrupt() → arun_structured 返回 interrupted

    Given:工具模式,executor.astream_events 正常产出 token,
        aget_state 返回含 pending interrupt 的 state
    When:arun_structured 执行任务
    Then:结果 status == interrupted;interrupts 为 pending 列表;
        output 为 None(interrupted 不带 output)
    """
    _monkeypatch_llm(monkeypatch)
    executor = _InterruptExecutor(
        stream_chunks=["思考中", "需要确认"],
        initial_pending_ids=["i-1"],
    )
    _monkeypatch_executor(monkeypatch, executor)

    agent = TeamAgent(name="t", tools=[_echo], system_prompt="sys", checkpointer=MemorySaver())
    # checkpointer 启用时 _build_run_config 需从外层 config 解析 thread_id
    config = {"configurable": {"thread_id": "test-thread-interrupt"}}

    result = asyncio.run(agent.arun_structured("task", config))

    assert isinstance(result, AgentTurnResult)
    assert result.status == "interrupted"
    assert result.is_interrupted is True
    assert result.is_completed is False
    assert result.output is None
    assert len(result.interrupts) == 1
    assert isinstance(result.interrupts[0], Interrupt)
    assert result.interrupts[0].id == "i-1"
    # aget_state 至少被调用一次(_aget_pending_interrupts)
    assert len(executor.aget_state_calls) >= 1


# ── arun_structured 纯文本模式 completed ─────────────────────────


def test_arun_structured_completed_pure_text(monkeypatch):
    """纯文本模式 → arun_structured 返回 completed(不检查 interrupt)

    Given:纯文本模式(无 tools/agent_executor),LLM 流式产出 "answer"
    When:arun_structured 执行任务
    Then:status == completed;output == "answer";interrupts 为空
    """
    _monkeypatch_llm(monkeypatch)
    agent = TeamAgent(name="t", system_prompt="sys")
    _set_fake_model(agent, chunks=["answer"])

    result = asyncio.run(agent.arun_structured("task"))

    assert isinstance(result, AgentTurnResult)
    assert result.status == "completed"
    assert result.is_completed is True
    assert result.output == "answer"
    assert result.interrupts == []


# ── aresume_structured 恢复后完成 ─────────────────────────────────


def test_aresume_structured_completes_after_resume(monkeypatch):
    """interrupt → resume(approve) → completed

    Given:工具模式 + checkpointer,executor 初始有 pending interrupt,
        resume behavior 为 complete(清空 pending 返回最终消息)
    When:aresume_structured({"choice_id": "approve"})
    Then:status == completed;output 为最终消息;
        ainvoke 收到 Command(resume=...);thread_id 经 _build_run_config 隔离
    """
    _monkeypatch_llm(monkeypatch)
    executor = _InterruptExecutor(
        stream_chunks=["等待确认"],
        initial_pending_ids=["i-1"],
        resume_behavior="complete",
    )
    _monkeypatch_executor(monkeypatch, executor)

    agent = TeamAgent(name="t", tools=[_echo], system_prompt="sys", checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "test-thread-resume-complete"}}

    result = asyncio.run(agent.aresume_structured({"choice_id": "approve"}, config))

    assert isinstance(result, AgentTurnResult)
    assert result.status == "completed"
    assert result.is_completed is True
    assert result.output == "已恢复执行"
    # ainvoke 收到 Command(resume=...)
    assert len(executor.ainvoke_calls) == 1
    command, _config = executor.ainvoke_calls[0]
    assert isinstance(command, Command)
    assert command.resume == {"choice_id": "approve"}
    # 内层 run_config 的 thread_id 经 _build_run_config 改写为隔离形式
    assert "configurable" in _config
    assert _config["configurable"]["thread_id"] == "team:t:test-thread-resume-complete"


# ── aresume_structured 拒绝 → cancelled ──────────────────────────


def test_aresume_structured_deny_returns_cancelled(monkeypatch):
    """interrupt → resume(deny) → cancelled(UserRejectedCommandError 语义)

    Given:工具模式 + checkpointer,executor 初始有 pending interrupt,
        resume behavior 为 deny(ainvoke raise UserRejectedCommandError)
    When:aresume_structured({"choice_id": "deny"})
    Then:status == cancelled;output 含"用户拒绝"语义;
        不含通用"任务执行失败"前缀(对齐 turn_runners.py:52-53)
    """
    _monkeypatch_llm(monkeypatch)
    executor = _InterruptExecutor(
        stream_chunks=["等待确认"],
        initial_pending_ids=["i-1"],
        resume_behavior="deny",
    )
    _monkeypatch_executor(monkeypatch, executor)

    agent = TeamAgent(name="t", tools=[_echo], system_prompt="sys", checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "test-thread-resume-deny"}}

    result = asyncio.run(agent.aresume_structured({"choice_id": "deny"}, config))

    assert isinstance(result, AgentTurnResult)
    assert result.status == "cancelled"
    assert result.is_completed is False
    assert result.output is not None
    # "用户已拒绝"是 aresume_structured 的 cancelled output 文本(对齐 turn_runners)
    assert "用户已拒绝" in result.output
    assert not result.output.startswith("任务执行失败")


# ── 多轮 interrupt → 多次 resume → completed ─────────────────────


def test_multi_round_interrupt(monkeypatch):
    """多次 interrupt → 多次 resume → completed

    Given:工具模式 + checkpointer,executor 初始有 pending interrupt i-1,
        resume behavior 为 interrupt_again(恢复后切换到下一组 pending i-2),
        需要再次 resume 才完成
    When:arun_structured 触发首次 interrupt(i-1);
        第一次 resume → 再次 interrupted(i-2);
        第二次 resume → completed
    Then:两轮 interrupt + 两次 resume,最终 completed
    """
    _monkeypatch_llm(monkeypatch)

    # 第一阶段 executor:arun_structured 用,产出 token + pending i-1
    first_executor = _InterruptExecutor(
        stream_chunks=["需要第一次确认"],
        initial_pending_ids=["i-1"],
    )
    _monkeypatch_executor(monkeypatch, first_executor)

    agent = TeamAgent(name="t", tools=[_echo], system_prompt="sys", checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "test-thread-multi-round"}}

    # 第一次 arun_structured → interrupted(i-1)
    first_result = asyncio.run(agent.arun_structured("task", config))
    assert first_result.status == "interrupted"
    assert first_result.interrupts[0].id == "i-1"

    # 切换为 resume executor:第一次 resume → 再次 interrupt(i-2)
    # 第二次 resume → complete
    # 用 _InterruptExecutor 模拟 interrupt_again 一次后 complete
    resume_executor = _InterruptExecutor(
        stream_chunks=["需要第二次确认"],
        initial_pending_ids=["i-1"],  # 当前 pending(i-1,由 _abuild_resume_command 读)
        resume_behavior="interrupt_again",
        next_pending_ids=["i-2"],  # resume 后切换到 i-2
    )
    agent.agent_executor = resume_executor

    # 第一次 resume → interrupted(i-2)
    second_result = asyncio.run(agent.aresume_structured({"choice_id": "approve"}, config))
    assert second_result.status == "interrupted"
    assert second_result.interrupts[0].id == "i-2"

    # 第二次 resume → complete:需要 ainvoke 返回 messages(无 __interrupt__),
    # 且 aget_state 返回空 pending
    final_executor = _InterruptExecutor(
        stream_chunks=["最终完成"],
        initial_pending_ids=["i-2"],
        resume_behavior="complete",
    )
    agent.agent_executor = final_executor

    third_result = asyncio.run(agent.aresume_structured({"choice_id": "approve"}, config))
    assert third_result.status == "completed"
    assert third_result.output == "已恢复执行"
