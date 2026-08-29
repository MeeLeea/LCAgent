"""TeamAgent 指标收集与类型化结果测试 - 对齐 agent/ 模块能力

覆盖:
- metrics 惰性属性(与 AgentCore.metrics 同构,兼容 object.__new__ 实例)
- 纯文本通道 LLM token 用量提取(最后 chunk usage_metadata)
- 工具模式流式指标:on_chat_model_end(LLM)/on_tool_end/on_tool_error(工具)
- turn 计数(astream 入口)
- arun_structured 类型化结果:completed / cancelled 状态映射
- UserRejectedCommandError 语义识别(对齐 turn_runners.py:52-53):
  _astream_with_tools re-raise / arun_structured cancelled / ainvoke 转文本

参考: agent/turn_runners.py 的指标提取逻辑与 _parse_turn_result 状态映射。
"""
from __future__ import annotations

import asyncio
from typing import ClassVar

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_core.tools import tool

from agent.turn_types import AgentTurnResult
from team.base import TeamAgent
from tools.terminal_tools import UserRejectedCommandError
from utils.metrics import MetricsCollector


@tool
def _echo(message: str) -> str:
    """回显输入文本"""
    return message


class _FakeModel:
    """假 chat model:可配置产出 chunk 序列或执行异常"""

    def __init__(self, chunks=None, error=None) -> None:
        self._chunks = chunks or []
        self._error = error

    async def astream(self, messages, config=None):
        if self._error is not None:
            raise self._error
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


class _FakeExecutor:
    """假 agent executor:按预设事件序列产出 astream_events

    error 不为 None 时,astream_events 在产出任意事件前直接 raise 该异常,
    用于模拟工具内部抛 UserRejectedCommandError 等冒到 astream_events
    调用方的异常(LangGraph 对未识别异常不转 on_tool_error 事件,直接 re-raise)
    """

    def __init__(self, events: list[dict], error=None) -> None:
        self._events = events
        self._error = error

    async def astream_events(self, messages, config=None, version="v2"):
        if self._error is not None:
            raise self._error
        for ev in self._events:
            yield ev


def _set_fake_model(agent) -> None:
    """把最新 _FakeLLM 的模型替换为指定假模型"""
    llm = _FakeLLM.instances[-1]
    agent.llm = llm
    return llm


def _monkeypatch_llm(monkeypatch) -> None:
    monkeypatch.setattr("team.base.LLMClient", _FakeLLM)


def _monkeypatch_executor(monkeypatch, executor) -> None:
    monkeypatch.setattr("langchain.agents.create_agent", lambda **kw: executor)


def _collect(agen) -> list[str]:
    """把异步生成器聚合成列表(asyncio.run 包装,避免依赖 pytest-asyncio)"""
    async def _gather():
        return [c async for c in agen]
    return asyncio.run(_gather())


def _stream_event(content: str) -> dict:
    return {
        "event": "on_chat_model_stream",
        "data": {"chunk": AIMessageChunk(content=content)},
        "name": "FakeModel",
    }


def _chat_model_end_event(content: str, tokens: int = 15) -> dict:
    usage = {"input_tokens": 10, "output_tokens": tokens - 10, "total_tokens": tokens}
    msg = AIMessage(
        content=content,
        response_metadata={"usage_metadata": usage},
    )
    return {"event": "on_chat_model_end", "data": {"output": msg}, "name": "FakeModel"}


def _tool_end_event(name: str, content: str = "done") -> dict:
    return {
        "event": "on_tool_end",
        "data": {"output": ToolMessage(content=content, tool_call_id="t1")},
        "name": name,
    }


def _tool_error_event(name: str) -> dict:
    return {
        "event": "on_tool_error",
        "data": {"error": RuntimeError("boom")},
        "name": name,
    }


# ── metrics 属性 ─────────────────────────────────────────────────

def test_metrics_lazy_initialization():
    """metrics 属性惰性初始化,object.__new__ 直构实例也可访问且结果缓存"""
    agent = object.__new__(TeamAgent)
    mc = agent.metrics
    assert isinstance(mc, MetricsCollector)
    assert agent.metrics is mc


# ── 纯文本通道 LLM 指标 ──────────────────────────────────────────

def test_pure_text_llm_metric_recorded(monkeypatch):
    """纯文本模式:从最后 chunk 提取 usage_metadata 记录 LLM 指标"""
    _monkeypatch_llm(monkeypatch)
    agent = TeamAgent(name="t")
    _set_fake_model(agent)._model = _FakeModel(chunks=[
        AIMessageChunk(content="hello", response_metadata={
            "usage_metadata": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        }),
    ])

    chunks = _collect(agent._astream_messages([{"role": "user", "content": "hi"}]))

    assert "".join(chunks) == "hello"
    summary = agent.metrics.get_summary()
    assert summary["llm"]["total_calls"] == 1
    assert summary["llm"]["total_tokens"] == 15
    assert summary["llm"]["by_provider"]["test"]["count"] == 1


def test_pure_text_llm_metric_error_chunk(monkeypatch):
    """纯文本模式 LLM 异常:yield 错误前缀,不产生 LLM 指标"""
    _monkeypatch_llm(monkeypatch)
    agent = TeamAgent(name="t")
    _set_fake_model(agent)._model = _FakeModel(error=RuntimeError("boom"))

    chunks = _collect(agent._astream_messages([{"role": "user", "content": "hi"}]))

    assert chunks[0].startswith("任务执行失败")
    assert agent.metrics.get_summary()["llm"]["total_calls"] == 0


# ── 工具模式流式指标 ─────────────────────────────────────────────

def test_tool_mode_metrics_recorded(monkeypatch):
    """工具模式:on_chat_model_end 记 LLM,on_tool_end/on_tool_error 记工具"""
    _monkeypatch_llm(monkeypatch)
    executor = _FakeExecutor([
        _stream_event("思考中"),
        _chat_model_end_event("调用工具"),
        _tool_end_event("echo", "done"),
        _stream_event("结果好"),
        _chat_model_end_event("最终回答"),
        _tool_error_event("echo"),
    ])
    _monkeypatch_executor(monkeypatch, executor)

    agent = TeamAgent(name="t", tools=[_echo], system_prompt="sys")
    chunks = _collect(agent._astream_with_tools("task"))

    assert "".join(chunks) == "思考中结果好"
    summary = agent.metrics.get_summary()
    assert summary["llm"]["total_calls"] == 2
    assert summary["tools"]["total_calls"] == 2
    echo_stats = summary["tools"]["by_name"]["echo"]
    assert echo_stats["count"] == 2
    assert echo_stats["failures"] == 1


def test_tool_timeout_metric_detected(monkeypatch):
    """工具超时(wrap 层 JSON 错误串)→ timed_out 计数而非失败"""
    _monkeypatch_llm(monkeypatch)
    executor = _FakeExecutor([
        _tool_end_event("slow", '{"error": "tool_timeout", "message": "超时"}'),
    ])
    _monkeypatch_executor(monkeypatch, executor)

    agent = TeamAgent(name="t", tools=[_echo], system_prompt="sys")
    _collect(agent._astream_with_tools("task"))

    slow_stats = agent.metrics.get_summary()["tools"]["by_name"]["slow"]
    assert slow_stats["timeouts"] == 1
    # 既有 MetricsCollector 语义:超时同时计失败(与 tests/utils/test_metrics.py 一致)
    assert slow_stats["failures"] == 1


def test_tool_mode_error_status_metric(monkeypatch):
    """工具返回 status="error"(ToolExecutionErrorMW 产物)→ 记为失败"""
    _monkeypatch_llm(monkeypatch)
    executor = _FakeExecutor([
        {
            "event": "on_tool_end",
            "data": {
                "output": ToolMessage(
                    content="执行出错,请检查参数",
                    tool_call_id="t1",
                    status="error",
                )
            },
            "name": "echo",
        },
    ])
    _monkeypatch_executor(monkeypatch, executor)

    agent = TeamAgent(name="t", tools=[_echo], system_prompt="sys")
    _collect(agent._astream_with_tools("task"))

    echo_stats = agent.metrics.get_summary()["tools"]["by_name"]["echo"]
    assert echo_stats["failures"] == 1
    assert echo_stats["timeouts"] == 0


# ── turn 计数 ────────────────────────────────────────────────────

def test_turn_count_incremented(monkeypatch):
    """astream 入口每次执行计一次 turn(ainvoke/arun_structured 均经此)"""
    _monkeypatch_llm(monkeypatch)
    agent = TeamAgent(name="t")

    _collect(agent.astream("hi"))
    _collect(agent.astream("again"))

    assert agent.metrics.get_summary()["session"]["turn_count"] == 2


# ── arun_structured 类型化结果 ───────────────────────────────────

def test_arun_structured_completed(monkeypatch):
    """成功路径 → AgentTurnResult.completed,output 为生成文本"""
    _monkeypatch_llm(monkeypatch)
    agent = TeamAgent(name="t")
    _set_fake_model(agent)._model = _FakeModel(chunks=[
        AIMessageChunk(content="answer"),
    ])

    result = asyncio.run(agent.arun_structured("task"))

    assert isinstance(result, AgentTurnResult)
    assert result.is_completed
    assert result.status == "completed"
    assert result.output == "answer"


def test_arun_structured_cancelled_on_error(monkeypatch):
    """LLM 调用失败 → AgentTurnResult.cancelled,output 为错误信息"""
    _monkeypatch_llm(monkeypatch)
    agent = TeamAgent(name="t")
    _set_fake_model(agent)._model = _FakeModel(error=RuntimeError("boom"))

    result = asyncio.run(agent.arun_structured("task"))

    assert result.status == "cancelled"
    assert result.is_completed is False
    assert result.output.startswith("任务执行失败")


def test_arun_structured_tool_mode_completed(monkeypatch):
    """工具模式成功路径 → completed,聚合所有 token 增量"""
    _monkeypatch_llm(monkeypatch)
    executor = _FakeExecutor([
        _stream_event("hello "),
        _chat_model_end_event(""),
        _stream_event("world"),
        _chat_model_end_event(""),
    ])
    _monkeypatch_executor(monkeypatch, executor)

    agent = TeamAgent(name="t", tools=[_echo], system_prompt="sys")
    result = asyncio.run(agent.arun_structured("task"))

    assert result.is_completed
    assert result.output == "hello world"
    assert agent.metrics.get_summary()["session"]["turn_count"] == 1


# ── 用户拒绝危险命令语义识别(对齐 agent/turn_runners.py:52-53) ────

def test_astream_with_tools_reraises_user_rejected(monkeypatch):
    """_astream_with_tools 遇 UserRejectedCommandError 时 re-raise(不混入
    通用"任务执行失败"错误文本),使上层 arun_structured 能单独识别为 cancelled
    """
    _monkeypatch_llm(monkeypatch)

    executor = _FakeExecutor(
        events=[],
        error=UserRejectedCommandError("rm -rf /"),
    )
    _monkeypatch_executor(monkeypatch, executor)

    agent = TeamAgent(name="t", tools=[_echo], system_prompt="sys")

    with pytest.raises(UserRejectedCommandError):
        _collect(agent._astream_with_tools("task"))


def test_arun_structured_cancelled_on_user_rejected(monkeypatch):
    """arun_structured 单独 catch UserRejectedCommandError → cancelled,
    output 为"用户拒绝"语义(不混入 TASK_ERROR_PREFIX),对齐 TurnRunners 模式
    """
    _monkeypatch_llm(monkeypatch)

    executor = _FakeExecutor(
        events=[],
        error=UserRejectedCommandError("rm -rf /"),
    )
    _monkeypatch_executor(monkeypatch, executor)

    agent = TeamAgent(name="t", tools=[_echo], system_prompt="sys")
    result = asyncio.run(agent.arun_structured("task"))

    assert result.status == "cancelled"
    assert result.is_completed is False
    # 关键断言:不含通用"任务执行失败"前缀,而是"用户拒绝"语义
    assert result.output is not None
    assert "用户拒绝" in result.output or "用户已拒绝" in result.output
    assert not result.output.startswith("任务执行失败")


def test_ainvoke_returns_error_text_on_user_rejected(monkeypatch):
    """ainvoke 契约返回 str 不抛异常:UserRejectedCommandError → 错误文本

    工作流节点(worker_exec 等)直接 await ainvoke 不带 try/except,
    若此处抛异常会破坏整个工作流图;故 ainvoke 须把异常转为错误文本返回。
    """
    _monkeypatch_llm(monkeypatch)

    executor = _FakeExecutor(
        events=[],
        error=UserRejectedCommandError("rm -rf /"),
    )
    _monkeypatch_executor(monkeypatch, executor)

    agent = TeamAgent(name="t", tools=[_echo], system_prompt="sys")
    result = asyncio.run(agent.ainvoke("task"))

    assert isinstance(result, str)
    assert result.startswith("任务执行失败")
    assert "用户拒绝" in result