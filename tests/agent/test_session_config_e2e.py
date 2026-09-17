"""会话配置注入通道端到端回归测试。

## 被锁定的框架契约（实测得出，非推断）

`SessionConfigMW.awrap_model_call` 通过 ``request.runtime.context`` 读取会话配置，
但 LangGraph **不会**把 ``config["configurable"]`` 映射到 ``runtime.context``：

- ``awrap_model_call`` 中 ``request.runtime`` 是 LangGraph ``Runtime``，字段为
  ``context / store / stream_writer / heartbeat / previous / execution_info /
  server_info / control`` —— **没有** ``config`` 属性，且 ``runtime.context`` 在调用方
  不传 ``context=`` 时恒为 ``None``。因此 ``config["configurable"]`` 只喂给 checkpointer。
- ``awrap_tool_call`` 中 ``request.runtime`` 是 ``ToolRuntime``，**有** ``.config`` ——
  所以 ``agent/workspace_mw.py`` / ``agent/tool_error_mw.py`` 读 ``runtime.config`` 一直正常。

结论：图调用处必须把同一份 ``{"configurable": {...}}`` **同时**经两条通道传入
（``config=`` 供 checkpointer，``context=`` 供中间件），见 ``agent/turn_runners.py``
与 ``agent/streaming.py``。本文件用**真实** ``create_agent`` + 真实图调用来守护这条契约，
而非用 ``SimpleNamespace`` 伪造 runtime（伪造正是该缺陷长期未被发现的遮蔽原因）。

Part A 锁定框架契约本身；Part B 守护生产调用点（删掉 ``context=config`` 即失败）。
"""
from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from agent.session_config_middleware import SessionConfigMW
from agent.streaming import Streaming
from agent.turn_runners import TurnRunners
from session.config import SessionConfig


class _EchoChatModel(BaseChatModel):
    """桩聊天模型：回复固定为 ``from:<tag>``，用于判定本次调用实际走了哪个模型。

    ``seen_systems`` 记录每次调用收到的 system 提示词，用于断言角色提示词确实生效。
    """

    tag: str
    seen_systems: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "echo"

    def _generate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.seen_systems.append(
            "\n".join(str(m.content) for m in messages if isinstance(m, SystemMessage))
        )
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=f"from:{self.tag}"))]
        )

    def bind_tools(self, tools: Any, **kwargs: Any) -> _EchoChatModel:
        """``BaseChatModel`` 默认实现抛 NotImplementedError，桩模型直接返回自身。"""
        return self


def _payload(thread_id: str, config: SessionConfig) -> dict[str, Any]:
    """构造与生产一致的图调用参数（``config`` 与 ``context`` 传同一份数据）。"""
    return {
        "configurable": {
            "thread_id": thread_id,
            "session_config": config.to_dict(),
        }
    }


def _build_agent(
    default: BaseChatModel,
    factory: Any,
) -> Any:
    """构建只挂 ``SessionConfigMW`` 的真实图，避免其他中间件干扰断言。"""
    return create_agent(
        model=default,
        tools=[],
        middleware=[SessionConfigMW(model_factory=factory, default_model=default)],
    )


# ============ Part A：真实框架契约 ============


@pytest.mark.anyio
async def test_context_kwarg_is_required_for_middleware_to_apply() -> None:
    """``context=`` 是唯一能填充 ``runtime.context`` 的通道；省略则中间件静默直通。

    同一张真实图上做对照：传 ``context=`` 时会话模型生效；不传时框架**不会**把
    ``config["configurable"]`` 自动暴露给中间件，因此回落到默认模型。
    """
    default = _EchoChatModel(tag="default")
    session = _EchoChatModel(tag="session")
    built: list[SessionConfig] = []

    def factory(config: SessionConfig) -> BaseChatModel:
        built.append(config)
        return session

    agent = _build_agent(default, factory)
    cfg = SessionConfig(provider="yunlan", model="m-new", system_prompt="会话提示")
    payload = _payload("t1", cfg)

    # 传 context=：中间件拿到会话配置并 override 模型与提示词
    with_ctx = await agent.ainvoke(
        {"messages": [HumanMessage(content="hi")]}, config=payload, context=payload
    )
    assert built and built[0] == cfg, "中间件未拿到会话配置：runtime.context 未被填充"
    assert with_ctx["messages"][-1].content == "from:session"
    assert "会话提示" in session.seen_systems[-1], "角色提示词未覆盖 system message"

    # 不传 context=：config["configurable"] 不会被映射到 runtime.context
    built.clear()
    without = await agent.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"configurable": {"thread_id": "t2"}},
    )
    assert built == [], "框架若把 configurable 自动映射到 runtime.context，本断言会失败"
    assert without["messages"][-1].content == "from:default"


@pytest.mark.anyio
async def test_two_sessions_use_their_own_model_and_prompt() -> None:
    """共享同一张图时，两个会话各自命中自己的模型与提示词，互不污染。"""
    default = _EchoChatModel(tag="default")
    per_model: dict[str, _EchoChatModel] = {
        "a-model": _EchoChatModel(tag="a-model"),
        "b-model": _EchoChatModel(tag="b-model"),
    }

    def factory(config: SessionConfig) -> BaseChatModel:
        assert config.model is not None
        return per_model[config.model]

    agent = _build_agent(default, factory)
    cfg_a = SessionConfig(provider="yunlan", model="a-model", system_prompt="提示A")
    cfg_b = SessionConfig(provider="zhipu", model="b-model", system_prompt="提示B")
    payload_a = _payload("ta", cfg_a)
    payload_b = _payload("tb", cfg_b)

    result_a = await agent.ainvoke(
        {"messages": [HumanMessage(content="hi")]}, config=payload_a, context=payload_a
    )
    result_b = await agent.ainvoke(
        {"messages": [HumanMessage(content="hi")]}, config=payload_b, context=payload_b
    )

    assert result_a["messages"][-1].content == "from:a-model"
    assert result_b["messages"][-1].content == "from:b-model"
    # 会话 A 的提示词不得泄漏进 B 的调用，反之亦然
    assert "提示A" in per_model["a-model"].seen_systems[-1]
    assert "提示B" not in per_model["a-model"].seen_systems[-1]
    assert "提示B" in per_model["b-model"].seen_systems[-1]
    assert "提示A" not in per_model["b-model"].seen_systems[-1]


@pytest.mark.anyio
async def test_astream_events_honours_context_kwarg() -> None:
    """流式路径同样依赖 ``context=``：``astream_events`` 经 ``**kwargs`` 透传该通道。

    省略 ``context=`` 时流式会话同样读不到配置，因此这里必须走真实图验证而非 mock。
    """
    default = _EchoChatModel(tag="default")
    session = _EchoChatModel(tag="session")
    built: list[SessionConfig] = []

    def factory(config: SessionConfig) -> BaseChatModel:
        built.append(config)
        return session

    agent = _build_agent(default, factory)
    cfg = SessionConfig(provider="yunlan", model="s-model", system_prompt="流式提示")
    payload = _payload("ts", cfg)

    async for _event in agent.astream_events(
        {"messages": [HumanMessage(content="hi")]},
        config=payload,
        context=payload,
        version="v2",
    ):
        pass

    assert built and built[0] == cfg, "astream_events 未把 context 透传到中间件"
    assert "流式提示" in session.seen_systems[-1]


# ============ Part B：生产调用点守护 ============


class _RecordingExecutor:
    """记录 ``ainvoke`` 收到的全部关键字参数，用于守护生产调用点。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(self, value: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"value": value, **kwargs})
        return {"messages": [AIMessage(content="ok")]}


class _StoreStub:
    """``_get_store`` 的最小替身：只提供 ``aresume_structured`` 需要的中断模式。"""

    async def aget_interrupt_mode(self, thread_id: str) -> str:
        return "chat"


class _TurnHost(TurnRunners):
    """最小 TurnRunners 宿主：只补齐被测路径触达的协作方法。"""

    def __init__(self, config: dict[str, Any], executor: _RecordingExecutor) -> None:
        self.agent_executor = executor
        self.metrics = SimpleNamespace(increment_turn=lambda: None)
        self._closed = False
        self._config = config
        self._store = _StoreStub()

    def _ensure_not_closed(self) -> None:
        return None

    def _current_sid(self, thread_id: str | None = None) -> str:
        return "t-turns"

    async def _ainvoke_config(self, thread_id: str | None = None) -> dict[str, Any]:
        return self._config

    def _temp_verbose(self, verbose: bool) -> Any:
        return nullcontext()

    def _thread_id_from_config(self, config: dict[str, Any]) -> str | None:
        return "t-turns"

    async def _ahandle_turn_completion(
        self, turn: Any, config: dict[str, Any], mode: str
    ) -> None:
        return None

    async def _arecord_tool_steps(
        self, messages: Any, input_msg: Any, session_id: str
    ) -> None:
        return None

    def _get_store(self) -> Any:
        return self._store

    async def _abuild_resume_command(
        self, config: dict[str, Any], payload: dict[str, Any]
    ) -> Any:
        return payload

    async def _acapture_pending_interrupt(self, config: dict[str, Any], mode: str) -> None:
        return None

    async def _aclear_pending_interrupt(self, thread_id: str) -> None:
        return None


@pytest.mark.anyio
async def test_turn_runners_pass_context_to_every_invoke() -> None:
    """三条结构化路径都必须把 config 同时作为 ``context`` 传给图。

    删掉任一处的 ``context=config``，本用例即失败——这是会话配置真正生效的前提。
    """
    config = _payload("t-turns", SessionConfig(provider="yunlan", model="m-new"))
    executor = _RecordingExecutor()
    host = _TurnHost(config, executor)

    await host.arun_structured("task")
    await host.achat_structured("msg")
    await host.aresume_structured({"choice_id": "x"})

    assert len(executor.calls) == 3, "三条结构化路径应各调用一次图"
    for call in executor.calls:
        assert "context" in call, "缺少 context= —— 会话配置中间件将读不到配置"
        # 生产代码传的是同一个对象，身份相等而非仅值相等
        assert call["context"] is call["config"]
        assert call["context"]["configurable"]["session_config"]["model"] == "m-new"


class _StreamRecorder:
    """记录 ``astream_events`` 收到的参数，且不产出事件。

    ``_arun_graph_events`` 在事件队列排空后会干净 ``return``，因此无需再补齐
    metrics / llm 等仅在被消费的事件分支中才用到的协作对象。
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def astream_events(self, inputs: Any, **kwargs: Any) -> Any:
        self.calls.append({"inputs": inputs, **kwargs})

        async def _empty() -> Any:
            return
            yield  # pragma: no cover - 仅为使其成为异步生成器

        return _empty()


class _StreamHost(Streaming):
    """最小 Streaming 宿主：``_arun_graph_events`` 只依赖 ``agent_executor``。"""

    def __init__(self, executor: Any) -> None:
        self.agent_executor = executor


@pytest.mark.anyio
async def test_arun_graph_events_passes_context_to_astream_events() -> None:
    """流式生产路径必须把 config 同时作为 ``context`` 传给 ``astream_events``。"""
    config = _payload("t-stream", SessionConfig(provider="yunlan", model="m-new"))
    recorder = _StreamRecorder()
    host = _StreamHost(recorder)

    async for _event in host._arun_graph_events(
        {"messages": [HumanMessage(content="hi")]},
        config,
        "t-stream",
        "trace-1",
    ):
        pass

    assert len(recorder.calls) == 1, "首轮即应干净结束，不应触发重试"
    call = recorder.calls[0]
    assert "context" in call, "缺少 context= —— 流式会话配置将读不到配置"
    assert call["context"] is config
    assert call["version"] == "v2"
