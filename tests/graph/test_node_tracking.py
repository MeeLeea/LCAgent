"""NodeTrackingHandler 回调钩子单元测试。

回归背景：``on_chat_model_stream`` 是 ``astream_events`` 的**事件名**而非
LangChain 回调系统派发的钩子（回调系统只派发 ``on_llm_new_token`` 等标准
方法），挂在 handler 上永远不会被调用——导致工作流节点执行期间前端收不到
任何 TOKEN 事件，SSE 在节点 LLM 流式期间完全静默，前端 90s 看门狗误判
超时拉黑线程，后续 interrupt 事件被静默丢弃（HITL 卡片不显示）。
本文件验证标准钩子 ``on_llm_new_token`` 的 token 转发与节点事件派发。
"""
from __future__ import annotations

import asyncio
import os
import sys
from typing import Any, TypedDict

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from graph.common.node_tracking import (
    NodeTrackingHandler,
    _extract_token_text,
)

# 项目根注入 sys.path（与既有 tests/ 模块同构）
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ==================== 辅助 ====================


def _make_handler(
    events: list[str],
    known_nodes: set[str] | None = None,
) -> NodeTrackingHandler:
    """构造把事件名追加进列表的 handler（便于断言触发顺序与次数）。"""

    def on_node_start(ev: Any) -> None:
        events.append(f"NODE_START:{ev.node}")

    def on_node_end(ev: Any) -> None:
        events.append(f"NODE_END:{ev.node}")

    def on_token(ev: Any) -> None:
        events.append(f"TOKEN:{ev.content}")

    return NodeTrackingHandler(
        known_nodes if known_nodes is not None else {"plan_node"},
        on_node_start=on_node_start,
        on_node_end=on_node_end,
        on_token=on_token,
    )


class _State(TypedDict, total=False):
    out: str


# ==================== 测试 ====================


def test_extract_token_text_variants():
    """_extract_token_text 覆盖 str / 内容块列表 / chunk.message 三种形态。

    Given: on_llm_new_token 的 token 参数可为 str 或内容块列表，chunk.message.content
           为结构化增量；
    When: 分别传入三种形态；
    Then: 均提取出纯文本；空内容返回空串。
    """
    # 纯 str token
    assert _extract_token_text("你好", None) == "你好"
    # 内容块列表 token
    assert _extract_token_text(
        [{"type": "text", "text": "段1"}, {"type": "text", "text": "段2"}], None
    ) == "段1段2"
    # chunk.message.content 优先
    chunk = type("C", (), {"message": type("M", (), {"content": "chunk文本"})()})
    assert _extract_token_text("token文本", chunk) == "chunk文本"
    # 空内容 → 空串（调用方据此跳过，不发空 TOKEN 事件）
    assert _extract_token_text("", None) == ""
    assert _extract_token_text([], None) == ""


def test_token_callback_fires_via_standard_hook():
    """回归：标准钩子 on_llm_new_token 必须触发 TOKEN 事件转发。

    Given: NodeTrackingHandler 挂在 config["callbacks"] 上（workflow_adapter 同构）；
    When: 流式 chat model 逐 token 产出；
    Then: on_token 回调收到非空 TOKEN 事件（历史缺陷中此处为 0，前端看门狗
          因 SSE 静默误判超时，导致后续 interrupt 事件被静默丢弃）。
    """
    events: list[str] = []
    handler = _make_handler(events)

    async def run() -> list[str]:
        model = GenericFakeChatModel(messages=iter([AIMessage(content="流式增量内容")]))
        chunks: list[str] = []
        async for chunk in model.astream(
            [("user", "hi")], config={"callbacks": [handler]}
        ):
            text = getattr(chunk, "content", None)
            if isinstance(text, str) and text:
                chunks.append(text)
        return chunks

    chunks = asyncio.run(run())

    token_events = [e for e in events if e.startswith("TOKEN:")]
    assert chunks, "fake 模型应产出流式 chunk"
    assert token_events, "on_llm_new_token 必须触发 TOKEN 事件（回归死代码缺陷）"
    # 转发内容与模型产出一致（按 chunk 顺序拼接相等）
    assert "".join(t.split("TOKEN:", 1)[1] for t in token_events) == "".join(chunks)


def test_workflow_node_emits_tokens_and_node_events():
    """端到端：外层 StateGraph 节点内透传 callbacks-only config，事件齐全。

    Given: 节点内部以仅含 callbacks 的显式 config 调模型（复刻 TeamAgent
           _build_run_config 透传路径）；
    When: graph.ainvoke 全程携带 NodeTrackingHandler；
    Then: NODE_START/NODE_END 各一次（known_nodes 过滤），且节点执行期间
          有 TOKEN 事件流动（前端看门狗不再遇到长静默窗口）。
    """

    async def plan_node(state: _State, config: RunnableConfig | None = None) -> dict:
        # 复刻 team/base.py::_build_run_config：显式 config 只透传 callbacks
        inner_config: dict[str, Any] = {}
        if config and config.get("callbacks"):
            inner_config["callbacks"] = config["callbacks"]
        model = GenericFakeChatModel(messages=iter([AIMessage(content="计划正文")]))
        async for _ in model.astream([("user", "任务")], config=inner_config):
            pass
        return {"out": "done", "messages": [AIMessage(content="计划正文")]}

    builder = StateGraph(_State)
    builder.add_node("plan_node", plan_node)
    builder.add_edge(START, "plan_node")
    builder.add_edge("plan_node", END)
    graph = builder.compile()

    events: list[str] = []
    handler = _make_handler(events)
    asyncio.run(graph.ainvoke({"out": ""}, config={"callbacks": [handler]}))

    node_starts = [e for e in events if e.startswith("NODE_START")]
    node_ends = [e for e in events if e.startswith("NODE_END")]
    token_events = [e for e in events if e.startswith("TOKEN:")]

    assert node_starts == ["NODE_START:plan_node"]
    assert node_ends == ["NODE_END:plan_node"]
    assert token_events, "节点执行期间必须有 TOKEN 事件流动"


def test_unknown_node_filtered_and_on_token_none_noop():
    """known_nodes 过滤与 on_token 缺省安全。

    Given: handler 的 known_nodes 不含当前节点、或未传 on_token；
    When: 节点执行 / token 到达；
    Then: 不派发 NODE_* 事件、不抛异常（fail-open）。
    """
    events: list[str] = []
    handler = _make_handler(events, known_nodes={"other_node"})

    async def run() -> None:
        model = GenericFakeChatModel(messages=iter([AIMessage(content="内容")]))
        async for _ in model.astream([("user", "hi")], config={"callbacks": [handler]}):
            pass

    asyncio.run(run())

    assert not [e for e in events if e.startswith("NODE_")], "非 known 节点应被过滤"

    # on_token=None：on_llm_new_token 直接 return，不抛异常
    silent = NodeTrackingHandler({"plan_node"})
    silent.on_llm_new_token("token", chunk=None)  # 不应抛异常
