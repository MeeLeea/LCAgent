"""节点级进度跟踪回调处理器。

NodeTrackingHandler 通过 ``config["callbacks"]`` 注入 ``graph.ainvoke``，
利用 LangGraph 节点执行时写入的 ``metadata["langgraph_node"]`` 字段识别业务节点，
在节点开始/结束/异常时构造 AgentEvent 并转发给外部回调。
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph

from utils.events import AgentEvent

logger = logging.getLogger(__name__)

NodeCallback = Callable[[AgentEvent], None]


def build_tracking_callbacks(
    graph: StateGraph,
    *,
    on_node_start: NodeCallback | None = None,
    on_node_end: NodeCallback | None = None,
    on_node_error: NodeCallback | None = None,
    on_token: NodeCallback | None = None,
) -> list[NodeTrackingHandler]:
    """构造节点级进度跟踪回调处理器列表（可直接挂到 ``config["callbacks"]``）。

    从编译图推导 known_nodes（排除 ``__`` 开头的内部哨兵/分支节点，子图与内部
    agent 节点不在其中会被过滤），统一封装 ``NodeTrackingHandler`` 的构造细节，
    供 workflow_runner 与 workflow_adapter 复用。on_token 仅在需要 TOKEN 级流式的
    调用方（如 workflow_adapter）传入，其余业务节点回调按需透传。
    """
    known_nodes = {
        n.id for n in graph.get_graph().nodes.values() if not n.id.startswith("__")
    }
    return [
        NodeTrackingHandler(
            known_nodes,
            on_node_start=on_node_start,
            on_node_end=on_node_end,
            on_node_error=on_node_error,
            on_token=on_token,
        )
    ]


def _extract_node_output(output: Any) -> str:
    """从节点返回值提取该节点的产出文本（供 NODE_END 事件携带）。

    节点函数统一返回 ``{"xxx": result, "messages": [AIMessage(content=result)]}``，
    本函数取 ``output["messages"]`` 最后一条消息的文本作为节点产出。
    """
    if not isinstance(output, dict):
        return ""
    try:
        messages = output.get("messages")
        if not messages:
            return ""
        last = messages[-1]
        if not isinstance(last, AIMessage):
            return ""
        content = last.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
            return "".join(parts)
        return str(content)
    except Exception as error:
        logger.debug("提取节点产出失败: %s", error)
        return ""


def _content_blocks_to_text(content: Any) -> str:
    """把 str / 内容块列表统一提取为纯文本（空内容返回空串）。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return ""


def _extract_token_text(token: Any, chunk: Any) -> str:
    """从 ``on_llm_new_token`` 的入参提取流式增量文本。

    优先取 ``chunk.message.content``（chat 模型的结构化增量块），
    为空时回退 ``token``（str 或内容块列表，见 BaseCallbackHandler 签名）。
    """
    if chunk is not None:
        message = getattr(chunk, "message", None)
        text = _content_blocks_to_text(getattr(message, "content", None))
        if text:
            return text
    return _content_blocks_to_text(token)


class NodeTrackingHandler(BaseCallbackHandler):
    """LangGraph 节点级进度跟踪回调处理器。

    通过 ``config["callbacks"]`` 注入 ``graph.invoke``，利用 LangGraph 节点执行时
    写入的 ``metadata["langgraph_node"]`` 字段识别业务节点（子图/内部 agent 的节点
    不在 known_nodes 中会被过滤），在节点开始/结束/异常时构造 AgentEvent
    （NODE_START / NODE_END / NODE_ERROR）并转发给外部回调。

    此外捕获 ``on_llm_new_token``（LangChain 回调系统派发的标准流式 token 钩子）：
    节点内 TeamAgent 经 ``_build_run_config`` 透传 callbacks 后，内部 LLM 的
    token 增量到达本 handler，构造 AgentEvent.token 转发给外部回调，实现
    workflow 节点执行期间的前端 TOKEN 级流式。注意：回调系统只派发标准钩子，
    ``on_chat_model_stream`` 是 ``astream_events`` 的**事件名**而非回调钩子，
    挂在 handler 上永远不会被调用（历史缺陷，已由此钩子替代）。
    """

    def __init__(
        self,
        known_nodes: set[str],
        on_node_start: NodeCallback | None = None,
        on_node_end: NodeCallback | None = None,
        on_node_error: NodeCallback | None = None,
        on_token: NodeCallback | None = None,
    ) -> None:
        self.known_nodes = known_nodes
        self.on_node_start = on_node_start
        self.on_node_end = on_node_end
        self.on_node_error = on_node_error
        self.on_token = on_token
        self._active: dict[str, str] = {}
        self.run_inline = True

    def on_chain_start(self, serialized: Any, inputs: Any, *, run_id: str, **kwargs: Any) -> None:
        metadata = kwargs.get("metadata") or {}
        node = metadata.get("langgraph_node")
        if node in self.known_nodes:
            self._active[run_id] = node
            if self.on_node_start:
                self.on_node_start(AgentEvent.node_start(node=node))

    def on_chain_end(self, output: Any, *, run_id: str, **kwargs: Any) -> None:
        node = self._active.pop(run_id, None)
        if node and self.on_node_end:
            self.on_node_end(
                AgentEvent.node_end(node=node, content=_extract_node_output(output))
            )

    def on_chain_error(self, error: BaseException, *, run_id: str, **kwargs: Any) -> None:
        node = self._active.pop(run_id, None)
        if node and self.on_node_error:
            self.on_node_error(AgentEvent.node_error(node=node))

    def on_llm_new_token(
        self,
        token: str | list[str | dict[str, Any]],
        *,
        chunk: Any = None,
        **kwargs: Any,
    ) -> None:
        """LLM token 增量 → TOKEN 事件转发（回调系统派发的标准流式钩子）。"""
        if not self.on_token:
            return
        content = _extract_token_text(token, chunk)
        if content:
            self.on_token(AgentEvent.token(text=content))
