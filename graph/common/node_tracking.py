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

from utils.events import AgentEvent

logger = logging.getLogger(__name__)

NodeCallback = Callable[[AgentEvent], None]


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


class NodeTrackingHandler(BaseCallbackHandler):
    """LangGraph 节点级进度跟踪回调处理器。

    通过 ``config["callbacks"]`` 注入 ``graph.invoke``，利用 LangGraph 节点执行时
    写入的 ``metadata["langgraph_node"]`` 字段识别业务节点（子图/内部 agent 的节点
    不在 known_nodes 中会被过滤），在节点开始/结束/异常时构造 AgentEvent
    （NODE_START / NODE_END / NODE_ERROR）并转发给外部回调。

    此外捕获 ``on_chat_model_stream``：节点内 TeamAgent.astream 透传 callbacks
    后，LLM token 增量事件到达本 handler，构造 AgentEvent.token 转发给外部回调，
    实现 workflow 节点执行期间的前端 TOKEN 级流式。
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

    def on_chat_model_stream(self, response: Any, **kwargs: Any) -> None:
        """LLM token 增量 → TOKEN 事件转发。"""
        if not self.on_token:
            return
        content = getattr(response, "content", None)
        if isinstance(content, str) and content:
            self.on_token(AgentEvent.token(text=content))
