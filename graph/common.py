"""
工作流共享组件 - 节点进度跟踪与通用运行器

从 graph/simple.py 提取的公共逻辑，供所有工作流复用：
- NodeCallback 类型别名
- _NodeTrackingHandler 节点级进度回调
- run_compiled_workflow 通用工作流运行器
"""
from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

# 节点进度回调类型:接收节点名
NodeCallback = Callable[[str], None]


class NodeTrackingHandler(BaseCallbackHandler):
    """LangGraph 节点级进度跟踪回调处理器。

    通过 ``config["callbacks"]`` 注入 ``graph.invoke``，利用 LangGraph 节点执行时
    写入的 ``metadata["langgraph_node"]`` 字段识别业务节点（子图/内部 agent 的节点
    不在 known_nodes 中会被过滤），在节点开始/结束/异常时转发给外部回调。
    """

    def __init__(
        self,
        known_nodes: set[str],
        on_node_start: NodeCallback | None = None,
        on_node_end: NodeCallback | None = None,
        on_node_error: NodeCallback | None = None,
    ) -> None:
        self.known_nodes = known_nodes
        self.on_node_start = on_node_start
        self.on_node_end = on_node_end
        self.on_node_error = on_node_error
        # run_id -> 节点名：on_chain_end/on_chain_error 无节点名参数，需按 run_id 反查
        self._active: dict[str, str] = {}

    def on_chain_start(self, serialized: Any, inputs: Any, *, run_id: str, **kwargs: Any) -> None:
        metadata = kwargs.get("metadata") or {}
        node = metadata.get("langgraph_node")
        if node in self.known_nodes:
            self._active[run_id] = node
            if self.on_node_start:
                self.on_node_start(node)

    def on_chain_end(self, output: Any, *, run_id: str, **kwargs: Any) -> None:
        node = self._active.pop(run_id, None)
        if node and self.on_node_end:
            self.on_node_end(node)

    def on_chain_error(self, error: BaseException, *, run_id: str, **kwargs: Any) -> None:
        node = self._active.pop(run_id, None)
        if node and self.on_node_error:
            self.on_node_error(node)


def run_compiled_workflow(
    graph,
    task: str,
    state_fields: dict[str, str] | None = None,
    raw_context: str = "",
    on_node_start: NodeCallback | None = None,
    on_node_end: NodeCallback | None = None,
    on_node_error: NodeCallback | None = None,
) -> dict:
    """
    通用工作流运行器 - 所有工作流共享的 invoke 逻辑

    各工作流只需提供自己特有的初始状态字段（state_fields），通用字段
    （task / raw_context / context_summary）由本函数自动填充。

    Args:
        graph: 编译好的 LangGraph StateGraph
        task: 用户任务
        state_fields: 工作流特有的初始状态字段（值为空串的占位）
            如 ``{"plan": "", "worker_result": "", "final_answer": ""}``
        raw_context: 原始记忆文本，为空则不注入记忆
        on_node_start: 节点开始回调，接收节点名
        on_node_end: 节点结束回调，接收节点名
        on_node_error: 节点异常回调，接收节点名

    Returns:
        工作流执行结果字典
    """
    thread_id = f"workflow-{uuid.uuid4().hex[:8]}"

    initial_state: dict[str, str] = {
        "task": task,
        "raw_context": raw_context,
        "context_summary": "",
    }
    if state_fields:
        initial_state.update(state_fields)

    config: dict = {"configurable": {"thread_id": thread_id}}
    if on_node_start or on_node_end or on_node_error:
        known_nodes = {
            n.id for n in graph.get_graph().nodes.values() if not n.id.startswith("__")
        }
        config["callbacks"] = [
            NodeTrackingHandler(
                known_nodes,
                on_node_start=on_node_start,
                on_node_end=on_node_end,
                on_node_error=on_node_error,
            )
        ]

    return graph.invoke(initial_state, config=config)
