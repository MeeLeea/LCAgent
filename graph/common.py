"""
工作流共享组件 - 节点进度跟踪与通用运行器

从 graph/simple.py 提取的公共逻辑，供所有工作流复用：
- NodeCallback 类型别名
- NodeTrackingHandler 节点级进度回调
- arun_compiled_workflow 通用异步工作流运行器

异步化说明：
    运行器使用 ``await graph.ainvoke(...)`` 而非同步 ``graph.invoke()``，
    避免在 asyncio 事件循环中阻塞。NodeTrackingHandler 设置 ``run_inline=True``，
    使同步回调方法在 ainvoke 期间直接运行于事件循环线程（而非线程池），
    从而保证回调中对 asyncio.Queue 等非线程安全对象的操作无需额外加锁。
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
        # ainvoke 期间同步回调默认经 run_in_executor 在线程池执行；置为 True 使回调
        # 直接运行于事件循环线程，避免回调中对 asyncio.Queue 等非线程安全对象的跨线程访问。
        self.run_inline = True

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


async def arun_compiled_workflow(
    graph,
    task: str,
    state_fields: dict[str, str] | None = None,
    raw_context: str = "",
    thread_id: str | None = None,
    on_node_start: NodeCallback | None = None,
    on_node_end: NodeCallback | None = None,
    on_node_error: NodeCallback | None = None,
) -> dict:
    """
    通用异步工作流运行器 - 所有工作流共享的 ainvoke 逻辑

    各工作流只需提供自己特有的初始状态字段（state_fields），通用字段
    （task / raw_context / context_summary）由本函数自动填充。

    使用 ``await graph.ainvoke(...)`` 异步执行，不阻塞事件循环；
    节点进度回调通过 ``run_inline=True`` 的 NodeTrackingHandler 在事件循环线程触发。

    Args:
        graph: 编译好的 LangGraph StateGraph
        task: 用户任务
        state_fields: 工作流特有的初始状态字段（值为空串的占位）
            如 ``{"plan": "", "worker_result": "", "final_answer": ""}``
        raw_context: 原始记忆文本，为空则不注入记忆
        thread_id: 会话线程 ID。为 None 时自动生成（无持久化绑定）；
            传入显式值时配合 checkpointer 编译的图可实现状态持久化。
        on_node_start: 节点开始回调，接收节点名
        on_node_end: 节点结束回调，接收节点名
        on_node_error: 节点异常回调，接收节点名

    Returns:
        工作流执行结果字典
    """
    tid = thread_id or f"workflow-{uuid.uuid4().hex[:8]}"

    initial_state: dict[str, str] = {
        "task": task,
        "raw_context": raw_context,
        "context_summary": "",
    }
    if state_fields:
        initial_state.update(state_fields)

    config: dict = {"configurable": {"thread_id": tid}}
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

    return await graph.ainvoke(initial_state, config=config)
