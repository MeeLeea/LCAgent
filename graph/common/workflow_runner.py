"""通用异步工作流运行器 — 所有工作流共享的 ainvoke 逻辑。

arun_compiled_workflow 封装了跨轮次记忆压缩、长期记忆召回注入、
节点进度回调、workspace 隔离等通用逻辑，各工作流只需提供
自己特有的初始状态字段（state_fields）。
"""
from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any

from langgraph.graph import StateGraph

from graph.common.node_tracking import NodeCallback, NodeTrackingHandler
from utils.events import AgentEvent
from utils.logging_config import TraceContext

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


async def arun_compiled_workflow(
    graph: StateGraph,
    task: str,
    state_fields: dict[str, str] | None = None,
    raw_context: str = "",
    thread_id: str | None = None,
    workspace_path: str | None = None,
    on_node_start: NodeCallback | None = None,
    on_node_end: NodeCallback | None = None,
    on_node_error: NodeCallback | None = None,
    max_history_chars: int = 6000,
    memory: Any | None = None,
    memory_thread_id: str | None = None,
    is_run_mode: bool = False,
) -> dict:
    """通用异步工作流运行器 - 所有工作流共享的 ainvoke 逻辑

    各工作流只需提供自己特有的初始状态字段（state_fields），通用字段
    （task / raw_context / context_summary）由本函数自动填充。

    Args:
        graph: 编译好的 LangGraph StateGraph
        task: 用户任务
        state_fields: 工作流特有的初始状态字段（值为空串的占位）
        raw_context: 原始记忆文本，为空则不注入记忆
        thread_id: 会话线程 ID。为 None 时自动生成（无持久化绑定）
        workspace_path: 会话绑定的工作空间绝对路径
        on_node_start: 节点开始回调
        on_node_end: 节点结束回调
        on_node_error: 节点异常回调
        max_history_chars: 跨轮次记忆摘要的最大字符数
        memory: MemoryManager 实例（长期记忆召回与结果沉淀）
        memory_thread_id: 长期记忆使用的会话线程 ID
        is_run_mode: 是否运行模式（决定 DONE 事件是否标记为重要记忆）

    Returns:
        工作流执行结果字典
    """
    tid = thread_id or f"workflow-{uuid.uuid4().hex[:8]}"

    with TraceContext(trace_id=tid, thread_id=tid):
        return await _arun_with_trace(
            graph, task, state_fields, raw_context, tid,
            workspace_path, on_node_start, on_node_end, on_node_error,
            max_history_chars, memory, memory_thread_id, is_run_mode,
        )


async def _arun_with_trace(
    graph: StateGraph,
    task: str,
    state_fields: dict[str, str] | None,
    raw_context: str,
    tid: str,
    workspace_path: str | None,
    on_node_start: NodeCallback | None,
    on_node_end: NodeCallback | None,
    on_node_error: NodeCallback | None,
    max_history_chars: int,
    memory: Any | None,
    memory_thread_id: str | None,
    is_run_mode: bool,
) -> dict:
    """在 TraceContext 内执行工作流主体。"""
    logger.info("工作流开始执行 [thread=%s]: %s", tid, task[:120])

    configurable: dict[str, Any] = {"thread_id": tid}
    if workspace_path is not None:
        configurable["workspace_path"] = workspace_path
    config: dict = {"configurable": configurable}

    previous_summary = await _aget_previous_workflow_summary(graph, config, max_history_chars)
    if previous_summary:
        raw_context = (
            f"{raw_context}\n\n【上一轮工作流记录】\n{previous_summary}".strip()
            if raw_context
            else f"【上一轮工作流记录】\n{previous_summary}"
        )

    if memory is not None and memory_thread_id:
        recalled = await memory.recall_text(memory_thread_id)
        if recalled:
            raw_context = (
                f"{raw_context}\n\n{recalled}".strip() if raw_context else recalled
            )

    initial_state: dict[str, str] = {
        "task": task,
        "raw_context": raw_context,
        "context_summary": "",
    }
    if state_fields:
        initial_state.update(state_fields)

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

    result = await graph.ainvoke(initial_state, config=config)

    if memory is not None and memory_thread_id:
        final_answer = result.get("final_answer") or ""
        if final_answer:
            await memory.consume_event(
                AgentEvent.done(
                    content=final_answer,
                    thread_id=memory_thread_id,
                    role="assistant",
                    is_important=is_run_mode,
                )
            )

    logger.info("工作流执行完成 [thread=%s]", tid)
    return result


async def _aget_previous_workflow_summary(
    graph: StateGraph,
    config: dict[str, Any],
    max_chars: int = 10000,
) -> str:
    """读取指定 thread 上一轮工作流状态并压缩为摘要。

    .. deprecated::
        该函数在 workflow 会话化后将由 checkpoint messages + summary 通道取代。
        当前保留以兼容 CLI/scheduler 等既有调用路径。

    无 checkpointer / 无历史 / 读取失败时返回空串（静默降级）。
    """
    try:
        state = await graph.aget_state(config)
    except Exception as error:
        logger.debug(
            "读取上一轮工作流状态失败 [thread=%s]: %s",
            config.get("configurable", {}).get("thread_id"),
            error,
        )
        return ""
    if state is None:
        return ""

    values = getattr(state, "values", None) or {}
    fields = {k: v for k, v in values.items() if v}
    relevant = [
        f"{k}: {v}"
        for k, v in fields.items()
        if k in ("task", "plan", "worker_result", "final_answer")
    ]
    if not relevant:
        return ""

    summary = "\n".join(relevant)
    if len(summary) > max_chars:
        summary = summary[:max_chars] + "\n...(上一轮工作流记录过长，已截断)"
    return summary
