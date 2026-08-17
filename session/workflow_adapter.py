"""WorkflowAdapter — workflow(graph/team 链路)的统一调度适配器。

使 workflow 链路获得与 AgentCore 会话链路同级的 SessionManager 门面接入
（统一会话/记忆/锁/SSE），上层（CLI/API）可无差别调度两类执行体。

实现 SessionAgent 协议（SessionManager 对 agent 依赖的 11 个接口）：
    arun_events / aresume_events(不支持) / session / _current_sid /
    set_current_session / checkpoint_info / aget_execution_history /
    aclear_history / manually_compact / aclose

能力边界（OMO 式渐进）：
- 事件为节点级：NODE_START / NODE_END / NODE_ERROR → DONE（无 TOKEN 流，
  内层 team agent 的同步 invoke 不可见）
- aresume_events 不支持（workflow 无 HITL 中断语义）
- 跨轮次上下文由 checkpoint messages 通道 + 长期记忆(recall_text)承载

执行流程（arun_events）：
1. 从 session_id 反解 workflow 名称（registry.workflow_name_of）
2. build_workflow 构建图（checkpointer + compaction 默认启用）
3. 执行前：checkpoint 历史消息 + 长期记忆拼 raw_context 注入
4. graph.ainvoke 边执行边经 NodeTrackingHandler 队列吐出节点事件
5. 结束后 yield DONE(final_answer)；记忆沉淀由 SessionManager 消费事件完成
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import AIMessage

from graph.common import NodeTrackingHandler, _build_compaction_middleware
from memory.manager import MemoryManager
from session.registry import SessionRegistry
from utils.compaction import CompactionConfig
from utils.events import AgentEvent
from utils.logging_config import TraceContext, generate_trace_id

logger = logging.getLogger(__name__)

# 注入 raw_context 的历史消息预览上限（每条截断字符数）
_HISTORY_MSG_PREVIEW_CHARS = 200
# 注入 raw_context 的历史消息条数上限
_HISTORY_MSG_COUNT = 5


class WorkflowAdapter:
    """Workflow 执行体适配器 — 使 workflow 接入统一 SessionManager 门面。

    Args:
        registry: SessionRegistry 实例（与 AgentCore/chat 会话共享，
            保证 workflow 会话与普通会话在同一注册表中管理）。
        memory: MemoryManager 实例（与 SessionManager 共享同一实例，
            执行前 recall 注入；记忆沉淀由 SessionManager 消费事件完成）。
        checkpointer: LangGraph checkpointer；为 None 时复用 registry 的。
        compaction_config: 消息通道压缩配置；为 None 时 build_workflow
            使用默认配置（阈值 50）。
    """

    def __init__(
        self,
        registry: SessionRegistry,
        memory: MemoryManager | None = None,
        checkpointer: Any = None,
        compaction_config: CompactionConfig | None = None,
    ) -> None:
        self._registry = registry
        self._memory = memory
        self._checkpointer = checkpointer if checkpointer is not None else registry.checkpointer
        self._compaction_config = compaction_config
        self._closed = False

    # ============ SessionAgent 协议：属性 ============

    @property
    def session(self) -> SessionRegistry:
        """会话注册表（workflow 会话与 chat 会话共用）。"""
        return self._registry

    @property
    def checkpoint_info(self) -> dict[str, Any]:
        """checkpoint 后端元信息（backend 类型 + 文件路径）。"""
        cp = self._checkpointer
        if cp is not None:
            cls_name = type(cp).__name__
            return {
                "checkpoint_backend": "sqlite" if "Sqlite" in cls_name else "memory",
                "checkpoint_file": (
                    getattr(cp, "checkpoint_file", "(内存)")
                    if hasattr(cp, "checkpoint_file")
                    else "(内存)"
                ),
            }
        return {"checkpoint_backend": "memory", "checkpoint_file": "(内存)"}

    def _current_sid(self, thread_id: str | None = None) -> str:
        """解析目标会话 ID（为 None 时使用 registry 当前会话）。"""
        if thread_id is not None:
            return thread_id
        return self._registry.current_session_id

    def set_current_session(self, session_id: str) -> None:
        """设置当前会话 ID（委托 registry）。"""
        self._registry.current_session_id = session_id

    # ============ SessionAgent 协议：执行 ============

    async def arun_events(
        self,
        message: str,
        *,
        thread_id: str | None = None,
        is_run_mode: bool = False,
    ) -> AsyncIterator[AgentEvent]:
        """纯执行接口：运行 workflow 图，yield 节点级 AgentEvent 流。

        事件顺序：NODE_START/NODE_END/NODE_ERROR* → DONE(final_answer)
        或 ERROR。SessionManager 负责锁/记忆提交/SSE 转发。

        Args:
            message: 用户任务
            thread_id: 目标 workflow 会话 ID（workflow-{name}-thread-{suffix}）
            is_run_mode: True=执行模式（DONE 事件标记 is_important）
        """
        self._ensure_not_closed()
        tid = self._current_sid(thread_id)
        trace_id = generate_trace_id()

        with TraceContext(trace_id=trace_id, thread_id=tid):
            logger.info("workflow arun_events [%s]: %s", tid, message[:100])
            async for ev in self._arun_graph_events(message, tid, trace_id, is_run_mode):
                yield ev

    async def aresume_events(
        self,
        payload: dict[str, Any],
        *,
        thread_id: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """恢复中断会话 — workflow 无 HITL 中断语义，不支持。"""
        raise NotImplementedError("WorkflowAdapter 不支持 aresume_events（无 HITL 中断）")

    # ============ SessionAgent 协议：历史 ============

    async def aget_execution_history(
        self, session_id: str | None = None
    ) -> list[dict[str, Any]]:
        """获取执行历史（复用 SessionStore，按会话隔离）。"""
        sid = self._current_sid(session_id)
        store = getattr(self._registry, "_store", None)
        if store is None:
            return []
        return await store.aget_history(sid)

    async def aclear_history(self, session_id: str | None = None) -> None:
        """清空执行历史。"""
        sid = self._current_sid(session_id)
        store = getattr(self._registry, "_store", None)
        if store is not None:
            await store.aclear_history(sid)

    # ============ SessionAgent 协议：压缩 ============

    async def manually_compact(
        self,
        force: bool = False,
        thread_id: str | None = None,
    ) -> dict[str, Any] | None:
        """手动触发 workflow 消息通道压缩（CLI compact 命令）。

        读取 workflow 会话的 checkpoint 消息 → 构造 compaction 中间件
        （从任一角色 agent 的 llm）→ arun_compaction → aupdate_state 写回。

        Args:
            force: 为 True 时跳过阈值检查强制压缩
            thread_id: 目标 workflow 会话 ID

        Returns:
            {"summary", "messages_before", "messages_after"} 或 None（无消息/失败）
        """
        tid = self._current_sid(thread_id)
        msgs = await self._registry.aget_messages(tid)
        if not msgs:
            return None

        workflow_name = self._registry.workflow_name_of(tid)
        if not workflow_name:
            return None

        # 读取当前 state 中的已有摘要
        existing_summary = ""
        config = {"configurable": {"thread_id": tid}}
        try:
            getter = getattr(self._checkpointer, "aget_tuple", None)
            tup = await getter(config) if callable(getter) else None
            if tup is not None and tup.checkpoint is not None:
                existing_summary = (
                    tup.checkpoint.get("channel_values", {}).get("summary", "") or ""
                )
        except Exception as error:
            logger.debug("读取 workflow 摘要失败: %s", error)

        # 惰性构造压缩中间件（从任一角色 agent 的 llm）
        from graph.registry import build_workflow

        _graph, agents = build_workflow(workflow_name, checkpointer=self._checkpointer)
        mw = _build_compaction_middleware(
            next(iter(agents.values())), self._compaction_config
        )
        if mw is None:
            return None

        update = await mw.arun_compaction(msgs, existing_summary=existing_summary, force=force)
        if update is None:
            return None

        await _graph.aupdate_state(config, update)
        return {
            "summary": update["summary"],
            "messages_before": len(msgs),
            "messages_after": len(update["messages"]) - 1,
        }

    # ============ SessionAgent 协议：生命周期 ============

    async def aclose(self) -> None:
        """优雅关闭（幂等）。checkpointer/Store 由 MemoryContext 负责关闭。"""
        if self._closed:
            return
        self._closed = True
        logger.info("WorkflowAdapter 已关闭")

    def _ensure_not_closed(self) -> None:
        if self._closed:
            raise RuntimeError("WorkflowAdapter 已关闭,不再可用")

    # ============ 内部：workflow 执行 ============

    async def _arun_graph_events(
        self,
        task: str,
        tid: str,
        trace_id: str,
        is_run_mode: bool,
    ) -> AsyncIterator[AgentEvent]:
        """执行 workflow 图并产出节点级事件流。"""
        # 1. warm workspace 缓存（断点续跑后缓存可能为空）
        await self._registry.awarm_workspace(tid)

        # 2. 从会话 ID 反解 workflow 名称
        workflow_name = self._registry.workflow_name_of(tid)
        if not workflow_name:
            raise ValueError(f"会话 {tid} 不是 workflow 会话,无法执行工作流")

        # 3. 构建图（checkpointer 持久化 + compaction 默认启用）
        from graph.registry import build_workflow

        graph, _agents = build_workflow(workflow_name, checkpointer=self._checkpointer)

        # 4. 执行前上下文：checkpoint 历史消息 + 长期记忆
        raw_context = await self._build_raw_context(tid)

        configurable: dict[str, Any] = {"thread_id": tid}
        workspace_path = getattr(self._registry.get_context(tid), "workspace_path", None)
        if workspace_path:
            configurable["workspace_path"] = workspace_path
        config: dict[str, Any] = {"configurable": configurable}

        # 5. 节点事件队列 + NodeTrackingHandler（run_inline 回调直接 put_nowait）
        node_queue: asyncio.Queue[AgentEvent] = asyncio.Queue()
        known_nodes = {
            n.id for n in graph.get_graph().nodes.values() if not n.id.startswith("__")
        }

        def _on_token(ev: AgentEvent) -> None:
            """LLM token 增量 → TOKEN 事件入队(补充会话元数据)"""
            node_queue.put_nowait(
                AgentEvent.token(
                    text=ev.content,
                    thread_id=tid,
                    role="assistant",
                    trace_id=trace_id,
                )
            )

        config["callbacks"] = [
            NodeTrackingHandler(
                known_nodes,
                on_node_start=node_queue.put_nowait,
                on_node_end=node_queue.put_nowait,
                on_node_error=node_queue.put_nowait,
                on_token=_on_token,
            )
        ]

        initial_state: dict[str, str] = {
            "task": task,
            "raw_context": raw_context,
            "context_summary": "",
        }

        async def _run() -> dict[str, Any]:
            return await graph.ainvoke(initial_state, config=config)

        run_task = asyncio.create_task(_run())

        # 6. 边执行边吐出节点事件（NodeTrackingHandler 回调实时入队）
        final: Any = None
        while True:
            try:
                ev = node_queue.get_nowait()
            except asyncio.QueueEmpty:
                if run_task.done():
                    final = run_task.result()
                    break
                await asyncio.sleep(0.005)
                continue
            yield ev

        # 7. 终止事件：ERROR 或 DONE(final_answer)
        if isinstance(final, BaseException):
            logger.error("工作流执行异常 [%s]: %s", tid, final)
            yield AgentEvent.error(
                content=f"工作流执行失败: {final}",
                thread_id=tid,
                trace_id=trace_id,
            )
            return

        final_answer = (
            final.get("final_answer", "") if isinstance(final, dict) else ""
        ) or ""
        logger.info("工作流执行完成 [%s]", tid)
        yield AgentEvent.done(
            content=final_answer,
            thread_id=tid,
            role="assistant",
            is_important=is_run_mode,
            trace_id=trace_id,
        )

    async def _build_raw_context(self, tid: str) -> str:
        """构建执行前注入的 raw_context：checkpoint 历史消息 + 长期记忆。

        历史消息来自 workflow 会话的 checkpoint messages 通道（跨轮次延续），
        长期记忆来自 MemoryManager.recall_text（跨会话事实）。两者任一
        失败或为空时静默跳过。
        """
        parts: list[str] = []

        # 1. checkpoint 历史消息（上一轮/多轮的节点 AIMessage 产出）
        try:
            msgs = await self._registry.aget_messages(tid)
            history: list[str] = []
            for msg in msgs:
                if isinstance(msg, AIMessage) and msg.content:
                    text = str(msg.content)
                    if len(text) > _HISTORY_MSG_PREVIEW_CHARS:
                        text = text[:_HISTORY_MSG_PREVIEW_CHARS] + "..."
                    history.append(f"- {text}")
                if len(history) >= _HISTORY_MSG_COUNT:
                    break
            if history:
                parts.append("【历史执行记录】\n" + "\n".join(history))
        except Exception as error:
            logger.debug("读取 workflow 历史消息失败: %s", error)

        # 2. 长期记忆召回
        if self._memory is not None:
            try:
                recalled = await self._memory.recall_text(tid)
                if recalled:
                    parts.append(recalled)
            except Exception as error:
                logger.debug("长期记忆召回失败: %s", error)

        return "\n\n".join(parts)


__all__ = ["WorkflowAdapter"]