"""SessionManager — 对外门面 & 会话调度，三层架构的 Session 层。

接管所有上层流量、并发、流输出、checkpoint、API。
Agent 不再对外提供任何 achat / 流式接口，全部由 SessionManager 承担。

核心职责：
1. 统一对外暴露 achat_stream() 流式接口、普通 chat 接口
2. thread 并发锁、串行执行、会话隔离
3. 加载/保存 LangGraph checkpoint 短期快照
4. 消费 Agent 事件流：实时吐出 SSE 文本流 + 增量更新会话状态
5. 调用 Memory 服务做长期记忆沉淀

标准化数据流：
  1. 客户端请求 → Session
  2. Session 加锁、加载 checkpoint
  3. Session 调用 Memory 召回长期记忆
  4. Session 调用 Agent.arun_events() 传入上下文 + 用户输入
  5. Agent 内部跑 graph，持续产出 AgentEvent
  6. Session 实时消费事件：对外吐出流式文本 + 增量保存会话状态
  7. 异步丢给 Memory 处理长期记忆
  8. 执行结束，Session 释放锁
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from memory.manager import MemoryManager

from .registry import SessionRegistry

logger = logging.getLogger(__name__)


class SessionManager:
    """对外门面 & 会话调度 — 三层架构的 Session 层。

    封装 AgentCore + MemoryManager + SessionRegistry 为统一门面，
    所有上层流量（API / CLI / Remote）只通过 SessionManager 访问 Agent。

    Args:
        agent: AgentCore 实例（纯执行内核）
        memory: MemoryManager 实例（独立记忆基础设施）。
                为 None 时记忆功能不可用（流式接口仍可工作，但无记忆沉淀）。
                入口程序应通过 MemoryContext 创建后注入。
    """

    def __init__(
        self,
        agent: Any,
        memory: MemoryManager | None = None,
    ) -> None:
        self._agent = agent
        self._memory = memory

        # Per-thread asyncio locks：同一 thread 串行执行，不同 thread 并行
        self._thread_locks: dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()

    # ============ 属性暴露 ============

    @property
    def agent(self) -> Any:
        """底层 AgentCore 实例（供工具/技能/LLM 管理等 agent 级操作使用）。"""
        return self._agent

    @property
    def memory(self) -> MemoryManager | None:
        """MemoryManager 实例（长期记忆召回/消费/压缩/清理）。"""
        return self._memory

    @property
    def session(self) -> SessionRegistry:
        """SessionRegistry 实例（会话生命周期管理）。"""
        return self._agent.session

    # ============ 并发锁管理 ============

    async def _get_thread_lock(self, thread_id: str) -> asyncio.Lock:
        """获取或创建 per-thread asyncio.Lock。

        同一 thread_id 的请求串行执行，不同 thread_id 并行。
        锁池无界增长可接受：thread 数量受 SessionRegistry 管控，
        且锁对象本身极轻量。
        """
        async with self._locks_guard:
            if thread_id not in self._thread_locks:
                self._thread_locks[thread_id] = asyncio.Lock()
            return self._thread_locks[thread_id]

    def _resolve_thread_id(self, thread_id: str | None) -> str:
        """解析目标会话 ID（为 None 时使用当前会话）。"""
        return self._agent._current_sid(thread_id)

    # ============ 对外流式接口 ============

    async def achat_stream(
        self,
        message: str,
        thread_id: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """流式对话 — 唯一对外流式接口。

        数据流：
        1. 加锁（per-thread 串行）
        2. 提交用户消息到 Memory（非阻塞防抖）
        3. 调用 Agent.arun_events() 获取 AgentEvent 流
        4. 实时消费事件：转 SSE dict 吐出 + 异步提交记忆
        5. 释放锁

        事件格式（SSE dict，向后兼容现有前端协议）：
          {"type": "token", "content": str}
          {"type": "tool_call", "id", "name", "args"}
          {"type": "tool_result", "id", "name", "content"}
          {"type": "interrupt", "prompt", "choices"}
          {"type": "cancelled", "content"}
          {"type": "error", "content"}
          {"type": "done"}

        Args:
            message: 用户消息
            thread_id: 目标会话线程 ID（为 None 时使用当前会话）
        """
        tid = self._resolve_thread_id(thread_id)
        lock = await self._get_thread_lock(tid)

        async with lock:
            # 提交用户消息到记忆（非阻塞）
            if self._memory is not None:
                await self._memory.submit_user_message(tid, message)

            # 消费 Agent 事件流
            async for event in self._agent.arun_events(message, thread_id=tid):
                # 转发为 SSE dict（向后兼容前端协议）
                yield event.to_sse_dict()

                # 异步提交记忆（非阻塞防抖）
                if self._memory is not None and event.is_memory_worthy:
                    await self._memory.consume_event(event)

    async def aresume_stream(
        self,
        payload: dict[str, Any],
        thread_id: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """流式恢复中断会话 — 事件格式同 achat_stream。

        Args:
            payload: 恢复数据
            thread_id: 目标会话线程 ID（为 None 时使用当前会话）
        """
        tid = self._resolve_thread_id(thread_id)
        lock = await self._get_thread_lock(tid)

        async with lock:
            async for event in self._agent.aresume_events(payload, thread_id=tid):
                yield event.to_sse_dict()
                if self._memory is not None and event.is_memory_worthy:
                    await self._memory.consume_event(event)

    async def arun_stream(
        self,
        task: str,
        thread_id: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """流式执行任务 — 同 achat_stream 但 is_run_mode=True（标记重要记忆）。

        Args:
            task: 任务描述
            thread_id: 目标会话线程 ID（为 None 时使用当前会话）
        """
        tid = self._resolve_thread_id(thread_id)
        lock = await self._get_thread_lock(tid)

        async with lock:
            if self._memory is not None:
                await self._memory.submit_user_message(tid, task, important=True)

            async for event in self._agent.arun_events(
                task, thread_id=tid, is_run_mode=True
            ):
                yield event.to_sse_dict()
                if self._memory is not None and event.is_memory_worthy:
                    await self._memory.consume_event(event)

    # ============ 对外非流式接口 ============

    async def achat(
        self,
        message: str,
        thread_id: str | None = None,
    ) -> str:
        """非流式对话。

        内部调用 achat_stream 并收集全部 token 事件为最终输出。

        Args:
            message: 用户消息
            thread_id: 目标会话线程 ID

        Returns:
            助手回复文本

        Raises:
            RuntimeError: 会话被中断（需调用 aresume 恢复）
        """
        output_parts: list[str] = []
        async for ev_dict in self.achat_stream(message, thread_id=thread_id):
            ev_type = ev_dict.get("type")
            if ev_type == "token":
                output_parts.append(ev_dict.get("content", ""))
            elif ev_type == "error":
                return ev_dict.get("content", "")
            elif ev_type == "interrupt":
                raise RuntimeError("Agent turn interrupted; resume with aresume().")
        return "".join(output_parts)

    async def aresume(
        self,
        payload: dict[str, Any],
        thread_id: str | None = None,
    ) -> str:
        """非流式恢复中断会话。

        Args:
            payload: 恢复数据
            thread_id: 目标会话线程 ID

        Returns:
            助手回复文本

        Raises:
            RuntimeError: 会话再次被中断
        """
        output_parts: list[str] = []
        async for ev_dict in self.aresume_stream(payload, thread_id=thread_id):
            ev_type = ev_dict.get("type")
            if ev_type == "token":
                output_parts.append(ev_dict.get("content", ""))
            elif ev_type == "error":
                return ev_dict.get("content", "")
            elif ev_type == "interrupt":
                raise RuntimeError("Agent turn interrupted; resume with aresume().")
        return "".join(output_parts)

    async def arun(
        self,
        task: str,
        thread_id: str | None = None,
    ) -> str:
        """非流式执行任务（run 模式，标记重要记忆）。

        Args:
            task: 任务描述
            thread_id: 目标会话线程 ID

        Returns:
            执行结果文本

        Raises:
            RuntimeError: 会话被中断
        """
        output_parts: list[str] = []
        async for ev_dict in self.arun_stream(task, thread_id=thread_id):
            ev_type = ev_dict.get("type")
            if ev_type == "token":
                output_parts.append(ev_dict.get("content", ""))
            elif ev_type == "error":
                return ev_dict.get("content", "")
            elif ev_type == "interrupt":
                raise RuntimeError("Agent turn interrupted; resume with aresume().")
        return "".join(output_parts)

    # ============ 会话管理（委托 SessionRegistry） ============

    def new_session(self) -> str:
        """开启新会话。"""
        return self.session.new_session()

    def new_workflow_session(self, workflow_name: str) -> str:
        """开启新的专属工作流会话。"""
        return self.session.new_workflow_session(workflow_name)

    def set_current_session(self, session_id: str) -> None:
        """设置当前会话 ID。"""
        self._agent.set_current_session(session_id)

    @property
    def current_session_id(self) -> str:
        """当前会话 ID。"""
        return self.session.current_session_id

    async def alist_sessions(self, all_types: bool = False) -> list[str]:
        """列出所有可见会话。"""
        return await self.session.alist_sessions(all_types=all_types)

    async def aswitch_session(self, session_id: str) -> bool:
        """切换到指定会话。"""
        return await self.session.aswitch_session(session_id)

    async def adelete_session(self, session_id: str) -> bool:
        """删除会话。

        删除前先清理该会话的 thread 级长期记忆（与写流水线串行化，避免
        清完又被防抖 flush 回写“复活”），agent 级跨会话共享记忆不受影响。
        """
        if self._memory is not None:
            try:
                await self._memory.clear(session_id, release_lock=True)
            except Exception as error:  # 记忆清理失败不应阻断会话删除
                logger.warning("清理会话长期记忆失败 [sid=%s]: %s", session_id, error)
        return await self.session.adelete_session(session_id)

    async def aget_messages(self, session_id: str | None = None) -> list[Any]:
        """获取会话消息列表。"""
        return await self.session.aget_messages(session_id)

    async def aexport_session(
        self, session_id: str | None = None, fmt: str = "text"
    ) -> str:
        """导出会话为可读文本。"""
        return await self.session.aexport_session(session_id, fmt=fmt)

    async def asummarize(self, session_id: str | None = None) -> dict[str, Any]:
        """获取会话摘要统计。"""
        return await self.session.asummarize(session_id)

    # ============ 记忆管理（委托 MemoryManager） ============

    async def aget_memory_summary(self) -> dict[str, Any]:
        """获取记忆摘要统计。"""
        sid = self.current_session_id
        session_info = await self.session.asummarize()
        cp_info = self._agent.checkpoint_info
        long_term_count = await self._memory.count_facts(sid) if self._memory else 0
        # agent 级（跨会话共享）长期记忆条数：与 thread 级计数并列展示
        agent_fact_count = (
            await self._memory.count_agent_facts() if self._memory else 0
        )
        return {
            "thread_id": sid,
            "checkpoint_messages": session_info["checkpoint_messages"],
            "checkpoint_backend": cp_info["checkpoint_backend"],
            "checkpoint_file": cp_info["checkpoint_file"],
            "long_term_count": long_term_count,
            "agent_fact_count": agent_fact_count,
            "total_threads": session_info["total_sessions"],
        }

    async def acompress_memory(self) -> dict[str, Any]:
        """压缩长期记忆。"""
        if self._memory is None:
            return {"success": False, "error": "MemoryManager 未初始化"}
        sid = self.current_session_id
        return await self._memory.compress(sid)

    async def aclear_long_term_memory(self, session_id: str | None = None) -> int:
        """清空长期记忆。"""
        if self._memory is None:
            return 0
        sid = self._resolve_thread_id(session_id)
        return await self._memory.clear(sid)

    # ============ 执行历史 ============

    async def aget_execution_history(
        self, session_id: str | None = None
    ) -> list[dict[str, Any]]:
        """获取执行历史。"""
        sid = self._resolve_thread_id(session_id)
        return await self._agent.aget_execution_history(sid)

    async def aclear_history(self, session_id: str | None = None) -> None:
        """清空执行历史。"""
        sid = self._resolve_thread_id(session_id)
        await self._agent.aclear_history(sid)

    async def manually_compact(
        self, force: bool = False, thread_id: str | None = None
    ) -> dict[str, Any] | None:
        """手动触发上下文压缩。"""
        return await self._agent.manually_compact(force=force, thread_id=thread_id)

    # ============ 生命周期 ============

    async def aclose(self) -> None:
        """优雅关闭：刷新记忆 buffer、关闭中间件、释放 Agent 资源。

        MemoryManager 的关闭由 MemoryContext.aclose() 负责（入口程序调用），
        此处仅刷新 buffer 以防数据丢失，然后释放 Agent 资源。
        """
        if self._memory is not None:
            try:
                await self._memory.flush_all()
                await self._memory.shutdown()
            except Exception as error:
                logger.warning("MemoryManager 关闭异常: %s", error, exc_info=True)

        await self._agent.aclose()


def create_workflow_session_manager(
    registry: SessionRegistry,
    memory: MemoryManager | None = None,
    checkpointer: Any = None,
) -> SessionManager:
    """创建 workflow 链路的统一 SessionManager 门面（绑定 WorkflowAdapter）。

    workflow(graph/team 链路)与 AgentCore 会话链路共享同一 SessionRegistry /
    MemoryManager / checkpointer，使上层（CLI/API）以同一门面无差别调度两类执行体。

    Args:
        registry: SessionRegistry 实例（与 AgentCore 共享）
        memory: MemoryManager 实例（与 chat 门面共享同一实例）
        checkpointer: LangGraph checkpointer；为 None 时 WorkflowAdapter
                     复用 registry 的 checkpointer

    Returns:
        绑定 WorkflowAdapter 的 SessionManager（提供 arun_stream / achat_stream /
        锁 / 记忆提交与沉淀 / 手动压缩等统一能力）
    """
    # 延迟导入避免启动期依赖（WorkflowAdapter 依赖 graph 模块）
    from .workflow_adapter import WorkflowAdapter

    adapter = WorkflowAdapter(
        registry=registry,
        memory=memory,
        checkpointer=checkpointer,
    )
    return SessionManager(adapter, memory=memory)


__all__ = ["SessionManager", "create_workflow_session_manager"]
