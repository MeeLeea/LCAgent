"""WorkflowAdapter — workflow(graph/team 链路)的统一调度适配器。

使 workflow 链路获得与 AgentCore 会话链路同级的 SessionManager 门面接入
（统一会话/记忆/锁/SSE），上层（CLI/API）可无差别调度两类执行体。

实现 SessionAgent 协议（SessionManager 对 agent 依赖的接口）：
    arun_events / aresume_events / aretry_events / session / _current_sid /
    set_current_session / checkpoint_info / aget_execution_history /
    aclear_history / manually_compact / aclose

能力边界（OMO 式渐进）：
- 事件为节点级：NODE_START / NODE_END / NODE_ERROR → INTERRUPT / DONE
  （内层 team agent 的 token 增量经 NodeTrackingHandler.on_token 转发
  例外：节点内 TeamAgent.astream 透传 callbacks 后 LLM token 增量可见）
- arun_events 与 aresume_events 同构：均基于 _arun_input_events 处理
  graph.ainvoke 的事件队列 + 终止事件
- 跨轮次上下文由 checkpoint messages 通道 + 长期记忆(recall_text)承载
- 失败现场保存：节点异常时 checkpointer 保留 pending task（失败节点），
  SessionStore.failures 持久化失败节点名 + 异常信息（供重启后展示）
- 图级自动重试：瞬时错误（should_retry 判定）自动 ainvoke(None) 从失败
  节点续跑（指数退避，最多 RETRY_ATTEMPTS 次）；耗尽后落库失败记录
- aretry_events：手动从失败断点续跑（前置 aget_state 校验断点存在）

执行流程（arun_events）：
1. 从 session_id 反解 workflow 名称（registry.workflow_name_of）
2. build_workflow 构建图（checkpointer + compaction 默认启用）
3. 执行前：checkpoint 历史消息 + 长期记忆拼 raw_context 注入
4. graph.ainvoke 边执行边经 NodeTrackingHandler 队列吐出节点事件
5. 结束后：__interrupt__ → INTERRUPT；否则 DONE(final_answer)；
   记忆沉淀由 SessionManager 消费事件完成

恢复流程（aresume_events / aretry_events）：
1. 复用 _awarm_and_build_workflow 重建 graph + config（与 arun_events 一致）
2. aresume_events 用 Command(resume=payload) 恢复被 interrupt 的节点
3. aretry_events 用 ainvoke(None) 从失败节点断点续跑（checkpointer 的
   pending task 即失败节点；已成功节点不重跑）

allow: SIZE_OK — 文件 ~290 pure LOC 超 250 上限；arun_events/aresume_events
共享 _WorkflowRuntime + _awarm_and_build_workflow + _arun_input_events 三件套
是单文件内的内聚拆分，进一步拆到 session/workflow_runtime.py 会违反
"唯一改动文件" 任务约束。manually_compact/_build_raw_context 是 SessionAgent
协议的一部分，无法外移。
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.types import Command

from agent.compaction import CompactionConfig
from graph.common import _build_compaction_middleware, build_tracking_callbacks
from llm.llm_client import RETRY_ATTEMPTS, RETRY_MAX_DELAY, should_retry
from llm.message_utils import build_interrupt_event
from memory.manager import MemoryManager
from session.registry import SessionRegistry
from skmng.manager import SkillManager, default_skills_dir
from utils.events import AgentEvent, EventType
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
        # 技能管理器惰性创建（首次 list_skills/aload_skill 时扫描默认技能目录）
        self._skills: SkillManager | None = None

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

        事件顺序：NODE_START/NODE_END/NODE_ERROR* → INTERRUPT 或 DONE(final_answer)
        或 ERROR。SessionManager 负责锁/记忆提交/SSE 转发。

        Args:
            message: 用户任务
            thread_id: 目标 workflow 会话 ID（workflow-{name}-thread-{suffix}）
            is_run_mode: True=执行模式（DONE 事件标记 is_important）
        """
        self._ensure_not_closed()
        tid = self._current_sid(thread_id)
        trace_id = generate_trace_id()

        # 新任务开跑前清除上一轮的失败记录（stale 断点不再有意义）
        await self._aclear_failure_record(tid)

        with TraceContext(trace_id=trace_id, thread_id=tid):
            logger.info("workflow arun_events [%s]: %s", tid, message[:100])
            runtime = await self._awarm_and_build_workflow(tid, trace_id)
            # 执行前上下文：checkpoint 历史消息 + 长期记忆
            raw_context = await self._build_raw_context(tid)
            initial_state: dict[str, str] = {
                "task": message,
                "raw_context": raw_context,
                "context_summary": "",
            }
            async for ev in self._arun_input_events(runtime, initial_state, is_run_mode):
                yield ev

    async def aresume_events(
        self,
        payload: dict[str, Any],
        *,
        thread_id: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """恢复中断会话 — 用 Command(resume=payload) 恢复外层 graph。

        与 arun_events 同构：复用 _awarm_and_build_workflow + _arun_input_events，
        仅 graph.ainvoke 的输入不同（initial_state vs resume_command）。

        外层 graph 从被 interrupt 的节点恢复，节点内
        run_team_turn_with_interrupt（team/interrupt.py）的 interrupt() 返回 resume_value，
        调 aresume_structured 恢复内层 TeamAgent。

        Args:
            payload: 恢复数据（多 interrupt 时可为 {interrupt_id: value} 映射）
            thread_id: 目标 workflow 会话 ID（为 None 时用当前会话）
        """
        self._ensure_not_closed()
        tid = self._current_sid(thread_id)
        trace_id = generate_trace_id()

        with TraceContext(trace_id=trace_id, thread_id=tid):
            logger.info("workflow aresume_events [%s]", tid)
            runtime = await self._awarm_and_build_workflow(tid, trace_id)
            resume_command = Command(resume=payload)
            async for ev in self._arun_input_events(
                runtime, resume_command, is_run_mode=False
            ):
                yield ev

    async def aretry_events(
        self,
        *,
        thread_id: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """从失败断点手动重试 — ainvoke(None) 让图从失败节点续跑。

        与 aresume_events 的区别：interrupt 挂起需 Command(resume=payload)
        恢复；节点异常失败时 checkpointer 保留失败节点为 pending task，
        ainvoke(None) 即从该节点重跑（已成功节点不重跑，state 延续）。

        前置校验 aget_state：无 checkpoint 或 next 为空（图已跑完/从未跑过）
        时 yield ERROR，避免 ainvoke(None) 抛 EmptyInputError 或空转。

        Args:
            thread_id: 目标 workflow 会话 ID（为 None 时用当前会话）
        """
        self._ensure_not_closed()
        tid = self._current_sid(thread_id)
        trace_id = generate_trace_id()

        with TraceContext(trace_id=trace_id, thread_id=tid):
            logger.info("workflow aretry_events [%s]", tid)
            runtime = await self._awarm_and_build_workflow(tid, trace_id)

            # 前置校验：checkpoint 必须有待执行节点（失败断点）
            state = await runtime.graph.aget_state(runtime.config)
            if state is None or not state.next:
                logger.warning("workflow 重试被拒绝 [%s]: 无可恢复的失败断点", tid)
                yield AgentEvent.error(
                    content="无可恢复的失败断点（工作流可能已完成或从未执行）",
                    thread_id=tid,
                    trace_id=trace_id,
                )
                return

            # input=None：LangGraph 从 checkpoint 的 pending task（失败节点）续跑
            async for ev in self._arun_input_events(runtime, None, is_run_mode=False):
                yield ev

    # ============ SessionAgent 协议：历史 ============

    async def aget_failure_record(
        self, session_id: str | None = None
    ) -> dict[str, Any] | None:
        """读取最近一次失败记录（供 stream-status 端点查询）。

        返回 {"node": 失败节点名, "error": 错误信息, "ts": 时间戳} 或 None。
        Store 不可达时静默返回 None（不阻塞查询路径）。
        """
        sid = self._current_sid(session_id)
        store = getattr(self._registry, "_store", None)
        if store is None:
            return None
        try:
            return await store.aget_failure(sid)
        except Exception as error:
            logger.warning("读取失败记录失败 [%s]: %s", sid, error)
            return None

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

    # ============ 技能（对齐 AgentCore.SkillOps） ============

    def _get_skill_manager(self) -> SkillManager:
        """惰性创建技能管理器（扫描默认技能目录 .agents/skills）。"""
        if self._skills is None:
            self._skills = SkillManager(default_skills_dir())
        return self._skills

    def list_skills(self) -> list[dict[str, str]]:
        """列出所有本地可用技能（与 AgentCore.list_skills 同构）。"""
        return self._get_skill_manager().list_skills()

    @property
    def auto_match_skills(self) -> bool:
        """workflow 图构建时是否启用技能自动匹配（默认 True）。"""
        return True

    async def aload_skill(self, name: str, thread_id: str | None = None) -> bool:
        """加载技能到指定 workflow 会话（写入 state.active_skills）。

        技能名存入 workflow 会话的 per-thread checkpoint state，由节点经
        ``create_llm_node``（team/node_factory.py）读取 ``state["active_skills"]`` 后注入提示词。
        与 ``SkillOps.aload_skill`` 语义一致，仅图来源不同。

        Args:
            name: 技能名（目录名或 frontmatter name）
            thread_id: 目标 workflow 会话 ID（为 None 时使用当前会话）

        Returns:
            True=成功加载；False=技能不存在或目标会话非 workflow 会话
        """
        if self._get_skill_manager().get_skill(name) is None:
            return False
        tid = self._current_sid(thread_id)
        workflow_name = self._registry.workflow_name_of(tid)
        if not workflow_name:
            return False

        from graph.common import build_workflow

        graph, _agents = build_workflow(workflow_name, checkpointer=self._checkpointer)
        config = {"configurable": {"thread_id": tid}}
        state = await graph.aget_state(config)
        current: set[str] = set(
            (state.values if state and state.values else {}).get("active_skills") or []
        )
        current.add(name)
        await graph.aupdate_state(config, {"active_skills": sorted(current)})
        return True

    async def aclear_skills(self, thread_id: str | None = None) -> None:
        """清空指定 workflow 会话的手动加载技能（写入空列表）。

        注：只清 state.active_skills，不影响 TeamAgent.fixed_skills 类属性
        （角色级固定依赖由 build_skill_block 独立合并，不经 state 通道）。

        Args:
            thread_id: 目标 workflow 会话 ID（为 None 时使用当前会话）
        """
        tid = self._current_sid(thread_id)
        workflow_name = self._registry.workflow_name_of(tid)
        if not workflow_name:
            return

        from graph.common import build_workflow

        graph, _agents = build_workflow(workflow_name, checkpointer=self._checkpointer)
        config = {"configurable": {"thread_id": tid}}
        await graph.aupdate_state(config, {"active_skills": []})

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
            {"messages_before", "messages_after"} 或 None（无消息/失败）
        """
        tid = self._current_sid(thread_id)
        msgs = await self._registry.aget_messages(tid)
        if not msgs:
            return None

        workflow_name = self._registry.workflow_name_of(tid)
        if not workflow_name:
            return None

        # 惰性构造压缩中间件（从任一角色 agent 的 llm）
        from graph.common import build_workflow

        _graph, agents = build_workflow(workflow_name, checkpointer=self._checkpointer)
        mw = _build_compaction_middleware(
            next(iter(agents.values())), self._compaction_config
        )
        if mw is None:
            return None

        # 工作流 State 无 summary 通道：existing_summary 固定为空串（D1-A），
        # 手动 compact 仅压缩 messages
        update = await mw.arun_compaction(msgs, existing_summary="", force=force)
        if update is None:
            return None

        # summary 已不再持久化到任何通道，写回前丢弃以免污染 state
        # （LangGraph 对未声明通道写入静默丢弃，不会崩，但保持语义干净）
        update.pop("summary", None)
        config: dict[str, Any] = {"configurable": {"thread_id": tid}}
        await _graph.aupdate_state(config, update)
        return {
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

    @dataclass(slots=True)
    class _WorkflowRuntime:
        """graph.ainvoke 执行所需的运行时上下文。

        - graph: build_workflow 构建的编译图
        - config: 含 thread_id / workspace_path / callbacks 的 LangGraph 配置
        - node_queue: NodeTrackingHandler 回调实时入队的节点事件队列
        - tid: 会话线程 ID（已写入 config.configurable.thread_id，此处冗余存储
          便于 _arun_input_events 直接消费）
        - trace_id: 当前 turn 的追踪 ID（构造 NodeTrackingHandler 用）
        """

        graph: Any
        config: dict[str, Any]
        node_queue: asyncio.Queue[AgentEvent]
        tid: str
        trace_id: str

    async def _awarm_and_build_workflow(
        self,
        tid: str,
        trace_id: str,
    ) -> _WorkflowRuntime:
        """warm workspace 缓存 + 构建 graph + 构造 config + callbacks。

        arun_events 与 aresume_events 的共享前置步骤：
        1. awarm_workspace（断点续跑后缓存可能为空）
        2. 反解 workflow_name，构建 graph（checkpointer + compaction 默认启用）
        3. 构造 config（thread_id + workspace_path）
        4. 构造 node_queue + NodeTrackingHandler（节点事件 + LLM token 增量入队）

        不含 raw_context 注入（仅 arun_events 需要）和 graph.ainvoke 调用。
        """
        # 1. warm workspace 缓存
        await self._registry.awarm_workspace(tid)

        # 2. 反解 workflow 名称 + 构建图
        workflow_name = self._registry.workflow_name_of(tid)
        if not workflow_name:
            raise ValueError(f"会话 {tid} 不是 workflow 会话,无法执行工作流")

        from graph.common import build_workflow

        graph, _agents = build_workflow(workflow_name, checkpointer=self._checkpointer)

        # 3. 构造 config（thread_id + workspace_path）
        # 从 SessionContext.config.configurable 读 workspace_path（SessionContext
        # 无顶级 workspace_path 属性，workspace_path 存于 config.configurable）；
        # 与 awarm_workspace 的 warm 缓存保持一致读源。
        configurable: dict[str, Any] = {"thread_id": tid}
        ctx = self._registry.get_context(tid)
        workspace_path = ctx.config.get("configurable", {}).get("workspace_path")
        if workspace_path:
            configurable["workspace_path"] = workspace_path
        node_queue: asyncio.Queue[AgentEvent] = asyncio.Queue()

        # 4. NodeTrackingHandler：节点事件 + LLM token 增量入队
        def _on_token(ev: AgentEvent) -> None:
            """LLM token 增量 → TOKEN 事件入队（补充会话元数据）。"""
            node_queue.put_nowait(
                AgentEvent.token(
                    text=ev.content,
                    thread_id=tid,
                    role="assistant",
                    trace_id=trace_id,
                )
            )

        config: dict[str, Any] = {
            "configurable": configurable,
            "callbacks": build_tracking_callbacks(
                graph,
                on_node_start=node_queue.put_nowait,
                on_node_end=node_queue.put_nowait,
                on_node_error=node_queue.put_nowait,
                on_token=_on_token,
            ),
        }

        return self._WorkflowRuntime(
            graph=graph,
            config=config,
            node_queue=node_queue,
            tid=tid,
            trace_id=trace_id,
        )

    async def _arun_input_events(
        self,
        runtime: _WorkflowRuntime,
        graph_input: Any,
        is_run_mode: bool,
    ) -> AsyncIterator[AgentEvent]:
        """执行 graph.ainvoke 并吐出节点事件 + 终止事件。

        arun_events 传 initial_state；aresume_events 传 Command(resume=...)；
        aretry_events 传 None（从 checkpoint pending task 续跑）。

        图级自动重试：final 为瞬时异常（should_retry 判定）且未达
        RETRY_ATTEMPTS 时，指数退避后以 input=None 重跑失败节点
        （checkpointer 保留 pending task，已成功节点不重跑）。
        事件泵跨尝试共用同一 node_queue，第二轮 NODE_START 会继续
        流给前端（节点红→running→绿）。

        终止事件逻辑：
        - final 为异常（不可重试/已耗尽）→ 落库失败记录 + ERROR
        - final 含 __interrupt__ → 清失败记录 + INTERRUPT
        - 否则 → 清失败记录 + DONE(final_answer)
        """
        last_error_node = ""  # 最近一次 NODE_ERROR 事件的节点名
        attempt = 0
        current_input = graph_input
        tid = runtime.tid
        trace_id = runtime.trace_id

        while True:
            run_task = asyncio.create_task(
                runtime.graph.ainvoke(current_input, config=runtime.config)
            )

            # 边执行边吐出节点事件（NodeTrackingHandler 回调实时入队）
            # finally 兜底：外层被取消（dispatch_task.cancel / 客户端断开）时，
            # 同步取消 run_task，避免 graph.ainvoke 继续执行后续节点。
            final: Any = None
            try:
                while True:
                    try:
                        ev = runtime.node_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        if run_task.done():
                            # run_task 异常时 result() 会 raise，捕获为返回值；
                            # CancelledError 非 Exception，自然向上传播（外层取消仍生效）
                            try:
                                final = run_task.result()
                            except Exception as e:
                                final = e
                            break
                        await asyncio.sleep(0.005)
                        continue
                    if ev.event_type == EventType.NODE_ERROR:
                        last_error_node = ev.node
                    yield ev
            finally:
                if not run_task.done():
                    run_task.cancel()
                    try:
                        await run_task
                    except (asyncio.CancelledError, Exception):  # noqa: S110 - 主动取消 run_task
                        # 仅吞 run_task 自身的取消/执行异常；
                        # 外层 CancelledError 不在此处被吞，会在 finally 完成后继续向上传播
                        pass

            # 终止事件判定：瞬时异常 → 自动重试（input=None 从失败节点续跑）
            if isinstance(final, BaseException):
                if should_retry(final) and attempt < RETRY_ATTEMPTS - 1:
                    attempt += 1
                    delay = min(RETRY_MAX_DELAY, float(2**attempt))
                    logger.warning(
                        "工作流瞬时异常，%.1fs 后从失败节点重试(第 %d/%d 次) [%s]: %s",
                        delay,
                        attempt,
                        RETRY_ATTEMPTS - 1,
                        tid,
                        final,
                    )
                    await asyncio.sleep(delay)
                    current_input = None
                    continue

                # 不可重试或已耗尽：落库失败记录（节点名+异常信息）+ ERROR
                logger.error("工作流执行异常 [%s]: %s", tid, final)
                await self._aset_failure_record(tid, last_error_node, str(final))
                yield AgentEvent.error(
                    content=f"工作流执行失败: {final}",
                    thread_id=tid,
                    trace_id=trace_id,
                )
                return

            # Todo 3.1：检测 __interrupt__，yield INTERRUPT（对照 interrupts.py:166-196
            # + streaming.py:323-326）。interrupt 时 final_answer 为空。
            if isinstance(final, dict) and "__interrupt__" in final:
                interrupts = final["__interrupt__"] or []
                if interrupts:
                    # 失败已被 interrupt 场景取代，清除旧失败记录
                    await self._aclear_failure_record(tid)
                    ev_dicts = [
                        build_interrupt_event(getattr(intr, "value", None))
                        for intr in interrupts
                    ]
                    # 多 interrupt 合并 prompt（同 interrupts.py:179-189 的批量语义）
                    prompt = "\n\n".join(d.get("prompt", "") for d in ev_dicts)
                    choices = ev_dicts[0].get("choices") if ev_dicts else None
                    # 多 interrupt 时把各分组待确认项拼接成一个列表（user_confirmation
                    # 才有 items；human_choice/dangerous_command 走 d.get("items") or [] 为空）
                    items = [it for d in ev_dicts for it in (d.get("items") or [])]
                    logger.info("工作流执行被中断 [%s] interrupts=%d", tid, len(interrupts))
                    yield AgentEvent.interrupt(
                        prompt=prompt,
                        choices=choices,
                        items=items,
                        thread_id=tid,
                        trace_id=trace_id,
                    )
                    return

            # 重试成功（或正常完成）：清除失败记录
            await self._aclear_failure_record(tid)
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
            return

    async def _aset_failure_record(self, tid: str, node: str, error: str) -> None:
        """写入失败记录到 SessionStore（Store 不可达或写失败时静默跳过）。"""
        store = getattr(self._registry, "_store", None)
        if store is None:
            return
        try:
            await store.aset_failure(tid, node=node, error=error)
        except Exception as e:
            logger.warning("写入失败记录失败 [%s]: %s", tid, e)

    async def _aclear_failure_record(self, tid: str) -> None:
        """清除失败记录（Store 不可达或删失败时静默跳过）。"""
        store = getattr(self._registry, "_store", None)
        if store is None:
            return
        try:
            await store.aclear_failure(tid)
        except Exception as e:
            logger.warning("清除失败记录失败 [%s]: %s", tid, e)

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