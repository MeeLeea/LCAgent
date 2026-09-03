"""事件流引擎 Mixin - AgentCore 的 astream_events 消费与 AgentEvent 事件流输出。

从 agent_core.py 抽离，职责：
- _arun_graph_events：消费 LangGraph astream_events，映射为 AgentEvent
  （处理重试、UserRejectedCommandError、LLM 异常映射、孤儿 tool_call 补发）
- arun_events / aresume_events：纯执行接口，输出标准化 AgentEvent 事件流

依赖 AgentCore 实例属性：agent_executor / _session_registry /
_session_store / metrics / verbose / llm。
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from llm.llm_client import RETRY_ATTEMPTS, RETRY_MAX_DELAY, should_retry
from llm.message_utils import extract_llm_error, stringify_content
from tools.terminal_tools import UserRejectedCommandError
from utils.events import AgentEvent
from utils.logging_config import TraceContext, generate_trace_id

if TYPE_CHECKING:
    from .agent_core import AgentCore

logger = logging.getLogger(__name__)

# 工具执行心跳间隔（秒）：长耗时工具执行期间定期发 tool_running 事件，
# 重置前端 watchdog（WATCHDOG_MS=90s），防止 60s+ 工具超时误触发响应超时。
# 15s 间隔留 6 倍余量，即使丢 1-2 个心跳也不误触发。
HEARTBEAT_INTERVAL = 15


class Streaming:
    """事件流引擎 Mixin（供 AgentCore 多继承使用，自身不初始化状态）"""

    # ============ 纯执行接口（yield AgentEvent，不处理记忆/会话/并发） ============

    async def _arun_graph_events(
        self,
        input_or_command: Any,
        config: dict[str, Any],
        thread_id: str,
        trace_id: str,
        collected_output: list[str] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """内部方法：直接消费 LangGraph astream_events，映射为 AgentEvent。

        处理重试、UserRejectedCommandError、LLM 异常映射。
        不处理：interrupt 检查、记忆提交、工具步骤记录。

        Args:
            input_or_command: 输入消息 dict 或 Command(resume=...)
            config: LangGraph 配置
            thread_id: 会话线程 ID（写入事件）
            trace_id: 追踪 ID
            collected_output: 可选列表，收集 TOKEN 文本用于最终 DONE 事件
        """
        graph = self.agent_executor
        recorded_msg_ids: set[str] = set()
        # 已发出 TOOL_CALL 事件的 tool_call_id 集合：
        # on_chat_model_end 检测到 tool_calls 时提前发出，on_tool_start 据此去重
        emitted_tool_call_ids: set[str] = set()
        # run_id → tool_call_id：on_tool_start 按 run_id 登记、on_tool_end 取用。
        # on_tool_start/on_tool_end 事件 data 不含 tool_call_id（LangChain 事件未暴露），
        # 需借助 on_chat_model_end 拿到的 AIMessage.tool_calls id 桥接，保证前端
        # tool_call / tool_result 事件 id 一致，避免工具卡片永远停留在"执行中"。
        active_tool_call_ids: dict[str, str] = {}
        # tool_call_id → tool_name：心跳事件携带工具名供前端显示
        active_tool_names: dict[str, str] = {}
        # 工具名 → 本轮 tool_call id 队列：on_chat_model_end 填充，on_tool_start 按事件 name 取用
        pending_tool_call_ids: dict[str, list[str]] = {}

        for attempt in range(RETRY_ATTEMPTS):
            emitted = False
            try:
                # 用独立 task 消费 graph 事件到 queue，主循环从 queue 取。
                # 超时发心跳，不 cancel graph 的 async generator
                # （wait_for cancel __anext__ 会导致 generator 终止，后续事件丢失）。
                _event_queue: asyncio.Queue = asyncio.Queue()
                _consume_exc: BaseException | None = None

                async def _consume_graph(_q: asyncio.Queue) -> None:
                    nonlocal _consume_exc
                    try:
                        async for _ev in graph.astream_events(
                            input_or_command,
                            config=config,
                            version="v2",
                        ):
                            await _q.put(_ev)
                    except BaseException as _exc:
                        _consume_exc = _exc
                    finally:
                        await _q.put(None)

                _consumer_task = asyncio.create_task(_consume_graph(_event_queue))
                while True:
                    try:
                        _item = await asyncio.wait_for(
                            _event_queue.get(),
                            timeout=HEARTBEAT_INTERVAL,
                        )
                    except TimeoutError:
                        # 工具执行期间长时间无事件：发心跳重置前端 watchdog
                        for _tc_id in active_tool_call_ids.values():
                            yield AgentEvent.tool_running(
                                tool_call_id=_tc_id,
                                name=active_tool_names.get(_tc_id, ""),
                                thread_id=thread_id,
                                trace_id=trace_id,
                            )
                        continue
                    if _item is None:
                        break
                    ev = _item
                    event_name = ev.get("event", "") if isinstance(ev, dict) else ""
                    metadata = ev.get("metadata") if isinstance(ev, dict) else None
                    data = ev.get("data") if isinstance(ev, dict) else None
                    metadata_dict = metadata if isinstance(metadata, dict) else {}
                    data_dict = data if isinstance(data, dict) else {}

                    if event_name == "on_chat_model_stream":
                        node = metadata_dict.get("langgraph_node", "")
                        chunk = data_dict.get("chunk")
                        if isinstance(chunk, AIMessageChunk) and node in ("agent", "model"):
                            text = stringify_content(chunk.content)
                            if text:
                                emitted = True
                                if collected_output is not None:
                                    collected_output.append(text)
                                yield AgentEvent.token(
                                    text, thread_id=thread_id, trace_id=trace_id
                                )
                    elif event_name == "on_tool_start":
                        # on_tool_start 事件 data 不含 tool_call_id，用 run_id 登记映射恢复；
                        # 未命中时按事件 name 从本轮 AIMessage.tool_calls 队列取 id。
                        run_id = ev.get("run_id", "")
                        tc_id = data_dict.get("tool_call_id") or ""
                        if not tc_id:
                            event_name_top = ev.get("name", "")
                            bucket = pending_tool_call_ids.get(event_name_top)
                            if bucket:
                                tc_id = bucket.pop(0)
                            elif event_name_top:
                                tc_id = event_name_top
                        if tc_id:
                            active_tool_call_ids[run_id] = tc_id
                        # on_chat_model_end 已提前发出 TOOL_CALL 时跳过（去重）
                        if tc_id and tc_id in emitted_tool_call_ids:
                            continue
                        name = ev.get("name") or data_dict.get("name")
                        if tc_id:
                            active_tool_names[tc_id] = name or ""
                        tool_input = data_dict.get("input")
                        emitted = True
                        yield AgentEvent.tool_call(
                            tool_call_id=tc_id,
                            name=name or "",
                            args=tool_input,
                            thread_id=thread_id,
                            trace_id=trace_id,
                        )
                    elif event_name == "on_tool_end":
                        output = data_dict.get("output")
                        # output 就是工具返回值：字符串直接用本身；
                        # 若是 ToolMessage 等含 content 的对象则取其 content。
                        content = (
                            getattr(output, "content", output)
                            if output is not None
                            else None
                        )
                        # ID 桥接：优先 run_id 映射 → pending 队列按名匹配 →
                        # ToolMessage.tool_call_id → 工具名兜底
                        run_id = ev.get("run_id", "")
                        tc_id = active_tool_call_ids.pop(run_id, "")
                        if not tc_id:
                            event_name_top = ev.get("name", "")
                            bucket = pending_tool_call_ids.get(event_name_top)
                            if bucket:
                                tc_id = bucket.pop(0)
                        if not tc_id:
                            tc_id = (
                                data_dict.get("tool_call_id")
                                or getattr(output, "tool_call_id", None)
                                or ev.get("name", "")
                            )
                        if tc_id:
                            active_tool_names.pop(tc_id, None)
                        emitted = True
                        yield AgentEvent.tool_result(
                            tool_call_id=tc_id,
                            name=ev.get("name") or data_dict.get("name", "") or "",
                            content=stringify_content(content),
                            thread_id=thread_id,
                            trace_id=trace_id,
                        )
                    elif event_name == "on_tool_error":
                        # 工具内部抛异常（MCP 崩溃 / 路径解析 bug / 权限错误等），
                        # LangGraph 不发射 on_tool_end，而是发射 on_tool_error。
                        # 发 tool_result 携带错误信息（而非 error 终止事件），
                        # 让前端工具卡片显示失败 + LLM 能看到错误并调整策略。
                        error_obj = data_dict.get("error")
                        error_msg = (
                            str(error_obj) if error_obj is not None
                            else "工具执行异常（未知错误）"
                        )
                        # on_tool_error 的 data 直接含 tool_call_id（复现验证），
                        # 但仍走 ID 桥接兜底，与 on_tool_end 保持一致
                        run_id = ev.get("run_id", "")
                        tc_id = active_tool_call_ids.pop(run_id, "")
                        if not tc_id:
                            tc_id = data_dict.get("tool_call_id", "")
                        if not tc_id:
                            event_name_top = ev.get("name", "")
                            bucket = pending_tool_call_ids.get(event_name_top)
                            if bucket:
                                tc_id = bucket.pop(0)
                        if not tc_id:
                            tc_id = ev.get("name", "")
                        if tc_id:
                            active_tool_names.pop(tc_id, None)
                        name = ev.get("name") or data_dict.get("name", "") or ""
                        emitted = True
                        yield AgentEvent.tool_result(
                            tool_call_id=tc_id,
                            name=name,
                            content=f"[工具执行失败] {error_msg}",
                            thread_id=thread_id,
                            trace_id=trace_id,
                        )
                    elif event_name == "on_chat_model_end":
                        output = data_dict.get("output")
                        if isinstance(output, AIMessage):
                            msg_id = getattr(output, "id", None) or str(id(output))
                            if msg_id not in recorded_msg_ids:
                                recorded_msg_ids.add(msg_id)
                                llm = getattr(self, "llm", None)
                                self.metrics.extract_and_record_llm_usage(
                                    output,
                                    provider=getattr(llm, "provider", ""),
                                    model=getattr(llm, "model", "") or "",
                                )
                            # 提前发出 TOOL_CALL：LLM 回复完成时 tool_calls 已确定，
                            # 无需等待 LangGraph 路由到工具节点（on_tool_start），
                            # 前端可更早显示工具名 + "执行中"
                            # 新一轮工具调用开始：清空上一轮遗留的 id 队列
                            # （上一轮的 tools 已在执行完毕后进入下一轮 model 调用）
                            pending_tool_call_ids = {}
                            for tc in getattr(output, "tool_calls", None) or []:
                                tc_id = tc.get("id", "") if isinstance(tc, dict) else ""
                                tc_name = tc.get("name", "") if isinstance(tc, dict) else ""
                                if tc_id:
                                    pending_tool_call_ids.setdefault(tc_name, []).append(tc_id)
                                if tc_id and tc_id not in emitted_tool_call_ids:
                                    emitted_tool_call_ids.add(tc_id)
                                    emitted = True
                                    yield AgentEvent.tool_call(
                                        tool_call_id=tc_id,
                                        name=tc_name,
                                        args=tc.get("args") if isinstance(tc, dict) else None,
                                        thread_id=thread_id,
                                        trace_id=trace_id,
                                    )
                if not _consumer_task.done():
                    _consumer_task.cancel()
                if _consume_exc is not None:
                    raise _consume_exc
                return
            except UserRejectedCommandError:
                await self._arepair_rejected_tool_calls(config)
                await self._aclear_pending_interrupt(thread_id)
                yield AgentEvent.cancelled(
                    "用户已拒绝执行危险命令，当前任务已取消。",
                    thread_id=thread_id,
                    trace_id=trace_id,
                )
                return
            except Exception as error:
                # 并行工具调用场景：第一个工具崩后 Pregel 立即 raise，
                # 其余已 on_tool_start 但未 on_tool_end/on_tool_error 的 tool_call
                # 残留在 active_tool_call_ids。补发失败 tool_result，
                # 避免前端工具卡片永远卡在"执行中"。
                orphan_tc_ids = list(active_tool_call_ids.values())
                active_tool_call_ids.clear()
                for orphan_tc_id in orphan_tc_ids:
                    yield AgentEvent.tool_result(
                        tool_call_id=orphan_tc_id,
                        name="",
                        content=f"[工具执行失败] 并行任务因其他工具异常被中断: {error}",
                        thread_id=thread_id,
                        trace_id=trace_id,
                    )
                if emitted or not should_retry(error) or attempt == RETRY_ATTEMPTS - 1:
                    yield AgentEvent.error(
                        extract_llm_error(error),
                        thread_id=thread_id,
                        trace_id=trace_id,
                    )
                    return
                await asyncio.sleep(min(RETRY_MAX_DELAY, 2**attempt))

    async def arun_events(
        self,
        message: str,
        *,
        thread_id: str | None = None,
        is_run_mode: bool = False,
    ) -> AsyncIterator[AgentEvent]:
        """纯执行接口：运行 graph，yield AgentEvent 事件流。

        这是三层架构中 Agent 层的核心接口。SessionManager 调用此方法获取
        事件流，负责并发锁、checkpoint、记忆沉淀等全部会话级职责。

        Agent 只做执行：
        - 组装 prompt（系统提示 + 会话上下文 + 召回记忆）
        - 执行 LangGraph 流程、LLM 推理、工具调用
        - 监听 graph 内部 astream_events
        - 统一输出标准化 AgentEvent 事件流

        事件流顺序：
        TOKEN* → (TOOL_CALL → TOOL_RESULT)* → INTERRUPT | CANCELLED | ERROR | DONE

        工具异常处理：
        - 工具内部抛异常时，LangGraph 发射 on_tool_error 事件（非 on_tool_end），
          本方法将其映射为 TOOL_RESULT 事件（content 以 "[工具执行失败]" 前缀标记），
          让前端工具卡片显示失败 + LLM 能看到错误并调整策略。
        - 并行工具调用场景下，第一个工具崩后 Pregel 立即 raise，其余已 on_tool_start
          但未 on_tool_end/on_tool_error 的 tool_call（孤儿）在 except 块中补发
          失败 TOOL_RESULT，避免前端工具卡片永远卡在"执行中"。
        - 异常最终逃逸到 except Exception，发 ERROR 事件终止流。

        Args:
            message: 用户消息
            thread_id: 目标会话线程 ID（为 None 时使用当前会话）
            is_run_mode: True=执行模式（DONE 事件标记 is_important），
                         False=对话模式
        """
        self._ensure_not_closed()
        tid = self._current_sid(thread_id)
        trace_id = generate_trace_id()

        # warm workspace 缓存：断点续跑/进程重启后缓存可能为空，
        # 从 DB 加载 workspace_path 到缓存，使 get_context 同步读可命中
        reg = getattr(self, "_session_registry", None)
        if reg is not None:
            await reg.awarm_workspace(tid)

        with TraceContext(trace_id=trace_id, thread_id=tid):
            logger.info("arun_events: %s", message[:100])
            config = self._invoke_config(thread_id)
            input_msg = HumanMessage(content=message)
            collected_output: list[str] = []

            with self._temp_verbose(False):
                async for ev in self._arun_graph_events(
                    {"messages": [input_msg]},
                    config,
                    tid,
                    trace_id,
                    collected_output,
                ):
                    if ev.is_terminal:
                        # ERROR / CANCELLED → 记录 interrupt 状态后返回
                        if ev.event_type.value == "cancelled":
                            await self._acapture_pending_interrupt(config, "chat")
                        yield ev
                        return
                    yield ev

            # 流正常结束：检查是否被中断
            interrupt_ev = await self._acheck_interrupt_event(config, tid, trace_id)
            if interrupt_ev is not None:
                await self._acapture_pending_interrupt(config, "chat" if not is_run_mode else "run")
                yield interrupt_ev
                return

            # 正常完成：清理中断状态，yield DONE
            await self._aclear_pending_interrupt(tid)
            final_output = "".join(collected_output)
            self.metrics.increment_turn()
            yield AgentEvent.done(
                content=final_output,
                thread_id=tid,
                role="assistant",
                is_important=is_run_mode,
                trace_id=trace_id,
            )

    async def aresume_events(
        self,
        payload: dict[str, Any],
        *,
        thread_id: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """纯执行接口：恢复中断会话，yield AgentEvent 事件流。

        Args:
            payload: 恢复数据
            thread_id: 目标会话线程 ID（为 None 时使用当前会话）
        """
        self._ensure_not_closed()
        tid = self._current_sid(thread_id)
        trace_id = generate_trace_id()

        # warm workspace 缓存：恢复中断会话时确保 workspace_path 已加载
        reg = getattr(self, "_session_registry", None)
        if reg is not None:
            await reg.awarm_workspace(tid)

        with TraceContext(trace_id=trace_id, thread_id=tid):
            logger.info("aresume_events: thread=%s", tid)
            config = self._invoke_config(thread_id)

            # 从 SessionStore 读取 per-session 中断模式
            mode = await self._get_store().aget_interrupt_mode(tid)
            if mode is None:
                mode = "chat"

            collected_output: list[str] = []

            resume_command = await self._abuild_resume_command(config, payload)

            with self._temp_verbose(False):
                async for ev in self._arun_graph_events(
                    resume_command,
                    config,
                    tid,
                    trace_id,
                    collected_output,
                ):
                    if ev.is_terminal:
                        if ev.event_type.value == "cancelled":
                            await self._acapture_pending_interrupt(config, mode)
                        yield ev
                        return
                    yield ev

            # 流正常结束：检查是否再次被中断
            interrupt_ev = await self._acheck_interrupt_event(config, tid, trace_id)
            if interrupt_ev is not None:
                await self._acapture_pending_interrupt(config, mode)
                yield interrupt_ev
                return

            # 正常完成
            await self._aclear_pending_interrupt(tid)
            final_output = "".join(collected_output)
            self.metrics.increment_turn()
            yield AgentEvent.done(
                content=final_output,
                thread_id=tid,
                role="assistant",
                is_important=(mode == "run"),
                trace_id=trace_id,
            )