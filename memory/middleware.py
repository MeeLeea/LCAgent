"""长期记忆中间件 — 事件驱动写入 + prompt 注入读取。

分为两个组件：

**ThreadMemoryWriteMiddleware**（写服务，非 AgentMiddleware）：
- 接收 Agent 执行完成事件，非阻塞投递到内存 buffer
- 防抖缓冲：同一 thread 的事件合并，20s 窗口后批量处理
- Fact 处理流水线：消息过滤 → LLM 抽取 → 去重 → 写入 Store
- 使用 ThreadMemoryLockPool 保护同一 thread 的写入

**ThreadMemoryReadMiddleware**（读，AgentMiddleware）：
- 在 ``awrap_model_call`` 中从 Store 读取 thread 的 facts
- 将 facts 组装为文本片段，追加到 SystemMessage
- 非阻塞更新 ``last_used_at``（用于 LRU 淘汰）

设计参照 ``docs/# 长期记忆模块改造 TODO‑List.md``。
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ContextT, ModelRequest
from langchain_core.messages import SystemMessage

from .lock_pool import ThreadMemoryLockPool
from .models import (
    MemoryCategory,
    MemoryInputEvent,
    ThreadFactItem,
    _naive_now,
    judge_long_term_memory,
)
from .store import ThreadMemoryStore

logger = logging.getLogger(__name__)

# ============ 运行时配置常量 ============

MEMORY_BUFFER_DELAY_SECONDS = 20
"""防抖缓冲窗口（秒）：同一 thread 的新事件重置计时"""

MAX_FACT_PER_THREAD = 50
"""单 thread 最大 fact 条数"""

MAX_BUFFER_MESSAGE_COUNT = 30
"""单 thread 缓冲区上限（防溢出）"""


class ThreadMemoryWriteMiddleware:
    """长期记忆写入服务：事件接收 + 防抖 + Fact 抽取流水线。

    非 AgentMiddleware，由 AgentCore 直接调用 ``submit_event``。
    不阻塞 LangGraph 主执行链路。

    Args:
        memory_store: ThreadMemoryStore 实例（Store 业务封装）
        lock_pool: ThreadMemoryLockPool 实例（per-thread 并发锁）
        llm_getter: 返回当前 LLMClient 的 callable（支持 LLM 热切换）
        buffer_delay_seconds: 防抖缓冲窗口（秒），默认 MEMORY_BUFFER_DELAY_SECONDS
        max_buffer_messages: 单 thread 缓冲区上限，默认 MAX_BUFFER_MESSAGE_COUNT
    """

    def __init__(
        self,
        memory_store: ThreadMemoryStore,
        lock_pool: ThreadMemoryLockPool,
        llm_getter: Callable[[], Any],
        buffer_delay_seconds: int | None = None,
        max_buffer_messages: int | None = None,
    ) -> None:
        self._store = memory_store
        self._lock_pool = lock_pool
        self._llm_getter = llm_getter
        self._buffer_delay_seconds = buffer_delay_seconds if buffer_delay_seconds is not None else MEMORY_BUFFER_DELAY_SECONDS
        self._max_buffer_messages = max_buffer_messages if max_buffer_messages is not None else MAX_BUFFER_MESSAGE_COUNT

        # 防抖 buffer: thread_id → [(role, content, important), ...]
        self._buffer: dict[str, list[tuple[str, str, bool]]] = {}
        # 定时器: thread_id → asyncio.Task
        self._timers: dict[str, asyncio.Task] = {}
        # 保护 buffer 和 timers 的并发访问
        self._buffer_lock = asyncio.Lock()

    def bind_llm(self, llm_getter: Callable[[], Any]) -> None:
        """运行时替换 LLM 获取器（支持 provider 热切换后即时生效）。

        入口创建 Agent 后调用，将记忆抽取的 LLM 来源动态绑定到
        ``agent.llm``，保证主对话与记忆抽取始终使用同一当前 LLM。

        Args:
            llm_getter: 返回当前 LLMClient 的 callable
        """
        self._llm_getter = llm_getter

    # ============ 事件接收 & 防抖 ============

    async def submit_event(
        self,
        thread_id: str,
        role: str,
        content: str,
        important: bool = False,
    ) -> None:
        """非阻塞投递事件到 buffer。

        新消息重置该 thread 的防抖计时器。不 await 业务处理，
        不影响 LangGraph 主执行链路。

        Args:
            thread_id: 会话线程 ID
            role: 消息角色 (user / assistant / system)
            content: 消息文本内容
            important: 是否用户显式标记为重要
        """
        if not thread_id or not content.strip():
            return

        async with self._buffer_lock:
            buf = self._buffer.setdefault(thread_id, [])
            buf.append((role, content, important))

            # 限制单 thread 缓存条数
            if len(buf) > self._max_buffer_messages:
                self._buffer[thread_id] = buf[-self._max_buffer_messages:]

            # 重置防抖计时器
            old_timer = self._timers.get(thread_id)
            if old_timer is not None and not old_timer.done():
                old_timer.cancel()

            self._timers[thread_id] = asyncio.create_task(
                self._a_delayed_flush(thread_id)
            )

    async def _a_delayed_flush(self, thread_id: str) -> None:
        """防抖延迟后执行 flush。

        被 cancel 时不执行（防抖语义：新消息到达时取消旧计时）。
        """
        try:
            await asyncio.sleep(self._buffer_delay_seconds)
        except asyncio.CancelledError:
            return
        await self._aflush_thread(thread_id)

    # ============ Fact 处理流水线 ============

    async def _aflush_thread(self, thread_id: str) -> None:
        """处理该 thread 的 buffer：执行完整 Fact 抽取流水线。

        流水线：消息过滤 → LLM 抽取 → 无效过滤 → 去重 → 写入 Store → 淘汰
        """
        async with self._buffer_lock:
            messages = self._buffer.pop(thread_id, None)
            self._timers.pop(thread_id, None)

        if not messages:
            return

        lock = await self._lock_pool.get(thread_id)
        async with lock:
            try:
                await self._a_run_pipeline(thread_id, messages)
            except Exception as error:
                logger.error(
                    "Fact 抽取流水线失败 [thread=%s]: %s",
                    thread_id,
                    error,
                    exc_info=True,
                )

    async def _a_run_pipeline(
        self,
        thread_id: str,
        messages: list[tuple[str, str, bool]],
    ) -> None:
        """执行完整 Fact 处理流水线。"""

        # ① 构造 MemoryInputEvent 并分类
        events: list[tuple[MemoryInputEvent, MemoryCategory]] = []
        for role, content, important in messages:
            event = MemoryInputEvent(
                event_type="message",
                content=content,
                is_user_explicit_remember=important,
                is_reusable=not important,  # 非显式标记的默认可复用
            )
            category = judge_long_term_memory(event)
            if category != MemoryCategory.SKIP:
                events.append((event, category))

        if not events:
            return

        # ② LLM fact 抽取（从原始消息中提取结构化事实）
        facts_raw = await self._a_extract_facts(thread_id, messages)
        if not facts_raw:
            return

        # ③ 无效内容过滤
        facts_raw = [f for f in facts_raw if f.get("content", "").strip()]
        if not facts_raw:
            return

        # ④ thread 内本地去重
        existing = await self._store.query_facts(thread_id)
        existing_contents = {f.content for f in existing}

        # ⑤ 组装 ThreadFactItem 并批量写入
        new_items: list[ThreadFactItem] = []
        for fact_data in facts_raw:
            content = fact_data["content"]
            if content in existing_contents:
                continue
            existing_contents.add(content)  # 防止本批次内部重复

            category = fact_data.get("category", "conv")
            # 验证 category 合法性
            valid_categories = {c.value for c in MemoryCategory if c != MemoryCategory.SKIP}
            if category not in valid_categories:
                category = MemoryCategory.IMPORTANT_CONVERSATION.value

            item = ThreadFactItem(
                fact_id=uuid.uuid4().hex,
                thread_id=thread_id,
                content=content,
                category=category,
                confidence=fact_data.get("confidence", 0.8),
            )
            new_items.append(item)

        if new_items:
            await self._store.save_facts_batch(thread_id, new_items)
            logger.info(
                "thread %s: 写入 %d 条长期记忆",
                thread_id,
                len(new_items),
            )

        # ⑥ LRU 淘汰
        await self._store.prune_facts(thread_id)

    async def _a_extract_facts(
        self,
        thread_id: str,
        messages: list[tuple[str, str, bool]],
    ) -> list[dict[str, Any]]:
        """LLM fact 抽取：从对话消息中提取结构化事实。

        输出 JSON 列表，每项包含 content、category、confidence。

        Returns:
            fact 字典列表，可能为空
        """
        # 构建对话文本
        lines = []
        for role, content, _ in messages:
            lines.append(f"[{role}] {content}")
        conversation_text = "\n".join(lines)

        system_prompt = (
            "你是一个记忆抽取助手。请从以下对话中提取值得长期记住的事实。\n\n"
            "提取范围：\n"
            "1. 用户事实偏好（用户告知的个人信息、习惯、偏好）\n"
            "2. 经验教训（工具使用踩坑、方案不可行、稳定结论）\n"
            "3. 业务实体（项目配置、关键路径、接口、长期目标）\n"
            "4. 重要对话（用户显式标记'记住'、技术选型决策）\n\n"
            "过滤掉：\n"
            "- 一次性问答（如'今天天气'）\n"
            "- 工具原始输出（大段代码/文件内容）\n"
            "- 临时路径、临时变量\n"
            "- 未确认的猜想\n\n"
            "输出格式：JSON 数组，每项包含：\n"
            '{"content": "事实内容", "category": "分类", "confidence": 0.8}\n\n'
            "category 取值：user_fact / lesson / business / conv\n"
            "如果没有值得提取的事实，返回空数组 []\n"
            "只输出 JSON，不要其他文字。"
        )

        try:
            llm = self._llm_getter()
            response = await asyncio.to_thread(
                llm.chat,
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": conversation_text},
                ],
            )
            return self._parse_facts_response(response)
        except Exception as error:
            logger.warning("LLM fact 抽取失败 [thread=%s]: %s", thread_id, error)
            return []

    @staticmethod
    def _parse_facts_response(response: str) -> list[dict[str, Any]]:
        """解析 LLM 返回的 JSON fact 列表。

        容错处理：尝试从响应中提取 JSON 数组。
        """
        if not response or not response.strip():
            return []

        text = response.strip()

        # 尝试直接解析
        try:
            result = json.loads(text)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

        # 尝试从 markdown 代码块中提取
        if "```" in text:
            for block in text.split("```"):
                block = block.strip()
                if block.startswith("json"):
                    block = block[4:].strip()
                if block.startswith("["):
                    try:
                        result = json.loads(block)
                        if isinstance(result, list):
                            return result
                    except json.JSONDecodeError:
                        continue

        # 尝试找到第一个 [ 和最后一个 ]
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                result = json.loads(text[start : end + 1])
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass

        logger.debug("无法解析 fact 响应: %s", text[:200])
        return []

    # ============ 生命周期管理 ============

    async def cleanup_thread(self, thread_id: str) -> None:
        """thread 销毁时清理 buffer 和定时器。"""
        async with self._buffer_lock:
            self._buffer.pop(thread_id, None)
            timer = self._timers.pop(thread_id, None)

        if timer is not None and not timer.done():
            timer.cancel()

    async def flush_all(self) -> None:
        """立即处理所有 buffer 中的待处理事件（用于 Agent 关闭前）。"""
        async with self._buffer_lock:
            thread_ids = list(self._buffer.keys())

        for tid in thread_ids:
            await self._aflush_thread(tid)

    async def shutdown(self) -> None:
        """关闭中间件：取消所有定时器，清理 buffer。"""
        async with self._buffer_lock:
            timers = list(self._timers.values())
            self._buffer.clear()
            self._timers.clear()

        for timer in timers:
            if not timer.done():
                timer.cancel()


class ThreadMemoryReadMiddleware(AgentMiddleware):
    """长期记忆读取中间件 — 在 model 调用前注入 thread facts 到 SystemMessage。

    在 ``awrap_model_call`` 中：
    1. 从 runtime context 提取 thread_id
    2. 调用 ``ThreadMemoryStore.query_facts(thread_id)`` 读取 facts
    3. 将 facts 组装为文本片段，追加到 SystemMessage
    4. 非阻塞更新 ``last_used_at``（用于 LRU 淘汰）

    长期记忆只注入 prompt，不修改原始 messages 列表。
    原始对话保留在 checkpointer。

    Args:
        memory_store: ThreadMemoryStore 实例
        recall_limit: 注入时最多使用的 fact 条数（取最近 N 条）；
                      为 None 时注入全部 facts（对应配置键 memory_recall_limit）
    """

    FACTS_HEADER = "【长期记忆】\n"

    def __init__(
        self, memory_store: ThreadMemoryStore, recall_limit: int | None = None
    ) -> None:
        self._store = memory_store
        self._recall_limit = recall_limit

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[Any]],
    ) -> Any:
        """注入长期记忆到 system message，然后调用 handler。"""
        thread_id = self._extract_thread_id(request)
        if not thread_id:
            return await handler(request)

        # 读取 facts
        try:
            facts = await self._store.query_facts(thread_id)
        except Exception as error:
            logger.debug("读取长期记忆失败 [thread=%s]: %s", thread_id, error)
            return await handler(request)

        if not facts:
            return await handler(request)

        # 按 recall_limit 截取最近 N 条（对应配置键 memory_recall_limit）
        if self._recall_limit and len(facts) > self._recall_limit:
            facts = facts[-self._recall_limit:]

        # 组装 facts 文本
        fact_text = self._format_facts(facts)

        # 注入到 SystemMessage
        new_request = self._inject_facts(request, fact_text)

        # 非阻塞更新 last_used_at
        for fact in facts:
            asyncio.create_task(self._store.touch_fact(thread_id, fact.fact_id))

        return await handler(new_request)

    # ============ 内部方法 ============

    @staticmethod
    def _extract_thread_id(request: ModelRequest[ContextT]) -> str | None:
        """从 ModelRequest 的 runtime context 中提取 thread_id。

        LangGraph 在 ainvoke 时将 config 存入 runtime.context，
        config 结构为 ``{"configurable": {"thread_id": "..."}}``。
        """
        try:
            context = request.runtime.context
            if context is None:
                return None
            if isinstance(context, dict):
                configurable = context.get("configurable")
                if isinstance(configurable, dict):
                    tid = configurable.get("thread_id")
                    if isinstance(tid, str):
                        return tid
            # 某些 LangGraph 版本中 context 是对象
            configurable = getattr(context, "configurable", None)
            if isinstance(configurable, dict):
                tid = configurable.get("thread_id")
                if isinstance(tid, str):
                    return tid
        except Exception:
            pass
        return None

    def _format_facts(self, facts: list[ThreadFactItem]) -> str:
        """将 facts 列表格式化为文本片段。

        按 category 分组，每条 fact 一行。
        """
        if not facts:
            return ""

        lines: list[str] = []
        for fact in facts:
            category_label = {
                MemoryCategory.USER_FACT.value: "用户事实",
                MemoryCategory.LESSON_EXPERIENCE.value: "经验教训",
                MemoryCategory.BUSINESS_ENTITY.value: "业务信息",
                MemoryCategory.IMPORTANT_CONVERSATION.value: "重要决策",
            }.get(fact.category, "记忆")

            lines.append(f"- [{category_label}] {fact.content}")

        return self.FACTS_HEADER + "\n".join(lines) + "\n"

    @staticmethod
    def _inject_facts(
        request: ModelRequest[ContextT], fact_text: str
    ) -> ModelRequest[ContextT]:
        """将 facts 文本追加到 SystemMessage，返回新的 ModelRequest。"""
        if request.system_message is not None:
            new_content = [
                *request.system_message.content_blocks,
                {"type": "text", "text": f"\n{fact_text}"},
            ]
        else:
            new_content = [{"type": "text", "text": fact_text}]

        new_sys_msg = SystemMessage(
            content=[
                c if isinstance(c, dict) else {"type": "text", "text": str(c)}
                for c in new_content
            ]
        )
        return request.override(system_message=new_sys_msg)


__all__ = [
    "MAX_BUFFER_MESSAGE_COUNT",
    "MAX_FACT_PER_THREAD",
    "MEMORY_BUFFER_DELAY_SECONDS",
    "ThreadMemoryReadMiddleware",
    "ThreadMemoryWriteMiddleware",
]
