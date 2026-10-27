"""长期记忆中间件 — 事件驱动写入 + prompt 注入读取。

分为两个组件：

**ThreadMemoryWriteMiddleware**（写服务，非 AgentMiddleware）：
- 接收 Agent 执行完成事件，非阻塞投递到内存 buffer
- 防抖缓冲：同一 thread 的事件合并，20s 窗口后批量处理
- Fact 处理流水线：消息过滤 → LLM 抽取 → 去重 → 写入 Store
- 使用 ThreadMemoryLockPool 保护同一 thread 的写入
- agent namespace（跨进程共享）的批量写入与 LRU 淘汰使用 AgentMemoryLock
  保护（基于 OS 级文件锁，跨进程互斥）

**ThreadMemoryReadMiddleware**（读，AgentMiddleware）：
- 在 ``awrap_model_call`` 中从 Store 读取 thread 的 facts
- 将 facts 组装为文本片段，追加为 ``request.messages`` 末尾的 user 消息
- 非阻塞更新 ``last_used_at``（用于 LRU 淘汰）

设计参照 ``docs/# 长期记忆模块改造 TODO‑List.md``。
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ContextT, ModelRequest
from langchain_core.messages import HumanMessage

from .config import (
    MEMORY_BUFFER_DELAY_SECONDS,
    MEMORY_CONFIDENCE_FLOOR,
    MEMORY_LOW_CONFIDENCE_THRESHOLD,
    MEMORY_MAX_BUFFER_MESSAGES,
)
from .lock_pool import AgentMemoryLock, ThreadMemoryLockPool
from .models import (
    EVIDENCE_INFERRED,
    PERSISTENCE_GLOBAL,
    PERSISTENCE_SESSION,
    VALID_EVIDENCES,
    VALID_PERSISTENCES,
    MemoryCategory,
    MemoryInputEvent,
    ThreadFactItem,
    judge_long_term_memory,
    normalize_topic_key,
)
from .store import ThreadMemoryStore

logger = logging.getLogger(__name__)

# 记忆链路 LLM 解析器：thread_id 可为 None（agent 级操作）；返回 LLMClient
ThreadLLMResolver = Callable[[str | None], Awaitable[Any]]


def default_thread_llm_resolver(getter: Callable[[], Any]) -> ThreadLLMResolver:
    """把 zero-arg getter 适配为 thread-aware async resolver（占位/测试用）。"""
    async def _resolve(thread_id: str | None) -> Any:
        return getter()
    return _resolve


# 控制流 / HITL 确认内容标记：TOOL_RESULT 含这些标记说明是 interrupt
# （危险命令确认 / ask_human）在事件流层的映射残留，不是真实工具失败，
# 不得计入失败计数、不得沉淀为 lesson（历史 bug：安全确认被逐字记成
# 跨会话"经验教训"）。事件流层已排除（agent/streaming.py），此处为
# 记忆层独立防线，防未来新路径再次把控制流信号转成 TOOL_RESULT。
_CONTROL_FLOW_MARKERS: tuple[str, ...] = (
    "GraphInterrupt",
    "GraphBubbleUp",
    "dangerous_command",
    "human_choice",
)

# 错误格式化层（agent/streaming.py、agent/tool_error_mw.py）附加的
# 反思指令 / workspace 提示后缀：它们是驱动 LLM 下一轮自愈的指令文本，
# 不是事实内容，lesson 蒸馏前必须剥离。
_ERROR_BOILERPLATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"。?请反思失败原因（[^）]*），修正后重试"),
    re.compile(r"。?请反思失败原因后修正重试"),
    re.compile(r"。?工作空间根目录为 [^。]+"),
    re.compile(r"。?文件类工具请基于工作空间根目录[^。]+"),
)

# lesson 蒸馏前对超长错误原文的截断上限（控制 LLM 输入成本）
_LESSON_SOURCE_MAX_CHARS = 3000
# 蒸馏产物长度上限（LLM 不守约时兜底截断）
_LESSON_MAX_CHARS = 300


def _is_tool_failure(content: str) -> bool:
    """检测 TOOL_RESULT 内容是否表示工具执行失败/超时。

    与 ``midware.terminal_retry_cap_mw.is_timeout_content`` 逻辑一致，
    额外补充工具执行失败标记。memory 层独立实现以避免跨层引用。
    含控制流 / HITL 标记的内容不算失败（见 ``_CONTROL_FLOW_MARKERS``）。
    """
    if not content:
        return False
    if any(marker in content for marker in _CONTROL_FLOW_MARKERS):
        return False
    return (
        '"error_type": "timeout"' in content
        or '"error": "tool_timeout"' in content
        or "[工具执行失败]" in content
        or "执行超时" in content
        or "命令超时" in content
    )


def _strip_tool_error_boilerplate(content: str) -> str:
    """剥离错误文本中的反思指令 / workspace 提示后缀并截断超长内容。

    供 lesson 蒸馏前预处理：错误原文可能内嵌完整命令脚本（数 KB），
    逐字入库会污染跨会话共享的 agent namespace。

    Args:
        content: 原始 TOOL_RESULT 错误文本

    Returns:
        剥离指令性后缀、截断超长后的纯净错误记录
    """
    cleaned = content
    for pattern in _ERROR_BOILERPLATE_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    cleaned = cleaned.strip().rstrip("。")
    if len(cleaned) > _LESSON_SOURCE_MAX_CHARS:
        cleaned = cleaned[:_LESSON_SOURCE_MAX_CHARS] + "…(已截断)"
    return cleaned


class ThreadMemoryWriteMiddleware:
    """长期记忆写入服务：事件接收 + 防抖 + Fact 抽取流水线。

    非 AgentMiddleware，由 AgentCore 直接调用 ``submit_event``。
    不阻塞 LangGraph 主执行链路。

    Args:
        memory_store: ThreadMemoryStore 实例（Store 业务封装）
        lock_pool: ThreadMemoryLockPool 实例（per-thread 并发锁）
        llm_getter: 返回当前 LLMClient 的 thread-aware async resolver（支持
            按会话解析 provider/model，失败回落默认 LLM）
        buffer_delay_seconds: 防抖缓冲窗口（秒），默认 MEMORY_BUFFER_DELAY_SECONDS
        max_buffer_messages: 单 thread 缓冲区上限，默认 MEMORY_MAX_BUFFER_MESSAGES
        agent_lock: agent 级跨进程互斥锁（保护 agent namespace 的批量写入与
            LRU 淘汰）。为 None 时内部自建进程内锁 ``AgentMemoryLock(path=None)``，
            供测试等无外部注入场景使用
        confidence_floor: agent 级（跨会话共享）写入的置信度门槛。路由到
            agent 级但 confidence 低于该值的 fact 降级写入 thread 级，
            防止低置信内容污染全局。默认取 MEMORY_CONFIDENCE_FLOOR
    """

    def __init__(
        self,
        memory_store: ThreadMemoryStore,
        lock_pool: ThreadMemoryLockPool,
        llm_getter: ThreadLLMResolver,
        buffer_delay_seconds: int | None = None,
        max_buffer_messages: int | None = None,
        agent_lock: AgentMemoryLock | None = None,
        confidence_floor: float | None = None,
    ) -> None:
        self._store = memory_store
        self._lock_pool = lock_pool
        self._llm_getter = llm_getter
        # agent 级互斥锁：agent namespace 跨进程共享，写流水线必须持锁
        self._agent_lock = agent_lock if agent_lock is not None else AgentMemoryLock()
        self._buffer_delay_seconds = buffer_delay_seconds if buffer_delay_seconds is not None else MEMORY_BUFFER_DELAY_SECONDS
        self._max_buffer_messages = max_buffer_messages if max_buffer_messages is not None else MEMORY_MAX_BUFFER_MESSAGES
        self._confidence_floor = confidence_floor if confidence_floor is not None else MEMORY_CONFIDENCE_FLOOR

        # 防抖 buffer: thread_id → [(role, content, important), ...]
        self._buffer: dict[str, list[tuple[str, str, bool]]] = {}
        # 定时器: thread_id → asyncio.Task[None]（_a_delayed_flush 无返回值）
        self._timers: dict[str, asyncio.Task[None]] = {}
        # 保护 buffer 和 timers 的并发访问
        self._buffer_lock = asyncio.Lock()
        # 失败计数：(thread_id, tool_name) → 同类工具失败累计次数，
        # 供 judge_long_term_memory 确定性判定经验教训（失败 ≥2 次）
        self._failure_counts: dict[tuple[str, str], int] = {}

    def bind_llm(self, llm_getter: ThreadLLMResolver) -> None:
        """运行时替换 LLM 解析器（支持 provider 热切换后即时生效）。

        入口创建 Agent 后调用，把记忆抽取的 LLM 来源替换为 thread-aware
        resolver：按 thread_id 解析该会话的 provider/model，无会话配置或
        解析失败时回落默认 LLM（``agent.llm``），避免跨 provider 误发请求。

        Args:
            llm_getter: 返回当前 LLMClient 的 thread-aware async resolver
        """
        self._llm_getter = llm_getter

    # ============ 事件接收 & 防抖 ============

    async def submit_event(
        self,
        thread_id: str,
        role: str,
        content: str,
        important: bool = False,
        event_type: str = "message",
        tool_name: str = "",
    ) -> None:
        """非阻塞投递事件到 buffer。

        新消息重置该 thread 的防抖计时器。不 await 业务处理，
        不影响 LangGraph 主执行链路。

        Args:
            thread_id: 会话线程 ID
            role: 消息角色 (user / assistant / system)
            content: 消息文本内容
            important: 是否用户显式标记为重要
            event_type: 事件类型（``"message"`` 或 ``"tool_result"``），
                供 :func:`judge_long_term_memory` 确定性判定
            tool_name: 工具名（仅 ``event_type="tool_result"`` 时有效），
                用于同类失败计数
        """
        if not thread_id or not content.strip():
            return

        async with self._buffer_lock:
            buf = self._buffer.setdefault(thread_id, [])
            buf.append((role, content, important, event_type, tool_name))

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
            except Exception:
                logger.exception("Fact 抽取流水线失败 [thread=%s]", thread_id)

    # 路由到 agent 级作用域的 category 集合（user_fact / lesson 跨会话共享）
    _AGENT_CATEGORIES: frozenset[str] = frozenset({
        MemoryCategory.USER_FACT.value,
        MemoryCategory.LESSON_EXPERIENCE.value,
    })

    async def _a_run_pipeline(
        self,
        thread_id: str,
        messages: list[tuple[str, str, bool, str, str]],
    ) -> None:
        """执行完整 Fact 处理流水线。

        ① 确定性预判定（judge_long_term_memory）：
           SKIP → 确定性丢弃；LESSON（失败≥2）→ 分类锁定 lesson，
           内容经 LLM 蒸馏（含 persistence / topic_key 判定）后入库；
           IMPORTANT_CONVERSATION / None → 进入 LLM 抽取
        ② 读取两级 existing（全量，含 superseded）：
           既作 content 精确去重基准，也提取既有 topic_key 清单供 LLM 复用
        ③ LLM fact 抽取（输出含 persistence / topic_key / evidence）
        ④ 无效过滤 + content 精确去重（两级，agent 优先）
        ⑤ persistence 优先路由作用域（session→thread / global→agent，
           低置信 global 降级 thread）+ 批内同 topic_key 合并
        ⑥ 写入：同 topic_key 的既有 active 旧条目标记 superseded（取代而非
           追加），新条目写入；agent 级在 agent 锁内执行
        ⑦ 分流 LRU 容量淘汰（非 active 优先）
        """

        # ① 确定性预判定 + 失败计数
        lesson_sources: list[str] = []
        extract_messages: list[tuple[str, str, bool]] = []

        for role, content, important, evt_type, tool_nm in messages:
            failure_count = 0
            if evt_type == "tool_result" and _is_tool_failure(content):
                key = (thread_id, tool_nm)
                self._failure_counts[key] = self._failure_counts.get(key, 0) + 1
                failure_count = self._failure_counts[key]

            event = MemoryInputEvent(
                event_type=evt_type,
                content=content,
                is_user_explicit_remember=important,
                failure_repeat_count=failure_count,
            )
            result = judge_long_term_memory(event)

            if result is MemoryCategory.SKIP:
                continue  # 确定性丢弃（如单次失败、非显式 tool_result）
            if result is MemoryCategory.LESSON_EXPERIENCE:
                # 确定性 lesson：同类失败 ≥2 次，分类绕过 LLM 判定，
                # 但内容必须先剥离指令后缀，再交 ①-b 蒸馏成简短教训。
                lesson_sources.append(_strip_tool_error_boilerplate(content))
                continue

            # IMPORTANT_CONVERSATION 或 None → 进入 LLM 抽取
            # （important 字段供 _a_extract_facts 加标注提高优先级）
            extract_messages.append((role, content, important))

        # ② 读取两级 existing 作为去重基准 + 提取既有 topic_key 清单。
        #    全量读取（含 superseded）：content 完全相同即视为"已记过"跳过，
        #    与既有语义一致；topic_key 清单供抽取/蒸馏 prompt 复用（防漂移）。
        existing_thread = await self._store.query_facts(thread_id)
        existing_agent = await self._store.query_agent_facts()
        existing_topic_keys = sorted({
            f.topic_key
            for f in [*existing_agent, *existing_thread]
            if f.topic_key
        })

        # ①-b lesson 蒸馏：LLM 把失败记录提炼为可复用教训并判定适用范围
        #     （session=会话局部环境状态 / global=跨会话普适）；
        #     LLM 失败 / 判定无教训时丢弃该条（宁缺勿污染共享空间），
        #     绝不回退逐字存原文。
        distilled_items: list[ThreadFactItem] = []
        for source in lesson_sources:
            lesson, persistence, topic_key = await self._a_distill_lesson(
                thread_id, source, existing_topic_keys
            )
            if not lesson:
                continue
            distilled_items.append(
                self._build_fact_item(
                    lesson,
                    MemoryCategory.LESSON_EXPERIENCE.value,
                    {
                        "confidence": 0.9,
                        "persistence": persistence,
                        "topic_key": topic_key,
                        "evidence": EVIDENCE_INFERRED,
                    },
                    self._resolve_scope(
                        MemoryCategory.LESSON_EXPERIENCE.value,
                        persistence,
                        0.9,
                    ),
                )
            )

        # ③ LLM fact 抽取：仅对需抽取的消息（失败≥2 的已确定性处理）
        facts_raw: list[dict[str, Any]] = []
        if extract_messages:
            facts_raw = await self._a_extract_facts(
                thread_id, extract_messages, existing_topic_keys
            )

        # ④ 无效内容过滤
        facts_raw = [f for f in facts_raw if f.get("content", "").strip()]

        # ⑤ content 精确去重（两级 existing 全量基准，agent 优先）
        agent_seen: set[str] = {f.content for f in existing_agent}
        thread_seen: set[str] = {f.content for f in existing_thread}

        agent_items: list[ThreadFactItem] = []
        thread_items: list[ThreadFactItem] = []

        for item in [*distilled_items, *self._facts_raw_to_items(facts_raw)]:
            if item.content in agent_seen or item.content in thread_seen:
                continue
            (agent_items if item.scope == "agent" else thread_items).append(item)
            # 新 content 登记进对应 seen，防批内重复
            (agent_seen if item.scope == "agent" else thread_seen).add(item.content)

        # ⑥ 写入：同 topic_key 取代（先写新条目，再标记旧条目 superseded——
        #    中断时宁可瞬态双版本由读取侧收敛，不可丢主题记忆）+ 容量淘汰。
        #    agent 级写入/supersede/prune 必须持有 agent 锁，且仅在本次确实
        #    产生了 agent facts 时才执行——空批次也 prune 会在多进程并发时
        #    各自按本地快照计算溢出并删除不相干集合，造成过量淘汰。
        if agent_items:
            merged_agent = self._merge_batch_by_topic(agent_items)
            supersede_ids = self._collect_supersede_targets(
                merged_agent, existing_agent
            )
            async with self._agent_lock:
                await self._store.save_agent_facts_batch(merged_agent)
                if supersede_ids:
                    await self._store.supersede_agent_facts(supersede_ids)
                    logger.info(
                        "[长期记忆-agent级] thread=%s 同主题取代 %d 条旧记忆",
                        thread_id,
                        len(supersede_ids),
                    )
                logger.info(
                    "[长期记忆-agent级] thread=%s 写入 %d 条 (跨会话共享, namespace=global_facts)",
                    thread_id,
                    len(merged_agent),
                )
                await self._store.prune_agent_facts()

        if thread_items:
            merged_thread = self._merge_batch_by_topic(thread_items)
            thread_supersede_ids = self._collect_supersede_targets(
                merged_thread, existing_thread
            )
            await self._store.save_facts_batch(thread_id, merged_thread)
            if thread_supersede_ids:
                await self._store.supersede_facts(thread_id, thread_supersede_ids)
                logger.info(
                    "[长期记忆-session级] thread=%s 同主题取代 %d 条旧记忆",
                    thread_id,
                    len(thread_supersede_ids),
                )
            logger.info(
                "[长期记忆-session级] thread=%s 写入 %d 条 (会话隔离, namespace=thread_facts)",
                thread_id,
                len(merged_thread),
            )
        await self._store.prune_facts(thread_id)

    def _resolve_scope(
        self, category: str, persistence: str, confidence: float
    ) -> str:
        """解析 fact 的写入作用域（namespace 路由）。

        路由优先级：
        1. ``persistence``（LLM 判定的适用范围）优先：``session`` → thread
           级；``global`` → agent 级（但 confidence 低于门槛时降级 thread 级）
        2. persistence 缺失/非法 → 回退旧 category 路由（user_fact/lesson →
           agent 级，同样受置信度门槛约束），保持向后兼容

        Args:
            category: 记忆分类（MemoryCategory 值）
            persistence: 持久层级（"session" / "global" / 空）
            confidence: 置信度

        Returns:
            ``"agent"`` 或 ``"thread"``
        """
        to_agent = (
            persistence == PERSISTENCE_GLOBAL
            if persistence in VALID_PERSISTENCES
            else category in self._AGENT_CATEGORIES
        )
        # 低置信内容不进跨会话共享空间：降级 thread 级（会话隔离）
        if to_agent and confidence < self._confidence_floor:
            return "thread"
        return "agent" if to_agent else "thread"

    def _facts_raw_to_items(
        self, facts_raw: list[dict[str, Any]]
    ) -> list[ThreadFactItem]:
        """把 LLM 抽取的原始 dict 列表组装为 ThreadFactItem（含作用域路由）。

        category 合法性校验 + 兜底（兜底为 conv）；persistence / topic_key /
        evidence 非法值安全回退（空 persistence 走旧 category 路由，
        无效 topic_key 视为无主题，无效 evidence 回退 stated）。
        """
        valid_categories = {c.value for c in MemoryCategory if c != MemoryCategory.SKIP}
        items: list[ThreadFactItem] = []
        for fact_data in facts_raw:
            content = fact_data["content"]
            category = fact_data.get("category", "conv")
            if category not in valid_categories:
                category = MemoryCategory.IMPORTANT_CONVERSATION.value
            confidence = fact_data.get("confidence", 0.8)
            persistence = fact_data.get("persistence", "")
            if not isinstance(persistence, str) or persistence not in VALID_PERSISTENCES:
                persistence = ""
            evidence = fact_data.get("evidence", "stated")
            if not isinstance(evidence, str) or evidence not in VALID_EVIDENCES:
                evidence = "stated"
            topic_key = normalize_topic_key(str(fact_data.get("topic_key", "")))
            items.append(
                self._build_fact_item(
                    content,
                    category,
                    {
                        "confidence": confidence,
                        "persistence": persistence,
                        "topic_key": topic_key,
                        "evidence": evidence,
                    },
                    self._resolve_scope(category, persistence, confidence),
                )
            )
        return items

    @staticmethod
    def _merge_batch_by_topic(
        items: list[ThreadFactItem],
    ) -> list[ThreadFactItem]:
        """批内同 topic_key 合并：同一批次出现同主题多条时只保留一条。

        保留置信度最高者；置信度相同保留后出现者（更新表述覆盖旧的）。
        无 topic_key 的条目不参与合并，原样保留。

        Args:
            items: 待写入的 fact 列表

        Returns:
            合并后的 fact 列表（无主题条目在前，主题代表条目按首次出现序）
        """
        result: list[ThreadFactItem] = []
        by_topic: dict[str, ThreadFactItem] = {}
        for item in items:
            if not item.topic_key:
                result.append(item)
                continue
            prev = by_topic.get(item.topic_key)
            if prev is None or item.confidence >= prev.confidence:
                by_topic[item.topic_key] = item
        result.extend(by_topic.values())
        return result

    @staticmethod
    def _collect_supersede_targets(
        new_items: list[ThreadFactItem],
        existing_active: list[ThreadFactItem],
    ) -> list[str]:
        """收集将被新条目取代的旧 fact_id（同 topic_key 的 active 条目）。

        Args:
            new_items: 本批次待写入条目（已批内合并）
            existing_active: 该 namespace 既有的 active 条目（含 superseded
                的全量列表传入亦可，非 active 会被 supersede 方法跳过）

        Returns:
            待标记 superseded 的 fact_id 列表
        """
        new_topics = {item.topic_key for item in new_items if item.topic_key}
        if not new_topics:
            return []
        return [
            fact.fact_id
            for fact in existing_active
            if fact.topic_key and fact.topic_key in new_topics
        ]

    @staticmethod
    def _build_fact_item(
        content: str,
        category: str,
        fact_data: dict[str, Any],
        scope: str,
    ) -> ThreadFactItem:
        """组装单条 ThreadFactItem（含 category 合法性校验后的字段）。

        两分支（agent / thread）复用，避免重复组装逻辑。

        Args:
            content: fact 文本内容（已 strip）
            category: 经合法性校验后的 category 值
            fact_data: LLM 返回的原始 fact 字典（取 confidence / topic_key /
                evidence 等附加字段；topic_key 由调用方先行归一化或留空）
            scope: ``"agent"`` 或 ``"thread"``

        Returns:
            新构建的 ThreadFactItem（thread_id 由 store 层按 scope 自动覆写）
        """
        return ThreadFactItem(
            fact_id=uuid.uuid4().hex,
            content=content,
            category=category,
            confidence=fact_data.get("confidence", 0.8),
            scope=scope,
            topic_key=fact_data.get("topic_key", ""),
            evidence=fact_data.get("evidence", "stated"),
        )

    async def _a_distill_lesson(
        self,
        thread_id: str,
        source: str,
        existing_topic_keys: list[str] | None = None,
    ) -> tuple[str, str, str]:
        """LLM 蒸馏：把工具重复失败记录提炼为简短教训并判定适用范围。

        确定性 lesson 路径（同类失败 ≥2 次）分类锁定 lesson、不依赖 LLM 判定，
        但内容必须蒸馏——错误原文可能内嵌完整命令脚本（数 KB），逐字入库会
        污染共享 namespace。同时由 LLM 判定 ``persistence``（教训的适用范围）：
        会话局部环境状态（服务未初始化等）默认 ``session``（thread 级隔离），
        仅跨会话普适的教训才 ``global``——修复"一次性事件被误升为普适教训"。

        Args:
            thread_id: 会话线程 ID（仅用于日志）
            source: 已剥离指令后缀的错误记录（可能为空）
            existing_topic_keys: 既有主题键清单（供 LLM 复用，防同主题键漂移）

        Returns:
            ``(lesson, persistence, topic_key)`` 三元组；LLM 失败、判定无教训
            或 source 为空时返回 ``("", "", "")``（该条 lesson 丢弃，不回退存原文）
        """
        if not source:
            return "", "", ""
        keys_hint = ", ".join(existing_topic_keys) if existing_topic_keys else "（无）"
        system_prompt = (
            "你是经验教训提炼助手。下面是一条工具重复失败的记录，"
            "请把它蒸馏成一条简短、可复用的经验教训，并判定其适用范围。\n\n"
            "输出 JSON 对象（只输出 JSON，不要其他文字）：\n"
            '{"lesson": "教训文本", "persistence": "session或global", "topic_key": "主题键"}\n\n'
            "要求：\n"
            "1. lesson 为中文一句话，不超过 120 字，直接给结论，不要前缀和解释；"
            "指出失败根因与规避方法，不得复述大段命令或文件内容\n"
            "2. persistence 判定（教训的适用范围）：\n"
            "   - session：失败根因是当前会话/环境的临时状态（如某服务未初始化、"
            "资源尚未就绪、本次任务特有配置）——换一个会话该教训不再成立\n"
            "   - global：根因具有跨会话普适性（如工具参数误用、路径规则、通用 API 行为）\n"
            "   - 拿不准时选 session（宁可作用域小，也不要污染全局）\n"
            "3. topic_key：英文 snake_case 主题键（≤50 字符，如 tool_timeout、"
            f"path_resolution）；已有主题键清单：[{keys_hint}]，同一主题必须复用清单中的键；"
            "无合适主题输出空字符串 \"\"\n"
            "4. 若记录只是用户拒绝执行、安全确认暂停、瞬时网络抖动等无可复用教训，"
            '返回 {"lesson": "", "persistence": "session", "topic_key": ""}'
        )
        try:
            llm = await self._llm_getter(thread_id)
            response = await asyncio.to_thread(
                llm.chat,
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"失败记录：\n{source}"},
                ],
            )
            return self._parse_distill_response(response or "")
        except Exception as error:
            logger.warning("lesson 蒸馏失败 [thread=%s]: %s", thread_id, error)
            return "", "", ""

    _DISTILL_EMPTY_SENTINELS: frozenset[str] = frozenset({
        "[]", "[ ]", '""', "''", "无", "（空）", "(空)", "{}",
    })

    @classmethod
    def _parse_distill_response(cls, response: str) -> tuple[str, str, str]:
        """解析蒸馏 LLM 的 JSON 响应为 ``(lesson, persistence, topic_key)``。

        容错层次：
        1. 直接 ``json.loads`` 得 dict → 取字段（persistence/topic_key 非法值回退）
        2. 失败则截取首个 ``{`` 到末个 ``{`` 之间再解析
        3. 仍失败且响应为非空裸文本（模型未守约输出 JSON）→ 整段视为教训文本，
           persistence 保守取 session、topic_key 留空（无主题，退化为追加）
        4. 空响应 / 空哨兵值（``[]``、``无`` 等）→ 三元空组（丢弃该条）
        """
        text = (response or "").strip()
        if not text or text in cls._DISTILL_EMPTY_SENTINELS:
            return "", "", ""

        parsed: dict[str, Any] | None = None
        try:
            result = json.loads(text)
            if isinstance(result, dict):
                parsed = result
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end > start:
                try:
                    result = json.loads(text[start : end + 1])
                    if isinstance(result, dict):
                        parsed = result
                except json.JSONDecodeError:
                    parsed = None

        if parsed is not None:
            lesson = str(parsed.get("lesson", "")).strip()
            if len(lesson) > _LESSON_MAX_CHARS:
                lesson = lesson[:_LESSON_MAX_CHARS]
            if not lesson:
                return "", "", ""
            persistence = parsed.get("persistence", "")
            if persistence not in VALID_PERSISTENCES:
                persistence = PERSISTENCE_SESSION
            topic_key = normalize_topic_key(str(parsed.get("topic_key", "")))
            return lesson, persistence, topic_key

        # 裸文本回退：视为教训文本，保守 session 级（不污染跨会话共享空间）
        lesson = text[:_LESSON_MAX_CHARS]
        return lesson, PERSISTENCE_SESSION, ""

    async def _a_extract_facts(
        self,
        thread_id: str,
        messages: list[tuple[str, str, bool]],
        existing_topic_keys: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """LLM fact 抽取：从对话消息中提取结构化事实。

        输出 JSON 列表，每项包含 content / category / confidence /
        persistence / topic_key / evidence。

        Args:
            thread_id: 会话线程 ID
            messages: 待抽取消息列表
            existing_topic_keys: 既有主题键清单（prompt 中要求 LLM 复用，
                防止同一主题在不同会话生成不同键导致合并失效）

        Returns:
            fact 字典列表，可能为空
        """
        # 构建对话文本（important 消息加标注，提高 LLM 抽取优先级）
        lines = []
        for role, content, important in messages:
            marker = " [用户明确要求记住]" if important else ""
            lines.append(f"[{role}]{marker} {content}")
        conversation_text = "\n".join(lines)

        keys_hint = ", ".join(existing_topic_keys) if existing_topic_keys else "（无）"
        system_prompt = (
            "你是一个记忆抽取助手。请从以下对话中提取值得长期记住的事实。\n\n"
            "提取范围：\n"
            "1. 用户事实偏好（用户告知的个人信息、习惯、偏好）\n"
            "2. 经验教训（工具使用踩坑、方案不可行、稳定结论）\n"
            "3. 业务实体（项目配置、关键路径、接口、长期目标）\n"
            "4. 重要对话（用户显式标记'记住'、技术选型决策）\n\n"
            "过滤掉：\n"
            "- 模型访问失败、网络异常、临时错误等非事实内容\n"
            "- 一次性问答（如'今天天气'）\n"
            "- 一次性格式要求/单次任务约束（如'本次输出用 markdown'、'这次用 json'）\n"
            "- 仅当前会话生效的临时偏好/指令\n"
            "- 工具原始输出（大段代码/文件内容）\n"
            "- 临时路径、临时变量\n"
            "- 未确认的猜想\n\n"
            "判定原则：\n"
            "- 用户陈述习惯/长期偏好（如'我习惯用 markdown'）→ user_fact\n"
            "- 用户给本次任务的一次性格式指定 → 过滤，不入库\n"
            "- 不确定时倾向于过滤（宁缺勿污染跨会话共享空间）\n\n"
            "输出格式：JSON 数组，每项包含：\n"
            '{"content": "事实内容", "category": "分类", "confidence": 0.8, '
            '"persistence": "session或global", "topic_key": "主题键", '
            '"evidence": "tested/stated/inferred"}\n\n'
            "字段说明：\n"
            "- persistence（持久层级，决定记忆共享范围）：\n"
            "  * session：仅本次会话/当前环境成立——环境未初始化、服务启动顺序、"
            "本次任务特有约束、临时状态\n"
            "  * global：跨会话普适——通用接口契约、用户长期偏好、与具体会话"
            "无关的稳定教训\n"
            "  * 拿不准时选 session（宁可作用域小，也不要污染全局）\n"
            "- topic_key（主题键，用于版本取代）：同一主题再次出现新说法时，"
            "输出相同 topic_key 可使新条目取代旧条目，避免新旧并存自相矛盾\n"
            "  * 英文 snake_case，≤50 字符，如 dim_y_formula、cli_usage\n"
            f"  * 已有主题键清单：[{keys_hint}]；新事实若与清单中某主题是同一"
            "主题，必须复用该键\n"
            "  * 全新主题给新键；无法归纳主题时输出空字符串 \"\"\n"
            "- evidence（证据来源）：tested=经实测验证 / stated=对话中明确陈述"
            "（默认）/ inferred=未经实测的推断\n\n"
            "category 取值：user_fact / lesson / business / conv\n"
            "如果没有值得提取的事实，返回空数组 []\n"
            "只输出 JSON，不要其他文字。"
        )

        try:
            llm = await self._llm_getter(thread_id)
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
        """thread 销毁时清理 buffer、定时器与失败计数。"""
        async with self._buffer_lock:
            self._buffer.pop(thread_id, None)
            timer = self._timers.pop(thread_id, None)

        if timer is not None and not timer.done():
            timer.cancel()

        # 清理该 thread 相关的失败计数，避免跨会话串号
        keys = [k for k in self._failure_counts if k[0] == thread_id]
        for k in keys:
            del self._failure_counts[k]

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
    """长期记忆读取中间件 — 在 model 调用前把 facts 注入为尾随 user 消息。

    在 ``awrap_model_call`` 中：
    1. 从 runtime context 提取 thread_id
    2. 并行读取 agent 级 + thread 级 facts（仅 ``status == "active"``）
    3. 按 content 精确去重归并（agent 优先），同 ``topic_key`` 的多条
       active 条目收敛为 ``create_time`` 最新的一条（历史存量兜底）
    4. 将 facts 组装为文本片段（推断类/低置信附标注），追加为
       ``request.messages`` 末尾的 user 消息
    5. 非阻塞更新 ``last_used_at``（用于 LRU 淘汰）

    system message 保持静态且只有一条（利于跨轮次 KV 缓存复用）。
    长期记忆只注入本次请求的副本，不修改原始 messages 列表，也不写 state。
    原始对话保留在 checkpointer。

    Args:
        memory_store: ThreadMemoryStore 实例
        recall_limit: 注入时最多使用的 fact 条数（取最近 N 条）；
                      为 None 时注入全部 facts（对应配置键 memory_recall_limit）
        low_confidence_threshold: 低置信标注阈值——confidence 低于该值的条目
                      注入时附"（低置信）"标注；默认取 MEMORY_LOW_CONFIDENCE_THRESHOLD
    """

    FACTS_HEADER = "【长期记忆】\n"

    def __init__(
        self,
        memory_store: ThreadMemoryStore,
        recall_limit: int | None = None,
        low_confidence_threshold: float | None = None,
    ) -> None:
        self._store = memory_store
        self._recall_limit = recall_limit
        self._low_confidence_threshold = (
            low_confidence_threshold
            if low_confidence_threshold is not None
            else MEMORY_LOW_CONFIDENCE_THRESHOLD
        )

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[Any]],
    ) -> Any:
        """注入长期记忆为尾随 user 消息，然后调用 handler。

        并行读取 agent 级 + thread 级的 **active** facts，按 content 精确去重
        归并（agent 级优先，保留 agent 级版本），同 topic_key 收敛为最新一条，
        按 ``create_time`` 升序排序，截取 ``recall_limit`` 条追加为
        ``request.messages`` 末尾的 user 消息。对合并后的全部 facts 非阻塞
        touch：agent 级调 ``touch_agent_fact``，thread 级调 ``touch_fact``。
        """
        thread_id = self._extract_thread_id(request)
        if not thread_id:
            return await handler(request)

        # 并行读取 agent 级 + thread 级 facts（仅 active：superseded/stale 过滤）
        try:
            agent_facts, thread_facts = await asyncio.gather(
                self._store.query_agent_facts(active_only=True),
                self._store.query_facts(thread_id, active_only=True),
            )
        except Exception as error:
            logger.debug("读取长期记忆失败 [thread=%s]: %s", thread_id, error)
            return await handler(request)

        # 合并 + 去重（agent 优先）+ 同主题收敛 + 按 create_time 升序
        facts = self._merge_facts(agent_facts, thread_facts)

        if not facts:
            return await handler(request)

        # 按 recall_limit 截取最近 N 条（在主题收敛之后截取，保证同主题
        # 只占一个名额；对应配置键 memory_recall_limit）
        if self._recall_limit and len(facts) > self._recall_limit:
            facts = facts[-self._recall_limit:]

        # 组装 facts 文本（推断/低置信附标注）
        fact_text = self._format_facts(facts)

        # 注入为尾随 user 消息
        new_request = self._inject_facts(request, fact_text)

        # 非阻塞更新 last_used_at：按 scope 分发
        # - agent 级 facts 调 touch_agent_fact（作用于 agent namespace）
        # - thread 级 facts 调 touch_fact（作用于 thread namespace）
        for fact in facts:
            if fact.scope == "agent":
                asyncio.create_task(self._store.touch_agent_fact(fact.fact_id))
            else:
                asyncio.create_task(self._store.touch_fact(thread_id, fact.fact_id))

        return await handler(new_request)

    @staticmethod
    def _merge_facts(
        agent_facts: list[ThreadFactItem],
        thread_facts: list[ThreadFactItem],
    ) -> list[ThreadFactItem]:
        """合并 agent 级 + thread 级 facts。

        归并顺序：
        1. agent 级优先遍历，按 content 精确去重（保留 agent 级版本）
        2. 按 ``create_time`` 升序排序
        3. 同 ``topic_key`` 的多条 active 条目收敛为最新一条（``create_time``
           最大者）——历史存量数据可能存在同主题并存，写入侧合并生效前的
           兜底；``topic_key`` 为空（旧数据）不参与收敛，保持原有为

        Args:
            agent_facts: agent 级 facts 列表（应为 active）
            thread_facts: thread 级 facts 列表（应为 active）

        Returns:
            去重合并后的 facts 列表（按 ``create_time`` 升序）
        """
        seen: set[str] = set()
        deduped: list[ThreadFactItem] = []

        # agent 级优先：同 content 时保留 agent 级版本
        for fact in agent_facts:
            if fact.content in seen:
                continue
            seen.add(fact.content)
            deduped.append(fact)

        for fact in thread_facts:
            if fact.content in seen:
                continue
            seen.add(fact.content)
            deduped.append(fact)

        deduped.sort(key=lambda f: f.create_time)

        # 同主题收敛：升序遍历，同 topic_key 后者（更新）覆盖前者
        no_topic: list[ThreadFactItem] = []
        by_topic: dict[str, ThreadFactItem] = {}
        for fact in deduped:
            if fact.topic_key:
                by_topic[fact.topic_key] = fact
            else:
                no_topic.append(fact)

        merged = [*no_topic, *by_topic.values()]
        merged.sort(key=lambda f: f.create_time)
        return merged

    # ============ 内部方法 ============

    @staticmethod
    def _extract_thread_id(request: ModelRequest[ContextT]) -> str | None:
        """从 ModelRequest 的 runtime context 中提取 thread_id。

        ``runtime.context`` **不会**自动由 ``config["configurable"]`` 填充，
        它只来自图调用处的 ``context=`` 参数（见 ``agent/turn_runners.py`` /
        ``agent/streaming.py``）。该参数传入的结构为
        ``{"configurable": {"thread_id": "..."}}``，与此处的解析保持一致。
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
        except Exception as error:
            logger.debug("提取 thread_id 失败: %s", error)
        return None

    def _format_facts(self, facts: list[ThreadFactItem]) -> str:
        """将 facts 列表格式化为文本片段。

        按 category 分组，每条 fact 一行。证据来源与置信度标注：
        - ``evidence == "inferred"`` → 附"（未实测）"标注（推断与事实显式区分）
        - ``confidence < low_confidence_threshold`` → 附"（低置信）"标注
        - 两者同时命中时叠加标注
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

            suffix = ""
            if fact.evidence == EVIDENCE_INFERRED:
                suffix += "（未实测）"
            if fact.confidence < self._low_confidence_threshold:
                suffix += "（低置信）"

            lines.append(f"- [{category_label}] {fact.content}{suffix}")

        return self.FACTS_HEADER + "\n".join(lines) + "\n"

    @staticmethod
    def _inject_facts(
        request: ModelRequest[ContextT], fact_text: str
    ) -> ModelRequest[ContextT]:
        """将 facts 文本追加为尾随 user 消息，返回新的 ModelRequest。

        注入到 ``request.messages`` 末尾（而非 ``request.system_message``）：
        - system message 保持静态且只有一条，跨轮次前缀可复用（利于 KV 缓存）；
        - 记忆块位于 payload 尾部，其变化不会使前面的 system 与历史失效；
        - 只覆盖本次请求的副本，不写 state，故不会在 checkpoint 中累积。

        注意：``request.messages`` 直接引用 ``state["messages"]``（见 langchain
        ``agents/factory.py`` 的 ``messages=state["messages"]``），**必须新建列表**，
        原地 append 会污染 checkpoint。
        """
        return request.override(
            messages=[*request.messages, HumanMessage(content=fact_text)]
        )


__all__ = [
    "ThreadLLMResolver",
    "ThreadMemoryReadMiddleware",
    "ThreadMemoryWriteMiddleware",
    "default_thread_llm_resolver",
]
