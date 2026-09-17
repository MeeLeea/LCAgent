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
- 将 facts 组装为文本片段，追加到 SystemMessage
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
from langchain_core.messages import SystemMessage

from .config import MEMORY_BUFFER_DELAY_SECONDS, MEMORY_MAX_BUFFER_MESSAGES
from .lock_pool import AgentMemoryLock, ThreadMemoryLockPool
from .models import (
    MemoryCategory,
    MemoryInputEvent,
    ThreadFactItem,
    judge_long_term_memory,
)
from .store import ThreadMemoryStore

logger = logging.getLogger(__name__)

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

    与 ``agent.terminal_retry_cap_mw.is_timeout_content`` 逻辑一致，
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
        llm_getter: 返回当前 LLMClient 的 callable（支持 LLM 热切换）
        buffer_delay_seconds: 防抖缓冲窗口（秒），默认 MEMORY_BUFFER_DELAY_SECONDS
        max_buffer_messages: 单 thread 缓冲区上限，默认 MEMORY_MAX_BUFFER_MESSAGES
        agent_lock: agent 级跨进程互斥锁（保护 agent namespace 的批量写入与
            LRU 淘汰）。为 None 时内部自建进程内锁 ``AgentMemoryLock(path=None)``，
            供测试等无外部注入场景使用
    """

    def __init__(
        self,
        memory_store: ThreadMemoryStore,
        lock_pool: ThreadMemoryLockPool,
        llm_getter: Callable[[], Any],
        buffer_delay_seconds: int | None = None,
        max_buffer_messages: int | None = None,
        agent_lock: AgentMemoryLock | None = None,
    ) -> None:
        self._store = memory_store
        self._lock_pool = lock_pool
        self._llm_getter = llm_getter
        # agent 级互斥锁：agent namespace 跨进程共享，写流水线必须持锁
        self._agent_lock = agent_lock if agent_lock is not None else AgentMemoryLock()
        self._buffer_delay_seconds = buffer_delay_seconds if buffer_delay_seconds is not None else MEMORY_BUFFER_DELAY_SECONDS
        self._max_buffer_messages = max_buffer_messages if max_buffer_messages is not None else MEMORY_MAX_BUFFER_MESSAGES

        # 防抖 buffer: thread_id → [(role, content, important), ...]
        self._buffer: dict[str, list[tuple[str, str, bool]]] = {}
        # 定时器: thread_id → asyncio.Task[None]（_a_delayed_flush 无返回值）
        self._timers: dict[str, asyncio.Task[None]] = {}
        # 保护 buffer 和 timers 的并发访问
        self._buffer_lock = asyncio.Lock()
        # 失败计数：(thread_id, tool_name) → 同类工具失败累计次数，
        # 供 judge_long_term_memory 确定性判定经验教训（失败 ≥2 次）
        self._failure_counts: dict[tuple[str, str], int] = {}

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
           内容经 LLM 蒸馏后入库（绝不逐字存错误原文，见 ①-b）；
           IMPORTANT_CONVERSATION / None → 进入 LLM 抽取
        ② LLM fact 抽取：仅对需抽取的消息调用 _a_extract_facts
        ③④⑤⑥ 不变：无效过滤 → 两级去重 → 按 category 分流 → 写入 → LRU 淘汰
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

        # ①-b lesson 蒸馏：LLM 把失败记录提炼为一条可复用教训；
        #     LLM 失败 / 判定无教训时丢弃该条（宁缺勿污染 agent 级共享空间），
        #     绝不回退逐字存原文。
        direct_agent_items: list[ThreadFactItem] = []
        for source in lesson_sources:
            distilled = await self._a_distill_lesson(thread_id, source)
            if not distilled:
                continue
            direct_agent_items.append(
                self._build_fact_item(
                    distilled,
                    MemoryCategory.LESSON_EXPERIENCE.value,
                    {"confidence": 0.9},
                    scope="agent",
                )
            )

        # ② LLM fact 抽取：仅对需抽取的消息（失败≥2 的已确定性处理）
        facts_raw: list[dict[str, Any]] = []
        if extract_messages:
            facts_raw = await self._a_extract_facts(thread_id, extract_messages)

        # ③ 无效内容过滤
        facts_raw = [f for f in facts_raw if f.get("content", "").strip()]

        # ④ 读取两级 existing 作为去重基准（agent 级别优先，避免跨作用域重复）
        existing_thread = await self._store.query_facts(thread_id)
        existing_agent = await self._store.query_agent_facts()

        # ⑤ 按 category 分流组装 ThreadFactItem（蒸馏后的确定性 lesson 同样
        #    参与两级去重——历史 bug：直接构造的 lesson 绕过查重，同一命令
        #    反复确认会累积多条近似重复的巨型 fact）
        agent_seen: set[str] = {f.content for f in existing_agent}
        thread_seen: set[str] = {f.content for f in existing_thread}

        agent_items: list[ThreadFactItem] = []
        thread_items: list[ThreadFactItem] = []

        for item in direct_agent_items:
            if item.content in agent_seen or item.content in thread_seen:
                continue
            agent_seen.add(item.content)
            agent_items.append(item)

        for fact_data in facts_raw:
            content = fact_data["content"]
            # 合法性校验 + 兜底（兜底为 conv，走 thread 分支）
            category = fact_data.get("category", "conv")
            valid_categories = {c.value for c in MemoryCategory if c != MemoryCategory.SKIP}
            if category not in valid_categories:
                category = MemoryCategory.IMPORTANT_CONVERSATION.value

            if category in self._AGENT_CATEGORIES:
                if content in agent_seen or content in thread_seen:
                    continue
                agent_seen.add(content)
                agent_items.append(
                    self._build_fact_item(content, category, fact_data, scope="agent")
                )
            else:
                if content in thread_seen or content in agent_seen:
                    continue
                thread_seen.add(content)
                thread_items.append(
                    self._build_fact_item(content, category, fact_data, scope="thread")
                )

        # ⑥ 分流批量写入 + LRU 淘汰（agent 级写入与 prune 必须持有 agent 锁，
        #    且仅在本次确实产生了 agent facts 时才执行——空批次也 prune 会在
        #    多进程并发时各自按本地快照计算溢出并删除不相交的集合，造成过量淘汰）
        if agent_items:
            async with self._agent_lock:
                await self._store.save_agent_facts_batch(agent_items)
                logger.info(
                    "[长期记忆-agent级] thread=%s 写入 %d 条 (跨会话共享, namespace=global_facts)",
                    thread_id,
                    len(agent_items),
                )
                await self._store.prune_agent_facts()

        if thread_items:
            await self._store.save_facts_batch(thread_id, thread_items)
            logger.info(
                "[长期记忆-session级] thread=%s 写入 %d 条 (会话隔离, namespace=thread_facts)",
                thread_id,
                len(thread_items),
            )
        await self._store.prune_facts(thread_id)

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
            fact_data: LLM 返回的原始 fact 字典（取 confidence 等附加字段）
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
        )

    async def _a_distill_lesson(self, thread_id: str, source: str) -> str:
        """LLM 蒸馏：把工具重复失败记录提炼为一条简短可复用的经验教训。

        确定性 lesson 路径（同类失败 ≥2 次）分类锁定 lesson、不依赖 LLM 判定，
        但内容必须蒸馏——错误原文可能内嵌完整命令脚本（数 KB），逐字入库会
        污染跨会话共享的 agent namespace。

        Args:
            thread_id: 会话线程 ID（仅用于日志）
            source: 已剥离指令后缀的错误记录（可能为空）

        Returns:
            蒸馏后的教训文本；LLM 失败、判定无教训或 source 为空时返回
            空字符串（该条 lesson 丢弃，不回退存原文）
        """
        if not source:
            return ""
        system_prompt = (
            "你是经验教训提炼助手。下面是一条工具重复失败的记录，"
            "请把它蒸馏成一条简短、可复用的经验教训，要求：\n"
            "1. 中文一句话，不超过 120 字，直接给结论，不要前缀和解释\n"
            "2. 指出失败根因与规避方法，不得复述大段命令或文件内容\n"
            "3. 若记录只是用户拒绝执行、安全确认暂停、瞬时网络抖动等"
            "无可复用教训，返回空字符串\n"
            "只输出教训文本本身，不要其他内容。"
        )
        try:
            llm = self._llm_getter()
            response = await asyncio.to_thread(
                llm.chat,
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"失败记录：\n{source}"},
                ],
            )
            distilled = (response or "").strip()
            # LLM 以空数组/空引号表达"无教训"时归一为空（该条 lesson 丢弃）
            if distilled in {"[]", "[ ]", '""', "''", "无", "（空）", "(空)"}:
                return ""
            if len(distilled) > _LESSON_MAX_CHARS:
                distilled = distilled[:_LESSON_MAX_CHARS]
            return distilled
        except Exception as error:
            logger.warning("lesson 蒸馏失败 [thread=%s]: %s", thread_id, error)
            return ""

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
        # 构建对话文本（important 消息加标注，提高 LLM 抽取优先级）
        lines = []
        for role, content, important in messages:
            marker = " [用户明确要求记住]" if important else ""
            lines.append(f"[{role}]{marker} {content}")
        conversation_text = "\n".join(lines)

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
            "- 不确定时倾向于过滤（宁缺勿污染 agent 级共享空间）\n\n"
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
    """长期记忆读取中间件 — 在 model 调用前注入 facts 到 SystemMessage。

    在 ``awrap_model_call`` 中：
    1. 从 runtime context 提取 thread_id
    2. 并行读取 agent 级 + thread 级 facts
    3. 按 content 精确去重归并（agent 优先），按 ``create_time`` 升序排序
    4. 将 facts 组装为文本片段，追加到 SystemMessage
    5. 非阻塞更新 thread 级 facts 的 ``last_used_at``（用于 LRU 淘汰）

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
        """注入长期记忆到 system message，然后调用 handler。

        并行读取 agent 级 + thread 级 facts，按 content 精确去重归并
        （agent 级优先，保留 agent 级版本），按 ``create_time`` 升序排序，
        截取 ``recall_limit`` 条注入 SystemMessage。对合并后的全部 facts
        非阻塞 touch：agent 级调 ``touch_agent_fact``，thread 级调 ``touch_fact``。
        """
        thread_id = self._extract_thread_id(request)
        if not thread_id:
            return await handler(request)

        # 并行读取 agent 级 + thread 级 facts
        try:
            agent_facts, thread_facts = await asyncio.gather(
                self._store.query_agent_facts(),
                self._store.query_facts(thread_id),
            )
        except Exception as error:
            logger.debug("读取长期记忆失败 [thread=%s]: %s", thread_id, error)
            return await handler(request)

        # 合并 + 去重（agent 优先）+ 按 create_time 升序
        facts = self._merge_facts(agent_facts, thread_facts)

        if not facts:
            return await handler(request)

        # 按 recall_limit 截取最近 N 条（对应配置键 memory_recall_limit）
        if self._recall_limit and len(facts) > self._recall_limit:
            facts = facts[-self._recall_limit:]

        # 组装 facts 文本
        fact_text = self._format_facts(facts)

        # 注入到 SystemMessage
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

        agent 级优先遍历，按 content 精确去重（保留 agent 级版本），
        合并后按 ``create_time`` 升序排序。

        Args:
            agent_facts: agent 级 facts 列表
            thread_facts: thread 级 facts 列表

        Returns:
            去重合并后的 facts 列表（按 ``create_time`` 升序）
        """
        seen: set[str] = set()
        merged: list[ThreadFactItem] = []

        # agent 级优先：同 content 时保留 agent 级版本
        for fact in agent_facts:
            if fact.content in seen:
                continue
            seen.add(fact.content)
            merged.append(fact)

        for fact in thread_facts:
            if fact.content in seen:
                continue
            seen.add(fact.content)
            merged.append(fact)

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
        # 收集 SystemMessage 内容块（ContentBlock 可能是 dict 子类如 TextContentBlock，
        # 也可能不是如 AudioContentBlock；非 dict 元素统一包成 text 格式 dict）
        new_content: list[str | dict[Any, Any]] = []
        if request.system_message is not None:
            for c in request.system_message.content_blocks:
                if isinstance(c, dict):
                    new_content.append(dict(c))
                else:
                    new_content.append({"type": "text", "text": str(c)})
            new_content.append({"type": "text", "text": f"\n{fact_text}"})
        else:
            new_content.append({"type": "text", "text": fact_text})

        new_sys_msg = SystemMessage(content=new_content)
        return request.override(system_message=new_sys_msg)


__all__ = [
    "ThreadMemoryReadMiddleware",
    "ThreadMemoryWriteMiddleware",
]
