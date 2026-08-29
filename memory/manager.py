"""MemoryManager — 独立记忆基础设施，三层架构的 Memory 层。

接管全部记忆职责：
- 事件过滤：从 AgentEvent 流中筛选值得沉淀的事件
- 记忆判定：基于 MemoryInputEvent 规则判定分类
- LLM 压缩/摘要/去重/冲突修正
- 长期记忆入库（fact / lesson / entity / decision）
- 长期记忆召回注入 Agent prompt

与 AgentCore 和 SessionManager 平级，互不耦合：
- Agent 不直接调用 MemoryManager（记忆注入由 LangGraph 中间件完成）
- SessionManager 调用 MemoryManager 做记忆召回和事件消费
- MemoryManager 内部封装 ThreadMemoryStore + 读写中间件

核心数据流：
  AgentEvent 流 → MemoryManager.consume_event()
    → 事件过滤 → submit to ThreadMemoryWriteMiddleware
    → 防抖 buffer → LLM fact 抽取 → 去重 → 写入 ThreadMemoryStore

  SessionManager 调用 recall() → 返回 facts 文本 → 注入 Agent prompt
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .lock_pool import ThreadMemoryLockPool
from .middleware import (
    ThreadMemoryReadMiddleware,
    ThreadMemoryWriteMiddleware,
)
from .models import MemoryCategory, ThreadFactItem
from .store import ThreadMemoryStore

if TYPE_CHECKING:
    from utils.events import AgentEvent

logger = logging.getLogger(__name__)

# 召回时默认取的 fact 条数上限
_DEFAULT_RECALL_LIMIT = 10

# 记忆分类标签（用于召回文本格式化）
_CATEGORY_LABELS: dict[str, str] = {
    MemoryCategory.USER_FACT.value: "用户事实",
    MemoryCategory.LESSON_EXPERIENCE.value: "经验教训",
    MemoryCategory.BUSINESS_ENTITY.value: "业务信息",
    MemoryCategory.IMPORTANT_CONVERSATION.value: "重要决策",
}


class MemoryManager:
    """独立记忆基础设施 — 接管记忆判定、压缩、摘要、长期存储。

    封装 ThreadMemoryStore + 读写中间件为统一门面，
    SessionManager 只需调用 recall / consume_event / compress / clear。

    Args:
        memory_store: ThreadMemoryStore 实例（长期记忆 Store 封装）
        lock_pool: ThreadMemoryLockPool 实例（per-thread 并发锁）
        llm_getter: 返回当前 LLMClient 的 callable（支持 LLM 热切换）
        recall_limit: 召回长期记忆时的默认条数上限
        buffer_delay_seconds: 防抖缓冲窗口（秒），透传给写中间件
        max_buffer_messages: 单 thread 缓冲区上限，透传给写中间件
    """

    def __init__(
        self,
        memory_store: ThreadMemoryStore,
        lock_pool: ThreadMemoryLockPool,
        llm_getter: Callable[[], Any],
        recall_limit: int = _DEFAULT_RECALL_LIMIT,
        buffer_delay_seconds: int | None = None,
        max_buffer_messages: int | None = None,
    ) -> None:
        self._store = memory_store
        self._lock_pool = lock_pool
        self._llm_getter = llm_getter
        self._recall_limit = recall_limit

        # 写中间件：事件接收 + 防抖 + Fact 抽取流水线
        self._write_middleware = ThreadMemoryWriteMiddleware(
            memory_store=memory_store,
            lock_pool=lock_pool,
            llm_getter=llm_getter,
            buffer_delay_seconds=buffer_delay_seconds,
            max_buffer_messages=max_buffer_messages,
        )

        # 读中间件：awrap_model_call 时注入 thread facts 到 SystemMessage
        # （recall_limit 约束每次注入的 fact 条数，与显式召回 recall() 一致）
        self._read_middleware = ThreadMemoryReadMiddleware(
            memory_store, recall_limit=recall_limit
        )

    # ============ 属性暴露 ============

    @property
    def store(self) -> ThreadMemoryStore:
        """底层 ThreadMemoryStore（供直接操作 facts 时使用）。"""
        return self._store

    @property
    def read_middleware(self) -> ThreadMemoryReadMiddleware:
        """LangGraph 读中间件实例（供 create_agent middleware 参数使用）。"""
        return self._read_middleware

    @property
    def write_middleware(self) -> ThreadMemoryWriteMiddleware:
        """写中间件实例（供 flush_all / shutdown 等生命周期管理使用）。"""
        return self._write_middleware

    def bind_llm(self, llm_getter: Callable[[], Any]) -> None:
        """运行时替换 LLM 获取器，并同步到写中间件（支持 provider 热切换）。

        入口创建 Agent 后调用，将记忆组件（召回/压缩/事实抽取）的 LLM
        来源动态绑定到 ``agent.llm``，确保切换提供商后记忆链路不再使用
        启动时的旧 LLMClient。

        Args:
            llm_getter: 返回当前 LLMClient 的 callable
        """
        self._llm_getter = llm_getter
        self._write_middleware.bind_llm(llm_getter)

    # ============ 召回（读） ============

    async def recall(
        self, thread_id: str, limit: int | None = None
    ) -> list[ThreadFactItem]:
        """召回指定会话的长期记忆 facts。

        按 create_time 升序返回，截取最近 ``limit`` 条。
        非阻塞更新 ``last_used_at``（用于 LRU 淘汰），失败仅记日志。

        Args:
            thread_id: 会话线程 ID
            limit: 返回条数上限（为 None 时使用构造时设定的 recall_limit）

        Returns:
            facts 列表（可能为空）
        """
        if not thread_id:
            return []
        effective_limit = limit if limit is not None else self._recall_limit
        try:
            facts = await self._store.query_facts(thread_id)
        except Exception as error:
            logger.debug("召回长期记忆失败 [thread=%s]: %s", thread_id, error)
            return []

        # 非阻塞更新 last_used_at
        for fact in facts:
            asyncio.create_task(self._store.touch_fact(thread_id, fact.fact_id))

        return facts[-effective_limit:] if effective_limit and len(facts) > effective_limit else facts

    async def recall_text(
        self, thread_id: str, limit: int | None = None
    ) -> str:
        """召回长期记忆并格式化为文本片段（供注入 prompt）。

        Args:
            thread_id: 会话线程 ID
            limit: 返回条数上限

        Returns:
            格式化的记忆文本；无记忆时返回空字符串
        """
        facts = await self.recall(thread_id, limit)
        if not facts:
            return ""

        lines: list[str] = []
        for fact in facts:
            label = _CATEGORY_LABELS.get(fact.category, "记忆")
            lines.append(f"- [{label}] {fact.content}")

        return "【长期记忆】\n" + "\n".join(lines) + "\n"

    # ============ 召回（读） — agent 级（跨会话共享） ============

    async def recall_agent(
        self, limit: int | None = None
    ) -> list[ThreadFactItem]:
        """召回 agent 级长期记忆 facts（跨会话共享）。

        读取 ``user_fact`` / ``lesson`` 类的 agent 级记忆，按 ``create_time``
        升序返回，截取最近 ``limit`` 条。与 thread 级 :meth:`recall` 不同，
        agent 级记忆跨会话共享且无 thread 上下文，因此 **不进行 touch 更新**
        （store 未提供 agent 级 touch 方法），LRU 淘汰依赖
        :meth:`ThreadMemoryStore.prune_agent_facts`。

        Args:
            limit: 返回条数上限（为 None 时使用构造时设定的 recall_limit）

        Returns:
            agent 级 facts 列表（可能为空）；异常仅记日志返回 ``[]``
        """
        effective_limit = limit if limit is not None else self._recall_limit
        try:
            facts = await self._store.query_agent_facts()
        except Exception as error:
            logger.debug("召回 agent 级长期记忆失败: %s", error)
            return []

        return (
            facts[-effective_limit:]
            if effective_limit and len(facts) > effective_limit
            else facts
        )

    async def recall_agent_text(self, limit: int | None = None) -> str:
        """召回 agent 级长期记忆并格式化为文本片段（供注入 prompt）。

        复用 :data:`_CATEGORY_LABELS` 标签，输出形如 ``【长期记忆】\\n- [用户事实] ...``。

        Args:
            limit: 返回条数上限

        Returns:
            格式化的记忆文本；无记忆时返回空字符串
        """
        facts = await self.recall_agent(limit)
        if not facts:
            return ""

        lines: list[str] = []
        for fact in facts:
            label = _CATEGORY_LABELS.get(fact.category, "记忆")
            lines.append(f"- [{label}] {fact.content}")

        return "【长期记忆】\n" + "\n".join(lines) + "\n"

    async def count_agent_facts(self) -> int:
        """统计 agent 级长期记忆条数。"""
        return await self._store.count_agent_facts()

    async def clear_agent_facts(self) -> int:
        """清空 agent 级全部长期记忆。

        Returns:
            被清除的 fact 数量
        """
        return await self._store.clear_agent_facts()

    # ============ 事件消费（写） ============

    async def submit_user_message(
        self,
        thread_id: str,
        content: str,
        important: bool = False,
    ) -> None:
        """提交用户消息到记忆写中间件。

        在 Agent 执行前调用，将用户输入投递到防抖 buffer。
        非阻塞：事件进入 buffer 后立即返回，不影响主流程。

        Args:
            thread_id: 会话线程 ID
            content: 用户消息文本
            important: 是否用户显式标记为重要
        """
        if not thread_id or not content.strip():
            return
        try:
            await self._write_middleware.submit_event(
                thread_id, "user", content, important
            )
        except Exception as error:
            logger.debug("用户消息记忆提交失败: %s", error)

    async def consume_event(self, event: AgentEvent) -> None:
        """消费 Agent 执行事件，提取记忆并提交到写中间件。

        事件过滤逻辑：
        - DONE 事件：最终输出（role=assistant）→ 提交
        - TOOL_RESULT 事件：工具结果（可能含经验教训）→ 提交
        - TOKEN / TOOL_CALL / INTERRUPT 等不单独提交

        非阻塞：事件提交到防抖 buffer，不等待 LLM 处理。

        Args:
            event: Agent 执行事件
        """
        if not event.is_memory_worthy or not event.thread_id:
            return

        role = event.role or "assistant"
        try:
            await self._write_middleware.submit_event(
                event.thread_id,
                role,
                event.content,
                event.is_important,
            )
        except Exception as error:
            logger.debug("事件记忆消费失败: %s", error)

    # ============ 压缩 & 清理 ============

    async def compress(self, thread_id: str) -> dict[str, Any]:
        """压缩指定会话的长期记忆：读取全部 facts，用 LLM 生成摘要后替换。

        LLMClient 无异步 chat 接口，阻塞调用放入线程池。

        Args:
            thread_id: 会话线程 ID

        Returns:
            ``{"success": bool, "original_count": int, ...}``
        """
        facts = await self._store.query_facts(thread_id)
        if not facts:
            return {"success": False, "error": "没有长期记忆可压缩"}

        # 1. 拼接所有 facts 为文本
        history_lines: list[str] = []
        original_chars = 0
        for idx, fact in enumerate(facts, 1):
            line = f"[{idx}] ({fact.create_time}) [{fact.category}] {fact.content}"
            history_lines.append(line)
            original_chars += len(fact.content)
        history_text = "\n\n".join(history_lines)

        # 2. 调用 LLM 生成摘要
        system_prompt = (
            "你是一个记忆压缩助手。请将以下历史对话记录压缩成一份简洁的摘要，要求：\n"
            "1. 保留所有关键信息、用户意图、重要决策和事实\n"
            "2. 去除重复和冗余内容\n"
            "3. 按主题分条目组织，使用 '- ' 开头\n"
            "4. 保持事实准确，不要添加推测内容\n"
            "5. 用中文输出"
        )

        def _sync_summarize() -> str:
            try:
                llm = self._llm_getter()
                return llm.chat(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"以下是历史对话记录，请压缩成摘要:\n\n{history_text}"},
                    ]
                ).strip()
            except Exception:
                return ""

        summary = await asyncio.to_thread(_sync_summarize)
        if not summary:
            return {"success": False, "error": "LLM 调用失败或返回空摘要"}

        # 3. 用摘要替换全部 facts
        result = await self._store.replace_with_summary(thread_id, summary)
        compressed_chars = len(summary)

        logger.info(
            "thread %s 记忆压缩: %d 条 → 1 条摘要 (%d → %d 字符)",
            thread_id,
            result["original_count"],
            original_chars,
            compressed_chars,
        )

        return {
            "success": result["success"],
            "original_count": result["original_count"],
            "original_chars": original_chars,
            "compressed_chars": compressed_chars,
            "summary": summary,
        }

    async def clear(self, thread_id: str, release_lock: bool = False) -> int:
        """清空指定会话的 thread 级长期记忆 facts（线程安全）。

        为避免与会话删除 / 显式清记忆时发生竞态（防抖 flush 正在回写
        导致记忆“复活”），执行顺序为：

        1. 丢弃该 thread 未 flush 的缓冲事件（取消定时器 + 弹出 buffer），
           防止清完后被回写；
        2. 获取 per-thread 锁，与可能正在进行的 flush 流水线串行化；
        3. 删除 ``(thread_id, "thread_facts")`` 命名空间下的全部 facts
           （不影响 agent 级跨会话共享记忆）。

        Args:
            thread_id: 会话线程 ID
            release_lock: 会话销毁时置 True，删除完成后释放锁池缓存，
                避免长时间运行下锁对象泄漏

        Returns:
            被清除的 fact 数量
        """
        # 1. 丢弃未 flush 的缓冲事件，避免清完被回写
        await self._write_middleware.cleanup_thread(thread_id)
        # 2. 与可能正在进行的 flush 串行化
        lock = await self._lock_pool.get(thread_id)
        async with lock:
            cleared = await self._store.clear_thread_memory(thread_id)
        # 3. 会话销毁时释放锁缓存
        if release_lock:
            await self._lock_pool.cleanup(thread_id)
        return cleared

    async def count_facts(self, thread_id: str) -> int:
        """统计指定会话的长期记忆条数。"""
        return await self._store.count_facts(thread_id)

    # ============ 生命周期 ============

    async def flush_all(self) -> None:
        """立即处理所有 buffer 中的待处理事件（用于 Agent 关闭前）。"""
        await self._write_middleware.flush_all()

    async def shutdown(self) -> None:
        """关闭中间件：取消所有定时器，清理 buffer。"""
        await self._write_middleware.shutdown()

__all__ = ["MemoryManager"]
