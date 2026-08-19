"""技能与压缩 Mixin - AgentCore 的技能加载/清理与手动上下文压缩。

从 agent_core.py 抽离，职责：
- 技能列表 / 加载到会话 state / 清空会话技能
- 手动触发上下文压缩（CLI compact 命令入口）

依赖 AgentCore 实例属性：skill_manager / agent_executor /
_compaction_middleware / metrics / verbose / session。
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .agent_core import AgentCore

logger = logging.getLogger(__name__)


class SkillOps:
    """技能与压缩 Mixin（供 AgentCore 多继承使用，自身不初始化状态）"""

    # ============ 技能阅读(Skills) ============

    def list_skills(self) -> list[dict[str, str]]:
        """列出所有本地可用技能"""
        return self.skill_manager.list_skills()

    async def aload_skill(self, name: str, thread_id: str | None = None) -> bool:
        """异步加载技能到指定会话（写入 LCAgentState.active_skills）

        技能名存入 per-thread state（随 checkpoint 持久化），
        由 SkillInjectionMW 在 model 调用时读取并注入提示词。
        无需重建 Graph，也无需维护实例级 active_skills。

        Args:
            name: 技能名(目录名或 frontmatter name)
            thread_id: 目标会话线程 ID（为 None 时使用当前会话）

        Returns:
            True=成功加载, False=技能不存在
        """
        if self.skill_manager.get_skill(name) is None:
            return False
        config = self._invoke_config(thread_id)
        state = await self.agent_executor.aget_state(config)
        current: set[str] = set(
            (state.values if state and state.values else {}).get("active_skills") or []
        )
        current.add(name)
        await self.agent_executor.aupdate_state(
            config, {"active_skills": sorted(current)}
        )
        return True

    async def aclear_skills(self, thread_id: str | None = None):
        """异步清空指定会话的技能（写入空列表到 state）

        Args:
            thread_id: 目标会话线程 ID（为 None 时使用当前会话）
        """
        config = self._invoke_config(thread_id)
        await self.agent_executor.aupdate_state(config, {"active_skills": []})

    # ============ 长上下文裁剪 ============

    async def manually_compact(
        self,
        force: bool = False,
        thread_id: str | None = None,
    ) -> dict[str, Any] | None:
        """手动触发一次上下文压缩（CLI 命令 compact 调用）

        与 before_model 中间件使用相同的压缩逻辑（增量摘要 + 工具输出 Prune），
        但通过 update_state（asyncio.to_thread 包裹）直接写入 checkpoint，
        不依赖 LangGraph 中间件上下文。

        Args:
            force: 为 True 时跳过 max_messages 阈值检查，允许在消息数
                   未超阈值时强制压缩（仍需消息数 > keep_recent 才能安全切割）。
            thread_id: 目标会话线程 ID（为 None 时使用当前会话）

        Returns:
            {"summary": str, "messages_before": int, "messages_after": int} 或 None
        """
        msgs = await self.session.aget_messages(session_id=thread_id)
        if not msgs:
            return None

        # 读取当前 state 中的已有摘要
        config = self._invoke_config(thread_id)
        executor = self.agent_executor
        state = await executor.aget_state(config)
        existing_summary = ""
        if state and state.values:
            existing_summary = state.values.get("summary", "") or ""

        # 调用中间件的手动压缩接口
        mw = getattr(self, "_compaction_middleware", None)
        if mw is None:
            return None

        update = await mw.arun_compaction(msgs, existing_summary=existing_summary, force=force)
        if update is None:
            if self.verbose:
                logger.debug("压缩: 消息不足或无法安全切割，无需压缩")
            return None

        import time as _time
        _compact_start = _time.time()

        messages_before = len(msgs)
        await executor.aupdate_state(config, update)
        messages_after = len(update["messages"]) - 1  # 减去 RemoveMessage 标记
        summary_length = len(update.get("summary", ""))

        # 记录压缩指标
        self.metrics.record_compaction(
            trigger="manual",
            messages_before=messages_before,
            messages_after=messages_after,
            summary_length=summary_length,
            duration_ms=(_time.time() - _compact_start) * 1000,
        )

        if self.verbose:
            logger.info("压缩: %d → %d 条消息，摘要已更新", messages_before, messages_after)

        return {
            "summary": update["summary"],
            "messages_before": messages_before,
            "messages_after": messages_after,
        }