"""技能注入中间件 - 从 LCAgentState.active_skills 读取技能，在 model 调用前注入提示词。

设计要点：
- ``active_skills`` 存入 ``LCAgentState``（随 checkpoint per-thread 隔离），
  不再依赖 AgentCore 实例属性，实现真正的无状态化。
- 中间件在 ``awrap_model_call`` 时从 state 读取技能列表 + 自动匹配，
  将技能指引块追加到 system message，无需重建 Graph 或维护 per-thread SystemMessage。
- 所有会话共享同一个编译图，技能隔离完全由 checkpoint state 保证。
"""
from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any, cast

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ContextT, ModelRequest
from langchain_core.messages import HumanMessage, SystemMessage

from tools.skills import SkillManager

logger = logging.getLogger(__name__)


class SkillInjectionMW(AgentMiddleware):
    """从 state 读取活跃技能并注入 system prompt 的中间件。

    在 ``awrap_model_call`` 中：
    1. 从 ``state["active_skills"]`` 读取手动加载的技能名列表
    2. 若开启自动匹配，从最后一条 HumanMessage 提取任务文本，匹配相关技能
    3. 合并去重后渲染技能指引块，追加到 system message

    Args:
        skill_manager: 技能管理器（本地 .agents/skills 读取）
        auto_match: 是否在每次 model 调用时自动匹配技能
    """

    def __init__(
        self,
        skill_manager: SkillManager,
        auto_match: bool = True,
    ) -> None:
        self.skill_manager = skill_manager
        self.auto_match = auto_match

    def _compute_skill_block(self, state: dict[str, Any]) -> str:
        """从 state 计算应注入的技能指引块。

        合并手动加载的技能（state["active_skills"]）与自动匹配的技能。
        """
        names: set[str] = set(state.get("active_skills") or [])

        if self.auto_match:
            messages = state.get("messages", [])
            # 从最后一条 HumanMessage 提取任务文本用于匹配
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage):
                    names.update(self.skill_manager.match_skills(msg.content))
                    break

        if not names:
            return ""
        return self.skill_manager.render_block(sorted(names))

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[Any]],
    ) -> Any:
        """异步版本：注入技能指引块到 system message。"""
        skill_block = self._compute_skill_block(request.state)
        if not skill_block:
            return await handler(request)
        return await handler(self._inject(request, skill_block))

    @staticmethod
    def _inject(
        request: ModelRequest[ContextT], skill_block: str
    ) -> ModelRequest[ContextT]:
        """将技能指引块追加到 system message，返回新的 ModelRequest。"""
        if request.system_message is not None:
            new_content = [
                *request.system_message.content_blocks,
                {"type": "text", "text": f"\n{skill_block}"},
            ]
        else:
            new_content = [{"type": "text", "text": skill_block}]
        new_sys_msg = SystemMessage(
            content=cast("list[str | dict[str, str]]", new_content)
        )
        return request.override(system_message=new_sys_msg)


__all__ = ["SkillInjectionMW"]
