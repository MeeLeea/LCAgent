"""
技能注入中间件 - 从 LCAgentState.active_skills 读取技能,在 model 调用前注入提示词

设计要点:
- ``active_skills`` 存入 ``LCAgentState``(随 checkpoint per-thread 隔离),
  不再依赖 AgentCore 实例属性,实现真正的无状态化。
- 中间件在 ``awrap_model_call`` 时从 state 读取技能列表 + 自动匹配,
  将技能指引块作为**尾随 user 消息**注入 ``request.messages`` 末尾,无需重建 Graph
  或维护 per-thread SystemMessage;system message 因此保持静态且只有一条,利于 KV 缓存。
- 所有会话共享同一个编译图,技能隔离完全由 checkpoint state 保证。

改调 skmng.core.build_skill_block 统一合并逻辑(取代原 _compute_skill_block
内的重复实现),与 TeamAgent.build_skill_block / SkillInjector.build_skill_block
三处共用同一份三来源合并代码。
"""
from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ContextT, ModelRequest
from langchain_core.messages import HumanMessage

from skmng.core import build_skill_block
from skmng.manager import SkillManager

logger = logging.getLogger(__name__)


class SkillInjectionMW(AgentMiddleware):
    """从 state 读取活跃技能并注入尾随 user 消息的中间件。

    在 ``awrap_model_call`` 中:
    1. 从 ``state["active_skills"]`` 读取手动加载的技能名列表(active_names 通道)
    2. 若开启自动匹配,从最后一条 HumanMessage 提取任务文本,匹配相关技能
    3. 经 skmng.core.build_skill_block 合并去重后渲染技能指引块,追加到
       ``request.messages`` 末尾(尾随 user 消息,不动 system message)

    Args:
        skill_manager: 技能管理器(本地 .agents/skills 读取)
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
        """从 state 计算应注入的技能指引块

        经 skmng.core.build_skill_block 统一合并三来源:
        - active_names: state["active_skills"](手动加载,per-thread 隔离)
        - match_skills: auto_match 开启时从最后一条 HumanMessage 提取任务文本匹配
        - fixed_skills: 中间件层无角色级固定依赖,默认空

        与 TeamAgent.build_skill_block / SkillInjector.build_skill_block 共用同一份逻辑。
        """
        # 从最后一条 HumanMessage 提取任务文本用于匹配
        task = ""
        if self.auto_match:
            messages = state.get("messages", [])
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage):
                    task = msg.content
                    break

        active_names = tuple(state.get("active_skills") or [])
        block = build_skill_block(
            self.skill_manager,
            task,
            active_names=active_names,
            fixed_skills=(),
            auto_match=self.auto_match,
        )
        return block

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[Any]],
    ) -> Any:
        """异步版本:把技能指引块作为尾随 user 消息注入。"""
        skill_block = self._compute_skill_block(request.state)
        if not skill_block:
            return await handler(request)
        return await handler(self._inject(request, skill_block))

    @staticmethod
    def _inject(
        request: ModelRequest[ContextT], skill_block: str
    ) -> ModelRequest[ContextT]:
        """把技能指引块追加为尾随 user 消息,返回新的 ModelRequest。

        注入到 ``request.messages`` 末尾(而非 ``request.system_message``):
        - system message 保持**静态**且只有一条,跨轮次前缀可复用(利于 KV 缓存);
        - 技能块位于 payload 尾部,内容变化不会使前面的 system 与历史失效;
        - 只覆盖本次请求的副本,不写 state,故不会在 checkpoint 中累积。

        注意:``request.messages`` 直接引用 ``state["messages"]``(见 langchain
        ``agents/factory.py`` 的 ``messages=state["messages"]``),**必须新建列表**,
        原地 append 会污染 checkpoint。
        """
        return request.override(
            messages=[*request.messages, HumanMessage(content=skill_block)]
        )


__all__ = ["SkillInjectionMW"]
