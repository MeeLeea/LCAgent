"""
提示词注入器协议 - 独立放置,避免 team 层 import 拉入 graph 依赖

任何提供 inject_into_prompt 的对象皆可注入(鸭子类型),
使 TeamAgent 及其子类无需依赖 graph 层即可作为 PromptInjector 传入节点函数。

active_names 参数:运行时手动加载的技能名列表(由节点函数从
state["active_skills"] 取值传入),注入器将其与自动匹配结果合并注入,
实现"手动加载技能在 graph/team 节点生效"(修复原实现 1/3/4 读不到 state 的行为不一致)。
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class PromptInjector(Protocol):
    """提示词注入器协议(鸭子类型,兼容 skmng.injector.SkillInjector)"""

    def inject_into_prompt(
        self,
        prompt: str,
        task: str,
        active_names: Sequence[str] = (),
    ) -> str:
        """把技能指引块追加到 prompt 末尾,返回注入后的提示词

        Args:
            prompt: 渲染后的节点提示词
            task: 用户任务描述(用于自动匹配)
            active_names: 手动加载的技能名(由节点函数从 state 取值传入)
        """
