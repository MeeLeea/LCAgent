"""
提示词注入器协议 - 独立放置,避免 team 层 import 拉入 graph 依赖

任何提供 inject_into_prompt(prompt, task) 的对象皆可注入(鸭子类型),
使 TeamAgent 及其子类无需依赖 graph 层即可作为 PromptInjector 传入节点函数。

注:Commit 4 将为 inject_into_prompt 增加 active_names 参数,协议同步更新。
"""
from __future__ import annotations

from typing import Protocol


class PromptInjector(Protocol):
    """提示词注入器协议(鸭子类型,兼容 skmng.injector.SkillInjector)"""

    def inject_into_prompt(self, prompt: str, task: str) -> str:
        """把技能指引块追加到 prompt 末尾,返回注入后的提示词"""
