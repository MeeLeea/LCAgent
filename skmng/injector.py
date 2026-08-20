"""
工作流技能注入器 - build 时构造,节点 prompt 层注入技能指引块

不走 AgentMiddleware(那是 create_agent 内部机制),而是复用
skmng.manager.SkillManager 的确定性打分匹配 + skmng.core 的三来源合并:

1. core.build_skill_block(sm, task, active_names, fixed_skills, auto_match)
   合并"角色固定(fixed_skills)" + "手动加载(active_names)" + "自动匹配(match_skills)"
2. core.inject_into_prompt(prompt, block) 把指引块追加到 prompt 末尾(防重复)

工作流节点渲染 prompt 后,调用 injector.inject_into_prompt(prompt, task)
即可完成注入。SkillInjector 是 build 时 partial 绑定的单例,运行时无 state
通道 —— active_names 必须由节点函数从 state.get("active_skills") 取值
显式传入(见 Commit 4:inject_into_prompt 签名将增加 active_names 参数)。

Args:
    skills_dir: 技能目录路径;为 None 时使用默认目录(<项目根>/.agents/skills)
    auto_match: 是否开启自动匹配(False 时仅注入 fixed + active,不走 match)
"""
from __future__ import annotations

from skmng.core import build_skill_block, inject_into_prompt
from skmng.manager import SkillManager, default_skills_dir


class SkillInjector:
    """工作流技能注入器 - 节点 prompt 层注入技能指引块

    当前阶段(Commit 2a):inject_into_prompt(prompt, task) 两参数,
    active_names/fixed_skills 均默认空,行为与原 graph.common.SkillInjector
    一致(仅走 auto_match 单通道)。Commit 4 将为 inject_into_prompt
    增加 active_names 参数,由节点函数从 state 取值传入。
    """

    def __init__(
        self,
        skills_dir: str | None = None,
        auto_match: bool = True,
    ) -> None:
        self.skill_manager = SkillManager(skills_dir or default_skills_dir())
        self.auto_match = auto_match

    def build_skill_block(self, task: str) -> str:
        """根据任务匹配技能并渲染指引块(当前仅 auto_match 单通道)

        Args:
            task: 用户任务描述(用于技能匹配)

        Returns:
            技能指引块文本;未命中任何技能或未开启自动匹配时返回空串
        """
        return build_skill_block(
            self.skill_manager,
            task,
            active_names=(),
            fixed_skills=(),
            auto_match=self.auto_match,
        )

    def inject_into_prompt(self, prompt: str, task: str) -> str:
        """将技能指引块追加到 prompt 末尾(已含 skill 块时跳过)

        Args:
            prompt: 渲染后的节点提示词
            task: 用户任务描述

        Returns:
            注入技能指引块后的提示词
        """
        block = self.build_skill_block(task)
        return inject_into_prompt(prompt, block)
