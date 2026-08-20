"""
工作流技能注入器 - build 时构造,节点 prompt 层注入技能指引块

不走 AgentMiddleware(那是 create_agent 内部机制),而是复用
skmng.manager.SkillManager 的确定性打分匹配 + skmng.core 的三来源合并:

1. core.build_skill_block(sm, task, active_names, fixed_skills, auto_match)
   合并"角色固定(fixed_skills)" + "手动加载(active_names)" + "自动匹配(match_skills)"
2. core.inject_into_prompt(prompt, block) 把指引块追加到 prompt 末尾(防重复)

工作流节点渲染 prompt 后,调用 injector.inject_into_prompt(prompt, task, active_names)
即可完成注入。SkillInjector 是 build 时 partial 绑定的单例,运行时无 state
通道 —— active_names 必须由节点函数从 state.get("active_skills") 取值
显式传入(节点函数内部 state 通道可达,injector 不可缓存以免跨会话污染)。

Args:
    skills_dir: 技能目录路径;为 None 时使用默认目录(<项目根>/.agents/skills)
    auto_match: 是否开启自动匹配(False 时仅注入 fixed + active,不走 match)
"""
from __future__ import annotations

from collections.abc import Sequence

from skmng.core import build_skill_block, inject_into_prompt
from skmng.manager import SkillManager, default_skills_dir


class SkillInjector:
    """工作流技能注入器 - 节点 prompt 层注入技能指引块

    inject_into_prompt(prompt, task, active_names) 三参数:
    active_names 由节点函数从 state.get("active_skills") 取值传入,
    使手动加载的技能在 graph 节点生效(修复原行为不一致)。
    SkillInjector 绝不缓存 active_names(跨会话污染)。
    """

    def __init__(
        self,
        skills_dir: str | None = None,
        auto_match: bool = True,
    ) -> None:
        self.skill_manager = SkillManager(skills_dir or default_skills_dir())
        self.auto_match = auto_match

    def build_skill_block(
        self,
        task: str,
        active_names: Sequence[str] = (),
    ) -> str:
        """根据任务匹配技能并渲染指引块(合并 active_names + auto_match)

        Args:
            task: 用户任务描述(用于技能匹配)
            active_names: 手动加载的技能名(由节点函数从 state 取值传入)

        Returns:
            技能指引块文本;未命中任何技能或未开启自动匹配时返回空串
        """
        return build_skill_block(
            self.skill_manager,
            task,
            active_names=tuple(active_names),
            fixed_skills=(),
            auto_match=self.auto_match,
        )

    def inject_into_prompt(
        self,
        prompt: str,
        task: str,
        active_names: Sequence[str] = (),
    ) -> str:
        """将技能指引块追加到 prompt 末尾(已含 skill 块时跳过)

        Args:
            prompt: 渲染后的节点提示词
            task: 用户任务描述
            active_names: 手动加载的技能名(由节点函数从 state 取值传入)

        Returns:
            注入技能指引块后的提示词
        """
        block = self.build_skill_block(task, active_names)
        return inject_into_prompt(prompt, block)
