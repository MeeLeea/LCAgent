"""
技能注入核心 - 统一三来源合并的技能指引块构建与 prompt 注入

三来源(在 build_skill_block 内部合并去重):
1. fixed_skills  - 角色级固定依赖(如 rtl_verification 的 vivado-2025.2),由类属性提供
2. active_names  - 运行时手动加载的技能(经 state["active_skills"] 传入,Commit 4 透传)
3. match_skills  - 任务文本自动匹配(确定性关键词打分,不调用 LLM)

设计要点:
- build_skill_block 产出 block 字符串,inject_into_prompt 负责追加(职责分离,
  便于 Commit 4 在 SkillInjector.inject_into_prompt 加 active_names 透传)
- inject_into_prompt 按 "【已加载的技能指引" 标记防重复注入
- SkillInjector 是 build 时 partial 绑定的单例,运行时无 state 通道;
  active_names 必须由节点函数从 state 取值显式传入(绝不在 injector 内缓存)
"""
from __future__ import annotations

from skmng.manager import SkillManager


def build_skill_block(
    sm: SkillManager,
    task: str,
    active_names: tuple[str, ...] = (),
    fixed_skills: tuple[str, ...] = (),
    auto_match: bool = True,
) -> str:
    """合并三来源技能并渲染为可注入 system prompt 的指引块

    合并去重顺序: fixed_skills + active_names + match_skills(后者仅 auto_match=True
    且 task 非空时参与),最终经 sorted 去重后交由 SkillManager.render_block 渲染。

    Args:
        sm: 技能管理器实例(持有 skills_dir)
        task: 用户任务描述(仅用于 auto_match 匹配打分)
        active_names: 运行时手动加载的技能名(由节点函数从 state 取值传入)
        fixed_skills: 角色级固定依赖技能名(由 TeamAgent 类属性提供)
        auto_match: 是否开启任务自动匹配(False 时仅注入 fixed + active)

    Returns:
        技能指引块文本;三来源均空时返回空串
    """
    # 收集三来源,保持插入序去重(后插入的同名技能不重复渲染)
    seen: set[str] = set()
    names: list[str] = []
    for src in (fixed_skills, active_names):
        for n in src:
            if n and n not in seen:
                seen.add(n)
                names.append(n)

    # 自动匹配(auto_match=False 或 task 为空时跳过,不影响 fixed + active)
    if auto_match and task and task.strip():
        for n in sm.match_skills(task):
            if n not in seen:
                seen.add(n)
                names.append(n)

    if not names:
        return ""
    return sm.render_block(sorted(names))


def inject_into_prompt(prompt: str, block: str) -> str:
    """把技能指引块追加到 prompt 末尾(已含 skill 块时跳过,防重复)

    Args:
        prompt: 渲染后的节点提示词
        block: 经 build_skill_block 计算好的技能指引块(空串则原样返回)

    Returns:
        注入技能指引块后的提示词
    """
    if not block:
        return prompt
    if "【已加载的技能指引" in prompt:
        return prompt
    return f"{prompt}\n\n{block}"
