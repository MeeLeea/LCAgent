"""
skmng - 技能管理统一包

收敛技能管理职责(原 tools/skills.py + tools/skill_tool.py +
graph/common.py:SkillInjector + team/base.py:PromptInjector 协议)。
本包按重构计划分阶段迁入:
  - Commit 1: manager.py + tool.py 原样迁入,旧行为不变
  - Commit 2a(当前): core.py + protocols.py + injector.py 迁入,
    SkillInjector 改调 core 三来源合并;PromptInjector 协议独立放置
  - 后续 Commit: middleware / ops 逐步迁入,active_names 透传(修行为不一致)
"""
from skmng.core import build_skill_block, inject_into_prompt
from skmng.injector import SkillInjector
from skmng.manager import SkillManager, default_skills_dir
from skmng.protocols import PromptInjector
from skmng.tool import read_skill

__all__ = [
    "PromptInjector",
    "SkillInjector",
    "SkillManager",
    "build_skill_block",
    "default_skills_dir",
    "inject_into_prompt",
    "read_skill",
]
