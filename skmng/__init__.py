"""
skmng - 技能管理统一包

收敛技能管理职责(原 tools/skills.py + tools/skill_tool.py +
graph/common.py:SkillInjector + team/base.py:PromptInjector 协议 +
agent/skill_mw.py:SkillInjectionMW + agent/skill_ops.py:SkillOps)。
本包按重构计划分阶段迁入:
  - Commit 1: manager.py + tool.py 原样迁入,旧行为不变
  - Commit 2a: core.py + protocols.py + injector.py 迁入,SkillInjector 改调 core
  - Commit 2b: TeamAgent 转发 core + rtl_verification 改 fixed_skills
  - Commit 3(当前): middleware.py + ops.py 迁入,
    SkillInjectionMW 改调 core 三来源合并;SkillOps 全量迁入(含 manually_compact)
  - 后续 Commit: active_names 透传(修行为不一致) + 收尾删旧 re-export
"""
from skmng.core import build_skill_block, inject_into_prompt
from skmng.injector import SkillInjector
from skmng.manager import SkillManager, default_skills_dir
from skmng.middleware import SkillInjectionMW
from skmng.ops import SkillOps
from skmng.protocols import PromptInjector
from skmng.tool import read_skill

__all__ = [
    "PromptInjector",
    "SkillInjectionMW",
    "SkillInjector",
    "SkillManager",
    "SkillOps",
    "build_skill_block",
    "default_skills_dir",
    "inject_into_prompt",
    "read_skill",
]
