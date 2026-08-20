"""
skmng - 技能管理统一包

收敛原有 tools/skills.py + tools/skill_tool.py 的技能管理职责。
本包按重构计划分阶段迁入：
  - Commit 1（当前）：manager.py + tool.py 原样迁入，旧行为不变
  - 后续 Commit：injector / middleware / ops / protocols / core 逐步迁入
"""
from skmng.manager import SkillManager, default_skills_dir
from skmng.tool import read_skill

__all__ = ["SkillManager", "default_skills_dir", "read_skill"]
