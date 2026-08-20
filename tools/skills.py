"""
技能阅读管理器 - 旧路径兼容

本模块已迁移到 skmng.manager,此处仅做 re-export 以保持向后兼容。
36 个文件通过 `from tools.skills import ...` 引用本模块,改动将在 Commit 5 统一删除。
"""
from skmng.manager import SkillManager, default_skills_dir

__all__ = ["SkillManager", "default_skills_dir"]
