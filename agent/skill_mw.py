"""技能注入中间件 - 旧路径兼容

本模块已迁移到 skmng.middleware,此处仅做 re-export 以保持向后兼容。
"""
from skmng.middleware import SkillInjectionMW

__all__ = ["SkillInjectionMW"]
