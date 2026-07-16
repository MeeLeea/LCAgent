"""
技能阅读工具 - 让 Agent 在任务中自行读取本地技能(SKILL.md)的指引

依赖 tools/skills.py 的 SkillManager。
"""
import os
from langchain_core.tools import tool
from typing import Dict, Any

from .skills import SkillManager


def _default_skills_dir() -> str:
    """默认技能目录: <项目根>/.agents/skills"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, ".agents", "skills")


@tool
def read_skill(skill_name: str = "") -> Dict[str, Any]:
    """
    读取本地技能(SKILL.md)的指引内容。

    当任务涉及某个专业领域(如提交 git、生成 pptx、查找技能等)时,
    应先调用本工具获取该技能的详细操作指引,再按指引完成任务。

    用法:
    - 不传 skill_name: 返回所有可用技能的名称与描述,供你判断该用哪个
    - 传入 skill_name: 返回该技能的完整指引正文

    Args:
        skill_name: 技能名称(如 git-commit、pptx、find-skills);留空则列出全部

    Returns:
        包含技能内容或可用列表的字典
    """
    manager = SkillManager(_default_skills_dir())

    if not skill_name or not skill_name.strip():
        skills = manager.list_skills()
        if not skills:
            return {
                "success": True,
                "count": 0,
                "skills": [],
                "message": "当前没有可用技能(目录为空或不存在)",
            }
        return {
            "success": True,
            "count": len(skills),
            "skills": skills,
            "message": "以下是所有可用技能,请选择相关的一个用 read_skill(<name>) 读取其指引",
        }

    name = skill_name.strip()
    content = manager.get_skill(name)
    if content is None:
        available = [s["name"] for s in manager.list_skills()]
        return {
            "success": False,
            "error": f"未找到技能: {name}",
            "available": available,
            "message": f"技能 '{name}' 不存在,可用技能: {', '.join(available) or '(无)'}",
        }

    return {
        "success": True,
        "skill_name": name,
        "content": content,
        "message": f"已读取技能 '{name}' 的指引,请遵循其中的步骤完成任务",
    }
