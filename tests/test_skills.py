import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.skills import SkillManager

SKILLS_DIR = os.path.join(ROOT, ".agents", "skills")


def _sm():
    return SkillManager(SKILLS_DIR)


def test_list_skills():
    names = [s["name"] for s in _sm().list_skills()]
    assert {"git-commit", "pptx", "find-skills"}.issubset(set(names))


def test_get_skill():
    content = _sm().get_skill("git-commit")
    assert content and "Git Commit" in content


def test_get_skill_missing():
    assert _sm().get_skill("nope") is None


def test_match_chinese_git():
    assert "git-commit" in _sm().match_skills("帮我把改动提交一下，写个提交信息")


def test_match_chinese_pptx():
    assert "pptx" in _sm().match_skills("做一个 pptx 演示文稿")


def test_match_none():
    assert _sm().match_skills("今天天气怎么样") == []


def test_render_block():
    block = _sm().render_block(["git-commit"])
    assert "技能" in block and "git-commit" in block
