"""skmng.core 单元测试 - 三来源合并去重 / inject_into_prompt 防重复 / auto_match 开关

覆盖 build_skill_block 与 inject_into_prompt 两个纯函数:
- 三来源(fixed_skills + active_names + auto_match)合并去重
- auto_match=False 时仍注入 fixed + active,仅跳过自动匹配
- inject_into_prompt 已含技能指引块时跳过(防重复注入)
- fixed_skills 与 active_names 独立传入,互不影响
"""
import pytest

from skmng.core import build_skill_block, inject_into_prompt
from skmng.manager import SkillManager


def _make_skill_manager(tmp_path) -> SkillManager:
    """构造含 git-helper 与 vivado-2025.2 两技能的 SkillManager"""
    skills_root = tmp_path / "skills"
    # git 技能:关键词 commit/git/提交
    git_dir = skills_root / "git-helper"
    git_dir.mkdir(parents=True)
    (git_dir / "SKILL.md").write_text(
        "---\nname: git-helper\ndescription: commit git 提交 推送\n---\n# git 指引\n提交前先 status",
        encoding="utf-8",
    )
    # vivado 技能:关键词 Vivado/FPGA/综合/比特流
    vivado_dir = skills_root / "vivado"
    vivado_dir.mkdir(parents=True)
    (vivado_dir / "SKILL.md").write_text(
        "---\nname: vivado-2025.2\ndescription: Vivado FPGA 综合 比特流 仿真\n---\n# vivado 指引\n创建工程 TCL",
        encoding="utf-8",
    )
    return SkillManager(str(skills_root))


def test_build_skill_block_auto_match_only(tmp_path):
    """仅 auto_match 通道:任务命中 git,渲染指引块"""
    sm = _make_skill_manager(tmp_path)
    block = build_skill_block(sm, "帮我提交代码", auto_match=True)
    assert "git-helper" in block
    assert "【已加载的技能指引" in block
    assert "提交前先 status" in block


def test_build_skill_block_no_match_returns_empty(tmp_path):
    """任务不命中任何技能时返回空串"""
    sm = _make_skill_manager(tmp_path)
    assert build_skill_block(sm, "算一下 1+1", auto_match=True) == ""


def test_build_skill_block_fixed_skills_always_injected(tmp_path):
    """fixed_skills 不依赖任务关键词,始终注入(角色级固定依赖语义)"""
    sm = _make_skill_manager(tmp_path)
    # 任务不含 Vivado 关键词,纯自动匹配不会命中 vivado
    assert "vivado-2025.2" not in build_skill_block(
        sm, "验证 RTL 功能", auto_match=True
    )
    # 但 fixed_skills 传入时,vivado 始终注入
    block = build_skill_block(
        sm, "验证 RTL 功能", fixed_skills=("vivado-2025.2",), auto_match=True
    )
    assert "vivado-2025.2" in block
    assert "创建工程 TCL" in block


def test_build_skill_block_active_names_injected(tmp_path):
    """active_names 手动加载的技能始终注入,与 auto_match 结果合并"""
    sm = _make_skill_manager(tmp_path)
    # 任务命中 git,active_names 手动加 vivado → 两者都注入
    block = build_skill_block(
        sm,
        "帮我提交代码",
        active_names=("vivado-2025.2",),
        auto_match=True,
    )
    assert "git-helper" in block
    assert "vivado-2025.2" in block


def test_build_skill_block_dedup_three_sources(tmp_path):
    """三来源含重复技能名时去重(同一技能只渲染一次)"""
    sm = _make_skill_manager(tmp_path)
    # 三来源都指向 git-helper,最终只渲染一次
    block = build_skill_block(
        sm,
        "帮我提交代码",  # auto_match 命中 git-helper
        active_names=("git-helper",),  # active 也含 git-helper
        fixed_skills=("git-helper", "vivado-2025.2"),  # fixed 也含 git-helper
        auto_match=True,
    )
    # git-helper 正文只出现一次(去重后 render_block 只渲染一个块)
    assert block.count("### 技能: git-helper") == 1
    assert block.count("### 技能: vivado-2025.2") == 1


def test_build_skill_block_auto_match_false_keeps_fixed_and_active(tmp_path):
    """auto_match=False 时跳过自动匹配,但 fixed + active 仍注入"""
    sm = _make_skill_manager(tmp_path)
    # 任务含 git 关键词,但 auto_match=False → 不自动匹配 git
    block = build_skill_block(
        sm,
        "帮我提交代码",
        active_names=("vivado-2025.2",),
        fixed_skills=("git-helper",),
        auto_match=False,
    )
    assert "git-helper" in block  # fixed 注入
    assert "vivado-2025.2" in block  # active 注入
    # auto_match 关闭,不因任务关键词额外命中 git-helper(已由 fixed 注入,去重无影响)


def test_build_skill_block_empty_when_all_sources_empty(tmp_path):
    """三来源均空时返回空串"""
    sm = _make_skill_manager(tmp_path)
    assert build_skill_block(sm, "", auto_match=True) == ""
    assert build_skill_block(sm, "任意任务", auto_match=False) == ""


def test_inject_into_prompt_appends_block():
    """inject_into_prompt 把 block 追加到 prompt 末尾"""
    prompt = "执行计划"
    block = "【已加载的技能指引(请在处理任务时遵循)】\n### 技能: x\n正文"
    result = inject_into_prompt(prompt, block)
    assert result.startswith(prompt)
    assert block in result


def test_inject_into_prompt_empty_block_noop():
    """block 为空串时原样返回 prompt"""
    prompt = "执行计划"
    assert inject_into_prompt(prompt, "") == prompt


def test_inject_into_prompt_skip_when_already_injected():
    """prompt 已含技能指引块标记时跳过(防重复注入)"""
    block = "【已加载的技能指引(请在处理任务时遵循)】\n### 技能: x\n正文"
    prompt_with_block = f"计划\n\n{block}"
    # 再次注入应原样返回(不重复追加)
    assert inject_into_prompt(prompt_with_block, block) == prompt_with_block


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
