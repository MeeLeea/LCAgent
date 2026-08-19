"""TeamAgent 技能注入内建化测试 - 验证 build_skill_block / inject_into_prompt

TeamAgent 现在内建 SkillManager(复用 tools.skills),自身满足 PromptInjector
协议(graph.common.SkillInjector 兼容),工作流节点不再依赖外部注入器:
- build_skill_block: 任务自动匹配技能并渲染指引块
- inject_into_prompt: 把指引块追加到 prompt 末尾(防重复)
"""
from team.base import PromptInjector, TeamAgent


class _FakeLLM:
    """替换真实 LLMClient,避免测试依赖 API 密钥与网络"""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def get_chat_model(self):
        return None

    def chat(self, messages) -> str:
        return ""


def _monkeypatch_llm(monkeypatch) -> None:
    """把 team.base 中的 LLMClient 替换为假实现"""
    monkeypatch.setattr("team.base.LLMClient", _FakeLLM)


def _make_skill_dir(tmp_path):
    """在临时目录构造一个 git 技能,返回技能目录路径"""
    skill_dir = tmp_path / "skills" / "git-helper"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: git-helper\n"
        "description: 处理 git 提交与推送操作\n"
        "---\n"
        "技能正文:提交前先 git status 检查。\n",
        encoding="utf-8",
    )
    return tmp_path / "skills"


def test_skill_manager_inited_with_default_dir(monkeypatch):
    """默认初始化 skill_manager(无 skills_dir 参数时指向默认目录)"""
    _monkeypatch_llm(monkeypatch)
    agent = TeamAgent(name="test")
    assert agent.skill_manager is not None
    assert agent.auto_match_skills is True
    # 默认目录存在时返回技能列表,不存在时静默返回空列表
    assert isinstance(agent.skill_manager.list_skills(), list)


def test_build_skill_block_matches_task(monkeypatch, tmp_path):
    """任务含技能关键词时,自动匹配并渲染指引块"""
    _monkeypatch_llm(monkeypatch)
    skills_dir = _make_skill_dir(tmp_path)
    agent = TeamAgent(name="test", skills_dir=str(skills_dir))

    block = agent.build_skill_block("帮我提交并推送代码")
    assert "git-helper" in block
    assert "【已加载的技能指引" in block
    assert "提交前先 git status" in block

    # 不相关任务不命中
    assert agent.build_skill_block("帮我算一下 1+1") == ""


def test_inject_into_prompt_appends_skill_block(monkeypatch, tmp_path):
    """inject_into_prompt 把指引块追加到 prompt 末尾,且不会重复注入"""
    _monkeypatch_llm(monkeypatch)
    skills_dir = _make_skill_dir(tmp_path)
    agent = TeamAgent(name="test", skills_dir=str(skills_dir))

    prompt = "执行计划:{plan}"
    injected = agent.inject_into_prompt(prompt, "提交代码")
    assert injected.startswith(prompt)
    assert "【已加载的技能指引" in injected

    # 二次注入跳过(已含标记)
    again = agent.inject_into_prompt(injected, "提交代码")
    assert again == injected


def test_auto_match_disabled_returns_empty(monkeypatch, tmp_path):
    """auto_match_skills=False 时 build_skill_block 恒为空串"""
    _monkeypatch_llm(monkeypatch)
    skills_dir = _make_skill_dir(tmp_path)
    agent = TeamAgent(
        name="test", skills_dir=str(skills_dir), auto_match_skills=False
    )

    assert agent.build_skill_block("提交代码") == ""
    assert agent.inject_into_prompt("计划", "提交代码") == "计划"


def test_team_agent_satisfies_prompt_injector_protocol(monkeypatch, tmp_path):
    """TeamAgent 实例自身可作为 PromptInjector 传给节点(协议兼容)"""
    _monkeypatch_llm(monkeypatch)
    skills_dir = _make_skill_dir(tmp_path)
    agent = TeamAgent(name="test", skills_dir=str(skills_dir))

    injector: PromptInjector = agent  # 类型层面即满足协议
    prompt = injector.inject_into_prompt("执行:{plan}", "提交代码")
    assert "【已加载的技能指引" in prompt