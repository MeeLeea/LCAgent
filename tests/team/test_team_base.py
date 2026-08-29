"""TeamAgent 基类测试 - 覆盖 system_prompt 自动解析与工作流模板缓存

重点验证「只需 prompt_file」的初始化语义:
- system_prompt 为空时自动从 AGENT.md 解析(剥离 ## workflow:* 小节)
- 显式传入的 system_prompt 优先
- __init__ 解析出的模板缓存被 get_template 复用,不重复读文件
"""
from typing import ClassVar

from team.base import TeamAgent


class _FakeLLM:
    """替换真实 LLMClient,避免测试依赖 API 密钥与网络"""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def get_chat_model(self):
        return None

    def chat(self, messages) -> str:
        return ""


class _FallbackAgent(TeamAgent):
    """带默认工作流模板的子类,验证缺失小节时回退"""

    default_templates: ClassVar[dict[str, str]] = {"manager_plan": "默认计划模板"}


def _monkeypatch_llm(monkeypatch) -> None:
    """把 team.base 中的 LLMClient 替换为假实现"""
    monkeypatch.setattr("team.base.LLMClient", _FakeLLM)


def test_auto_parse_system_prompt_from_prompt_file(tmp_path, monkeypatch):
    """只传 prompt_file 时,system_prompt 自动解析且剥离 workflow 小节"""
    _monkeypatch_llm(monkeypatch)

    agent_md = tmp_path / "AGENT.md"
    agent_md.write_text(
        "# 角色提示词\n"
        "你是测试经理。\n\n"
        "## 重要规则\n"
        "1. 规则A\n\n"
        "## workflow:manager_plan\n"
        "请制定计划:{task}\n",
        encoding="utf-8",
    )

    agent = TeamAgent(name="test", prompt_file=str(agent_md))

    assert "你是测试经理" in agent.system_prompt
    assert "## 重要规则" in agent.system_prompt
    # 模板小节不应混入系统提示词
    assert "workflow:" not in agent.system_prompt
    assert "请制定计划" not in agent.system_prompt
    # 模板可正常读取
    assert agent.get_template("manager_plan") == "请制定计划:{task}"


def test_explicit_system_prompt_wins(tmp_path, monkeypatch):
    """显式传入 system_prompt 时,不覆盖为 prompt_file 内容"""
    _monkeypatch_llm(monkeypatch)

    agent_md = tmp_path / "AGENT.md"
    agent_md.write_text("# 文件提示词\n", encoding="utf-8")

    agent = TeamAgent(
        name="test",
        system_prompt="显式提示词",
        prompt_file=str(agent_md),
    )

    assert agent.system_prompt == "显式提示词"


def test_missing_prompt_file_falls_back(tmp_path, monkeypatch):
    """prompt_file 不存在时 system_prompt 为空,get_template 回退默认模板"""
    _monkeypatch_llm(monkeypatch)

    agent = _FallbackAgent(
        name="test",
        prompt_file=str(tmp_path / "not_exists.md"),
    )

    assert agent.system_prompt == ""
    assert agent.get_template("manager_plan") == "默认计划模板"


def test_template_cache_reuses_init_parse(tmp_path, monkeypatch):
    """__init__ 已解析模板时,get_template 复用缓存,不重复读文件"""
    _monkeypatch_llm(monkeypatch)

    agent_md = tmp_path / "AGENT.md"
    agent_md.write_text(
        "# 提示词\n\n"
        "## workflow:manager_plan\n"
        "计划:{task}\n",
        encoding="utf-8",
    )

    agent = TeamAgent(name="test", prompt_file=str(agent_md))
    # 预热:确认模板已解析
    assert agent.get_template("manager_plan") == "计划:{task}"

    # 删除源文件后模板仍可读取 → 证明走的是 __init__ 建立的缓存而非重新读盘
    agent_md.unlink()
    assert agent.get_template("manager_plan") == "计划:{task}"
