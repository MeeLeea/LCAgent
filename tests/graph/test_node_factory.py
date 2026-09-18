"""
测试 graph/common.node_factory.create_llm_node 与 llm.config.load_agent_rules

覆盖:
- load_agent_rules: 默认仅返回「重要规则」通用节; include_tool_rules=True 追加「工具规则」
- load_agent_rules: 缺失文件时回退内置默认提示词(含同样小节); 无对应小节时返回空串
- create_llm_node: 按角色能力运行时解析(持有工具才注入「工具规则」)
- create_llm_node: base_prompts="" 关闭注入; 传字符串覆盖默认解析
- prompt 顺序: 基础提示词 → 节点模板 → 技能块
"""
from __future__ import annotations

import asyncio

import pytest

from graph.common import node_factory
from llm.config import load_agent_rules

RULES_HEADING = "## 重要规则"
TOOL_RULES_HEADING = "## 工具规则"


class _FakeAgent:
    """最小 TeamAgent 替身: get_template/render_template 返回固定模板, 可选持有工具。"""

    def __init__(self, template: str = "节点模板", tools: list | None = None) -> None:
        self._template = template
        self.tools = tools if tools is not None else []

    def get_template(self, name: str) -> str:
        return self._template

    @staticmethod
    def render_template(template: str, **kwargs) -> str:
        return template


class _FakeInjector:
    """最小注入器替身: 把技能块追加到 prompt 末尾,并记录收到的 prompt。"""

    def __init__(self) -> None:
        self.received: list[str] = []

    def inject_into_prompt(self, prompt: str, task: str, exclude_skills=()) -> str:
        self.received.append(prompt)
        return f"{prompt}\n\n【技能块】"


def _run_node(
    monkeypatch, agent: object | None = None, **factory_kwargs
) -> list[str]:
    """构造并执行节点, 返回 run_team_turn_with_interrupt 实际收到的 prompt 列表。"""
    prompts: list[str] = []

    async def _fake_run(agent, prompt, config=None) -> str:
        prompts.append(prompt)
        return "节点结果"

    monkeypatch.setattr(node_factory, "run_team_turn_with_interrupt", _fake_run)

    node_fn = node_factory.create_llm_node("tpl", "output", **factory_kwargs)
    result = asyncio.run(node_fn({"task": "任务"}, agent or _FakeAgent(), _FakeInjector()))
    assert result["output"] == "节点结果"
    return prompts


# ──────────────────────────────────────────────
# load_agent_rules: 按角色能力分节
# ──────────────────────────────────────────────


def test_load_agent_rules_default_returns_only_universal_section():
    """默认(无工具)仅返回「重要规则」, 不含「工具规则」。"""
    rules = load_agent_rules()

    assert rules.startswith(RULES_HEADING)
    assert "请用中文回答。" in rules
    assert TOOL_RULES_HEADING not in rules
    assert [line for line in rules.splitlines() if line.startswith("## ")] == [
        RULES_HEADING
    ]


def test_load_agent_rules_with_tool_rules_appends_tool_section():
    """include_tool_rules=True 时追加「工具规则」, 顺序为 重要规则 → 工具规则。"""
    rules = load_agent_rules(include_tool_rules=True)
    headings = [line for line in rules.splitlines() if line.startswith("## ")]

    assert headings == [RULES_HEADING, TOOL_RULES_HEADING]
    # 工具专属条款(定时任务流程)只出现在工具节
    universal, tool = rules.split(TOOL_RULES_HEADING)
    assert "schedule_task" in tool
    assert "schedule_task" not in universal


def test_load_agent_rules_missing_file_falls_back_to_builtin_sections(tmp_path):
    """文件缺失时回退内置默认提示词; 默认提示词与 AGENT.md 同构, 规则小节仍会被提取。"""
    missing = str(tmp_path / "nonexistent" / "AGENT.md")

    universal = load_agent_rules(missing)
    with_tool = load_agent_rules(missing, include_tool_rules=True)

    assert universal.startswith(RULES_HEADING)
    assert TOOL_RULES_HEADING not in universal
    assert TOOL_RULES_HEADING in with_tool


def test_load_agent_rules_without_sections_returns_empty(tmp_path):
    """文件存在但未定义规则小节时返回空串。"""
    prompt_file = tmp_path / "AGENT.md"
    prompt_file.write_text("# 标题\n\n没有规则小节\n", encoding="utf-8")

    assert load_agent_rules(str(prompt_file)) == ""


# ──────────────────────────────────────────────
# create_llm_node: 按角色能力注入
# ──────────────────────────────────────────────


def test_node_omits_tool_rules_for_agent_without_tools(monkeypatch):
    """纯文本角色(tools 为空): 仅注入「重要规则」, 顺序为 规则 → 模板 → 技能块。"""
    prompt = _run_node(monkeypatch, _FakeAgent(tools=[]))[0]

    assert prompt.startswith(RULES_HEADING)
    assert TOOL_RULES_HEADING not in prompt
    assert (
        prompt.index(RULES_HEADING) < prompt.index("节点模板") < prompt.index("【技能块】")
    )


def test_node_includes_tool_rules_for_agent_with_tools(monkeypatch):
    """持有工具的角色: 注入「重要规则」+「工具规则」。"""
    prompt = _run_node(monkeypatch, _FakeAgent(tools=[object()]))[0]

    assert prompt.startswith(RULES_HEADING)
    assert TOOL_RULES_HEADING in prompt
    assert prompt.index(RULES_HEADING) < prompt.index(TOOL_RULES_HEADING)


def test_node_tolerates_agent_without_tools_attribute(monkeypatch):
    """Agent 无 tools 属性时按无工具处理(兼容测试替身 / object.__new__)。"""

    class _Bare:
        def get_template(self, name: str) -> str:
            return "节点模板"

        @staticmethod
        def render_template(template: str, **kwargs) -> str:
            return template

    prompt = _run_node(monkeypatch, _Bare())[0]

    assert prompt.startswith(RULES_HEADING)
    assert TOOL_RULES_HEADING not in prompt


def test_create_llm_node_empty_base_prompts_disables_injection(monkeypatch):
    """base_prompts="" 时不注入基础提示词。"""
    prompt = _run_node(monkeypatch, base_prompts="")[0]

    assert prompt.startswith("节点模板")
    assert RULES_HEADING not in prompt


def test_create_llm_node_custom_base_prompts_overrides_default(monkeypatch):
    """base_prompts 传字符串时覆盖默认解析结果。"""
    prompt = _run_node(monkeypatch, base_prompts="自定义基础规则")[0]

    assert prompt.startswith("自定义基础规则\n\n节点模板")
    assert RULES_HEADING not in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
