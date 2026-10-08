import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from llm.config import (
    _DEFAULT_AGENT_CORE_PROMPT,
    AGENT_RULES_HEADING,
    AGENT_TOOL_RULES_HEADING,
    DEFAULTS,
    _load_agent_prompt,
    compose_role_system_prompt,
    load_agent_config,
    load_team_agent_role_entry,
    resolve_path,
)


def test_defaults_when_missing():
    cfg = load_agent_config("this_file_does_not_exist.json")
    assert cfg["max_iterations"] == DEFAULTS["max_iterations"]
    assert cfg["enable_mcp"] == DEFAULTS["enable_mcp"]
    assert cfg["name"] == DEFAULTS["name"]
    assert "skills_dir" in cfg


def test_name_in_defaults():
    """测试 name 字段有默认值且为非空字符串"""
    assert isinstance(DEFAULTS["name"], str)
    assert len(DEFAULTS["name"]) > 0


def test_load_custom_name(tmp_path):
    """测试配置文件中的 name 字段覆盖默认值"""
    p = tmp_path / "agent_config.json"
    p.write_text('{"name": "MyAgent"}', encoding="utf-8")
    cfg = load_agent_config(str(p))
    assert cfg["name"] == "MyAgent"


def test_load_real(tmp_path):
    p = tmp_path / "agent_config.json"
    p.write_text('{"max_iterations": 7, "enable_mcp": false}', encoding="utf-8")
    cfg = load_agent_config(str(p))
    assert cfg["max_iterations"] == 7
    assert cfg["enable_mcp"] is False
    # 未出现的键仍取默认
    assert cfg["verbose"] is True


def test_sampling_params_passthrough(tmp_path):
    """temperature/max_tokens 经 agent_config.json 透传（DEFAULTS 不含这两键，未配置时为 None）"""
    p = tmp_path / "agent_config.json"
    p.write_text('{"temperature": 0.3, "max_tokens": 4096}', encoding="utf-8")
    cfg = load_agent_config(str(p))
    assert cfg["temperature"] == 0.3
    assert cfg["max_tokens"] == 4096


def test_sampling_params_default_from_defaults():
    """未配置采样参数时，由 DEFAULTS 兜底（0.7/8192），供 LLMClient 内部读取"""
    cfg = load_agent_config("this_file_does_not_exist.json")
    assert cfg["temperature"] == 0.7
    assert cfg["max_tokens"] == 8192
    assert DEFAULTS["temperature"] == 0.7
    assert DEFAULTS["max_tokens"] == 8192


def test_load_global_config_has_sampling_params():
    """真实全局 agent/agent_config.json 应包含采样参数（供 main/scheduler/api 使用）"""
    import os

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cfg = load_agent_config(os.path.join(root, "agent", "agent_config.json"))
    assert isinstance(cfg.get("temperature"), float)
    assert isinstance(cfg.get("max_tokens"), int)


def test_resolve_path_absolute():
    assert resolve_path("C:\\x", "D:\\base") == "C:\\x"


def test_resolve_path_relative():
    assert resolve_path(".agents", "D:\\base") == os.path.join("D:\\base", ".agents")


def test_agent_core_prompt_in_defaults():
    """测试 agent_core_prompt 默认值存在"""
    assert isinstance(_DEFAULT_AGENT_CORE_PROMPT, str)
    assert len(_DEFAULT_AGENT_CORE_PROMPT) > 0


def test_load_prompt_from_agent_md(tmp_path):
    """测试从 AGENT.md 加载自定义 prompt"""
    prompt_file = tmp_path / "AGENT.md"
    custom_prompt = "# 这是一个自定义的提示词\n\n请按照规则执行任务。"
    prompt_file.write_text(custom_prompt, encoding="utf-8")

    result = _load_agent_prompt(str(prompt_file))
    assert result == custom_prompt


def test_prompt_fallback_to_default(tmp_path):
    """测试配置文件中没有指定 AGENT.md 或文件不存在时使用默认值"""
    result = _load_agent_prompt(str(tmp_path / "nonexistent/AGENT.md"))
    assert result == _DEFAULT_AGENT_CORE_PROMPT


def test_prompt_file_not_exists(tmp_path):
    """测试 AGENT.md 文件不存在时回退到默认值"""
    result = _load_agent_prompt(str(tmp_path / "nonexistent/AGENT.md"))
    assert result == _DEFAULT_AGENT_CORE_PROMPT


def test_prompt_file_empty(tmp_path):
    """测试 AGENT.md 文件为空时回退到默认值"""
    prompt_file = tmp_path / "AGENT.md"
    prompt_file.write_text("", encoding="utf-8")

    result = _load_agent_prompt(str(prompt_file))
    assert result == _DEFAULT_AGENT_CORE_PROMPT


# ──────────────────────────────────────────────
# compose_role_system_prompt: 角色提示词拼接基础规则
# ──────────────────────────────────────────────

ROLE_PROMPT = "# Agent 核心提示词\n你是一个芯片架构工程师（Architect）。"


def test_compose_role_prompt_with_tools_keeps_both_rule_sections():
    """持有工具的角色：基础规则两节 + 角色提示词都保留（修复"切角色丢通用规则"缺陷）。"""
    composed = compose_role_system_prompt(ROLE_PROMPT, role="architect", has_tools=True)
    headings = [ln for ln in composed.splitlines() if ln.startswith("## ")]

    assert headings == [AGENT_RULES_HEADING, AGENT_TOOL_RULES_HEADING]
    assert "请用中文回答。" in composed
    assert "schedule_task" in composed  # 工具规则专属条款
    assert "芯片架构工程师" in composed  # 角色提示词未被丢弃


def test_compose_role_prompt_without_tools_omits_tool_section():
    """无工具角色：仅注入「重要规则」，不注入「工具规则」（避免误导模型调用不存在的工具）。"""
    composed = compose_role_system_prompt(ROLE_PROMPT, role="manager", has_tools=False)
    headings = [ln for ln in composed.splitlines() if ln.startswith("## ")]

    assert headings == [AGENT_RULES_HEADING]
    assert AGENT_TOOL_RULES_HEADING not in composed
    assert "请用中文回答。" in composed
    assert "芯片架构工程师" in composed


def test_compose_role_prompt_puts_base_rules_before_role_prompt():
    """顺序固定为基础规则 → 角色提示词（缓存前缀稳定，避免逐轮抖动）。"""
    composed = compose_role_system_prompt(ROLE_PROMPT, role="architect")

    assert composed.index(AGENT_RULES_HEADING) < composed.index("芯片架构工程师")


def test_compose_role_prompt_default_role_is_not_concatenated():
    """role="default" 的提示词来源就是 agent/AGENT.md，拼接会导致规则小节重复。"""
    agent_md_like = f"{AGENT_RULES_HEADING}\n1. 规则\n\n{AGENT_TOOL_RULES_HEADING}\n1. 工具"
    composed = compose_role_system_prompt(agent_md_like, role="default")

    assert composed == agent_md_like
    assert composed.count(AGENT_RULES_HEADING) == 1
    assert composed.count(AGENT_TOOL_RULES_HEADING) == 1


def test_compose_role_prompt_empty_role_prompt_falls_back_to_rules():
    """角色提示词为空时回退为纯基础规则，不产生多余空白。"""
    composed = compose_role_system_prompt("", role="architect", has_tools=True)

    assert composed.startswith(AGENT_RULES_HEADING)
    assert composed == composed.strip()


# ──────────────────────────────────────────────
# max_context_tokens: token 阈值取代消息数阈值
# ──────────────────────────────────────────────


def test_defaults_use_max_context_tokens_not_messages():
    """DEFAULTS 暴露 max_context_tokens（默认 100000），不再有 max_context_messages。"""
    assert DEFAULTS["max_context_tokens"] == 100_000
    assert "max_context_messages" not in DEFAULTS


def test_global_agent_config_uses_max_context_tokens():
    """真实 agent/agent_config.json 使用 max_context_tokens，不再有 max_context_messages。"""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cfg = load_agent_config(os.path.join(root, "agent", "agent_config.json"))

    assert cfg["max_context_tokens"] == 100_000
    assert "max_context_messages" not in cfg


# ──────────────────────────────────────────────
# load_team_agent_role_entry: 角色自身条目（不与 default 合并）
# ──────────────────────────────────────────────


def test_load_team_agent_role_entry_returns_role_own_entry():
    """worker 自身条目声明 temperature=0.3，应原样返回该值。"""
    entry = load_team_agent_role_entry("worker", ROOT)

    assert entry["temperature"] == 0.3


def test_load_team_agent_role_entry_does_not_merge_default():
    """default 声明了 temperature=0.7，但 architect 自身未声明 → 结果不含 temperature 键。"""
    entry = load_team_agent_role_entry("architect", ROOT)

    assert "temperature" not in entry

