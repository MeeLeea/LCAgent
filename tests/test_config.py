import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent.config import load_agent_config, resolve_path, DEFAULTS, _DEFAULT_AGENT_CORE_PROMPT


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
    # 创建配置文件
    config_file = tmp_path / "agent_config.json"
    prompt_file = tmp_path / "AGENT.md"
    
    custom_prompt = "# 这是一个自定义的提示词\n\n请按照规则执行任务。"
    prompt_file.write_text(custom_prompt, encoding="utf-8")
    
    config_file.write_text(
        '{"agent_prompt_file": "' + str(prompt_file).replace('\\', '\\\\') + '"}',
        encoding="utf-8"
    )
    
    cfg = load_agent_config(str(config_file))
    assert cfg["agent_core_prompt"] == custom_prompt


def test_prompt_fallback_to_default(tmp_path):
    """测试配置文件中没有指定 AGENT.md 或文件不存在时使用默认值"""
    p = tmp_path / "agent_config.json"
    p.write_text(
        '{"max_iterations": 7, "agent_prompt_file": "nonexistent/AGENT.md"}',
        encoding="utf-8"
    )
    cfg = load_agent_config(str(p))
    # 应该回退到默认 prompt
    assert cfg["agent_core_prompt"] == _DEFAULT_AGENT_CORE_PROMPT


def test_prompt_file_not_exists(tmp_path):
    """测试 AGENT.md 文件不存在时回退到默认值"""
    config_file = tmp_path / "agent_config.json"
    config_file.write_text(
        '{"agent_prompt_file": "nonexistent/AGENT.md"}',
        encoding="utf-8"
    )
    
    cfg = load_agent_config(str(config_file))
    assert cfg["agent_core_prompt"] == _DEFAULT_AGENT_CORE_PROMPT


def test_prompt_file_empty(tmp_path):
    """测试 AGENT.md 文件为空时回退到默认值"""
    config_file = tmp_path / "agent_config.json"
    prompt_file = tmp_path / "AGENT.md"
    
    # 创建空文件
    prompt_file.write_text("", encoding="utf-8")
    
    config_file.write_text(
        '{"agent_prompt_file": "' + str(prompt_file).replace('\\', '\\\\') + '"}',
        encoding="utf-8"
    )
    
    cfg = load_agent_config(str(config_file))
    assert cfg["agent_core_prompt"] == _DEFAULT_AGENT_CORE_PROMPT
