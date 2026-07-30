import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent.config import load_agent_config, resolve_path, DEFAULTS


def test_defaults_when_missing():
    cfg = load_agent_config("this_file_does_not_exist.json")
    assert cfg["max_iterations"] == DEFAULTS["max_iterations"]
    assert cfg["enable_mcp"] == DEFAULTS["enable_mcp"]
    assert "skills_dir" in cfg


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
