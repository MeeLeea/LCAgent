"""测试团队角色目录发现：role_sw 的可用角色扫描与目录定位。

覆盖两件事：
1. ``_locate_team_agent_dir`` 扫描 team/ 精确定位角色目录，未命中抛 KeyError。
2. ``get_available_team_roles`` 返回统一配置中定义的内置角色。

注：会话级角色切换已改由会话配置实现（CLI ``cli/commands/role.py`` /
HTTP ``api/server.py::_resolve_role_patch``），历史遗留的
``arebuild_agent_from_team_dir``（就地改写共享 AgentCore 的全局路径）已删除，
故本文件不再覆盖切换行为。

断言约定：不硬编码具体 provider/model 名——验证的是"角色目录可被正确定位"
这一行为，而非绑定某个模型（避免默认配置调整导致测试失配）。
"""
import os

import pytest

from agent import role_sw
from agent.role_sw import _locate_team_agent_dir

# ============ 测试：目录定位 ============


def test_locate_team_agent_dir_finds_manager():
    # When: 定位内置 manager 角色
    path = _locate_team_agent_dir("manager")

    # Then: 返回的目录包含必需文件
    assert os.path.isdir(path)
    assert path.endswith("manager")
    assert os.path.isfile(os.path.join(path, "AGENT.md"))
    # 不再检查 agent_config.json，改检查统一配置


def test_locate_team_agent_dir_raises_on_missing():
    # When/Then: 未知角色抛 KeyError，错误信息含可用角色
    with pytest.raises(KeyError) as exc:
        _locate_team_agent_dir("nonexistent_role_xyz")
    assert "nonexistent_role_xyz" in str(exc.value)


def test_locate_team_agent_dir_returns_agent_dir_for_default():
    """default 角色不在 team/ 下，返回 agent/ 目录。"""
    path = _locate_team_agent_dir("default")

    assert os.path.isdir(path)
    assert os.path.basename(path) == "agent"


def test_get_available_team_roles_includes_builtin():
    """get_available_team_roles 返回内置角色"""
    roles = role_sw.get_available_team_roles()
    assert "manager" in roles
    assert "worker" in roles
    assert "terminator" in roles
    assert "architect" in roles
    assert "rtl_designer" in roles
    assert "rtl_verification" in roles
