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


# ============ 测试：共享角色配置解析（CLI / API 唯一实现） ============


def test_resolve_role_config_patch_builds_role_and_prompt():
    # When: 解析内置 manager 角色
    patch = role_sw.resolve_role_config_patch("manager")

    # Then: 产出含 role 与非空 system_prompt 的补丁
    assert patch.role == "manager"
    assert patch.system_prompt
    # 基础规则应已拼接进角色提示词（manager 无工具，仅「重要规则」）
    assert "重要规则" in patch.system_prompt


def test_resolve_role_config_patch_default_role_keeps_prompt_unconcatenated():
    # When: 解析 default 角色（提示词来源即 agent/AGENT.md，不应重复拼接）
    patch = role_sw.resolve_role_config_patch("default")

    # Then: role 为 default，system_prompt 非空
    assert patch.role == "default"
    assert patch.system_prompt


def test_resolve_role_config_patch_raises_for_unknown_role():
    with pytest.raises(KeyError):
        role_sw.resolve_role_config_patch("nonexistent_role_xyz")


def test_resolve_role_config_patch_explicit_fields_take_precedence():
    """显式字段（如 temperature/system_prompt）优先于角色配置。"""
    from session.config import SessionConfigPatch

    patch = role_sw.resolve_role_config_patch(
        "manager",
        explicit=SessionConfigPatch(system_prompt="自定义提示词", temperature=0.123),
    )
    assert patch.role == "manager"
    assert patch.system_prompt == "自定义提示词"
    assert patch.temperature == 0.123


def test_resolve_role_config_patch_same_across_cli_and_api_entry():
    """CLI 与 API 对同一角色解析出的补丁必须一致（防两入口漂移）。"""
    from session.config import SessionConfigPatch

    # API 路径（带显式补丁）与 CLI 路径（无显式补丁）对纯角色切换应等价
    from_api = role_sw.resolve_role_config_patch("worker", explicit=SessionConfigPatch(role="worker"))
    from_cli = role_sw.resolve_role_config_patch("worker")
    assert from_api == from_cli
