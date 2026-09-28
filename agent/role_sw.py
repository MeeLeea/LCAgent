"""团队角色目录发现 + 角色配置解析 - 扫描 team/<角色>/ 定位角色并合成配置补丁。

对外提供(供 CLI / API 委托调用):
    - get_available_team_roles: 扫描 team/ 列出可用角色
    - _locate_team_agent_dir: 扫描 team/ 精确定位角色目录
    - resolve_role_config_patch: 角色的唯一解析入口：定位目录→读配置→读 AGENT.md
      →剥离 workflow 小节→拼基础规则→合并 provider/model/采样参数，产出
      ``SessionConfigPatch``

从 agent_core.py 抽离,避免核心调度模块承载角色目录扫描逻辑。

**角色切换的写入口**仍只有两处(会话级角色切换由会话配置实现,写入
``SessionConfig.role`` / ``SessionConfig.system_prompt``,经
``agent/session_config_middleware.py::SessionConfigMW`` 在每次 model 调用时生效):
    - CLI:  ``cli/commands/role.py::_switch_role``
    - HTTP: ``api/server.py::_resolve_role_patch``

但这两处**不再各自复制解析逻辑**,统一委托 ``resolve_role_config_patch``,避免
"同一角色在两入口解析结果漂移"的历史缺陷。

历史遗留的 ``arebuild_agent_from_team_dir`` 已删除:它就地改写**共享** AgentCore
实例(``agent_core_prompt`` / ``llm`` + 重建共享图),会篡改**所有**会话,与
per-session 隔离模型冲突。
"""
from __future__ import annotations

import json
import os
from typing import Any

from session.config import SessionConfigPatch

# 项目根目录(基于本文件位置计算: agent/role_sw.py -> 上两级)
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 默认 agent/ 目录
_DEFAULT_AGENT_DIR = os.path.join(_BASE_DIR, "agent")
# 多 Agent 角色目录
_TEAM_DIR = os.path.join(_BASE_DIR, "team")
# team/ 下的非角色目录(基础设施,跳过)
_NON_ROLE_DIRS = frozenset({"__pycache__"})

def get_available_team_roles() -> list[str]:
    """扫描 team/ 目录,列出可用角色文件夹名"""
    if not os.path.isdir(_TEAM_DIR):
        return []

    available: list[str] = ["default"]
    
    # 从统一配置获取定义的角色
    unified_config_path = os.path.join(_BASE_DIR, "team", "team_agents.json")
    unified_roles: set[str] = set()
    if os.path.exists(unified_config_path):
        try:
            with open(unified_config_path, "r", encoding="utf-8") as f:
                team_config = json.load(f)
            unified_roles = set(team_config.keys()) - {"default"}
        except (OSError, json.JSONDecodeError):
            pass
    
    for entry in sorted(os.listdir(_TEAM_DIR)):
        if entry in _NON_ROLE_DIRS:
            continue
        sub_dir = os.path.join(_TEAM_DIR, entry)
        if not os.path.isdir(sub_dir):
            continue
        # 合法角色：在统一配置中定义，且有 AGENT.md
        has_unified = entry in unified_roles
        has_prompt = os.path.isfile(os.path.join(sub_dir, "AGENT.md"))
        # 兼容：过渡期保留 agent_config.json 检查
        has_config = os.path.isfile(os.path.join(sub_dir, "agent_config.json"))
        if (has_unified or has_config) and has_prompt:
            available.append(entry)

    return available

def _locate_team_agent_dir(agent_name: str) -> str:
    """扫描 team/ 目录,按文件夹名定位目标角色目录

    实现方式参考 graph/registry.py::_load_builtin_workflows():
    遍历 team/ 下的子目录,按 folder name 精确匹配用户输入的角色名。
    命中的目录必须同时包含 agent_config.json 与 AGENT.md 才视为合法角色。

    Args:
        agent_name: team/ 下的角色文件夹名(如 "manager"/"worker")

    Returns:
        角色目录的绝对路径

    Raises:
        KeyError: team/ 不存在、目标文件夹缺失,或缺少必需的配置/提示词文件
    """
    if not os.path.isdir(_TEAM_DIR):
        raise KeyError(f"team 目录不存在: {_TEAM_DIR}")

    available = get_available_team_roles()
    if agent_name not in available:
        roles = ", ".join(available) or "(空)"
        raise KeyError(f"未找到 team 角色: {agent_name}。可用角色: {roles}")

    if agent_name == "default":
        # 默认角色不在 team/ 下,直接返回空路径
        return _DEFAULT_AGENT_DIR
    # 直接拼接路径，不用再次扫描磁盘
    agent_dir = os.path.join(_TEAM_DIR, agent_name)
    return agent_dir


#: 角色配置可覆盖的采样类字段（provider/model 之外的写入字段）。
_ROLE_PATCH_FIELDS = ("provider", "model", "temperature", "max_tokens", "max_iterations")


def resolve_role_config_patch(
    role_name: str,
    *,
    base_dir: str = _BASE_DIR,
    explicit: SessionConfigPatch | None = None,
) -> SessionConfigPatch:
    """解析角色并合成为该会话的配置补丁（CLI / API 共用唯一实现）。

    完整流程：定位角色目录 → 加载 ``team/team_agents.json`` 角色配置 →
    读取 ``AGENT.md`` → 剥离 ``## workflow:*`` 小节 → 拼接基础规则得到 system prompt
    → 合并 provider/model/采样参数，产出 ``SessionConfigPatch``。

    字段优先级（``explicit`` 非 None 时）：显式请求字段 > 角色配置字段 > 不修改。
    ``explicit=None``（CLI 路径）时：角色配置中存在的字段一律覆盖。

    Args:
        role_name: 角色名（``team/<role>/`` 目录名，或 ``"default"``）。
        base_dir: 项目根目录（用于定位 ``team/team_agents.json``）。
        explicit: 调用方已显式给出的补丁；其非 None 字段优先于角色配置。

    Returns:
        合成后的 ``SessionConfigPatch``（含 role + system_prompt）。

    Raises:
        KeyError: 角色不存在（由 ``_locate_team_agent_dir`` 抛出）。
        FileNotFoundError: 角色提示词为空或无法读取。
    """
    # 延迟导入：避免 agent 包与 llm/team 在模块加载期形成循环依赖
    from llm.config import compose_role_system_prompt, load_team_agent_config
    from team.base import TeamAgent

    role_dir = _locate_team_agent_dir(role_name)
    config = load_team_agent_config(role_name, base_dir)
    content = TeamAgent._read_prompt_file(os.path.join(role_dir, "AGENT.md"))
    if content is None:
        raise FileNotFoundError(f"角色提示词为空: {role_name}")
    role_prompt, _ = TeamAgent.parse_prompt_sections(content)
    # 角色提示词拼接基础规则（主对话 Agent 持有工具，故附带「工具规则」）；
    # role="default" 时内部特判不拼接，避免 agent/AGENT.md 规则重复
    explicit_prompt = explicit.system_prompt if explicit is not None else None
    composed_prompt = explicit_prompt or compose_role_system_prompt(role_prompt, role=role_name)

    values: dict[str, Any] = {"role": role_name, "system_prompt": composed_prompt}
    for field in _ROLE_PATCH_FIELDS:
        explicit_value = getattr(explicit, field) if explicit is not None else None
        if explicit_value is not None:
            values[field] = explicit_value
        elif field in config and config[field] is not None:
            values[field] = config[field]
    return SessionConfigPatch(**values)
