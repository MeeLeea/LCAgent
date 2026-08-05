"""团队角色切换 - 从 team/<角色>/ 目录重建主对话 Agent 的角色

对外提供两个能力(供 AgentCore 委托调用):
    - _locate_team_agent_dir: 扫描 team/ 精确定位角色目录
    - arebuild_agent_from_team_dir: 读取角色 agent_config.json + AGENT.md,
      就地把传入的 AgentCore 切换为该角色的提示词/LLM

从 agent_core.py 抽离,避免核心调度模块承载角色目录扫描逻辑。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import TYPE_CHECKING

from agent.llm_client import LLMClient

if TYPE_CHECKING:
    from agent.agent_core import AgentCore

logger = logging.getLogger(__name__)

# 项目根目录(基于本文件位置计算: agent/role_sw.py -> 上两级)
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 多 Agent 角色目录
_TEAM_DIR = os.path.join(_BASE_DIR, "team")
# team/ 下的非角色目录(基础设施,跳过)
_NON_ROLE_DIRS = frozenset({"__pycache__"})

def get_available_team_roles() -> list[str]:
    """扫描 team/ 目录,列出可用角色文件夹名"""
    if not os.path.isdir(_TEAM_DIR):
        return []

    available: list[str] = []
    for entry in sorted(os.listdir(_TEAM_DIR)):
        if entry in _NON_ROLE_DIRS:
            continue
        sub_dir = os.path.join(_TEAM_DIR, entry)
        if not os.path.isdir(sub_dir):
            continue
        # 仅将同时具备 agent_config.json + AGENT.md 的目录视为合法角色
        has_config = os.path.isfile(os.path.join(sub_dir, "agent_config.json"))
        has_prompt = os.path.isfile(os.path.join(sub_dir, "AGENT.md"))
        if has_config and has_prompt:
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

    # 直接拼接路径，不用再次扫描磁盘
    agent_dir = os.path.join(_TEAM_DIR, agent_name)
    return agent_dir


async def arebuild_agent_from_team_dir(
    agent: AgentCore, agent_name: str, *, task: str = ""
) -> None:
    """按 team/ 角色文件夹名重建主对话 Agent 的角色(唯一对外入口)

    扫描 team/ 定位目标角色目录,读取其 agent_config.json 与 AGENT.md,
    复用现有构建链把主 AgentCore 切换为该角色的提示词/LLM:

    - 仅提示词变化 → 走 _update_system_prompt(),不重建 Graph(约快 100x)
    - provider/model 变化 → 重建 LLMClient 并走 _arebuild_agent_executor()

    整个过程就地修改传入的 AgentCore 实例,不返回新对象。

    Args:
        agent: 待切换角色的 AgentCore 实例(就地修改)
        agent_name: team/ 下的角色文件夹名(如 "manager"/"worker")
        task: 可选任务描述,用于切换后自动匹配注入技能

    Raises:
        KeyError: 角色文件夹不存在或缺少必需文件
        FileNotFoundError: AGENT.md 读取失败(内容为空)
    """
    agent._ensure_not_closed()

    # 1. 扫描 team/ 定位目标角色目录
    role_dir = _locate_team_agent_dir(agent_name)
    config_path = os.path.join(role_dir, "agent_config.json")
    prompt_path = os.path.join(role_dir, "AGENT.md")

    # 2. 读取角色配置与提示词(复用现有能力)
    from agent.config import load_agent_config
    from team.base import TeamAgent

    config = load_agent_config(config_path)
    content = TeamAgent._read_prompt_file(prompt_path)
    if content is None:
        raise FileNotFoundError(f"角色提示词文件为空或无法读取: {prompt_path}")

    # 剥离 ## workflow:* 小节,只取角色系统提示词
    role_prompt, _templates = TeamAgent.parse_prompt_sections(content)

    # provider/model 不在 load_agent_config 的 DEFAULTS 白名单内,
    # 需从原始 JSON 直接读取(缺省则沿用当前 LLM)
    def _read_raw_cfg() -> dict:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}

    raw_cfg: dict = await asyncio.to_thread(_read_raw_cfg)

    # 3. 判断是否需要切换 LLM(provider/model 变化)
    target_provider = (raw_cfg.get("provider") or agent.llm.provider).lower()
    target_model = raw_cfg.get("model") or agent.llm.model
    llm_changed = (
        target_provider != agent.llm.provider or target_model != agent.llm.model
    )

    async with agent._state_lock:
        # 更新角色核心提示词
        agent.agent_core_prompt = role_prompt
        agent.name = config.get("name", agent.name)
        agent.max_iterations = config.get("max_iterations", agent.max_iterations)

        if llm_changed:
            # LLM 变化:重建 LLMClient + 重建 executor
            agent.llm = LLMClient(
                provider=target_provider,
                model=target_model,
                config_file=agent.llm.config_file,
                temperature=agent.llm.temperature,
                max_tokens=agent.llm.max_tokens,
            )
            await agent._arebuild_agent_executor(task)
        else:
            # 仅提示词变化:更新系统提示词,不重建 Graph
            agent._update_system_prompt(task)

    if agent.verbose:
        logger.info(
            "已切换到 team 角色: %s (LLM %s)",
            agent_name,
            "已重建" if llm_changed else "未变",
        )
