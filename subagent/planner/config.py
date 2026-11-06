"""Planner 子程序运行时配置（一次性方案设计）。

配置来源（高 → 低）：
1. ``subagent/planner/config.json``（存在时）
2. 环境变量 ``LCAGENT_PLANNER_ROOT``（仅覆盖输出根目录，便于测试/隔离）
3. ``subagent/config`` 总表默认值

本模块是**瘦壳**：通用加载机制（合并/校验/LLM 设置解析）已下沉至
:mod:`subagent.common.config`，这里只保留 planner 特有的 dataclass 字段集、
默认值（引自总表）与路径常量。

设计约束：
- 模块级只依赖标准库与 ``subagent`` 内轻量模块，保证 ``tools`` 包在主对话
  进程 eager import 时不会因本模块拉起 ``langchain`` / ``team`` 重依赖；
- 需要读取 ``config/llm_config.json`` 的校验（provider 存在性、API Key）
  在 :func:`resolve_llm_settings` 底层惰性导入，只在子进程启动时执行。
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from subagent.common.config import (
    PROJECT_ROOT,
    build_config,
)
from subagent.common.config import (
    resolve_llm_settings as _resolve_llm_settings_common,
)
from subagent.config import (
    PLANNER_MAX_TOKENS,
    PLANNER_NAME,
    PLANNER_OUTPUT_ROOT_ENV,
    PLANNER_TEMPERATURE,
    PLANNER_TIMEOUT_SECONDS,
)

DEFAULT_CONFIG_FILE: str = os.path.join(
    PROJECT_ROOT, "subagent", "planner", "config.json"
)
DEFAULT_AGENT_PROMPT_FILE: str = os.path.join(
    PROJECT_ROOT, "subagent", "planner", "AGENT.md"
)
DEFAULT_OUTPUT_ROOT: str = os.path.join(PROJECT_ROOT, "data", "subagent", "planner")

# 输出根目录的环境变量覆盖键（测试 / 多实例隔离用；源自总表）
OUTPUT_ROOT_ENV: str = PLANNER_OUTPUT_ROOT_ENV

# dataclass 字段 → 解析器种类（语义见 subagent.common.config.build_config）
_FIELD_KINDS: dict[str, str] = {
    "name": "str",
    "provider": "nullable_str",
    "model": "nullable_str",
    "temperature": "float",
    "max_tokens": "int",
    "timeout_seconds": "float",
    "output_root": "output_root",
    "agent_prompt_file": "path",
}


@dataclass(frozen=True, slots=True)
class PlannerConfig:
    """Planner 子程序的一次性运行时配置。"""

    name: str = PLANNER_NAME
    provider: str | None = None
    model: str | None = None
    temperature: float = PLANNER_TEMPERATURE
    max_tokens: int = PLANNER_MAX_TOKENS
    timeout_seconds: float = PLANNER_TIMEOUT_SECONDS
    output_root: str = DEFAULT_OUTPUT_ROOT
    agent_prompt_file: str = DEFAULT_AGENT_PROMPT_FILE

    @classmethod
    def load(
        cls,
        config_file: str | None = None,
        base_dir: str = PROJECT_ROOT,
    ) -> PlannerConfig:
        """加载配置文件并与默认值合并。

        Args:
            config_file: 配置文件路径；None 时使用 ``subagent/planner/config.json``。
            base_dir: 相对路径锚定的基准目录（默认项目根）。

        Returns:
            合并后的配置；文件缺失或损坏时回退默认值（仅记 warning）。
        """
        return build_config(
            cls,
            config_file=config_file or DEFAULT_CONFIG_FILE,
            field_kinds=_FIELD_KINDS,
            env_key=OUTPUT_ROOT_ENV,
            base_dir=base_dir,
            label=PLANNER_NAME,
        )

    def resolved_output_root(self) -> str:
        """返回规范化后的输出根目录。"""
        return os.path.normpath(self.output_root)


def resolve_llm_settings(config: PlannerConfig) -> tuple[str, str]:
    """解析并校验 LLM 提供商与模型。

    在子进程启动前调用，保证「缺 API Key」这类问题在 ``start_planner_task``
    阶段同步暴露，而不是变成一个永远 ``running`` 的僵尸任务。实现统一在
    :func:`subagent.common.config.resolve_llm_settings`，本 wrapper 固定
    ``label="planner"`` 保持原有报错文案。

    Args:
        config: Planner 配置。

    Returns:
        ``(provider, model)``。

    Raises:
        ValueError: 未配置任何提供商、指定 provider 不存在、API Key 缺失
            或未解析出模型名。
    """
    return _resolve_llm_settings_common(config, label=PLANNER_NAME)


__all__ = [
    "DEFAULT_AGENT_PROMPT_FILE",
    "DEFAULT_CONFIG_FILE",
    "DEFAULT_OUTPUT_ROOT",
    "OUTPUT_ROOT_ENV",
    "PROJECT_ROOT",
    "PlannerConfig",
    "resolve_llm_settings",
]
