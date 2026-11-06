"""subagent 子程序配置模块（唯一事实源）。

集中声明 ``planner`` / ``executor`` 两个后台子程序的全部配置要素：

- 默认值常量（``PLANNER_*`` / ``EXECUTOR_*``）；provider / model 默认
  ``None``（自动选择），属结构性语义直接写在 dataclass 中；
- 统一配置文件 ``subagent/config.json``（已入库，含 ``"planner"`` /
  ``"executor"`` 两节）：``load_planner_config()`` / ``load_executor_config()``
  在子程序初始化时读取对应小节，把覆盖值合并到 dataclass 默认值上；
  文件缺失/损坏/小节缺失时记 warning 并整体回退默认值；环境变量
  ``LCAGENT_PLANNER_ROOT`` / ``LCAGENT_EXECUTOR_ROOT`` 仅覆盖输出根目录；
- 两个 frozen dataclass（``PlannerConfig`` / ``ExecutorConfig``）与各自的
  「字段声明表」；
- 通用加载机制（合并/校验）与 LLM 提供商解析在 :mod:`subagent.common.config`，
  本模块不重复实现，也不包含任何逐子程序重复的加载方法。

设计约束：
- 模块级只依赖标准库与 ``subagent.common.config``，保证 ``tools`` 包在主对话
  进程 eager import 时不会拉起 ``langchain`` / ``team`` 重依赖；
- 修改默认值只改本模块常量或 ``subagent/config.json``，不再有第二处字面量。
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from subagent.common.config import PROJECT_ROOT, build_config

# ---- 统一配置文件（入库）----

#: 共享配置文件：含 ``"planner"`` / ``"executor"`` 顶层小节。
DEFAULT_CONFIG_FILE: str = os.path.join(PROJECT_ROOT, "subagent", "config.json")

# ---- planner（一次性方案设计：单次 LLM 调用）----

PLANNER_NAME: str = "planner"
PLANNER_TEMPERATURE: float = 0.4
PLANNER_MAX_TOKENS: int = 8192
PLANNER_TIMEOUT_SECONDS: float = 900.0

# 输出根目录的环境变量覆盖键（测试 / 多实例隔离用）
PLANNER_OUTPUT_ROOT_ENV: str = "LCAGENT_PLANNER_ROOT"

PLANNER_DEFAULT_OUTPUT_ROOT: str = os.path.join(
    PROJECT_ROOT, "data", "subagent", "planner"
)
PLANNER_AGENT_PROMPT_FILE: str = os.path.join(
    PROJECT_ROOT, "subagent", "planner", "AGENT.md"
)

# dataclass 字段 → 通用解析器种类（语义见 subagent.common.config.build_config）
_PLANNER_FIELD_KINDS: dict[str, str] = {
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
    output_root: str = PLANNER_DEFAULT_OUTPUT_ROOT
    agent_prompt_file: str = PLANNER_AGENT_PROMPT_FILE


def load_planner_config(
    config_file: str | None = None,
    base_dir: str = PROJECT_ROOT,
) -> PlannerConfig:
    """初始化 Planner 配置：读取统一配置文件的 ``planner`` 小节并与默认值合并。

    Args:
        config_file: 配置文件路径；None 时使用 ``subagent/config.json``。
        base_dir: 相对路径锚定的基准目录（默认项目根）。

    Returns:
        合并后的配置；文件缺失/损坏/小节缺失时回退默认值（仅记 warning）。
    """
    return build_config(
        PlannerConfig,
        config_file=config_file or DEFAULT_CONFIG_FILE,
        field_kinds=_PLANNER_FIELD_KINDS,
        env_key=PLANNER_OUTPUT_ROOT_ENV,
        base_dir=base_dir,
        label=PLANNER_NAME,
        section=PLANNER_NAME,
    )


# ---- executor（多步 ReAct 工具循环）----

EXECUTOR_NAME: str = "executor"
EXECUTOR_TEMPERATURE: float = 0.2
EXECUTOR_MAX_TOKENS: int = 8192
# 整个任务的总预算（所有步骤共享的绝对 deadline，秒）
EXECUTOR_TIMEOUT_SECONDS: float = 1800.0
# 步数上限（每一步 = 一次 LLM 调用 + 本轮 tool_calls 执行）
EXECUTOR_MAX_STEPS: int = 24
# 单条工具观测（ToolMessage content）最大字符数，超出截断
EXECUTOR_MAX_OBSERVATION_CHARS: int = 6000
# read_file 单次读取的最大字节数
EXECUTOR_MAX_READ_BYTES: int = 262_144
# list_dir 返回条目数上限
EXECUTOR_MAX_LIST_ENTRIES: int = 500
# 单步 LLM 调用的最小剩余预算（秒）：小于它就不发起新一步，直接判定超时
EXECUTOR_MIN_STEP_SECONDS: float = 30.0

# 输出根目录的环境变量覆盖键（测试 / 多实例隔离用）
EXECUTOR_OUTPUT_ROOT_ENV: str = "LCAGENT_EXECUTOR_ROOT"

EXECUTOR_DEFAULT_OUTPUT_ROOT: str = os.path.join(
    PROJECT_ROOT, "data", "subagent", "executor"
)
EXECUTOR_AGENT_PROMPT_FILE: str = os.path.join(
    PROJECT_ROOT, "subagent", "executor", "AGENT.md"
)

# dataclass 字段 → 通用解析器种类（语义见 subagent.common.config.build_config）
_EXECUTOR_FIELD_KINDS: dict[str, str] = {
    "name": "str",
    "provider": "nullable_str",
    "model": "nullable_str",
    "temperature": "float",
    "max_tokens": "int",
    "timeout_seconds": "float",
    "max_steps": "int",
    "max_observation_chars": "int",
    "max_read_bytes": "int",
    "max_list_entries": "int",
    "output_root": "output_root",
    "agent_prompt_file": "path",
}


@dataclass(frozen=True, slots=True)
class ExecutorConfig:
    """Executor 子程序的运行时配置（多步工具循环 + 单次结果文档）。"""

    name: str = EXECUTOR_NAME
    provider: str | None = None
    model: str | None = None
    temperature: float = EXECUTOR_TEMPERATURE
    max_tokens: int = EXECUTOR_MAX_TOKENS
    timeout_seconds: float = EXECUTOR_TIMEOUT_SECONDS
    max_steps: int = EXECUTOR_MAX_STEPS
    max_observation_chars: int = EXECUTOR_MAX_OBSERVATION_CHARS
    max_read_bytes: int = EXECUTOR_MAX_READ_BYTES
    max_list_entries: int = EXECUTOR_MAX_LIST_ENTRIES
    output_root: str = EXECUTOR_DEFAULT_OUTPUT_ROOT
    agent_prompt_file: str = EXECUTOR_AGENT_PROMPT_FILE


def load_executor_config(
    config_file: str | None = None,
    base_dir: str = PROJECT_ROOT,
) -> ExecutorConfig:
    """初始化 Executor 配置：读取统一配置文件的 ``executor`` 小节并与默认值合并。

    Args:
        config_file: 配置文件路径；None 时使用 ``subagent/config.json``。
        base_dir: 相对路径锚定的基准目录（默认项目根）。

    Returns:
        合并后的配置；文件缺失/损坏/小节缺失时回退默认值（仅记 warning）。
    """
    return build_config(
        ExecutorConfig,
        config_file=config_file or DEFAULT_CONFIG_FILE,
        field_kinds=_EXECUTOR_FIELD_KINDS,
        env_key=EXECUTOR_OUTPUT_ROOT_ENV,
        base_dir=base_dir,
        label=EXECUTOR_NAME,
        section=EXECUTOR_NAME,
    )


__all__ = [
    "DEFAULT_CONFIG_FILE",
    "EXECUTOR_AGENT_PROMPT_FILE",
    "EXECUTOR_DEFAULT_OUTPUT_ROOT",
    "EXECUTOR_MAX_LIST_ENTRIES",
    "EXECUTOR_MAX_OBSERVATION_CHARS",
    "EXECUTOR_MAX_READ_BYTES",
    "EXECUTOR_MAX_STEPS",
    "EXECUTOR_MAX_TOKENS",
    "EXECUTOR_MIN_STEP_SECONDS",
    "EXECUTOR_NAME",
    "EXECUTOR_OUTPUT_ROOT_ENV",
    "EXECUTOR_TEMPERATURE",
    "EXECUTOR_TIMEOUT_SECONDS",
    "PLANNER_AGENT_PROMPT_FILE",
    "PLANNER_DEFAULT_OUTPUT_ROOT",
    "PLANNER_MAX_TOKENS",
    "PLANNER_NAME",
    "PLANNER_OUTPUT_ROOT_ENV",
    "PLANNER_TEMPERATURE",
    "PLANNER_TIMEOUT_SECONDS",
    "ExecutorConfig",
    "PlannerConfig",
    "load_executor_config",
    "load_planner_config",
]
