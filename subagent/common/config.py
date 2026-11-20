"""子程序配置模块（唯一事实源，stdlib-only）。

承载 ``planner`` / ``executor`` 两个后台子程序的全部配置，分两层：

- **通用机制**：:func:`build_config`（按「字段声明表」把统一配置文件
  ``subagent/config.json`` 的对应小节覆盖值合并到 dataclass 默认值上，含环境
  变量对 ``output_root`` 的优先覆盖、类型回退、相对路径解析）、
  :func:`resolve_llm_settings`（解析并校验 LLM 提供商与模型，惰性导入
  ``llm.llm_client``，缺 API Key 在派活阶段同步暴露）、``PROJECT_ROOT`` 及私有
  数值/路径/提供商辅助函数；
- **子程序配置要素**：默认值常量（``PLANNER_*`` / ``EXECUTOR_*``，provider /
  model 默认 ``None`` 属结构性语义直接写在 dataclass 中）、统一配置文件路径、
  两个 frozen dataclass（``PlannerConfig`` / ``ExecutorConfig``）与各自的
  「字段声明表」、加载入口 ``load_planner_config()`` / ``load_executor_config()``
  ——子程序初始化时读取 ``subagent/config.json``（已入库，含 ``"planner"`` /
  ``"executor"`` 两节）对应小节合并到默认值，文件/小节缺失或损坏时记 warning
  回退；环境变量 ``LCAGENT_PLANNER_ROOT`` / ``LCAGENT_EXECUTOR_ROOT`` 仅覆盖
  输出根目录。

设计约束：
- 模块级只依赖标准库，保证 ``tools`` 包在主对话进程 eager import 时不会因
  本模块拉起 ``langchain`` / ``team`` 重依赖；
- ``llm.llm_client`` 只允许在 :func:`resolve_llm_settings` 内部惰性导入；
- 修改默认值只改本模块常量或 ``subagent/config.json``，不再有第二处字面量。
"""
from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Protocol

logger = logging.getLogger(__name__)

# 项目根目录：<root>/subagent/common/config.py -> <root>
PROJECT_ROOT: str = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

#: 「字段声明表」支持的解析器种类。``output_root`` 由字段名特判（环境变量
#: ``env_key`` 优先覆盖），其余字段按声明值分派。
_SUPPORTED_KINDS: tuple[str, ...] = (
    "str",  # str(value)
    "nullable_str",  # 非空 str 保留，否则回退 None（provider/model）
    "float",  # 正数 float，非法回退默认
    "int",  # 正整数 int，非法回退默认
    "path",  # str 后按 base_dir 解析为绝对路径（agent_prompt_file）
)


class LLMSettingsLike(Protocol):
    """:func:`resolve_llm_settings` 需要读取的最小配置协议。"""

    provider: str | None
    model: str | None


def build_config[C: object](
    cls: type[C],
    *,
    config_file: str | None,
    field_kinds: Mapping[str, str],
    env_key: str,
    base_dir: str,
    label: str,
    section: str | None = None,
) -> C:
    """按「字段声明表」把统一配置文件对应小节的覆盖值合并到 dataclass 默认值。

    等价复刻原每子程序 ``XxxConfig.load`` 的合并语义：

    1. ``config_file`` 存在时读取 JSON 对象；缺失/损坏/非对象仅记 warning 回退。
       ``section`` 非空时按顶层键取子程序专属小节（如 ``"planner"``），小节缺失
       或非对象时记 warning 并整节回退默认值。
    2. ``output_root`` 的值 = ``os.environ[env_key]``（非空时优先，测试/隔离用）
       否则文件覆盖值，都缺省用 dataclass 默认值；随后按 ``base_dir`` 解析。
    3. 其余字段按 ``field_kinds`` 分派（含非法值回退默认值）。

    Args:
        cls: 目标 dataclass（frozen ``dataclass`` 由调用方保证）。
        config_file: 统一配置文件路径；``None`` 表示不读取文件。
        field_kinds: 字段名 → 解析器种类（见 :data:`_SUPPORTED_KINDS`）。
        env_key: ``output_root`` 的环境变量覆盖键；空串表示无覆盖。
        base_dir: 相对路径锚定的基准目录（默认项目根）。
        label: 子程序名（warning / 报错文案用，如 ``"planner"``）。
        section: 顶层配置小节键名；``None`` 表示把整个 JSON 对象当平铺小节
            （通用语义，供测试/其他调用方使用）。

    Returns:
        合并后的 ``cls`` 实例。

    Raises:
        ValueError: ``field_kinds`` 含不支持的解析器种类（纯防御分支）。
    """
    raw: dict[str, object] = {}
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict):
                if section is None:
                    raw = loaded
                else:
                    section_raw = loaded.get(section)
                    if isinstance(section_raw, dict):
                        raw = section_raw
                    else:
                        logger.warning(
                            "读取 %s 配置失败，回退默认值: 缺少 %s 小节", label, section
                        )
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("读取 %s 配置失败，回退默认值: %s", label, exc)

    defaults = cls()
    values: dict[str, object] = {}
    for key, kind in field_kinds.items():
        if key == "output_root":
            override = os.environ.get(env_key) or str(
                raw.get(key, getattr(defaults, key))
            )
            values[key] = _resolve(override, base_dir)
            continue
        if kind == "path":
            values[key] = _resolve(
                str(raw.get(key, getattr(defaults, key))), base_dir
            )
        elif kind == "nullable_str":
            value = raw.get(key, getattr(defaults, key))
            values[key] = value if isinstance(value, str) and value else None
        elif kind == "str":
            values[key] = str(raw.get(key, getattr(defaults, key)))
        elif kind == "float":
            values[key] = _as_float(raw.get(key), getattr(defaults, key))
        elif kind == "int":
            values[key] = _as_int(raw.get(key), getattr(defaults, key))
        else:
            raise ValueError(
                f"未知的字段解析器: {kind}"
                f"（支持: {', '.join(_SUPPORTED_KINDS)}）"
            )

    # 校验声明表覆盖了 dataclass 的全部实例字段（防字段漂移的纯防御分支）。
    declared = {name for name, _ in field_kinds.items()}
    actual = {f.name for f in fields(cls) if f.init}
    if declared != actual:
        raise ValueError(
            f"{label} 字段声明表与 dataclass 字段不一致: "
            f"声明 {sorted(declared)}，实际 {sorted(actual)}"
        )

    return cls(**values)


def resolve_llm_settings(
    config: LLMSettingsLike, *, label: str | None = None
) -> tuple[str, str]:
    """解析并校验 LLM 提供商与模型。

    在子进程启动前调用，保证「缺 API Key」这类问题在派活阶段（``start_xxx_task``）
    同步暴露，而不是变成一个永远 ``running`` 的僵尸任务。

    Args:
        config: 携带 ``provider`` / ``model`` 的子程序配置；若类型含 ``name``
            字段且 ``label`` 未显式提供，``label`` 缺省取 ``config.name``。
        label: 子程序名，用于 ``model`` 缺失时的报错文案（如 ``"planner"``）；
            缺省时从 ``config.name`` 推导。

    Returns:
        ``(provider, model)``。

    Raises:
        ValueError: 未配置任何提供商、指定 provider 不存在、API Key 缺失
            或未解析出模型名。
    """
    label = label or str(getattr(config, "name", "")) or "subagent"

    # 惰性导入：避免主对话进程 import tools 时加载 langchain 重依赖。
    from llm.llm_client import load_providers

    providers = load_providers()
    if not providers:
        raise ValueError("config/llm_config.json 中未配置任何 LLM 提供商")

    provider = config.provider or _pick_provider(providers)
    if provider not in providers:
        available = ", ".join(providers.keys())
        raise ValueError(f"不支持的提供商: {provider}（可用: {available}）")

    provider_config = providers[provider]
    api_key = provider_config.get("api_key") or os.environ.get(
        provider_config.get("env_key", ""), ""
    )
    if not api_key:
        raise ValueError(
            f"请提供 {provider_config.get('name', provider)} 的API密钥"
            f"（环境变量 {provider_config.get('env_key', '')} 或 "
            f"config/llm_config.json）"
        )

    model = config.model or provider_config.get("model")
    if not model:
        raise ValueError(
            f"提供商 {provider} 未配置默认模型，"
            f"请在 subagent/config.json 的 {label} 小节指定 model"
        )
    return provider, str(model)


def _pick_provider(providers: dict[str, dict[str, object]]) -> str:
    """选择第一个已配置 API Key 的提供商；都没有时退回第一个。"""
    for key, cfg in providers.items():
        if cfg.get("api_key") or os.environ.get(str(cfg.get("env_key", "")), ""):
            return key
    return next(iter(providers))


def _as_float(value: object, default: float) -> float:
    """把正数转换为 float；非法/非正数时回退默认值。"""
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return float(value)
    return default


def _as_int(value: object, default: int) -> int:
    """把正整数转换为 int；非法/非正数时回退默认值。"""
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return default


def _resolve(path: str, base_dir: str) -> str:
    """把相对路径解析为基于 ``base_dir`` 的绝对路径。"""
    if not path:
        return path
    return os.path.normpath(path if os.path.isabs(path) else os.path.join(base_dir, path))


# ==================== 子程序配置要素（默认值总表）====================

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

# dataclass 字段 → 通用解析器种类（语义见本模块 build_config）
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

# dataclass 字段 → 通用解析器种类（语义见本模块 build_config）
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
    "PROJECT_ROOT",
    "ExecutorConfig",
    "LLMSettingsLike",
    "PlannerConfig",
    "build_config",
    "load_executor_config",
    "load_planner_config",
    "resolve_llm_settings",
]
