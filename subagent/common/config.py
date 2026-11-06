"""子程序配置通用机制（stdlib-only）。

承载 ``planner`` / ``executor`` 两个子程序 ``config.py`` 中**逐字相同**的加载
与校验逻辑，作为唯一事实源：

- :func:`build_config` — 按「字段声明表」把本地 ``config.json`` 覆盖值合并到
  dataclass 默认值上（含环境变量对 ``output_root`` 的优先覆盖、类型回退、
  相对路径解析）；
- :func:`resolve_llm_settings` — 解析并校验 LLM 提供商与模型（惰性导入
  ``llm.llm_client``，缺 API Key 在派活阶段同步暴露）；
- ``PROJECT_ROOT`` 及私有数值/路径/提供商辅助函数。

各子程序 ``config.py`` 保留为**瘦壳**：dataclass 字段集、默认值（引自
``subagent.config`` 总表）与 ``load()`` 委托。

设计约束：
- 模块级只依赖标准库，保证 ``tools`` 包在主对话进程 eager import 时不会因
  本模块拉起 ``langchain`` / ``team`` 重依赖；
- ``llm.llm_client`` 只允许在 :func:`resolve_llm_settings` 内部惰性导入。
"""
from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from dataclasses import fields
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
) -> C:
    """按「字段声明表」把本地配置文件覆盖值合并到 dataclass 默认值。

    等价复刻原每子程序 ``XxxConfig.load`` 的合并语义：

    1. ``config_file`` 存在时读取 JSON 对象；缺失/损坏/非对象仅记 warning 回退。
    2. ``output_root`` 的值 = ``os.environ[env_key]``（非空时优先，测试/隔离用）
       否则文件覆盖值，都缺省用 dataclass 默认值；随后按 ``base_dir`` 解析。
    3. 其余字段按 ``field_kinds`` 分派（含非法值回退默认值）。

    Args:
        cls: 目标 dataclass（frozen ``dataclass`` 由调用方保证）。
        config_file: 配置文件路径；``None`` 表示不存在本地文件。
        field_kinds: 字段名 → 解析器种类（见 :data:`_SUPPORTED_KINDS`）。
        env_key: ``output_root`` 的环境变量覆盖键；空串表示无覆盖。
        base_dir: 相对路径锚定的基准目录（默认项目根）。
        label: 子程序名（warning 文案用，如 ``"planner"``）。

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
                raw = loaded
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
    config: LLMSettingsLike, *, label: str
) -> tuple[str, str]:
    """解析并校验 LLM 提供商与模型。

    在子进程启动前调用，保证「缺 API Key」这类问题在派活阶段（``start_xxx_task``）
    同步暴露，而不是变成一个永远 ``running`` 的僵尸任务。

    Args:
        config: 携带 ``provider`` / ``model`` 的子程序配置。
        label: 子程序名，用于 ``model`` 缺失时的报错文案（如 ``"planner"``）。

    Returns:
        ``(provider, model)``。

    Raises:
        ValueError: 未配置任何提供商、指定 provider 不存在、API Key 缺失
            或未解析出模型名。
    """
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
            f"提供商 {provider} 未配置默认模型，请在 {label} config.json 指定 model"
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


__all__ = [
    "PROJECT_ROOT",
    "LLMSettingsLike",
    "build_config",
    "resolve_llm_settings",
]
