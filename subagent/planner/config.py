"""Planner 子程序运行时配置（一次性方案设计）。

配置来源（高 → 低）：
1. ``subagent/planner/config.json``（存在时）
2. 环境变量 ``LCAGENT_PLANNER_ROOT``（仅覆盖输出根目录，便于测试/隔离）
3. 本模块内置默认值

设计约束：
- 模块级只依赖标准库，保证 ``tools`` 包在主对话进程 eager import 时
  不会因本模块拉起 ``langchain`` / ``team`` 依赖；
- 需要读取 ``config/llm_config.json`` 的校验（provider 存在性、API Key）
  放在 :func:`resolve_llm_settings` 内部惰性导入，只在子进程启动时执行。
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# 项目根目录：<root>/subagent/planner/config.py -> <root>
PROJECT_ROOT: str = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

DEFAULT_CONFIG_FILE: str = os.path.join(
    PROJECT_ROOT, "subagent", "planner", "config.json"
)
DEFAULT_AGENT_PROMPT_FILE: str = os.path.join(
    PROJECT_ROOT, "subagent", "planner", "AGENT.md"
)
DEFAULT_OUTPUT_ROOT: str = os.path.join(PROJECT_ROOT, "data", "subagent", "planner")

# 输出根目录的环境变量覆盖键（测试 / 多实例隔离用）
OUTPUT_ROOT_ENV: str = "LCAGENT_PLANNER_ROOT"


@dataclass(frozen=True, slots=True)
class PlannerConfig:
    """Planner 子程序的一次性运行时配置。"""

    name: str = "planner"
    provider: str | None = None
    model: str | None = None
    temperature: float = 0.4
    max_tokens: int = 8192
    timeout_seconds: float = 900.0
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
        path = config_file or DEFAULT_CONFIG_FILE
        raw: dict[str, object] = {}
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    loaded = json.load(fh)
                if isinstance(loaded, dict):
                    raw = loaded
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("读取 planner 配置失败，回退默认值: %s", exc)

        defaults = cls()
        provider = raw.get("provider", defaults.provider)
        model = raw.get("model", defaults.model)
        output_root = os.environ.get(OUTPUT_ROOT_ENV) or str(
            raw.get("output_root", defaults.output_root)
        )
        agent_prompt_file = str(
            raw.get("agent_prompt_file", defaults.agent_prompt_file)
        )

        return cls(
            name=str(raw.get("name", defaults.name)),
            provider=provider if isinstance(provider, str) and provider else None,
            model=model if isinstance(model, str) and model else None,
            temperature=_as_float(raw.get("temperature"), defaults.temperature),
            max_tokens=_as_int(raw.get("max_tokens"), defaults.max_tokens),
            timeout_seconds=_as_float(
                raw.get("timeout_seconds"), defaults.timeout_seconds
            ),
            output_root=_resolve(output_root, base_dir),
            agent_prompt_file=_resolve(agent_prompt_file, base_dir),
        )

    def resolved_output_root(self) -> str:
        """返回规范化后的输出根目录。"""
        return os.path.normpath(self.output_root)


def resolve_llm_settings(config: PlannerConfig) -> tuple[str, str]:
    """解析并校验 LLM 提供商与模型。

    在子进程启动前调用，保证「缺 API Key」这类问题在 ``start_planner_task``
    阶段同步暴露，而不是变成一个永远 ``running`` 的僵尸任务。

    Args:
        config: Planner 配置。

    Returns:
        ``(provider, model)``。

    Raises:
        ValueError: 未配置任何提供商、指定 provider 不存在、API Key 缺失
            或未解析出模型名。
    """
    # 惰性导入：避免主对话进程 import tools 时加载 langchain。
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
        raise ValueError(f"提供商 {provider} 未配置默认模型，请在 planner config.json 指定 model")
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
    "DEFAULT_AGENT_PROMPT_FILE",
    "DEFAULT_CONFIG_FILE",
    "DEFAULT_OUTPUT_ROOT",
    "OUTPUT_ROOT_ENV",
    "PROJECT_ROOT",
    "PlannerConfig",
    "resolve_llm_settings",
]
