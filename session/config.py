"""会话基础配置 - 每个会话（Session / thread_id）独立的 provider / model / 角色。

设计要点：
- **唯一事实源**是 ``SessionStore`` 的 ``session_config`` 命名空间（见 session/store.py），
  不是 checkpoint state；checkpoint 只保存只读快照用于诊断。
- 本模块只定义**不可变数据类型 + 结构校验 + 序列化**，不 import ``llm`` / ``agent`` /
  其余 session 模块，避免循环导入并保持 Session 层依赖轻量。
- provider / role / model 的**存在性**校验由调用方注入候选列表完成
  （API / CLI / Registry 层各自持有 load_providers() / get_available_team_roles()），
  本模块只做类型与取值范围的校验。
- ``SessionConfig`` 是 frozen dataclass，可安全跨协程共享：
  请求开始处捕获一份快照，本轮执行期间配置恒定（前端中途改配置只影响下一轮）。
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any

#: ``max_iterations`` 兜底默认值（正常路径由 AgentCore 的配置推导，不依赖此常量）。
DEFAULT_MAX_ITERATIONS = 25

#: temperature 合法区间（与主流 provider 的取值域一致）。
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0

#: ``config["configurable"]`` 中承载会话配置的键名（middleware 据此读取）。
CONFIGURABLE_KEY = "session_config"

#: Store 中 ``session_config`` 命名空间使用的 kind（4 层 namespace 的最后一层）。
STORE_KIND = "session_config"


class SessionConfigError(ValueError):
    """会话配置非法（字段类型错误 / 取值越界 / 候选不存在）。"""


@dataclass(frozen=True, slots=True)
class SessionConfig:
    """单个会话的基础配置（不可变）。

    Attributes:
        provider: 提供商标识（``config/llm_config.json`` 的 providers 键）。
        model: 模型名；``None`` 表示使用该 provider 的默认模型。
        role: 团队角色名（``team/<role>/`` 目录名）；``None`` 表示不使用角色。
        system_prompt: 设置角色时解析出的系统提示词快照。在**写入时**解析并保存，
            避免每轮重新读文件导致同一会话中途漂移。
        temperature: 采样温度；``None`` 表示使用 provider / 全局默认。
        max_tokens: 最大生成 token；``None`` 表示使用 provider / 全局默认。
        max_iterations: 该会话的图递归上限（= LangGraph ``recursion_limit``）。
        version: 乐观并发版本号，每次写回自增；用于诊断"本轮实际用了哪一版配置"。
    """

    provider: str
    model: str | None = None
    role: str | None = None
    system_prompt: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        """转 JSON 兼容 dict（LangGraph Store 只接受可序列化基础值）。"""
        return {
            "provider": self.provider,
            "model": self.model,
            "role": self.role,
            "system_prompt": self.system_prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_iterations": self.max_iterations,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SessionConfig:
        """从 dict 还原；容忍历史数据缺字段（缺失即取默认值）。

        Raises:
            SessionConfigError: provider 缺失或为空。
        """
        provider = data.get("provider")
        if not isinstance(provider, str) or not provider.strip():
            raise SessionConfigError(f"session_config 缺少合法 provider: {provider!r}")
        return cls(
            provider=provider,
            model=_opt_str(data.get("model")),
            role=_opt_str(data.get("role")),
            system_prompt=_opt_str(data.get("system_prompt")),
            temperature=_opt_float(data.get("temperature")),
            max_tokens=_opt_int(data.get("max_tokens")),
            max_iterations=_opt_int(data.get("max_iterations")) or DEFAULT_MAX_ITERATIONS,
            version=_opt_int(data.get("version")) or 1,
        )

    def apply(self, patch: SessionConfigPatch) -> SessionConfig:
        """应用一次部分更新，``version`` 自增。

        patch 中为 ``None`` 的字段表示**不修改**；因此无法用 patch 把
        model / role 重置为 ``None``，需要时用完整 ``set`` 语义（整体替换）。
        """
        changes: dict[str, Any] = {"version": self.version + 1}
        for field_name in (
            "provider",
            "model",
            "role",
            "system_prompt",
            "temperature",
            "max_tokens",
            "max_iterations",
        ):
            value = getattr(patch, field_name)
            if value is not None:
                changes[field_name] = value
        updated = replace(self, **changes)
        validate_session_config(updated)
        return updated


@dataclass(frozen=True, slots=True)
class SessionConfigPatch:
    """会话配置的部分更新请求；``None`` 表示该字段不修改。"""

    provider: str | None = None
    model: str | None = None
    role: str | None = None
    system_prompt: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    max_iterations: int | None = None


def validate_session_config(
    config: SessionConfig,
    *,
    providers: Iterable[str] | None = None,
    roles: Iterable[str] | None = None,
    models: Iterable[str] | None = None,
) -> None:
    """校验配置的结构合法性，并在调用方提供候选集合时校验存在性。

    候选集合参数由调用方按自身能力注入（API 层通常三者都有，CLI 层可能有），
    未注入的维度只做结构校验。

    Raises:
        SessionConfigError: 任一校验失败。
    """
    if not isinstance(config.provider, str) or not config.provider.strip():
        raise SessionConfigError("provider 必须是非空字符串")
    if not isinstance(config.max_iterations, int) or config.max_iterations < 1:
        raise SessionConfigError(f"max_iterations 必须是 >=1 的整数，收到 {config.max_iterations!r}")
    if not isinstance(config.version, int) or config.version < 1:
        raise SessionConfigError(f"version 必须是 >=1 的整数，收到 {config.version!r}")
    if config.temperature is not None:
        if not isinstance(config.temperature, (int, float)):
            raise SessionConfigError(f"temperature 必须是数字，收到 {config.temperature!r}")
        if not (MIN_TEMPERATURE <= float(config.temperature) <= MAX_TEMPERATURE):
            raise SessionConfigError(
                f"temperature 必须位于 [{MIN_TEMPERATURE}, {MAX_TEMPERATURE}]，收到 {config.temperature!r}"
            )
    if config.max_tokens is not None and (
        not isinstance(config.max_tokens, int) or config.max_tokens < 1
    ):
        raise SessionConfigError(f"max_tokens 必须是 >=1 的整数，收到 {config.max_tokens!r}")

    if providers is not None and config.provider not in set(providers):
        raise SessionConfigError(f"未知 provider: {config.provider!r}")
    if roles is not None and config.role is not None and config.role not in set(roles):
        raise SessionConfigError(f"未知角色: {config.role!r}")
    if models is not None and config.model is not None and config.model not in set(models):
        raise SessionConfigError(f"未知模型: {config.model!r}")


def session_config_to_configurable(config: SessionConfig) -> dict[str, Any]:
    """构造注入 ``config["configurable"]`` 的会话配置片段。

    与 ``thread_id`` / ``workspace_path`` 同处 configurable：本版本的 LangGraph 会把
    ``configurable`` 映射到 ``request.runtime.context``，middleware 据此读取
    （项目既有先例见 ``memory/middleware.py::_extract_thread_id``）。
    """
    return {CONFIGURABLE_KEY: config.to_dict()}


def session_config_from_runtime_context(context: Any) -> SessionConfig | None:
    """从 ``request.runtime.context`` 提取会话配置。

    兼容两种形态：``context`` 是 dict（含 ``configurable`` 子 dict），
    或 ``context`` 是带 ``configurable`` 属性的对象。任一环节缺失返回 ``None``
    （middleware 据此回退到构建期默认模型，保证旧调用路径不炸）。
    """
    configurable: Any = None
    if isinstance(context, Mapping):
        configurable = context.get("configurable")
    else:
        configurable = getattr(context, "configurable", None)
    if not isinstance(configurable, Mapping):
        return None
    raw = configurable.get(CONFIGURABLE_KEY)
    if not isinstance(raw, Mapping):
        return None
    try:
        return SessionConfig.from_dict(raw)
    except SessionConfigError:
        return None


def _opt_str(value: Any) -> str | None:
    if value is None:
        return None
    return value if isinstance(value, str) else str(value)


def _opt_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _opt_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


__all__ = [
    "CONFIGURABLE_KEY",
    "DEFAULT_MAX_ITERATIONS",
    "MAX_TEMPERATURE",
    "MIN_TEMPERATURE",
    "STORE_KIND",
    "SessionConfig",
    "SessionConfigError",
    "SessionConfigPatch",
    "session_config_from_runtime_context",
    "session_config_to_configurable",
    "validate_session_config",
]
