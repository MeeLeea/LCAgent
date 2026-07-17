"""Provider and model selection commands."""

from __future__ import annotations

import json
import os
import sys

from .types import CommandContext, CommandOutcome, HANDLED, LlmLike


def select_provider(config_file: str, select_menu) -> str:
    from llm_client import list_providers

    providers = list_providers(config_file)
    # 环境变量和本地配置文件任一提供密钥，都应在菜单中标记为已配置。
    configured_keys = _configured_provider_keys(config_file, providers)
    options = []
    for key, config in providers.items():
        env_key = str(config["env_key"])
        has_key = bool(os.environ.get(env_key)) or key in configured_keys
        mark = "✓" if has_key else " "
        label = f"[{mark}] {key:10s} ({config['name']})  模型: {', '.join(config['models'])}"
        options.append((label, key))
    selected = select_menu("选择大模型提供商 ([✓] = 已配置 API Key)", options, current="zhipu")
    return selected if selected else "zhipu"


def create_llm(provider: str, config_file: str) -> LlmLike:
    from llm_client import create_client

    try:
        return create_client(provider=provider, config_file=config_file)
    except ValueError as error:
        print(f"\n错误: {error}")
        sys.exit(1)


def switch_provider(context: CommandContext, user_input: str) -> CommandOutcome:
    # 无参数时进入菜单；带参数时直接切换，二者最终走同一替换流程。
    if user_input.lower() == "switch":
        providers = context.list_providers()
        options = [(f"{provider}  ({providers[provider]['name']})", provider) for provider in providers]
        selected = context.select_menu("选择提供商", options, current=context.llm.provider)
        if selected is None:
            return HANDLED
        new_provider = str(selected)
    else:
        new_provider = user_input[7:].strip().lower()
    try:
        new_llm = context.create_llm(new_provider)
        context.replace_llm(new_llm)
        info = context.llm.get_info()
        context.print(f"\n已切换到: {info['provider_name']} ({info['model']})")
    except SystemExit:
        return HANDLED
    except (AttributeError, KeyError, RuntimeError, ValueError) as error:
        context.print(f"\n切换失败: {error}")
    return HANDLED


def choose_model(context: CommandContext) -> CommandOutcome:
    info = context.llm.get_info()
    models = context.llm.list_models()
    selected = context.select_menu(
        f"选择模型 [{info['provider_name']}]",
        [(model, model) for model in models],
        current=context.llm.model,
    )
    if selected is None:
        return HANDLED
    # 选择当前模型时不重建 Agent，避免无意义地刷新执行器。
    if selected == context.llm.model:
        context.print(f"\n模型未变: {context.llm.model}")
        return HANDLED
    return _switch_model(context, str(selected))


def switch_model(context: CommandContext, user_input: str) -> CommandOutcome:
    low = user_input.lower()
    if low.startswith("model:"):
        new_model = user_input[6:].strip()
    else:
        parts = user_input.split(None, 1)
        new_model = parts[1].strip() if len(parts) > 1 else ""
    if not new_model:
        context.print("用法: model:<模型名>  或  model <模型名>")
        context.print("示例: model:glm-4-flash")
        return HANDLED
    return _switch_model(context, new_model)


def _switch_model(context: CommandContext, model: str) -> CommandOutcome:
    try:
        context.llm.switch_model(model)
        context.replace_llm(context.llm)
        info = context.llm.get_info()
        context.print(f"\n已切换模型: {info['model']} (提供商: {info['provider_name']})")
    except (AttributeError, KeyError, RuntimeError, ValueError) as error:
        context.print(f"\n切换失败: {error}")
    return HANDLED


def _configured_provider_keys(config_file: str, providers: dict[str, dict[str, object]]) -> set[str]:
    if not os.path.exists(config_file):
        return set()
    try:
        with open(config_file, "r", encoding="utf-8") as file:
            config = json.load(file)
    except (json.JSONDecodeError, OSError):
        return set()
    # 仅返回已定义提供商，忽略配置文件中遗留或拼写错误的条目。
    return {
        key
        for key in providers
        if config.get("providers", {}).get(key, {}).get("api_key")
    }
