"""
统一大模型封装 - 基于LangChain，支持多提供商(智谱/千问/DeepSeek/Kimi)
所有提供商均兼容 OpenAI API 格式，使用 langchain.chat_models.init_chat_model 统一调用
"""
import json
import os
import re
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from tenacity import (
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

# 提供商配置(单一来源: 见 config/llm_config.json 的 providers 字段)
DEFAULT_CONFIG_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "llm_config.json",
)

# ============ 瞬时错误自动重试 ============

# 可安全自动重试的 HTTP 状态码(限流/服务过载)
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
RETRY_ATTEMPTS = 3  # 含首次在内最多尝试次数
RETRY_BASE_DELAY = 1.0  # 指数退避基数(秒)
RETRY_MAX_DELAY = 8.0  # 单次最长等待(秒)

# 网关/模型服务过载时常见的纯文本关键字
_RETRYABLE_KEYWORDS = (
    "service temporarily unavailable",
    "service unavailable",
    "temporarily unavailable",
    "internal server error",
    "bad gateway",
    "gateway timeout",
    "too many requests",
    "server overloaded",
    "temporary failure",
)


def should_retry(e: Exception) -> bool:
    """判断异常是否属于可安全自动重试的瞬时错误(限流/服务过载/网络抖动)。

    覆盖场景:
      - 429 / 5xx 等可重试状态码(异常对象带 status_code，或字符串中出现状态码)
      - 连接/超时类异常(ConnectionError / TimeoutError 等)
      - 网关纯文本提示(如 "Service temporarily unavailable")

    Args:
        e: 待判断的异常对象

    Returns:
        是否应自动重试
    """
    # 连接/超时类异常
    if isinstance(e, (ConnectionError, TimeoutError, OSError)):
        return True

    # 异常对象直接暴露状态码
    status = getattr(e, "status_code", None)
    if status is not None:
        try:
            if int(status) in RETRYABLE_STATUS:
                return True
        except (TypeError, ValueError):
            pass

    text = str(e)
    low = text.lower()

    # 字符串中标注的状态码，如 "Error code: 500" / "status code: 429" / "HTTP 503"
    status_match = re.search(
        r"(?:error code|status[ _-]?code|http[ _-]?status|status)\s*[:=]?\s*(\d{3})",
        text,
        re.IGNORECASE,
    )
    if status_match:
        try:
            if int(status_match.group(1)) in RETRYABLE_STATUS:
                return True
        except ValueError:
            pass

    # 纯文本网关提示关键字
    return any(k in low for k in _RETRYABLE_KEYWORDS)


def load_providers(config_file: str | None = None) -> dict[str, dict[str, Any]]:
    """从配置文件读取提供商定义(配置即唯一来源,不再在代码中维护)

    config/llm_config.json 结构:
    {
        "providers": {
            "<name>": {
                "name": "...", "base_url": "...", "env_key": "...",
                "model": "...(可选覆盖)", "models": [...],
                "api_key": "...(可选)"
            }
        },
        "tavily": {"api_key": "..."}
    }
    """
    path = config_file or DEFAULT_CONFIG_FILE
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("providers", {})
    except (OSError, json.JSONDecodeError):
        return {}


def _make_retryer() -> Retrying:
    """构建针对瞬时错误(429/5xx/连接超时)的指数退避重试器。

    Returns:
        配置好的 tenacity.Retrying 实例，失败最终会原样抛出原始异常
    """
    return Retrying(
        retry=retry_if_exception(should_retry),
        wait=wait_exponential(
            multiplier=RETRY_BASE_DELAY,
            min=RETRY_BASE_DELAY,
            max=RETRY_MAX_DELAY,
        ),
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        reraise=True,
    )


class LLMClient:
    """统一的大模型调用接口，基于LangChain，支持多提供商"""

    def __init__(
        self,
        provider: str | None = "zhipu",
        api_key: str | None = None,
        model: str | None = None,
        config_file: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ):
        """
        初始化LLM客户端

        Args:
            provider: 提供商名称(见 config/llm_config.json 的 providers 字段)
            api_key: API密钥，不提供则从环境变量或配置文件获取
            model: 模型名称，不提供则使用提供商默认模型
            config_file: 配置文件路径，用于读取API密钥
            temperature: 温度参数
            max_tokens: 最大生成token数
        """
        provider = (provider or "openai").lower()
        self.config_file = config_file or DEFAULT_CONFIG_FILE
        providers = load_providers(self.config_file)
        if provider not in providers:
            available = ", ".join(providers.keys()) or "(配置文件中未定义任何提供商)"
            raise ValueError(
                f"不支持的提供商: {provider}\n"
                f"支持的提供商: {available}"
            )

        self.provider = provider
        self.provider_config = providers[provider]
        self.temperature = temperature
        self.max_tokens = max_tokens

        # 获取API密钥: 优先参数传入 > 配置文件 > 环境变量
        self.api_key = (
            api_key
            or self.provider_config.get("api_key")
            or os.environ.get(self.provider_config.get("env_key", ""))
        )

        if not self.api_key:
            raise ValueError(
                f"请提供 {self.provider_config['name']} 的API密钥\n"
                f"  方式1: 设置环境变量 {self.provider_config['env_key']}\n"
                f"  方式2: 在 config/llm_config.json 中配置\n"
                f"  方式3: 初始化时传入 api_key 参数"
            )

        # 设置模型: 参数 > 配置文件 model
        self.model = model or self.provider_config.get("model")

        # 创建 LangChain 统一聊天模型(init_chat_model)
        self.client = self._create_chat_model()

    def _create_chat_model(self) -> BaseChatModel:
        """创建统一聊天模型(init_chat_model, OpenAI 兼容接口)"""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "model_provider": "openai",
            "api_key": self.api_key,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        base_url = self.provider_config.get("base_url")
        if base_url:  # base_url 可选:仅当配置中提供时才传入
            kwargs["base_url"] = base_url
        return init_chat_model(**kwargs)

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None
    ) -> str:
        """
        发送对话请求

        Args:
            messages: 消息列表，格式为 [{"role": "user/assistant/system", "content": "..."}]
            temperature: 温度参数（覆盖默认值）
            max_tokens: 最大生成token数（覆盖默认值）

        Returns:
            模型生成的回复文本
        """
        # 转换为LangChain消息格式
        langchain_messages = self._to_langchain_messages(messages)

        # 如果有临时参数，通过 bind 轻量覆盖，避免重建客户端
        client = self.client
        if temperature is not None or max_tokens is not None:
            overrides: dict[str, Any] = {}
            if temperature is not None:
                overrides["temperature"] = temperature
            if max_tokens is not None:
                overrides["max_tokens"] = max_tokens
            client = client.bind(**overrides)

        try:
            response = _make_retryer()(client.invoke, langchain_messages)
            return response.content
        except Exception as e:
            raise RuntimeError(f"[{self.provider_config['name']}] 调用失败: {e!s}")

    def chat_with_history(
        self,
        user_input: str,
        history: list[dict[str, str]],
        system_prompt: str | None = None,
        **kwargs
    ) -> str:
        """带历史记录的对话"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history)
        messages.append({"role": "user", "content": user_input})
        return self.chat(messages, **kwargs)

    def extract_json(self, text: str) -> dict[str, Any] | None:
        """从文本中提取JSON对象"""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
        return None

    def get_chat_model(self) -> BaseChatModel:
        """获取统一聊天模型实例（供Agent使用）"""
        return self.client

    def switch_model(self, model: str):
        """
        运行时切换模型(仅限当前提供商支持的模型)

        Args:
            model: 模型名称(必须在当前提供商 providers.<provider>.models 列表中)
        """
        available_models = self.provider_config.get("models", [])
        if model not in available_models:
            raise ValueError(
                f"不支持的模型: {model}\n"
                f"当前提供商 [{self.provider_config['name']}] 可用模型: {', '.join(available_models)}"
            )
        self.model = model
        self.client = self._create_chat_model()

    def list_models(self) -> list[str]:
        """列出当前提供商支持的所有模型"""
        return list(self.provider_config.get("models", []))

    def get_info(self) -> dict[str, str]:
        """获取当前客户端信息"""
        return {
            "provider": self.provider,
            "provider_name": self.provider_config["name"],
            "model": self.model,
            "base_url": self.provider_config["base_url"]
        }

    @staticmethod
    def _to_langchain_messages(messages: list[dict[str, str]]) -> list[BaseMessage]:
        """将字典消息列表转换为LangChain消息对象"""
        result = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                result.append(SystemMessage(content=content))
            elif role == "assistant":
                result.append(AIMessage(content=content))
            else:
                result.append(HumanMessage(content=content))
        return result
