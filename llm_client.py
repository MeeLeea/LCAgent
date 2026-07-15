"""
统一大模型封装 - 基于LangChain，支持多提供商(智谱/千问/DeepSeek/Kimi)
所有提供商均兼容 OpenAI API 格式，使用 langchain.chat_models.init_chat_model 统一调用
"""
import os
import json
from typing import Optional, Dict, List, Any
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage


# 提供商配置
PROVIDERS = {
    "zhipu": {
        "name": "智谱AI",
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "env_key": "ZHIPU_API_KEY",
        "default_model": "glm-4.7",
        "models": ["glm-4", "glm-4-flash", "glm-4-long", "glm-4.7-flash", "glm-4.7-long"]
    },
    "qwen": {
        "name": "通义千问",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "env_key": "DASHSCOPE_API_KEY",
        "default_model": "qwen-plus",
        "models": ["qwen-plus", "qwen-turbo", "qwen-max"]
    },
    "deepseek": {
        "name": "DeepSeek",
        "base_url": "https://api.deepseek.com",
        "env_key": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-chat",
        "models": ["deepseek-chat", "deepseek-reasoner"]
    },
    "kimi": {
        "name": "Kimi (Moonshot)",
        "base_url": "https://api.moonshot.cn/v1",
        "env_key": "MOONSHOT_API_KEY",
        "default_model": "moonshot-v1-8k",
        "models": ["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"]
    }
}


class LLMClient:
    """统一的大模型调用接口，基于LangChain，支持多提供商"""

    def __init__(
        self,
        provider: Optional[str] = "openai",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        config_file: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ):
        """
        初始化LLM客户端

        Args:
            provider: 提供商名称 (zhipu/qwen/deepseek/kimi)
            api_key: API密钥，不提供则从环境变量或配置文件获取
            model: 模型名称，不提供则使用提供商默认模型
            config_file: 配置文件路径，用于读取API密钥
            temperature: 温度参数
            max_tokens: 最大生成token数
        """
        provider = provider.lower()
        if provider not in PROVIDERS:
            raise ValueError(
                f"不支持的提供商: {provider}\n"
                f"支持的提供商: {', '.join(PROVIDERS.keys())}"
            )

        self.provider = provider
        self.provider_config = PROVIDERS[provider]
        self.temperature = temperature
        self.max_tokens = max_tokens

        # 获取API密钥: 优先参数传入 > 配置文件 > 环境变量
        self.api_key = api_key
        if not self.api_key and config_file:
            self.api_key = self._load_api_key_from_config(config_file, provider)
        if not self.api_key:
            self.api_key = os.environ.get(self.provider_config["env_key"])

        if not self.api_key:
            raise ValueError(
                f"请提供 {self.provider_config['name']} 的API密钥\n"
                f"  方式1: 设置环境变量 {self.provider_config['env_key']}\n"
                f"  方式2: 在 llm_config.json 中配置\n"
                f"  方式3: 初始化时传入 api_key 参数"
            )

        # 设置模型
        self.model = self._load_model_from_config(config_file, provider) or self.provider_config["default_model"]

        # 创建 LangChain 统一聊天模型(init_chat_model)
        self.client = self._create_chat_model()

    def _create_chat_model(self) -> BaseChatModel:
        """创建统一聊天模型(init_chat_model, OpenAI 兼容接口)"""
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "model_provider": "openai",
            "api_key": self.api_key,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        base_url = self.provider_config.get("base_url")
        if base_url:  # base_url 可选:仅当 PROVIDERS 中提供时才传入
            kwargs["base_url"] = base_url
        return init_chat_model(**kwargs)

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
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
            overrides: Dict[str, Any] = {}
            if temperature is not None:
                overrides["temperature"] = temperature
            if max_tokens is not None:
                overrides["max_tokens"] = max_tokens
            client = client.bind(**overrides)

        try:
            response = client.invoke(langchain_messages)
            return response.content
        except Exception as e:
            raise RuntimeError(f"[{self.provider_config['name']}] 调用失败: {str(e)}")

    def chat_with_history(
        self,
        user_input: str,
        history: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """带历史记录的对话"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history)
        messages.append({"role": "user", "content": user_input})
        return self.chat(messages, **kwargs)

    def extract_json(self, text: str) -> Optional[Dict[str, Any]]:
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

    def switch_provider(
        self,
        provider: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        """运行时切换提供商"""
        provider = provider.lower()
        if provider not in PROVIDERS:
            raise ValueError(f"不支持的提供商: {provider}")

        self.provider = provider
        self.provider_config = PROVIDERS[provider]

        new_key = api_key or os.environ.get(self.provider_config["env_key"])
        if not new_key:
            raise ValueError(f"请提供 {self.provider_config['name']} 的API密钥")

        self.api_key = new_key
        self.model = model or self.provider_config["default_model"]
        self.client = self._create_chat_model()

    def switch_model(self, model: str):
        """
        运行时切换模型(仅限当前提供商支持的模型)

        Args:
            model: 模型名称(必须在当前提供商 PROVIDERS['models'] 列表中)
        """
        available_models = self.provider_config.get("models", [])
        if model not in available_models:
            raise ValueError(
                f"不支持的模型: {model}\n"
                f"当前提供商 [{self.provider_config['name']}] 可用模型: {', '.join(available_models)}"
            )
        self.model = model
        self.client = self._create_chat_model()

    def list_models(self) -> List[str]:
        """列出当前提供商支持的所有模型"""
        return list(self.provider_config.get("models", []))

    def get_info(self) -> Dict[str, str]:
        """获取当前客户端信息"""
        return {
            "provider": self.provider,
            "provider_name": self.provider_config["name"],
            "model": self.model,
            "base_url": self.provider_config["base_url"]
        }

    @staticmethod
    def _to_langchain_messages(messages: List[Dict[str, str]]) -> List[BaseMessage]:
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

    def _load_api_key_from_config(self, config_file: str, provider: str) -> Optional[str]:
        """从配置文件读取API密钥"""
        if not os.path.exists(config_file):
            return None
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config.get(provider, {}).get("api_key")
        except (json.JSONDecodeError, IOError):
            return None

    def _load_model_from_config(self, config_file: str, provider: str) -> Optional[str]:
        """从配置文件读取模型名称"""
        if not os.path.exists(config_file):
            return None
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config.get(provider, {}).get("model")
        except (json.JSONDecodeError, IOError):
            return None

def list_providers() -> Dict[str, Dict]:
    """列出所有支持的提供商"""
    return PROVIDERS


def create_client(
    provider: str = "zhipu",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    config_file: Optional[str] = None
) -> LLMClient:
    """创建LLM客户端的便捷函数"""
    return LLMClient(
        provider=provider,
        api_key=api_key,
        model=model,
        config_file=config_file
    )
