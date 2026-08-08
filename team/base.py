"""
TeamAgent 轻量基类 - 为多 Agent 工作流设计的轻量角色

与 AgentCore 的区别:
- 单轮任务执行(无会话记忆/checkpoint)
- 可选工具注入(不默认加载全部工具/MCP/技能)
- 更快的构建速度,适合团队协作场景
"""
import logging
import os
from typing import ClassVar

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool

from agent.llm_client import LLMClient

logger = logging.getLogger(__name__)

# AGENT.md 中工作流提示词小节的标题前缀(角色系统提示词加载时会被剥离)
WORKFLOW_SECTION_PREFIX = "## workflow:"


class TeamAgent:
    """轻量团队 Agent 基类,适用于单轮角色化任务执行"""
    
    # LLM 采样参数默认值(子类可通过类属性或 __init__ 参数覆盖)
    temperature: float = 0.7
    max_tokens: int = 2048
    
    # 工作流节点提示词的默认模板(子类覆盖;仅 AGENT.md 缺失或未定义小节时兜底)
    default_templates: ClassVar[dict[str, str]] = {}
    
    def __init__(
        self,
        name: str = "TeamAgent",
        system_prompt: str = "",
        tools: list[BaseTool] | None = None,
        max_iterations: int = 25,
        verbose: bool = False,
        provider: str = "zhipu",
        model: str | None = None,
        config_file: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        prompt_file: str | None = None,
    ):
        """
        初始化团队 Agent
        
        Args:
            name: Agent 名称
            system_prompt: 角色系统提示词(从 AGENT.md 加载)
            tools: 可选工具列表(为 None 或空列表时为纯文本模式)
            max_iterations: Agent 最大迭代次数(仅工具模式有效)
            verbose: 是否打印详细执行过程
            provider: LLM 提供商(如 "zhipu")
            model: LLM 模型名称(如 "glm-4-flash")
            config_file: LLM 配置文件路径(默认 config/llm_config.json)
            temperature: LLM 采样温度,不传则用类属性默认值
            max_tokens: LLM 最大生成 token 数,不传则用类属性默认值
            prompt_file: 角色 AGENT.md 路径,用于加载工作流节点提示词模板
        """
        # 参数优先,否则回退到类属性默认值
        self.temperature = temperature if temperature is not None else self.temperature
        self.max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        self.llm = LLMClient(
            provider=provider,
            model=model,
            config_file=config_file,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        self.name = name
        self.system_prompt = system_prompt
        self.prompt_file = prompt_file
        self.tools = list(tools) if tools else []
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        # 工作流模板缓存(懒加载:首次 get_template 时解析 AGENT.md)
        self._workflow_templates: dict[str, str] | None = None
        
        # 工具模式:创建轻量 agent executor
        if self.tools:
            self._create_tool_agent()
        else:
            self.agent_executor = None
    
    @staticmethod
    def _read_prompt_file(prompt_file: str | None) -> str | None:
        """读取提示词文件内容,文件不存在/为空/读取失败时返回 None"""
        if prompt_file and os.path.exists(prompt_file):
            try:
                with open(prompt_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        return content
            except (OSError, UnicodeDecodeError):
                pass
        return None
    
    @staticmethod
    def parse_prompt_sections(content: str) -> tuple[str, dict[str, str]]:
        """
        解析 AGENT.md 内容,拆分为系统提示词与工作流模板
        
        以 `## workflow:名称` 开头的行开始一个新的模板小节,直到下一个小节
        标题或文件结束。其余内容(含其他 `## ` 标题)归入系统提示词。
        
        Args:
            content: AGENT.md 文件内容
        
        Returns:
            (system_prompt, templates) 元组:
                - system_prompt: 剔除工作流小节后的系统提示词
                - templates: 工作流小节名 -> 模板文本 的字典
        """
        system: list[str] = []
        templates: dict[str, str] = {}
        current_name: str | None = None
        current: list[str] = []
        
        def flush() -> None:
            nonlocal current_name, current
            if current_name is None:
                system.extend(current)
            else:
                templates[current_name] = "\n".join(current).strip()
            current = []
            current_name = None
        
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith(WORKFLOW_SECTION_PREFIX):
                flush()
                current_name = stripped[len(WORKFLOW_SECTION_PREFIX):].strip()
            else:
                current.append(line)
        flush()
        
        return "\n".join(system).strip(), templates
    
    @staticmethod
    def render_template(template: str, **kwargs) -> str:
        """
        安全渲染提示词模板
        
        用字符串替换替换 {key} 占位符,而非 str.format(),
        因此模板中出现 JSON 花括号等字面量也不会报错。
        
        Args:
            template: 模板文本(含 {key} 占位符)
            **kwargs: 占位符 -> 值 的映射
        
        Returns:
            替换完成后的文本
        """
        result = template
        for key, value in kwargs.items():
            result = result.replace("{" + key + "}", str(value))
        return result
    
    def get_template(self, name: str) -> str:
        """
        懒加载角色 AGENT.md 的 `## workflow:{name}` 小节,缺失回退类默认模板
        
        Args:
            name: 工作流小节名(如 "manager_plan")
        
        Returns:
            模板文本
        """
        if self._workflow_templates is None:
            content = self._read_prompt_file(self.prompt_file)
            _, self._workflow_templates = (
                self.parse_prompt_sections(content) if content is not None else ("", {})
            )
        return self._workflow_templates.get(name, self.default_templates.get(name, ""))
    
    def _create_tool_agent(self) -> None:
        """创建轻量工具 agent(仅当有 tools 时调用)"""
        from langchain.agents import create_agent
        
        chat_model = self.llm.get_chat_model()
        
        # 不带 checkpointer,单轮执行无需持久化
        self.agent_executor = create_agent(
            model=chat_model,
            tools=self.tools,
            system_prompt=self.system_prompt,
        )
    
    def invoke(self, task: str) -> str:
        """
        执行单轮任务
        
        Args:
            task: 任务描述
            
        Returns:
            执行结果字符串
        """
        # 工具模式:通过 agent 循环执行
        if self.agent_executor:
            return self._invoke_with_tools(task)
        
        # 纯文本模式:直接 LLM 调用
        return self._invoke_pure_text(task)
    
    def _invoke_with_tools(self, task: str) -> str:
        """工具模式执行(带 ReAct 推理循环)"""
        if self.verbose:
            logger.info("[%s] 执行任务(工具模式): %s", self.name, task[:100])
        
        try:
            config = {"recursion_limit": self.max_iterations}
            result = self.agent_executor.invoke(
                {"messages": [HumanMessage(content=task)]},
                config=config
            )
            
            # 从返回的消息列表中提取最后一条 AIMessage
            messages = result.get("messages", [])
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.content:
                    return str(msg.content)
            
            return ""
            
        except Exception as e:
            error_msg = f"任务执行失败: {e!s}"
            if self.verbose:
                logger.error("[%s] %s", self.name, error_msg)
            return error_msg

    def _invoke_pure_text(self, task: str) -> str:
        """纯文本模式执行(单次 LLM 调用)"""
        if self.verbose:
            logger.info("[%s] 执行任务(纯文本模式): %s", self.name, task[:100])
        
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": task}
            ]
            response = self.llm.chat(messages)
            return response
            
        except Exception as e:
            error_msg = f"任务执行失败: {e!s}"
            if self.verbose:
                logger.error("[%s] %s", self.name, error_msg)
            return error_msg
