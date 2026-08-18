"""
TeamAgent 轻量基类 - 为多 Agent 工作流设计的轻量角色

与 AgentCore 的区别:
- 单轮任务执行(无会话记忆/checkpoint)
- 可选工具注入(不默认加载全部工具/MCP/技能)
- 更快的构建速度,适合团队协作场景
"""
import logging
import os
from collections.abc import AsyncIterator
from typing import ClassVar, Protocol

from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool

from llm.llm_client import LLMClient

logger = logging.getLogger(__name__)

# AGENT.md 中工作流提示词小节的标题前缀(角色系统提示词加载时会被剥离)
WORKFLOW_SECTION_PREFIX = "## workflow:"


class PromptInjector(Protocol):
    """提示词注入器协议(鸭子类型,兼容 graph.common.SkillInjector)

    仅声明工作流节点需要的注入接口,避免 TeamAgent 及其子类对 graph 层的
    强依赖(循环导入防护):任何提供 inject_into_prompt 的对象皆可注入。
    """

    def inject_into_prompt(self, prompt: str, task: str) -> str:
        """把技能指引块追加到 prompt 末尾,返回注入后的提示词"""


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
            system_prompt: 角色系统提示词(可选;为空且提供 prompt_file 时自动从中解析)
            tools: 可选工具列表(为 None 或空列表时为纯文本模式)
            max_iterations: Agent 最大迭代次数(仅工具模式有效)
            verbose: 是否打印详细执行过程
            provider: LLM 提供商(如 "zhipu")
            model: LLM 模型名称(如 "glm-4-flash")
            config_file: LLM 配置文件路径(默认 config/llm_config.json)
            temperature: LLM 采样温度,不传则用类属性默认值
            max_tokens: LLM 最大生成 token 数,不传则用类属性默认值
            prompt_file: 角色 AGENT.md 路径,同时提供系统提示词与工作流节点提示词模板
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
        self.prompt_file = prompt_file
        self.tools = list(tools) if tools else []
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        # 工作流模板缓存(None 表示尚未解析;解析过一次后不再重复读文件)
        self._workflow_templates: dict[str, str] | None = None
        
        # 角色系统提示词:显式传入优先;为空时自动从 prompt_file 解析
        # (parse_prompt_sections 会剥离 ## workflow:* 小节,与模板共用同一次文件读取)
        if not system_prompt and prompt_file:
            content = self._read_prompt_file(prompt_file)
            if content is not None:
                system_prompt, templates = self.parse_prompt_sections(content)
                self._workflow_templates = templates
        self.system_prompt = system_prompt
        
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
        获取角色 AGENT.md 的 `## workflow:{name}` 小节,缺失回退类默认模板
        
        模板在 __init__ 自动解析系统提示词时(或首次调用时)解析一次,
        之后复用缓存,避免重复读文件。
        
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

        # 延迟导入避免循环依赖(agent.workspace_middleware 顶层仅依赖 langchain,
        # 且 agent_core 早已在顶层导入该模块,此处只是保险)
        from agent.workspace_middleware import WorkspaceSecurityMiddleware

        chat_model = self.llm.get_chat_model()
        
        # 不带 checkpointer,单轮执行无需持久化;挂工作空间安全中间件,
        # 使工作流内工具调用同样受 workspace 隔离约束(路径解析 + 逃逸校验)
        self.agent_executor = create_agent(
            model=chat_model,
            tools=self.tools,
            system_prompt=self.system_prompt,
            middleware=[WorkspaceSecurityMiddleware()],
        )
    
    async def ainvoke(self, task: str, config: dict | None = None) -> str:
        """异步执行单轮任务(等价 invoke 的异步版,TOKEN 级流式聚合)

        与 invoke 的区别:内部经 ``astream`` 逐块收集生成文本,LLM token 增量
        经 ``config["callbacks"]``(异步同一事件循环,透传安全)流出到外层
        NodeTrackingHandler,实现 workflow 节点执行期间的前端 TOKEN 流式。

        Args:
            task: 任务描述
            config: 外层运行配置(RunnableConfig)。透传 configurable.workspace_path
                使工具调用受 workspace 隔离约束;透传 callbacks 使 token 事件可达外层。

        Returns:
            执行结果字符串
        """
        chunks: list[str] = []
        async for chunk in self.astream(task, config):
            chunks.append(chunk)
        return "".join(chunks)
    
    async def astream(self, task: str, config: dict | None = None) -> AsyncIterator[str]:
        """异步流式执行单轮任务,逐块产出生成文本(TOKEN 级增量)

        Args:
            task: 任务描述
            config: 外层运行配置(RunnableConfig),透传 workspace_path 与 callbacks

        Yields:
            生成文本增量块(工具模式为 ReAct 循环内各 LLM 调用的流式输出;
            纯文本模式为单次 LLM 调用的流式输出)
        """
        if self.agent_executor:
            async for chunk in self._astream_with_tools(task, config):
                yield chunk
        else:
            async for chunk in self._astream_pure_text(task, config):
                yield chunk
    
    async def _astream_with_tools(self, task: str, config: dict | None = None) -> AsyncIterator[str]:
        """工具模式异步流式执行(经 agent_executor.astream_events)"""
        if self.verbose:
            logger.info("[%s] 执行任务(工具模式·异步): %s", self.name, task[:100])
        
        try:
            run_config: dict = {"recursion_limit": self.max_iterations}
            # 仅提取 workspace_path 构造最小 config,严禁转发外层 config 的
            # thread_id 等运行时字段——checkpointer 由外层 workflow 图统一管理。
            # callbacks 可安全透传:异步执行与 LangGraph 主循环同一事件循环,
            # 不再有同步 invoke + to_thread 的跨线程回调风险。
            configurable = config.get("configurable") if config else None
            if isinstance(configurable, dict) and configurable.get("workspace_path"):
                run_config["configurable"] = {"workspace_path": configurable["workspace_path"]}
            if config and config.get("callbacks"):
                run_config["callbacks"] = config["callbacks"]
            
            async for ev in self.agent_executor.astream_events(
                {"messages": [HumanMessage(content=task)]},
                config=run_config,
                version="v2",
            ):
                if ev["event"] != "on_chat_model_stream":
                    continue
                chunk = ev["data"].get("chunk")
                content = getattr(chunk, "content", None)
                if isinstance(content, str) and content:
                    yield content
        except Exception as e:
            error_msg = f"任务执行失败: {e!s}"
            if self.verbose:
                logger.error("[%s] %s", self.name, error_msg)
            yield error_msg
    
    async def _astream_pure_text(self, task: str, config: dict | None = None) -> AsyncIterator[str]:
        """纯文本模式异步流式执行(经 chat model.astream)"""
        if self.verbose:
            logger.info("[%s] 执行任务(纯文本模式·异步): %s", self.name, task[:100])
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task},
        ]
        async for chunk in self._astream_messages(messages, config):
            yield chunk
    
    async def _astream_messages(
        self,
        messages: list[dict[str, str]],
        config: dict | None = None,
    ) -> AsyncIterator[str]:
        """对任意消息列表做异步流式(纯文本通道,透传 callbacks)

        供子类自定义消息结构的 async 业务方法复用(如 summarize_context)。

        Args:
            messages: 消息列表,格式 [{"role": "system/user", "content": "..."}]
            config: 外层运行配置(RunnableConfig),透传 callbacks

        Yields:
            生成文本增量块;LLM 调用失败时 yield 一次错误信息
        """
        try:
            run_config: dict = {}
            if config and config.get("callbacks"):
                run_config["callbacks"] = config["callbacks"]
            model = self.llm.get_chat_model()
            async for chunk in model.astream(messages, config=run_config):
                content = getattr(chunk, "content", None)
                if isinstance(content, str) and content:
                    yield content
        except Exception as e:
            error_msg = f"任务执行失败: {e!s}"
            if self.verbose:
                logger.error("[%s] %s", self.name, error_msg)
            yield error_msg

