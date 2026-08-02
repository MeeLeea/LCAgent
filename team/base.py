"""
TeamAgent 轻量基类 - 为多 Agent 工作流设计的轻量角色

与 AgentCore 的区别:
- 单轮任务执行(无会话记忆/checkpoint)
- 可选工具注入(不默认加载全部工具/MCP/技能)
- 更快的构建速度,适合团队协作场景
"""
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool

from agent.llm_client import LLMClient


class TeamAgent:
    """轻量团队 Agent 基类,适用于单轮角色化任务执行"""
    
    # LLM 采样参数默认值(子类可通过类属性或 __init__ 参数覆盖)
    temperature: float = 0.7
    max_tokens: int = 2048
    
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
        self.tools = list(tools) if tools else []
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        # 工具模式:创建轻量 agent executor
        if self.tools:
            self._create_tool_agent()
        else:
            self.agent_executor = None
    
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
            print(f"[{self.name}] 执行任务(工具模式): {task[:100]}...")
        
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
                print(f"[{self.name}] 错误: {error_msg}")
            return error_msg
    
    def _invoke_pure_text(self, task: str) -> str:
        """纯文本模式执行(单次 LLM 调用)"""
        if self.verbose:
            print(f"[{self.name}] 执行任务(纯文本模式): {task[:100]}...")
        
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
                print(f"[{self.name}] 错误: {error_msg}")
            return error_msg
