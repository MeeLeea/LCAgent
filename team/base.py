"""
TeamAgent 轻量基类 - 为多 Agent 工作流设计的轻量角色

与 AgentCore 的区别:
- 单轮任务执行(无会话记忆/checkpoint)
- 可选工具注入(不默认加载全部工具/MCP);工具模式挂载工具错误纠错与
  workspace 安全中间件,并做超时保护
- 内建技能注入(build_skill_block / inject_into_prompt,满足 PromptInjector 协议):
  三来源合并(角色级 fixed_skills 类属性 + 运行时 active_names + 自动匹配),
  统一转发 skmng.core 实现,使 TeamAgent 不再依赖外部注入器
- 更快的构建速度,适合团队协作场景
"""
import logging
import os
from collections.abc import AsyncIterator, Sequence
from typing import ClassVar

from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool

from agent.turn_types import AgentTurnResult
from llm.llm_client import LLMClient
from skmng.core import build_skill_block as _build_skill_block
from skmng.core import inject_into_prompt as _inject_into_prompt
from skmng.manager import SkillManager, default_skills_dir
from skmng.protocols import PromptInjector
from utils.metrics import MetricsCollector

logger = logging.getLogger(__name__)

# AGENT.md 中工作流提示词小节的标题前缀(角色系统提示词加载时会被剥离)
WORKFLOW_SECTION_PREFIX = "## workflow:"

# 任务执行失败时 astream 产出的错误文本前缀(arun_structured 据此判定 cancelled)
TASK_ERROR_PREFIX = "任务执行失败"

__all__ = ["PromptInjector", "TeamAgent"]


class TeamAgent:
    """轻量团队 Agent 基类,适用于单轮角色化任务执行"""

    # LLM 采样参数默认值(子类可通过类属性或 __init__ 参数覆盖)
    temperature: float = 0.7
    max_tokens: int = 2048

    # 工作流节点提示词的默认模板(子类覆盖;仅 AGENT.md 缺失或未定义小节时兜底)
    default_templates: ClassVar[dict[str, str]] = {}

    # 角色级固定技能依赖(子类覆盖;由 build_skill_block 统一合并注入,
    # 不走配置穿透、不走 state;aclear_skills 只清 state 不影响此处)。
    # 例: rtl_verification 设 ["vivado-2025.2"] 始终注入 Vivado Xsim 指引
    fixed_skills: ClassVar[list[str]] = []
    
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
        tool_timeout: float | None = None,
        prompt_file: str | None = None,
        skills_dir: str | None = None,
        auto_match_skills: bool = True,
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
            tool_timeout: 工具执行超时秒数(0 或 None 时使用 tools.tool_wrapper 的
                默认超时策略:全局 60 秒 + 工具级覆盖如 ask_human 600 秒)
            prompt_file: 角色 AGENT.md 路径,同时提供系统提示词与工作流节点提示词模板
            skills_dir: 技能目录路径(默认 <项目根>/.agents/skills)
            auto_match_skills: 是否开启技能自动匹配(经 build_skill_block/inject_into_prompt)
            注:经 team/factory.py 构建时,采样参数由角色 agent_config.json 解析后传入
            (未配置则回退 llm/config.py 的 DEFAULTS)
        """
        # 参数优先,否则回退到类属性默认值
        self.temperature = temperature if temperature is not None else self.temperature
        self.max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        # 工具超时:0 或 None 视为未配置,由 wrap_tools_with_timeout 落默认策略
        self.tool_timeout = (
            tool_timeout if tool_timeout is not None and tool_timeout > 0 else None
        )
        # 技能管理器(本地 .agents/skills 读取 + 任务自动匹配)
        self.skill_manager = SkillManager(skills_dir or default_skills_dir())
        self.auto_match_skills = auto_match_skills
        
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
    
    def build_skill_block(
        self,
        task: str,
        active_names: Sequence[str] = (),
    ) -> str:
        """根据任务匹配技能并渲染指引块(内建技能注入能力)

        转发 skmng.core.build_skill_block 实现三来源合并:
        - fixed_skills(self 类属性,角色级固定依赖)
        - active_names(由节点函数从 state["active_skills"] 取值传入)
        - match_skills(auto_match_skills 开启时按任务文本自动匹配)

        Args:
            task: 用户任务描述(用于技能匹配)
            active_names: 手动加载的技能名(由节点函数从 state 取值传入)

        Returns:
            技能指引块文本;未命中任何技能或未开启自动匹配时返回空串
        """
        return _build_skill_block(
            self.skill_manager,
            task,
            active_names=tuple(active_names),
            fixed_skills=tuple(self.fixed_skills),
            auto_match=self.auto_match_skills,
        )

    def inject_into_prompt(
        self,
        prompt: str,
        task: str,
        active_names: Sequence[str] = (),
    ) -> str:
        """将技能指引块追加到 prompt 末尾(已含 skill 块时跳过)

        实现 PromptInjector 协议,工作流节点可直接把 TeamAgent 实例当注入器
        传给节点函数(如 worker_exec),无需 graph 层额外创建 SkillInjector。

        Args:
            prompt: 渲染后的节点提示词
            task: 用户任务描述
            active_names: 手动加载的技能名(由节点函数从 state 取值传入)

        Returns:
            注入技能指引块后的提示词
        """
        block = self.build_skill_block(task, active_names)
        return _inject_into_prompt(prompt, block)
    
    @property
    def metrics(self) -> MetricsCollector:
        """运行时指标收集器(惰性初始化,兼容 object.__new__ 创建的测试实例)

        与 AgentCore.metrics 同构:记录 LLM 调用(tokens/耗时)、工具执行
        (次数/失败/超时)与 turn 计数;经 astream 的流式事件与纯文本通道
        自动收集,调用方经 get_summary() 汇总。
        """
        mc = getattr(self, "_metrics", None)
        if mc is None:
            mc = MetricsCollector()
            self._metrics = mc
        return mc
    
    def _create_tool_agent(self) -> None:
        """创建轻量工具 agent(仅当有 tools 时调用)"""
        from langchain.agents import create_agent

        # 延迟导入避免循环依赖(agent.workspace_mw 顶层仅依赖 langchain,
        # 且 agent_core 早已在顶层导入该模块,此处只是保险)
        from agent.tool_error_mw import ToolExecutionErrorMW
        from agent.workspace_mw import WorkspaceSecurityMW
        from tools.tool_wrapper import wrap_tools_with_timeout

        chat_model = self.llm.get_chat_model()

        # 工具超时保护(防卡死):超时/异常由包装器转 JSON 错误字符串,
        # LLM 下一轮 ReAct 循环可读到并调整参数重试
        # getattr 兼容测试中 object.__new__ 创建的实例(无 tool_timeout 属性)
        wrapped_tools = wrap_tools_with_timeout(
            self.tools, getattr(self, "tool_timeout", None)
        )

        # 中间件链:工具错误纠错(异常 → ToolMessage(status="error") + 反思指令)
        # + 工作空间安全(路径解析 + 逃逸校验),使工作流内工具调用同样受
        # workspace 隔离约束
        self.agent_executor = create_agent(
            model=chat_model,
            tools=wrapped_tools,
            system_prompt=self.system_prompt,
            middleware=[ToolExecutionErrorMW(), WorkspaceSecurityMW()],
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
        # 每次 astream 视为一次 Agent turn(ainvoke/arun_structured 均经此入口)
        self.metrics.increment_turn()
        if self.agent_executor:
            async for chunk in self._astream_with_tools(task, config):
                yield chunk
        else:
            async for chunk in self._astream_pure_text(task, config):
                yield chunk
    
    async def arun_structured(self, task: str, config: dict | None = None) -> AgentTurnResult:
        """异步执行单轮任务并返回类型化结果(对齐 agent.TurnRunners.arun_structured)

        与 ainvoke 的区别:返回 ``AgentTurnResult``(completed / cancelled),
        调用方可区分"正常完成"与"LLM 调用失败/执行异常",工作流节点可据此
        走重试或降级路径,而非把错误文本当正常结果消费。

        Args:
            task: 任务描述
            config: 外层运行配置(RunnableConfig),透传 workspace_path 与 callbacks

        Returns:
            AgentTurnResult:
                - completed: 正常完成,output 为生成文本
                - cancelled: LLM 调用失败/执行异常,output 为错误信息
        """
        try:
            chunks: list[str] = []
            async for chunk in self.astream(task, config):
                chunks.append(chunk)
            output = "".join(chunks)
            if output.startswith(TASK_ERROR_PREFIX):
                return AgentTurnResult.cancelled(output)
            return AgentTurnResult.completed(output)
        except Exception as e:
            return AgentTurnResult.cancelled(f"{TASK_ERROR_PREFIX}: {e!s}")
    
    async def _astream_with_tools(self, task: str, config: dict | None = None) -> AsyncIterator[str]:
        """工具模式异步流式执行(经 agent_executor.astream_events)

        除产出 LLM token 增量外,顺带收集运行时指标:
        - on_chat_model_end → 记录 LLM 调用(token 用量,msg id 去重)
        - on_tool_end / on_tool_error → 记录工具执行(失败/超时)
        """
        if self.verbose:
            logger.info("[%s] 执行任务(工具模式·异步): %s", self.name, task[:100])
        
        recorded_msg_ids: set[str] = set()
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
                event_name = ev["event"]
                data_dict = ev.get("data", {}) if isinstance(ev.get("data"), dict) else {}
                if event_name == "on_chat_model_stream":
                    chunk = data_dict.get("chunk")
                    content = getattr(chunk, "content", None)
                    if isinstance(content, str) and content:
                        yield content
                elif event_name == "on_chat_model_end":
                    # 记录 LLM 指标(与 agent/streaming.py 同构,msg id 去重)
                    output = data_dict.get("output")
                    if output is not None:
                        msg_id = getattr(output, "id", None) or str(id(output))
                        if msg_id not in recorded_msg_ids:
                            recorded_msg_ids.add(msg_id)
                            llm = getattr(self, "llm", None)
                            self.metrics.extract_and_record_llm_usage(
                                output,
                                provider=getattr(llm, "provider", ""),
                                model=getattr(llm, "model", "") or "",
                            )
                elif event_name == "on_tool_end":
                    # 记录工具指标:超时(wrap 层转 JSON 错误串)/失败(status="error")
                    output = data_dict.get("output")
                    content_str = str(getattr(output, "content", ""))
                    timed_out = '"error": "tool_timeout"' in content_str
                    success = not timed_out and getattr(output, "status", "success") != "error"
                    self.metrics.record_tool_call(
                        name=ev.get("name") or "",
                        success=success,
                        timed_out=timed_out,
                    )
                elif event_name == "on_tool_error":
                    # 工具内部抛异常(LangGraph 不发 on_tool_end 而发 on_tool_error)
                    self.metrics.record_tool_call(
                        name=ev.get("name") or "",
                        success=False,
                    )
        except Exception as e:
            error_msg = f"{TASK_ERROR_PREFIX}: {e!s}"
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
            last_chunk = None
            async for chunk in model.astream(messages, config=run_config):
                last_chunk = chunk
                content = getattr(chunk, "content", None)
                if isinstance(content, str) and content:
                    yield content
            # 记录 LLM 指标:从最后一个 chunk 提取 usage_metadata
            # (对齐 agent.turn_runners 的提取逻辑,无 usage 时字符粗估)
            if last_chunk is not None:
                llm = getattr(self, "llm", None)
                self.metrics.extract_and_record_llm_usage(
                    last_chunk,
                    provider=getattr(llm, "provider", ""),
                    model=getattr(llm, "model", "") or "",
                )
        except Exception as e:
            error_msg = f"{TASK_ERROR_PREFIX}: {e!s}"
            if self.verbose:
                logger.error("[%s] %s", self.name, error_msg)
            yield error_msg

