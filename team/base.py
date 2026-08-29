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
from typing import Any, ClassVar

from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command, Interrupt

from agent.turn_types import AgentTurnResult
from llm.llm_client import LLMClient
from skmng.core import build_skill_block as _build_skill_block
from skmng.core import inject_into_prompt as _inject_into_prompt
from skmng.manager import SkillManager, default_skills_dir
from skmng.protocols import PromptInjector
from tools.terminal_tools import UserRejectedCommandError
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
        checkpointer: Any | None = None,
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
            checkpointer: LangGraph checkpointer 实例(如 MemorySaver),用于工具模式
                agent_executor 的 interrupt/resume 支持;None 时 agent_executor 不带
                checkpointer,interrupt 不可用(纯文本模式始终不受影响)
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
        # checkpointer: 工具模式 agent_executor 编译时注入,支持 interrupt/resume
        # (纯文本模式无 agent_executor,checkpointer 不参与执行)。None 时
        # agent_executor 不带 checkpointer,interrupt 检测会因 aget_state 缺失而
        # 安全跳过(见 _aget_pending_interrupts 的 except 守卫)
        self._checkpointer = checkpointer
        
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
            checkpointer=getattr(self, "_checkpointer", None),
        )
    
    async def ainvoke(self, task: str, config: dict | None = None) -> str:
        """异步执行单轮任务(等价 invoke 的异步版,TOKEN 级流式聚合)

        与 invoke 的区别:内部经 ``astream`` 逐块收集生成文本,LLM token 增量
        经 ``config["callbacks"]``(异步同一事件循环,透传安全)流出到外层
        NodeTrackingHandler,实现 workflow 节点执行期间的前端 TOKEN 流式。

        工具模式下若用户拒绝危险命令(工具 raise UserRejectedCommandError),
        ``astream`` 会 re-raise 该异常;此方法将其转为错误文本返回(契约:
        返回 str,不抛异常),与通用 Exception 路径产出同一前缀,使工作流
        节点(worker_exec 等)能消费错误文本而非被异常打断。

        Args:
            task: 任务描述
            config: 外层运行配置(RunnableConfig)。透传 configurable.workspace_path
                使工具调用受 workspace 隔离约束;透传 callbacks 使 token 事件可达外层。

        Returns:
            执行结果字符串;用户拒绝危险命令或 LLM 异常时返回错误文本
            (以 ``TASK_ERROR_PREFIX`` 开头)
        """
        chunks: list[str] = []
        try:
            async for chunk in self.astream(task, config):
                chunks.append(chunk)
        except UserRejectedCommandError:
            # 用户拒绝危险命令:转错误文本返回(ainvoke 契约不抛异常)
            # arun_structured 走自己的 cancelled 分支,不依赖此路径
            return f"{TASK_ERROR_PREFIX}: 用户拒绝执行危险命令"
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

        与 ainvoke 的区别:返回 ``AgentTurnResult``(completed / interrupted /
        cancelled),调用方可区分"正常完成"/"工具内 interrupt()"/"LLM 调用
        失败/执行异常/用户拒绝危险命令",工作流节点可据此走 resume / 重试 /
        降级路径,而非把错误文本当正常结果消费。

        执行路径:
        - **工具模式**(agent_executor is not None):走 ``astream_events``
          聚合 token(保留流式,经 callbacks 流出到外层 NodeTrackingHandler),
          流后调 ``aget_state`` 检查 pending interrupts(对照
          agent/streaming.py:322-326)。有 interrupt → interrupted;否则
          completed。``UserRejectedCommandError`` → cancelled。
        - **纯文本模式**(agent_executor is None):走 ``astream`` 聚合,
          返回 completed,**不检查 interrupt**(纯文本通道不进 ReAct 循环,
          不会触发工具内 interrupt())。

        异常处理对齐 agent/turn_runners.py 的模式(行 50-53):
        - ``UserRejectedCommandError``(工具内用户拒绝危险命令)→ cancelled,
          output 为"用户拒绝"语义,不混入通用"任务执行失败"前缀。
          TeamAgent 无 SessionStore,故不调用 ``_arepair_rejected_tool_calls``
          / ``_aclear_pending_interrupt`` 这两个 checkpoint 副作用(AgentCore 专属)。
        - 其他 ``Exception`` → cancelled,output 以
          ``TASK_ERROR_PREFIX`` 开头(兼容 ainvoke 路径产出同一前缀)。
        - ``astream`` 内部已把 LLM 异常 ``yield`` 成错误文本,此处的
          ``except Exception`` 兜底捕获 ``astream`` 调用本身抛出的异常
          (如 LangGraph ``astream_events`` 接口异常)。

        Args:
            task: 任务描述
            config: 外层运行配置(RunnableConfig),透传 workspace_path 与 callbacks;
                启用 checkpointer 时还透传 configurable.thread_id(经
                ``_build_run_config`` 改写为内层隔离 thread_id)

        Returns:
            AgentTurnResult:
                - completed: 正常完成,output 为生成文本
                - interrupted: 工具内 interrupt() 触发,interrupts 为 pending
                  interrupts 列表(仅工具模式 + checkpointer 时可能)
                - cancelled: LLM 调用失败/执行异常/用户拒绝危险命令,
                  output 为错误信息(用户拒绝时含"用户拒绝"语义)
        """
        try:
            # 每次 arun_structured 视为一个 turn(对齐 AgentCore turn_runners.py:58)。
            # arun_structured 直接调 _astream_with_tools / _astream_pure_text,
            # 绕过了 astream 的 increment_turn,需在此显式计数。
            self.metrics.increment_turn()
            if self.agent_executor is not None:
                # 工具模式:走 astream_events 聚合 token(保留流式),流后调
                # aget_state 检查 pending interrupts(对照 streaming.py:322-326)
                run_config = self._build_run_config(config)
                chunks: list[str] = []
                async for chunk in self._astream_with_tools(task, config):
                    chunks.append(chunk)
                output = "".join(chunks)
                if output.startswith(TASK_ERROR_PREFIX):
                    return AgentTurnResult.cancelled(output)
                interrupts = await self._aget_pending_interrupts(run_config)
                if interrupts:
                    return AgentTurnResult.interrupted(interrupts)
                return AgentTurnResult.completed(output)
            # 纯文本模式:astream 聚合,不检查 interrupt(纯文本不进 ReAct 循环)
            chunks = []
            async for chunk in self._astream_pure_text(task, config):
                chunks.append(chunk)
            output = "".join(chunks)
            if output.startswith(TASK_ERROR_PREFIX):
                return AgentTurnResult.cancelled(output)
            return AgentTurnResult.completed(output)
        except UserRejectedCommandError:
            # 用户拒绝危险命令:单独识别为 cancelled,语义不混入通用错误
            # (对齐 agent/turn_runners.py:52-53,但不做 checkpoint 修复)
            return AgentTurnResult.cancelled(
                "用户已拒绝执行危险命令,当前任务已取消。"
            )
        except Exception as e:
            return AgentTurnResult.cancelled(f"{TASK_ERROR_PREFIX}: {e!s}")

    async def aresume_structured(
        self,
        payload: dict,
        config: dict | None = None,
    ) -> AgentTurnResult:
        """异步恢复中断会话(对齐 agent.TurnRunners.aresume_structured)

        用 ``Command(resume=...)`` 恢复内层 agent_executor 的 pending
        interrupts(对照 turn_runners.py:93-125)。仅工具模式 + checkpointer
        时有意义;纯文本模式无 agent_executor,直接返回 cancelled。

        流程:
        1. ``_abuild_resume_command`` 构造 Command(多 pending interrupts
           时批量恢复,对照 interrupts.py:201-239)
        2. ``agent_executor.ainvoke(resume_command, config=...)`` 恢复
        3. ``_parse_turn_result`` 检查结果(对照 turn_runners.py:207-217)
        4. 若 completed,再次调 ``_aget_pending_interrupts`` 检查是否
           恢复后又被中断(如多个 dangerous_command 串行确认)

        异常处理:
        - ``UserRejectedCommandError`` → cancelled(同 arun_structured)
        - 其他 ``Exception`` → cancelled(output 以 TASK_ERROR_PREFIX 开头)

        Args:
            payload: 恢复数据(单 interrupt 时为裸值;多 interrupt 时可为
                ``{interrupt_id: value}`` 映射,由 _abuild_resume_command 处理)
            config: 外层运行配置(RunnableConfig),透传 thread_id / callbacks

        Returns:
            AgentTurnResult:
                - completed: 恢复后正常完成
                - interrupted: 恢复后再次被中断(多 interrupt 串行场景)
                - cancelled: 纯文本模式 / 用户拒绝 / 执行异常
        """
        if self.agent_executor is None:
            return AgentTurnResult.cancelled(
                f"{TASK_ERROR_PREFIX}: 纯文本模式不支持 resume"
            )
        try:
            run_config = self._build_run_config(config)
            resume_command = await self._abuild_resume_command(run_config, payload)
            result = await self.agent_executor.ainvoke(resume_command, config=run_config)
        except UserRejectedCommandError:
            return AgentTurnResult.cancelled(
                "用户已拒绝执行危险命令,当前任务已取消。"
            )
        except Exception as e:
            return AgentTurnResult.cancelled(f"{TASK_ERROR_PREFIX}: {e!s}")
        turn = self._parse_turn_result(result)
        # 恢复后再次检查 interrupt(多 interrupt 串行确认场景)
        if turn.is_completed:
            interrupts = await self._aget_pending_interrupts(run_config)
            if interrupts:
                return AgentTurnResult.interrupted(interrupts)
        return turn

    async def _aget_pending_interrupts(self, config: dict) -> list[Interrupt]:
        """流后读取 checkpoint 的 pending interrupts(对照 streaming.py:322-326)

        未启用 checkpointer 时返回空列表(agent_executor 无 aget_state 或
        aget_state 返回 None)。读取失败时记录警告并返回空列表,不抛异常
        (arun_structured 的调用方不应因状态查询失败而把 turn 当 cancelled)。

        Args:
            config: 内层 run_config(含 thread_id,由 _build_run_config 构造)

        Returns:
            pending interrupts 列表;无或读取失败时为空
        """
        if self._checkpointer is None:
            return []
        try:
            state = await self.agent_executor.aget_state(config)
        except (AttributeError, RuntimeError, TypeError, ValueError) as e:
            if self.verbose:
                logger.warning("[%s] 读取 pending interrupts 失败: %s", self.name, e)
            return []
        if state is None:
            return []
        return [
            intr
            for task in getattr(state, "tasks", []) or []
            for intr in getattr(task, "interrupts", []) or []
        ]

    def _parse_turn_result(self, result: dict) -> AgentTurnResult:
        """解析 ainvoke 返回的 dict,检查 __interrupt__(对照 turn_runners.py:207-217)

        Args:
            result: agent_executor.ainvoke 的返回 dict

        Returns:
            - interrupted: result["__interrupt__"] 非空时,interrupts 为该列表
            - completed: 否则取最后一条有 content 的 AIMessage 的 content;
              无则空串 completed
        """
        interrupts = result.get("__interrupt__")
        if interrupts:
            return AgentTurnResult.interrupted(list(interrupts))
        messages = result.get("messages", [])
        for msg in reversed(messages):
            content = getattr(msg, "content", None)
            if content:
                return AgentTurnResult.completed(str(content))
        return AgentTurnResult.completed("")

    async def _abuild_resume_command(self, config: dict, payload: dict) -> Command:
        """构造恢复命令:多 pending interrupts 时批量恢复(对照 interrupts.py:201-239)

        并行工具调用可能产生多个 pending interrupts(如多个危险命令确认)。
        将同一答案应用到所有 pending interrupts,避免用户逐个确认。

        Args:
            config: 内层 run_config(含 thread_id,用于 aget_state)
            payload: 调用方传入的恢复数据

        Returns:
            可直接传给 agent_executor.ainvoke 的 ``Command(resume=...)``
        """
        try:
            state = await self.agent_executor.aget_state(config)
        except (AttributeError, RuntimeError, TypeError, ValueError) as e:
            if self.verbose:
                logger.warning("[%s] 读取 pending interrupts 失败: %s", self.name, e)
            return Command(resume=payload)
        if state is None:
            return Command(resume=payload)
        interrupts = [
            intr
            for task in getattr(state, "tasks", []) or []
            for intr in getattr(task, "interrupts", []) or []
        ]
        if len(interrupts) <= 1:
            return Command(resume=payload)
        # payload 已是 {interrupt_id: value} 映射
        pending_ids = {intr.id for intr in interrupts}
        if isinstance(payload, dict) and payload and pending_ids.issuperset(payload.keys()):
            return Command(resume=payload)
        # 多中断 + 裸值:将同一答案应用到所有 pending interrupts
        return Command(resume={intr.id: payload for intr in interrupts})
    
    async def _astream_with_tools(self, task: str, config: dict | None = None) -> AsyncIterator[str]:
        """工具模式异步流式执行(经 agent_executor.astream_events)

        除产出 LLM token 增量外,顺带收集运行时指标:
        - on_chat_model_end → 记录 LLM 调用(token 用量,msg id 去重)
        - on_tool_end / on_tool_error → 记录工具执行(失败/超时)
        """
        if self.verbose:
            logger.info("[%s] 执行任务(工具模式·异步): %s", self.name, task[:100])
        
        recorded_msg_ids: set[str] = set()
        run_config = self._build_run_config(config)
        try:
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
        except UserRejectedCommandError:
            # 用户拒绝危险命令:不混入通用"任务执行失败"语义,
            # re-raise 给上层(arun_structured)单独识别为 cancelled,
            # 与 agent/turn_runners.py 的 _ahandle_rejected_command 同构
            # (TeamAgent 无 SessionStore/checkpoint,故不修复 checkpoint 状态)
            raise
        except Exception as e:
            error_msg = f"{TASK_ERROR_PREFIX}: {e!s}"
            if self.verbose:
                logger.error("[%s] %s", self.name, error_msg)
            yield error_msg

    def _build_run_config(self, config: dict | None) -> dict:
        """构造 agent_executor 用的内层 run config

        与 _astream_with_tools 旧逻辑的区别:启用 checkpointer 时,从外层
        config 的 configurable.thread_id 解析出内层隔离 thread_id
        ``f"team:{self.name}:{tid}"``,写入 run_config["configurable"]["thread_id"]。
        这样:
        - 不同外层会话的 TeamAgent turn 落到不同的内层 checkpoint 线程,
          避免 interrupt 状态串台
        - 不转发外层的其他 configurable 字段(如 workspace_path 显式取值另放,
          其余字段如 trace_id 不透传,保持内层 config 最小化)

        未启用 checkpointer 时,不写 thread_id 字段,行为与旧版一致
        (agent_executor 无 checkpointer,configurable 字段被忽略)。

        Args:
            config: 外层 RunnableConfig(可 None)

        Returns:
            内层 run_config 字典
        """
        run_config: dict = {"recursion_limit": self.max_iterations}
        # 内层 thread_id 隔离:启用 checkpointer 时从外层解析 tid 并改写
        # (未启用 checkpointer 时不写,agent_executor 无 checkpointer → 字段被忽略)
        # getattr 兼容测试中 object.__new__(TeamAgent) 创建的实例(无 _checkpointer 属性)
        if getattr(self, "_checkpointer", None) is not None:
            outer_tid = self._extract_outer_thread_id(config)
            if outer_tid:
                run_config["configurable"] = {"thread_id": f"team:{self.name}:{outer_tid}"}
        # workspace_path 隔离:工具调用受 workspace 约束(独立于 thread_id)
        configurable = config.get("configurable") if config else None
        if isinstance(configurable, dict) and configurable.get("workspace_path"):
            run_config.setdefault("configurable", {})["workspace_path"] = configurable["workspace_path"]
        # callbacks 透传:异步同事件循环,无跨线程风险
        if config and config.get("callbacks"):
            run_config["callbacks"] = config["callbacks"]
        return run_config

    @staticmethod
    def _extract_outer_thread_id(config: dict | None) -> str | None:
        """从外层 RunnableConfig 的 configurable.thread_id 提取 tid

        Args:
            config: 外层 RunnableConfig(可 None)

        Returns:
            外层 thread_id 字符串;无则 None
        """
        if not config:
            return None
        configurable = config.get("configurable")
        if not isinstance(configurable, dict):
            return None
        tid = configurable.get("thread_id")
        return tid if isinstance(tid, str) and tid else None
    
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

