"""
Agent核心调度模块 - 基于LangChain 1.x + LangGraph
使用 langchain.agents.create_agent 实现工具调用，支持ReAct式推理
支持动态加载本地工具 + MCP Server工具
"""
import asyncio
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Literal
from langchain.agents import create_agent
from langchain_core.messages import (
    HumanMessage, AIMessage, SystemMessage, ToolMessage, BaseMessage,
)
from langchain_core.tools import BaseTool
from langgraph.types import Command, Interrupt
from .llm_client import LLMClient
from .memory import AgentMemory
from .message_utils import StreamHandler
from tools.mcp_loader import load_mcp_tools, DEFAULT_CONFIG_FILE
from tools.skills import SkillManager, default_skills_dir
from tools.terminal_tools import UserRejectedCommandError

@dataclass(frozen=True, slots=True)
class AgentTurnResult:
    """Typed result for one LangGraph turn."""

    status: Literal["completed", "interrupted", "cancelled"]
    output: str | None = None
    interrupts: list[Interrupt] = field(default_factory=list)

    @classmethod
    def completed(cls, output: str) -> "AgentTurnResult":
        return cls(status="completed", output=output, interrupts=[])

    @classmethod
    def interrupted(cls, interrupts: list[Interrupt]) -> "AgentTurnResult":
        return cls(status="interrupted", output=None, interrupts=interrupts)

    @classmethod
    def cancelled(cls, output: str) -> "AgentTurnResult":
        return cls(status="cancelled", output=output, interrupts=[])

    @property
    def is_completed(self) -> bool:
        return self.status == "completed"

    @property
    def is_interrupted(self) -> bool:
        return self.status == "interrupted"


class AgentCore:
    """基于LangChain 1.x 的Agent核心调度器"""

    def __init__(
        self,
        llm_client: LLMClient,
        name: str = "LCAgent",
        memory_size: int = 10,
        long_term_memory_file: Optional[str] = None,
        checkpoint_file: Optional[str] = None,
        thread_id: Optional[str] = None,
        max_iterations: int = 25,
        verbose: bool = True,
        mcp_config_file: Optional[str] = None,
        enable_mcp: bool = True,
        skills_dir: Optional[str] = None,
        auto_match_skills: bool = True,
        max_context_messages: int = 0,
        context_trim_keep: int = 12,
        process_type: Optional[str] = None,
        agent_core_prompt: Optional[str] = None
    ):
        """
        初始化Agent核心

        Args:
            llm_client: LLM客户端实例
            name: Agent名称(为 None 时使用默认名)
            memory_size: 仅兼容旧 API(checkpoint 不限容量)
            long_term_memory_file: 长期记忆 JSON 文件(用于 compress)
            checkpoint_file: SQLite checkpoint 文件路径(为 None 时用内存)
            thread_id: 会话线程 ID(为 None 时自动生成)
            max_iterations: Agent最大迭代次数(langgraph recursion_limit)
            verbose: 是否打印详细执行过程
            mcp_config_file: MCP servers 配置文件路径
            enable_mcp: 是否启用 MCP 工具加载
            max_context_messages: 长上下文裁剪阈值(0=关闭);超过则自动摘要并开新会话
            context_trim_keep: 裁剪时保留的最近消息条数
            process_type: 进程类型标识(server/scheduler/feishu)，用于多进程隔离
            agent_core_prompt: Agent核心系统提示词(为 None 时使用配置默认值)
        """
        self.name = name
        self.llm = llm_client
        self.memory = AgentMemory(
            checkpoint_file=checkpoint_file,
            long_term_file=long_term_memory_file,
            thread_id=thread_id,
            short_term_size=memory_size,
            use_sqlite=checkpoint_file is not None,
            process_type=process_type
        )
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.max_context_messages = max_context_messages
        self.context_trim_keep = context_trim_keep
        self.compaction_summary = ""  # 长上下文裁剪后的历史摘要(注入 system prompt)

        # 存储核心提示词（从配置加载或使用默认值）
        from .config import _DEFAULT_AGENT_CORE_PROMPT
        self.agent_core_prompt = agent_core_prompt or _DEFAULT_AGENT_CORE_PROMPT

        # 本地工具（lazy import 打破潜在循环依赖）
        from tools import all_tools as _local_tools
        self.local_tools: List[BaseTool] = list(_local_tools)
        # MCP 工具(从 MCP Server 加载)
        self.mcp_tools: List[BaseTool] = []
        # 合并后的完整工具列表
        self.tools: List[BaseTool] = list(self.local_tools)

        # MCP 配置
        self.mcp_config_file = mcp_config_file or DEFAULT_CONFIG_FILE
        self.enable_mcp = enable_mcp

        # 技能阅读(本地 .agents/skills)
        if skills_dir is None:
            skills_dir = default_skills_dir()
        self.skill_manager = SkillManager(skills_dir)
        self.active_skills: set = set()      # 由 CLI (skill:<name>) 手动加载
        self.auto_match_skills = auto_match_skills  # 任务开始时自动匹配注入

        # 启动时加载 MCP 工具
        if enable_mcp:
            self.reload_mcp_tools()

        # 创建Agent
        self.agent_executor = self._create_agent_executor()

        # 执行历史
        self.execution_history: List[Dict[str, Any]] = []
        self._recorded_tool_call_ids: set[str] = set()

        # 流式事件处理器（组合模式）
        self.stream = StreamHandler(self)

        # 异步互斥锁：保护 tools / mcp_tools / agent_executor / active_skills 等共享状态
        self._state_lock = asyncio.Lock()

    def reload_mcp_tools(self) -> int:
        """
        重新加载 MCP 工具(同步兼容壳，已废弃)

        业务代码请使用 areload_mcp_tools() 异步方法。

        Returns:
            加载到的 MCP 工具数量
        """
        import warnings
        warnings.warn(
            "reload_mcp_tools() 已废弃，请使用 areload_mcp_tools()",
            DeprecationWarning,
            stacklevel=2,
        )
        return asyncio.run(self.areload_mcp_tools())

    async def areload_mcp_tools(self) -> int:
        """
        异步重新加载 MCP 工具

        使用 _state_lock 保护 tools 和 agent_executor 的并发修改。

        Returns:
            加载到的 MCP 工具数量
        """
        async with self._state_lock:
            try:
                count = await self._async_load_mcp_tools()
                # 合并工具列表
                self.tools = list(self.local_tools) + list(self.mcp_tools)
                # 重建Agent(保留已加载的技能)
                if hasattr(self, "agent_executor"):
                    await self._arebuild_agent_executor("")
                return count
            except Exception as e:
                print(f"[MCP] 重新加载失败: {e}")
                return 0

    async def _async_load_mcp_tools(self) -> int:
        """异步加载 MCP 工具"""
        tools = await load_mcp_tools(self.mcp_config_file)
        self.mcp_tools = tools
        if tools and self.verbose:
            print(f"[MCP] 已加载 {len(tools)} 个工具: {', '.join(t.name for t in tools)}")
        elif self.verbose:
            print("[MCP] 未加载到任何工具(可能配置为空或服务器未启用)")
        return len(tools)

    def _create_agent_executor(self, skill_block: str = ""):
        """创建LangGraph ReAct Agent"""
        chat_model = self.llm.get_chat_model()

        # create_agent 直接返回可调用的agent
        # system_prompt 参数作为系统提示词
        # checkpointer 让 Agent 自动持久化状态到 SQLite
        agent = create_agent(
            model=chat_model,
            tools=self.tools,
            system_prompt=self._get_system_prompt(skill_block),
            checkpointer=self.memory.get_checkpointer(),
        )
        return agent

    def _rebuild_agent_executor(self, task: str = "") -> None:
        """统一的 Agent 重建入口（同步，已废弃，请使用 _arebuild_agent_executor）

        根据当前技能状态和任务内容重新创建 agent_executor。

        Args:
            task: 任务描述，用于自动匹配技能（为空时不自动匹配）
        """
        skill_block = self._compute_skill_block(task)
        self.agent_executor = self._create_agent_executor(skill_block)

    async def _arebuild_agent_executor(self, task: str = "") -> None:
        """统一的 Agent 异步重建入口

        根据当前技能状态和任务内容重新创建 agent_executor。
        使用 _state_lock 保护共享状态。

        Args:
            task: 任务描述，用于自动匹配技能（为空时不自动匹配）
        """
        async with self._state_lock:
            skill_block = self._compute_skill_block(task)
            self.agent_executor = self._create_agent_executor(skill_block)

    def _handle_rejected_command(self, config: Dict[str, Any]) -> AgentTurnResult:
        """处理用户拒绝执行危险命令的情况

        当工具调用被用户拒绝时，修复 checkpoint 状态并返回取消结果。

        Args:
            config: LangGraph 配置对象

        Returns:
            状态为 'cancelled' 的 AgentTurnResult
        """
        self._repair_rejected_tool_calls(config)
        self._clear_pending_interrupt()
        return AgentTurnResult.cancelled("用户已拒绝执行危险命令，当前任务已取消。")

    def _handle_turn_completion(
        self,
        turn: AgentTurnResult,
        config: Dict[str, Any],
        mode: str,
        user_message: Optional[str] = None,
        save_assistant: bool = True,
        important: bool = False
    ) -> None:
        """统一处理 turn 完成后的状态更新和记忆存储

        根据 turn 状态更新 interrupt 状态，并选择性地保存消息到记忆。

        Args:
            turn: Agent 执行结果
            config: LangGraph 配置对象
            mode: 执行模式 ('run', 'chat', 等)
            user_message: 用户消息（如需保存到记忆）
            save_assistant: 是否保存 assistant 消息到记忆
            important: assistant 消息是否标记为重要
        """
        if turn.is_interrupted:
            self._capture_pending_interrupt(config, mode)
        elif turn.is_completed:
            self._clear_pending_interrupt()
            if user_message:
                self.memory.add("user", user_message)
            if save_assistant and turn.output:
                metadata = {"important": True} if important else {}
                self.memory.add("assistant", turn.output, metadata)

    def _get_system_prompt(self, skill_block: str = "") -> str:
        """获取系统提示词(可附加技能指引块)"""
        base = self.agent_core_prompt
        if skill_block:
            base = base + "\n" + skill_block
        if self.compaction_summary:
            base = base + (
                "\n\n【历史对话摘要(上文因过长已被自动裁剪压缩)】\n"
                + self.compaction_summary
            )
        return base

    def _invoke_config(self) -> Dict[str, Any]:
        return {**self.memory.get_config(), "recursion_limit": self.max_iterations}

    def _thread_id_from_config(self, config: Dict[str, Any]) -> str | None:
        configurable = config.get("configurable")
        if isinstance(configurable, dict):
            thread_id = configurable.get("thread_id")
            if isinstance(thread_id, str):
                return thread_id
        return None

    @contextmanager
    def _temp_verbose(self, verbose: bool):
        """临时设置 verbose 状态的上下文管理器
        
        Args:
            verbose: 临时要设置的 verbose 值
            
        Yields:
            None
            
        Example:
            with self._temp_verbose(False):
                # 这里 self.verbose 为 False
                self._do_something()
            # 这里 self.verbose 恢复原值
        """
        original = self.verbose
        self.verbose = verbose
        try:
            yield
        finally:
            self.verbose = original

    def _capture_pending_interrupt(self, config: Dict[str, Any], mode: str) -> None:
        self._pending_interrupt_thread_id = self._thread_id_from_config(config)
        self._pending_interrupt_mode = mode

    def _clear_pending_interrupt(self) -> None:
        self._pending_interrupt_thread_id = None
        self._pending_interrupt_mode = None

    def _record_tool_steps(self, result_messages: List[BaseMessage], input_msg: HumanMessage) -> None:
        # LangGraph 会返回当前线程的完整消息历史，按 tool_call id 去重避免重复记账。
        recorded_ids = getattr(self, "_recorded_tool_call_ids", None)
        if recorded_ids is None:
            recorded_ids = set()
            self._recorded_tool_call_ids = recorded_ids
        new_entries_by_call_id: dict[str, Dict[str, Any]] = {}
        for msg in result_messages:
            if msg in (input_msg,):
                continue

            if isinstance(msg, AIMessage):
                tool_calls = getattr(msg, "tool_calls", None)
                if tool_calls:
                    for tc in tool_calls:
                        call_id = tc.get("id")
                        if isinstance(call_id, str) and call_id in recorded_ids:
                            continue
                        step_count = len(self.execution_history) + 1
                        if self.verbose:
                            print(f"\n--- 步骤 {step_count} ---")
                            print(f"工具: {tc.get('name', 'unknown')}")
                            print(f"输入: {tc.get('args', {})}")
                        entry = {
                            "step": step_count,
                            "tool": tc.get("name"),
                            "input": tc.get("args"),
                            "observation": ""
                        }
                        self.execution_history.append(entry)
                        if isinstance(call_id, str):
                            recorded_ids.add(call_id)
                            new_entries_by_call_id[call_id] = entry
            elif hasattr(msg, "content") and hasattr(msg, "tool_call_id"):
                call_id = getattr(msg, "tool_call_id")
                entry = new_entries_by_call_id.get(call_id)
                if entry is not None:
                    entry["observation"] = str(msg.content)[:500]
                if self.verbose and entry is not None:
                    print(f"结果: {str(msg.content)[:200]}...")

    def _check_and_raise_if_interrupted(self, turn: AgentTurnResult) -> None:
        """检查 turn 是否被中断，如果是则抛出异常
        
        Args:
            turn: Agent 执行结果
            
        Raises:
            RuntimeError: 如果 turn 状态为 interrupted
        """
        if turn.status == "interrupted":
            raise RuntimeError("Agent turn interrupted; resume with resume_structured().")

    def _parse_turn_result(self, result: Dict[str, Any]) -> AgentTurnResult:
        interrupts = result.get("__interrupt__")
        if interrupts:
            return AgentTurnResult.interrupted(list(interrupts))

        # 找到最后一条有内容的 AIMessage
        messages = result.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                return AgentTurnResult.completed(str(msg.content))
        return AgentTurnResult.completed("")

    def _repair_rejected_tool_calls(self, config: Dict[str, Any]) -> None:
        """为 checkpoint 中未完成的工具调用补齐取消结果。"""
        state = self.agent_executor.get_state(config)
        messages = list(state.values.get("messages", []))
        existing_results = [message for message in messages if isinstance(message, ToolMessage)]
        answered_ids = {message.tool_call_id for message in existing_results}
        repairs = []
        for message in messages:
            if not isinstance(message, AIMessage):
                continue
            for tool_call in message.tool_calls:
                call_id = tool_call.get("id")
                if not isinstance(call_id, str) or call_id in answered_ids:
                    continue
                repairs.append(
                    ToolMessage(
                        content="用户拒绝执行危险命令，工具调用已取消。",
                        name=tool_call.get("name"),
                        tool_call_id=call_id,
                        status="error",
                    )
                )
                answered_ids.add(call_id)
        if repairs:
            # 保留并行调用已经产生的结果，再补齐缺失项，确保消息历史满足工具协议。
            self.agent_executor.update_state(
                config,
                {"messages": [*existing_results, *repairs]},
                as_node="tools",
            )

    async def _arepair_rejected_tool_calls(self, config: Dict[str, Any]) -> None:
        """异步修复 checkpoint 中未完成的工具调用（补齐取消结果）。

        使用 aupdate_state 替代 update_state，支持 AsyncSqliteSaver。
        """
        state = self.agent_executor.get_state(config)
        messages = list(state.values.get("messages", []))
        existing_results = [message for message in messages if isinstance(message, ToolMessage)]
        answered_ids = {message.tool_call_id for message in existing_results}
        repairs = []
        for message in messages:
            if not isinstance(message, AIMessage):
                continue
            for tool_call in message.tool_calls:
                call_id = tool_call.get("id")
                if not isinstance(call_id, str) or call_id in answered_ids:
                    continue
                repairs.append(
                    ToolMessage(
                        content="用户拒绝执行危险命令，工具调用已取消。",
                        name=tool_call.get("name"),
                        tool_call_id=call_id,
                        status="error",
                    )
                )
                answered_ids.add(call_id)
        if repairs:
            await self.agent_executor.aupdate_state(
                config,
                {"messages": [*existing_results, *repairs]},
                as_node="tools",
            )

    async def _ahandle_rejected_command(self, config: Dict[str, Any]) -> AgentTurnResult:
        """异步处理用户拒绝执行危险命令的情况

        当工具调用被用户拒绝时，异步修复 checkpoint 状态并返回取消结果。

        Args:
            config: LangGraph 配置对象

        Returns:
            状态为 'cancelled' 的 AgentTurnResult
        """
        await self._arepair_rejected_tool_calls(config)
        self._clear_pending_interrupt()
        return AgentTurnResult.cancelled("用户已拒绝执行危险命令，当前任务已取消。")

async def arun_structured(self, task: str) -> AgentTurnResult:
        """异步执行任务（结构化入口）

        使用 ainvoke / aupdate_state 替代同步 invoke / update_state，
        支持 AsyncSqliteSaver 和事件循环内调用。
        """
        await self._acompact_if_needed()
        config = self._invoke_config()
        input_msg = HumanMessage(content=task)

        await self._arebuild_agent_executor(task)

        try:
            result = await self.agent_executor.ainvoke(
                {"messages": [input_msg]},
                config=config
            )
        except UserRejectedCommandError:
            return await self._ahandle_rejected_command(config)

        self._record_tool_steps(result.get("messages", []), input_msg)
        turn = self._parse_turn_result(result)
        self._handle_turn_completion(turn, config, "run", task, important=True)

        return turn

    def run_structured(self, task: str) -> AgentTurnResult:
        """同步执行任务（已废弃，请使用 arun_structured）"""
        import warnings
        warnings.warn(
            "run_structured() 已废弃，请使用 arun_structured()",
            DeprecationWarning,
            stacklevel=2,
        )
        return asyncio.run(self.arun_structured(task))

    async def achat_structured(self, message: str) -> AgentTurnResult:
        """异步对话（结构化入口）"""
        await self._acompact_if_needed()
        config = self._invoke_config()

        with self._temp_verbose(False):
            await self._arebuild_agent_executor(message)
            try:
                result = await self.agent_executor.ainvoke(
                    {"messages": [HumanMessage(content=message)]},
                    config=config
                )
            except UserRejectedCommandError:
                return await self._ahandle_rejected_command(config)

        turn = self._parse_turn_result(result)
        self._handle_turn_completion(turn, config, "chat", message)

        return turn

    def chat_structured(self, message: str) -> AgentTurnResult:
        """同步对话（已废弃，请使用 achat_structured）"""
        import warnings
        warnings.warn(
            "chat_structured() 已废弃，请使用 achat_structured()",
            DeprecationWarning,
            stacklevel=2,
        )
        return asyncio.run(self.achat_structured(message))

    async def aresume_structured(self, payload: Dict[str, Any]) -> AgentTurnResult:
        """异步恢复中断会话（结构化入口）"""
        config = self._invoke_config()
        pending_thread_id = getattr(self, "_pending_interrupt_thread_id", None)
        current_thread_id = self._thread_id_from_config(config)

        if pending_thread_id is not None and current_thread_id != pending_thread_id:
            raise ValueError("Cannot resume interrupt on a different thread")

        try:
            result = await self.agent_executor.ainvoke(
                Command(resume=payload),
                config=config
            )
        except UserRejectedCommandError:
            return await self._ahandle_rejected_command(config)

        turn = self._parse_turn_result(result)

        if turn.is_interrupted:
            self._capture_pending_interrupt(
                config,
                getattr(self, "_pending_interrupt_mode", "chat")
            )
        elif turn.is_completed:
            mode = getattr(self, "_pending_interrupt_mode", None)
            self._clear_pending_interrupt()
            if mode == "run" and turn.output:
                self.memory.add("assistant", turn.output, {"important": True})

        return turn

    def resume_structured(self, payload: Dict[str, Any]) -> AgentTurnResult:
        """同步恢复中断会话（已废弃，请使用 aresume_structured）"""
        import warnings
        warnings.warn(
            "resume_structured() 已废弃，请使用 aresume_structured()",
            DeprecationWarning,
            stacklevel=2,
        )
        return asyncio.run(self.aresume_structured(payload))

    # ============ 流式接口（委托给 StreamHandler） ============

    async def astream_chat(self, message: str):
        """流式对话，委托给 self.stream。事件格式见 StreamHandler.astream_chat。"""
        async for ev in self.stream.astream_chat(message):
            yield ev

    async def astream_resume(self, payload: Dict[str, Any]):
        """流式恢复中断会话，委托给 self.stream。"""
        async for ev in self.stream.astream_resume(payload):
            yield ev

    async def arun(self, task: str) -> str:
        """
        异步执行任务（推荐使用）

        使用 arun_structured 异步执行，支持事件循环内调用。

        Args:
            task: 任务描述

        Returns:
            执行结果
        """
        print(f"\n{'='*50}")
        print(f"开始执行任务: {task}")
        print(f"{'='*50}\n")

        try:
            turn = await self.arun_structured(task)
            self._check_and_raise_if_interrupted(turn)
            output = turn.output or ""
            print(f"\n最终答案: {output}")
            return output
        except RuntimeError as e:
            if "interrupt" in str(e):
                raise
            error_msg = f"任务执行失败: {str(e)}"
            print(f"\n错误: {error_msg}")
            return error_msg
        except Exception as e:
            error_msg = f"任务执行失败: {str(e)}"
            print(f"\n错误: {error_msg}")
            return error_msg

    def run(self, task: str) -> str:
        """
        使用Agent执行任务（同步兼容壳，已废弃）

        业务代码请使用 arun() 异步方法。
        """
        import warnings
        warnings.warn(
            "run() 已废弃，请使用 arun()",
            DeprecationWarning,
            stacklevel=2,
        )
        return asyncio.run(self.arun(task))

    async def achat(self, message: str) -> str:
        """
        异步对话模式（推荐使用）

        Args:
            message: 用户消息

        Returns:
            助手回复
        """
        try:
            turn = await self.achat_structured(message)
            self._check_and_raise_if_interrupted(turn)
            return turn.output or ""
        except RuntimeError as e:
            if "interrupt" in str(e):
                raise
            return self._fallback_chat(message)
        except Exception as e:
            if e.__class__.__name__ in {"GraphInterrupt", "NodeInterrupt"}:
                raise
            return self._fallback_chat(message)

    def chat(self, message: str) -> str:
        """
        普通对话模式（同步兼容壳，已废弃）

        业务代码请使用 achat() 异步方法。
        """
        import warnings
        warnings.warn(
            "chat() 已废弃，请使用 achat()",
            DeprecationWarning,
            stacklevel=2,
        )
        return asyncio.run(self.achat(message))

    async def aresume(self, payload: Dict[str, Any]) -> str:
        """
        异步恢复中断会话（推荐使用）

        Args:
            payload: 恢复数据

        Returns:
            助手回复
        """
        turn = await self.aresume_structured(payload)
        self._check_and_raise_if_interrupted(turn)
        return turn.output or ""

    def _fallback_chat(self, message: str) -> str:
        """Agent 执行失败时降级为纯 LLM 对话"""
        history = self.memory.get_short_term()
        self.memory.add("user", message)
        response = self.llm.chat_with_history(
            user_input=message,
            history=history,
            system_prompt="你是一个有帮助的AI助手，请用中文回答。"
        )
        self.memory.add("assistant", response)
        return response

    def cot(self, task: str) -> str:
        """
        CoT链式思考模式（不调用工具，纯推理）

        Args:
            task: 任务描述

        Returns:
            推理结果
        """
        print(f"\n{'='*50}")
        print(f"链式思考模式: {task}")
        print(f"{'='*50}\n")

        system_prompt = (
            "你是一个智能助手，使用链式思考(Chain of Thought)来解决问题。\n"
            "请一步步分析问题，最后给出结论。\n"
            "请用中文回答。"
        )

        # CoT 模式不调用工具,直接用 LLM + checkpoint 历史
        context = self.memory.get_short_term() + self.memory.get_long_term(3)
        response = self.llm.chat_with_history(
            user_input=task,
            history=context,
            system_prompt=system_prompt,
            temperature=0.7
        )

        print(f"\n最终答案: {response}")

        # 存入记忆
        self.memory.add("user", task)
        self.memory.add("assistant", response, {"important": True})

        return response

    def switch_llm(self, llm_client: LLMClient):
        """
        切换LLM提供商（同步兼容壳，已废弃）

        业务代码请使用 aswitch_llm() 异步方法。
        """
        import warnings
        warnings.warn(
            "switch_llm() 已废弃，请使用 aswitch_llm()",
            DeprecationWarning,
            stacklevel=2,
        )
        asyncio.run(self.aswitch_llm(llm_client))

    async def aswitch_llm(self, llm_client: LLMClient):
        """
        异步切换LLM提供商

        使用 _state_lock 保护共享状态。

        Args:
            llm_client: 新的LLM客户端实例
        """
        async with self._state_lock:
            self.llm = llm_client
            await self._arebuild_agent_executor("")

    # ============ 技能阅读(Skills) ============

    def list_skills(self) -> List[Dict[str, str]]:
        """列出所有本地可用技能"""
        return self.skill_manager.list_skills()

    def load_skill(self, name: str) -> bool:
        """
        手动加载技能（同步兼容壳，已废弃）

        业务代码请使用 aload_skill() 异步方法。
        """
        import warnings
        warnings.warn(
            "load_skill() 已废弃，请使用 aload_skill()",
            DeprecationWarning,
            stacklevel=2,
        )
        return asyncio.run(self.aload_skill(name))

    async def aload_skill(self, name: str) -> bool:
        """
        异步加载技能到当前会话

        使用 _state_lock 保护 active_skills 和 agent_executor 的并发修改。

        Args:
            name: 技能名(目录名或 frontmatter name)

        Returns:
            True=成功加载, False=技能不存在
        """
        if self.skill_manager.get_skill(name) is None:
            return False
        async with self._state_lock:
            self.active_skills.add(name)
            await self._arebuild_agent_executor("")
        return True

    def clear_skills(self):
        """
        清空手动加载的技能（同步兼容壳，已废弃）

        业务代码请使用 aclear_skills() 异步方法。
        """
        import warnings
        warnings.warn(
            "clear_skills() 已废弃，请使用 aclear_skills()",
            DeprecationWarning,
            stacklevel=2,
        )
        asyncio.run(self.aclear_skills())

    async def aclear_skills(self):
        """异步清空手动加载的技能"""
        async with self._state_lock:
            self.active_skills.clear()
            await self._arebuild_agent_executor("")

    def set_auto_match(self, enabled: bool):
        """开关任务自动匹配技能"""
        self.auto_match_skills = enabled

    def _compute_skill_block(self, task: str) -> str:
        """
        计算当前应注入的技能指引块:
        - 自动匹配(若开启)命中的技能
        - 手动加载的技能(active_skills)
        两者合并去重后渲染
        """
        names = set(self.active_skills)
        if self.auto_match_skills and task:
            names.update(self.skill_manager.match_skills(task))
        if not names:
            return ""
        return self.skill_manager.render_block(sorted(names))

    # ============ 长上下文裁剪 ============

    async def _asummarize_messages(self, msgs: List[BaseMessage]) -> str:
        """异步生成对话历史摘要（供 _acompact_if_needed 使用）

        将 BaseMessage 列表格式化为文本，调用 LLM 异步接口生成中文摘要。

        Args:
            msgs: 待压缩的消息列表

        Returns:
            压缩后的中文摘要文本，失败时返回空字符串
        """
        lines = []
        for m in msgs:
            if isinstance(m, HumanMessage):
                role = "user"
            elif isinstance(m, AIMessage):
                role = "assistant"
            elif isinstance(m, SystemMessage):
                role = "system"
            else:
                role = "tool"
            content = getattr(m, "content", "")
            if isinstance(content, list):
                content = " ".join(str(x) for x in content)
            text = str(content).strip()
            if text:
                lines.append(f"{role}: {text}")
        if not lines:
            return ""
        prompt = (
            "请将以下对话历史压缩成一份简洁的中文摘要,保留关键决策、用户意图与事实,"
            "按主题分条列出,不要添加推测内容:"
        )
        try:
            summary = (await self.llm.achat([
                {"role": "system", "content": prompt},
                {"role": "user", "content": "\n".join(lines)},
            ])).strip()
        except Exception:
            summary = ""
        if not summary and self.verbose:
            print("[上下文摘要生成失败,跳过裁剪]")
        return summary

    async def _acompact_if_needed(self):
        """异步执行上下文裁剪（阈值 <= 0 时自动跳过）

        替代 _compact_if_needed() 的异步版本，避免在事件循环中调用
        self.llm.chat() 同步阻塞。
        """
        if self.max_context_messages <= 0:
            return
        msgs = self.memory.get_messages()
        if len(msgs) <= self.max_context_messages:
            return

        keep = min(self.context_trim_keep, max(len(msgs) - 1, 0))
        old = msgs[:-keep] if keep > 0 else msgs
        retained = msgs[-keep:] if keep > 0 else []

        summary = await self._asummarize_messages(old)
        if not summary:
            return

        self.compaction_summary = summary
        old_tid = self.memory.thread_id
        self.memory.new_thread()
        await self._arebuild_agent_executor("")

        if retained:
            await self.agent_executor.aupdate_state(
                self.memory.get_config(),
                {"messages": retained},
            )

    def _compact_if_needed(self):
        """调用 memory.maybe_compact 执行上下文裁剪(阈值 <= 0 时自动跳过)。

        已废弃：请使用 _acompact_if_needed() 异步方法。
        """
        import warnings
        warnings.warn(
            "_compact_if_needed() 已废弃，请使用 _acompact_if_needed()",
            DeprecationWarning,
            stacklevel=2,
        )
        asyncio.run(self._acompact_if_needed())

    def get_available_tools(self) -> List[str]:
        """获取可用工具名称列表"""
        return [t.name for t in self.tools]

    def get_execution_history(self) -> List[Dict[str, Any]]:
        """获取执行历史"""
        return self.execution_history

    def clear_history(self):
        """清空执行历史"""
        self.execution_history.clear()
        self._recorded_tool_call_ids = set()

    def get_memory_summary(self) -> Dict[str, Any]:
        """获取记忆摘要"""
        return self.memory.summarize()

    def compress_memory(self) -> Dict[str, Any]:
        """压缩长期记忆: 委托给 memory.compress_memory,用 LLM 生成摘要。"""
        def _summarize_text(text: str, prompt: str) -> str:
            try:
                return self.llm.chat([
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": text},
                ]).strip()
            except Exception:
                return ""

        return self.memory.compress_memory(_summarize_text)
