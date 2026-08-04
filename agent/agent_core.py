"""
Agent核心调度模块 - 基于LangChain 1.x + LangGraph
使用 langchain.agents.create_agent 实现工具调用，支持ReAct式推理
支持动态加载本地工具 + MCP Server工具
"""
import asyncio
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Any, Deque, List, Optional, Literal
from langchain.agents import create_agent
from langchain_core.messages import (
    HumanMessage, AIMessage, SystemMessage, ToolMessage, BaseMessage,
)
from langchain_core.tools import BaseTool
from langgraph.types import Command, Interrupt
from .llm_client import LLMClient
from .memory import AgentMemory
from .message_utils import StreamHandler
from .compaction import (
    CompactionConfig,
    LCAgentCompactionMiddleware,
    LCAgentState,
)
from tools.mcp_loader import DEFAULT_CONFIG_FILE
from tools.mcp_pool import MCPPool, ServerStatus
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
        agent_core_prompt: Optional[str] = None,
        max_execution_history: int = 100,
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
            max_execution_history: 执行历史最大条数(防止内存泄漏)
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

        # 压缩配置：before_model 中间件自动触发 + manually_compact 手动触发
        # summary 存入 LangGraph state（随 checkpoint 持久化），天然 per-thread 隔离
        self.compaction_config = CompactionConfig.from_kwargs(
            max_context_messages=max_context_messages,
            context_trim_keep=context_trim_keep,
        )

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

        # MCP 连接池（per-server 连接管理 + 健康探测 + 自动重连）
        self._mcp_pool = MCPPool(self.mcp_config_file)

        # 异步互斥锁：保护 tools / mcp_tools / agent_executor / active_skills 等共享状态
        # 必须在任何异步方法调用之前初始化，因为 areload_mcp_tools / _arebuild_agent_executor 等都会访问它
        self._state_lock = asyncio.Lock()

        # 技能阅读(本地 .agents/skills)
        if skills_dir is None:
            skills_dir = default_skills_dir()
        self.skill_manager = SkillManager(skills_dir)
        self.active_skills: set = set()      # 由 CLI (skill:<name>) 手动加载
        self.auto_match_skills = auto_match_skills  # 任务开始时自动匹配注入

        # 启动时加载 MCP 工具（通过连接池）
        if enable_mcp:
            asyncio.run(self.areload_mcp_tools())

        # 可变 SystemMessage：content 在每次 invoke 前动态更新（技能匹配），
        # 避免 create_agent 重新编译 Graph。model_node 闭包捕获此对象引用，
        # 修改 .content 即可让下次 LLM 调用看到最新提示词。
        self._system_message = SystemMessage(content=self._get_system_prompt(""))

        # 创建Agent（编译一次，后续不再因技能变化而重建）
        self.agent_executor = self._create_agent_executor()

        # 执行历史（deque 有界队列，防止内存泄漏）
        self.execution_history: Deque[Dict[str, Any]] = deque(maxlen=max_execution_history)
        self._max_execution_history = max_execution_history
        self._recorded_tool_call_ids: set[str] = set()

        # 流式事件处理器（组合模式）
        self.stream = StreamHandler(self)

    async def areload_mcp_tools(self) -> int:
        """
        异步重新加载 MCP 工具（通过 MCPPool 全量重连）

        使用 _state_lock 保护 tools 和 agent_executor 的并发修改。
        仅当工具签名（工具名集合）变化时才重建 Graph，否则只更新系统提示词。

        对于单个 server 的重连，推荐使用 areload_mcp_server(name)。

        Returns:
            加载到的 MCP 工具数量
        """
        async with self._state_lock:
            try:
                old_signature = getattr(self, "_tools_signature", frozenset())
                count = await self._async_load_mcp_tools()
                # 合并工具列表
                self.tools = list(self.local_tools) + list(self.mcp_tools)
                new_signature = frozenset(t.name for t in self.tools)

                if hasattr(self, "agent_executor"):
                    if new_signature != old_signature:
                        # 工具列表变化，必须重建 Graph
                        await self._arebuild_agent_executor("")
                    else:
                        # 工具列表不变，只更新系统提示词
                        self._update_system_prompt("")
                return count
            except Exception as e:
                print(f"[MCP] 重新加载失败: {e}")
                return 0

    async def areload_mcp_server(self, name: str) -> bool:
        """重连单个 MCP server（不影响其他 server）

        Args:
            name: server 名称

        Returns:
            True=重连成功
        """
        async with self._state_lock:
            try:
                old_signature = getattr(self, "_tools_signature", frozenset())
                success = await self._mcp_pool.reload_server(name)
                if not success:
                    if self.verbose:
                        print(f"[MCP] {name}: 重连失败或已移除")
                # 从池中获取最新工具列表
                self.mcp_tools = self._mcp_pool.get_all_tools()
                self.tools = list(self.local_tools) + list(self.mcp_tools)
                new_signature = frozenset(t.name for t in self.tools)

                if hasattr(self, "agent_executor"):
                    if new_signature != old_signature:
                        await self._arebuild_agent_executor("")
                    else:
                        self._update_system_prompt("")
                return success
            except Exception as e:
                print(f"[MCP] {name}: 重连失败 - {e}")
                return False

    async def _async_load_mcp_tools(self) -> int:
        """通过 MCPPool 初始化所有 MCP 连接"""
        tool_count = await self._mcp_pool.initialize()
        self.mcp_tools = self._mcp_pool.get_all_tools()
        if self.mcp_tools and self.verbose:
            # 按 server 分组展示
            for info in self._mcp_pool.get_server_infos():
                if info.status == ServerStatus.CONNECTED:
                    print(f"[MCP] {info.name}: {info.tool_count} 个工具 ({', '.join(info.tool_names)})")
                elif info.status == ServerStatus.ERROR:
                    print(f"[MCP] {info.name}: 连接失败 - {info.last_error}")
        elif self.verbose:
            print("[MCP] 未加载到任何工具(可能配置为空或服务器未启用)")
        return tool_count

    def _create_agent_executor(self, skill_block: str = ""):
        """创建LangGraph ReAct Agent（仅在工具列表或 LLM 变化时调用）

        集成压缩中间件（before_model 自动触发增量摘要 + 工具输出 Prune）。
        summary 存入 LCAgentState.summary，随 checkpoint per-thread 持久化。

        关键优化：system_prompt 传入可变 SystemMessage 对象（self._system_message），
        而非字符串。model_node 闭包捕获此对象引用，后续修改 .content 即可动态
        更新提示词，无需重新编译 Graph。
        """
        chat_model = self.llm.get_chat_model()

        # 更新 _system_message 内容（重建时同步当前技能状态）
        self._system_message.content = self._get_system_prompt(skill_block)

        # 压缩中间件：消息超阈值时自动增量摘要 + Prune 工具输出
        compaction_middleware = LCAgentCompactionMiddleware(
            model=chat_model,
            config=self.compaction_config,
        )

        # create_agent 直接返回可调用的agent
        # system_prompt 传入可变 SystemMessage 对象，后续可通过 _update_system_prompt 动态更新
        agent = create_agent(
            model=chat_model,
            tools=self.tools,
            system_prompt=self._system_message,
            checkpointer=self.memory.get_checkpointer(),
            state_schema=LCAgentState,
            middleware=[compaction_middleware],
        )
        # 保存中间件引用，供手动压缩使用
        self._compaction_middleware = compaction_middleware
        # 记录工具签名，用于检测工具列表是否变化
        self._tools_signature = frozenset(t.name for t in self.tools)
        return agent

    def _update_system_prompt(self, task: str = "") -> None:
        """动态更新系统提示词（不重建 Graph）

        通过修改 _system_message.content 实现：
        - model_node 闭包捕获的是 SystemMessage 对象引用
        - 修改 .content 后，下次 LLM 调用自动使用新提示词
        - 无需重新编译 LangGraph，性能提升约 100x

        Args:
            task: 任务描述，用于自动匹配技能（为空时不自动匹配）
        """
        skill_block = self._compute_skill_block(task)
        new_content = self._get_system_prompt(skill_block)
        # _system_message 可能不存在（测试中用 object.__new__ 创建）
        sys_msg = getattr(self, "_system_message", None)
        if sys_msg is None:
            self._system_message = SystemMessage(content=new_content)
        else:
            sys_msg.content = new_content

    async def _arebuild_agent_executor(self, task: str = "") -> None:
        """重建 Agent（仅在工具列表或 LLM 变化时使用）

        技能变化不需要重建——使用 _update_system_prompt 即可。
        此方法仅在以下场景调用：
        - MCP 工具列表变化（areload_mcp_tools 检测到工具签名不同）
        - LLM 切换（aswitch_llm，model 对象变化）

        注意：调用方必须已持有 _state_lock（此方法不再自行加锁，
        避免在 areload_mcp_tools 内部调用时死锁）。

        Args:
            task: 任务描述，用于自动匹配技能（为空时不自动匹配）
        """
        skill_block = self._compute_skill_block(task)
        self.agent_executor = self._create_agent_executor(skill_block)

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
        """获取系统提示词(可附加技能指引块)

        注意：历史对话摘要不再拼接到这里。
        压缩中间件会将摘要作为 SystemMessage 放入 messages 列表头部，
        与 system_prompt 分离，避免实例级共享状态污染。
        """
        base = self.agent_core_prompt
        if skill_block:
            base = base + "\n" + skill_block
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

    async def _arepair_rejected_tool_calls(self, config: Dict[str, Any]) -> None:
        """异步修复 checkpoint 中未完成的工具调用（补齐取消结果）。

        使用 aupdate_state 替代 asyncio.to_thread(update_state)，
        真正异步执行，不阻塞事件循环。
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

        使用 asyncio.to_thread 将同步 invoke 跑在线程池中，避免阻塞事件循环。
        压缩由 before_model 中间件自动触发，无需手动调用。
        系统提示词通过 _update_system_prompt 动态更新，不重建 Graph。
        """
        config = self._invoke_config()
        input_msg = HumanMessage(content=task)

        # 动态更新系统提示词（技能匹配），不重建 Graph
        self._update_system_prompt(task)

        try:
            result = await asyncio.to_thread(
                self.agent_executor.invoke,
                {"messages": [input_msg]},
                config=config,
            )
        except UserRejectedCommandError:
            return await self._ahandle_rejected_command(config)

        self._record_tool_steps(result.get("messages", []), input_msg)
        turn = self._parse_turn_result(result)
        self._handle_turn_completion(turn, config, "run", task, important=True)

        return turn

    async def achat_structured(self, message: str) -> AgentTurnResult:
        """异步对话（结构化入口）

        压缩由 before_model 中间件自动触发，无需手动调用。
        系统提示词通过 _update_system_prompt 动态更新，不重建 Graph。
        """
        config = self._invoke_config()

        with self._temp_verbose(False):
            self._update_system_prompt(message)
            try:
                result = await asyncio.to_thread(
                    self.agent_executor.invoke,
                    {"messages": [HumanMessage(content=message)]},
                    config=config,
                )
            except UserRejectedCommandError:
                return await self._ahandle_rejected_command(config)

        turn = self._parse_turn_result(result)
        self._handle_turn_completion(turn, config, "chat", message)

        return turn

    async def aresume_structured(self, payload: Dict[str, Any]) -> AgentTurnResult:
        """异步恢复中断会话（结构化入口）"""
        config = self._invoke_config()
        pending_thread_id = getattr(self, "_pending_interrupt_thread_id", None)
        current_thread_id = self._thread_id_from_config(config)

        if pending_thread_id is not None and current_thread_id != pending_thread_id:
            raise ValueError("Cannot resume interrupt on a different thread")

        try:
            result = await asyncio.to_thread(
                self.agent_executor.invoke,
                Command(resume=payload),
                config=config,
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

    async def aload_skill(self, name: str) -> bool:
        """
        异步加载技能到当前会话

        使用 _state_lock 保护 active_skills 的并发修改。
        技能变化只需更新系统提示词，不重建 Graph。

        Args:
            name: 技能名(目录名或 frontmatter name)

        Returns:
            True=成功加载, False=技能不存在
        """
        if self.skill_manager.get_skill(name) is None:
            return False
        async with self._state_lock:
            self.active_skills.add(name)
            self._update_system_prompt("")
        return True

    async def aclear_skills(self):
        """异步清空手动加载的技能（不重建 Graph）"""
        async with self._state_lock:
            self.active_skills.clear()
            self._update_system_prompt("")

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

    async def manually_compact(self) -> dict[str, Any] | None:
        """手动触发一次上下文压缩（CLI 命令 compact 调用）

        与 before_model 中间件使用相同的压缩逻辑（增量摘要 + 工具输出 Prune），
        但通过 aupdate_state 直接写入 checkpoint，不依赖 LangGraph 中间件上下文。

        Returns:
            {"summary": str, "messages_before": int, "messages_after": int} 或 None
        """
        msgs = self.memory.get_messages()
        if not msgs:
            return None

        # 读取当前 state 中的已有摘要
        config = self._invoke_config()
        state = self.agent_executor.get_state(config)
        existing_summary = ""
        if state and state.values:
            existing_summary = state.values.get("summary", "") or ""

        # 调用中间件的手动压缩接口
        mw = getattr(self, "_compaction_middleware", None)
        if mw is None:
            return None

        update = await mw.arun_compaction(msgs, existing_summary=existing_summary)
        if update is None:
            if self.verbose:
                print("[压缩] 消息数未超阈值，无需压缩")
            return None

        messages_before = len(msgs)
        # 通过 aupdate_state 写入压缩后的消息和摘要
        await self.agent_executor.aupdate_state(config, update)
        messages_after = len(update["messages"]) - 1  # 减去 RemoveMessage 标记

        if self.verbose:
            print(f"[压缩] {messages_before} → {messages_after} 条消息，摘要已更新")

        return {
            "summary": update["summary"],
            "messages_before": messages_before,
            "messages_after": messages_after,
        }

    def get_available_tools(self) -> List[str]:
        """获取可用工具名称列表"""
        return [t.name for t in self.tools]

    def get_execution_history(self) -> List[Dict[str, Any]]:
        """获取执行历史"""
        return list(self.execution_history)

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
