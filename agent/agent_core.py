"""
Agent核心调度模块 - 基于LangChain 1.x + LangGraph
使用 langgraph.prebuilt.create_react_agent 实现工具调用，支持ReAct式推理
支持动态加载本地工具 + MCP Server工具
"""
import asyncio
import os
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Literal
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, BaseMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command, Interrupt
from llm_client import LLMClient
from .memory import AgentMemory
from tools import all_tools as local_tools
from tools.mcp_loader import load_mcp_tools, DEFAULT_CONFIG_FILE
from tools.skills import SkillManager, default_skills_dir
from tools.terminal_tools import UserRejectedCommandError

_AGENT_CORE_PROMPT = (
    "你是一个智能助手，配备了多种工具（包括文件读写、目录管理、搜索、计算等）。\n"
    "\n"
    "【重要规则】\n"
    "1. 当用户要求创建文件、读写文件、创建目录、搜索信息等操作时，你【必须】调用相应工具来完成，"
    "绝对不要回复'我无法访问你的文件系统'、'我没有权限'、'请你自己保存'之类的话。\n"
    "2. 你确实拥有这些工具的能力，工具会在用户本地执行，可以真正创建和修改文件。\n"
    "3. 对话过程中创建文件、脚本、文件夹默认位置是项目根目录下的 'tests/' 目录（即 main.py 所在目录下的 tests/）。\n"
    "4. 如果用户要保存内容到文件，或创建文件，直接调用 write_file 工具，不要把内容贴出来让用户自己保存。\n"
    "5. 如果用户要创建目录，直接调用 create_workspace 工具，不要让用户手动操作\n"
    "6. 只有纯知识性问答（不需要操作文件/搜索/计算）才直接回答，不调用工具。\n"
    "7. 调用工具时，如果一次任务需要多步操作（例如先创建目录再写文件），请依次调用多个工具。\n"
    "8. 如果用户要求跑一下或者测试一下，直接调用相应工具或测试文件，不要让用户手动操作。\n"
    "9. 如果用户要求生成工具,直接在tools目录下创建，并使用@tool装饰器，不修改__init__.py文件。\n"
    "10. 危险命令(如 rm -rf、format、shutdown 等)会被安全策略拦截或要求用户确认，"
    "不要尝试使用破坏性命令；需要删除/移动文件时优先使用专门的文件工具。\n"
    "11. 当任务涉及某个专业领域(如提交 git、生成 pptx、查找技能等)时，"
    "应优先用 read_skill 工具读取对应技能的详细指引,并按指引完成。\n"
    "12. 当需要人工确认、选择或补充信息才能继续时，调用 ask_human 工具并提供结构化 choices，"
    "等待返回的结构化选择后再继续；不要用普通文本假装等待人工输入。\n"
    "13. 当用户要求在某个时间点（如'2分钟后'、'明天下午3点'、'下周一'）或按周期"
    "（如'每天9点'、'每周一'、'工作日下午5点半'）执行任务时，【必须】按以下三步流程操作，"
    "不要立即执行任务本身：\n"
    "    ① 调用 get_local_time 获取当前精确时间\n"
    "    ② 根据当前时间计算出 execute_time（ISO 8601，如 '2026-07-29T17:36:00'）或 cron 表达式\n"
    "    ③ 调用 schedule_task 登记任务，完成后回复'任务已登记，将于[时间]自动执行'\n"
    "    要点：\n"
    "    - task_text 只写任务本身（自然语言描述，去掉时间信息），【不要写代码或函数调用】\n"
    "    - 一次性任务 → task_type='one_time' + execute_time\n"
    "    - 周期任务   → task_type='periodic' + cron_expr（5字段：分 时 日 月 周）\n"
    "    - cron 常用示例：'0 9 * * *'=每天9点，'30 8 * * 1-5'=工作日8:30，'0 17 * * 5'=每周五17点\n"
    "14. 当用户查询或管理定时任务时（如'查一下定时任务'、'有哪些待执行任务'、'取消任务3'、'定时任务列表'），"
    "直接调用 list_scheduled_tasks（可按 status 过滤：pending/running/done/failed/cancelled）"
    "或 cancel_scheduled_task（传入任务 ID）。"
    "15. 当用户想删除已完成的任务或批量清理历史任务时，调用 delete_scheduled_task（删单个）或 cleanup_finished_tasks"
    "（清理所有 done/failed/cancelled）。如'删除任务5'、'清理已完成的任务'、'清一下历史任务'等。\n"
    "\n"
    "请用中文回答。"
)

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
        context_trim_keep: int = 12
    ):
        """
        初始化Agent核心

        Args:
            llm_client: LLM客户端实例
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
        """
        self.llm = llm_client
        self.memory = AgentMemory(
            checkpoint_file=checkpoint_file,
            long_term_file=long_term_memory_file,
            thread_id=thread_id,
            short_term_size=memory_size,
            use_sqlite=checkpoint_file is not None
        )
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.max_context_messages = max_context_messages
        self.context_trim_keep = context_trim_keep
        self.compaction_summary = ""  # 长上下文裁剪后的历史摘要(注入 system prompt)

        # 本地工具
        self.local_tools: List[BaseTool] = list(local_tools)
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

    def reload_mcp_tools(self) -> int:
        """
        重新加载 MCP 工具(同步入口)

        Returns:
            加载到的 MCP 工具数量
        """
        try:
            count = asyncio.run(self._async_load_mcp_tools())
            # 合并工具列表
            self.tools = list(self.local_tools) + list(self.mcp_tools)
            # 重建Agent(保留已加载的技能)
            if hasattr(self, "agent_executor"):
                self.agent_executor = self._create_agent_executor(
                    self._compute_skill_block("")
                )
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

        # create_react_agent 直接返回可调用的agent
        # prompt 参数作为系统提示词
        # checkpointer 让 Agent 自动持久化状态到 SQLite
        agent = create_react_agent(
            model=chat_model,
            tools=self.tools,
            prompt=self._get_system_prompt(skill_block),
            checkpointer=self.memory.get_checkpointer(),
        )
        return agent

    def _get_system_prompt(self, skill_block: str = "") -> str:
        """获取系统提示词(可附加技能指引块)"""
        base = _AGENT_CORE_PROMPT
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

    def _parse_turn_result(self, result: Dict[str, Any]) -> AgentTurnResult:
        interrupts = result.get("__interrupt__")
        if interrupts:
            return AgentTurnResult.interrupted(list(interrupts))

        for msg in reversed(result.get("messages", [])):
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

    def run_structured(self, task: str) -> AgentTurnResult:
        self._compact_if_needed()
        config = self._invoke_config()
        input_msg = HumanMessage(content=task)
        self.agent_executor = self._create_agent_executor(
            self._compute_skill_block(task)
        )
        try:
            result = self.agent_executor.invoke(
                {"messages": [input_msg]},
                config=config
            )
        except UserRejectedCommandError:
            self._repair_rejected_tool_calls(config)
            self._clear_pending_interrupt()
            return AgentTurnResult.cancelled("用户已拒绝执行危险命令，当前任务已取消。")
        self._record_tool_steps(result.get("messages", []), input_msg)
        turn = self._parse_turn_result(result)
        if turn.is_interrupted:
            self._capture_pending_interrupt(config, "run")
        elif turn.is_completed:
            self._clear_pending_interrupt()
            self.memory.add("user", task)
            self.memory.add("assistant", turn.output or "", {"important": True})
        return turn

    def chat_structured(self, message: str) -> AgentTurnResult:
        self._compact_if_needed()
        config = self._invoke_config()
        original_verbose = self.verbose
        self.verbose = False
        try:
            self.agent_executor = self._create_agent_executor(
                self._compute_skill_block(message)
            )
            try:
                result = self.agent_executor.invoke(
                    {"messages": [HumanMessage(content=message)]},
                    config=config
                )
            except UserRejectedCommandError:
                self._repair_rejected_tool_calls(config)
                self._clear_pending_interrupt()
                return AgentTurnResult.cancelled("用户已拒绝执行危险命令，当前任务已取消。")
        finally:
            self.verbose = original_verbose

        turn = self._parse_turn_result(result)
        if turn.is_interrupted:
            self._capture_pending_interrupt(config, "chat")
        elif turn.is_completed:
            self._clear_pending_interrupt()
            self.memory.add("user", message)
            self.memory.add("assistant", turn.output or "")
        return turn

    def resume_structured(self, payload: Dict[str, Any]) -> AgentTurnResult:
        config = self._invoke_config()
        pending_thread_id = getattr(self, "_pending_interrupt_thread_id", None)
        current_thread_id = self._thread_id_from_config(config)
        if pending_thread_id is not None and current_thread_id != pending_thread_id:
            raise ValueError("Cannot resume interrupt on a different thread")

        result = self.agent_executor.invoke(
            Command(resume=payload),
            config=config
        )
        turn = self._parse_turn_result(result)
        if turn.is_interrupted:
            self._capture_pending_interrupt(config, getattr(self, "_pending_interrupt_mode", "chat"))
        elif turn.is_completed:
            mode = getattr(self, "_pending_interrupt_mode", None)
            self._clear_pending_interrupt()
            if mode == "run":
                self.memory.add("assistant", turn.output or "", {"important": True})
        return turn

    def run(self, task: str) -> str:
        """
        使用Agent执行任务（自动决定是否调用工具）

        Args:
            task: 任务描述

        Returns:
            执行结果
        """
        print(f"\n{'='*50}")
        print(f"开始执行任务: {task}")
        print(f"{'='*50}\n")

        try:
            turn = self.run_structured(task)
            if getattr(turn, "status", None) == "interrupted":
                raise RuntimeError("Agent turn interrupted; resume with resume_structured().")
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

    def chat(self, message: str) -> str:
        """
        普通对话模式（也通过Agent执行，自动判断是否调用工具）

        与 run() 的区别：不打印步骤详情，不强制存入长期记忆。

        Args:
            message: 用户消息

        Returns:
            助手回复
        """
        try:
            turn = self.chat_structured(message)
            if getattr(turn, "status", None) == "interrupted":
                raise RuntimeError("Agent turn interrupted; resume with resume_structured().")
            return turn.output or ""

        except RuntimeError as e:
            if "interrupt" in str(e):
                raise
        except Exception as e:
            if e.__class__.__name__ in {"GraphInterrupt", "NodeInterrupt"}:
                raise
        return self._fallback_chat(message)

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
        切换LLM提供商

        Args:
            llm_client: 新的LLM客户端实例
        """
        self.llm = llm_client
        # 重新创建Agent(保留已加载的技能)
        self.agent_executor = self._create_agent_executor(self._compute_skill_block(""))

    # ============ 技能阅读(Skills) ============

    def list_skills(self) -> List[Dict[str, str]]:
        """列出所有本地可用技能"""
        return self.skill_manager.list_skills()

    def load_skill(self, name: str) -> bool:
        """
        手动将某技能加载进当前会话(注入后续 system prompt)

        Args:
            name: 技能名(目录名或 frontmatter name)

        Returns:
            True=成功加载, False=技能不存在
        """
        if self.skill_manager.get_skill(name) is None:
            return False
        self.active_skills.add(name)
        # 立即重建 Agent,使后续对话带上该技能指引
        self.agent_executor = self._create_agent_executor(self._compute_skill_block(""))
        return True

    def clear_skills(self):
        """清空手动加载的技能"""
        self.active_skills.clear()
        self.agent_executor = self._create_agent_executor(self._compute_skill_block(""))

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

    def _compact_if_needed(self):
        """调用 memory.maybe_compact 执行上下文裁剪(阈值 <= 0 时自动跳过)。"""
        def _summarize_messages(msgs: List[BaseMessage]) -> str:
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
                summary = self.llm.chat([
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": "\n".join(lines)},
                ]).strip()
            except Exception:
                summary = ""
            if not summary and self.verbose:
                print("[上下文摘要生成失败,跳过裁剪]")
            return summary

        def _recreate(summary: str):
            self.compaction_summary = summary
            self.agent_executor = self._create_agent_executor(
                self._compute_skill_block("")
            )
            return self.agent_executor

        self.memory.maybe_compact(
            max_context_messages=self.max_context_messages,
            context_trim_keep=self.context_trim_keep,
            summarize_callback=_summarize_messages,
            recreate_agent_callback=_recreate,
            verbose=self.verbose,
        )

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
