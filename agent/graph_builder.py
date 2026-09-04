"""Agent 构建 Mixin - AgentCore 的 LangGraph executor 创建与重建。

从 agent_core.py 抽离，职责：
- 创建 / 重建 LangGraph ReAct Agent（create_agent）
- 组装中间件链（工具错误纠错 + 压缩 + 技能注入 + 工作空间安全 + 外部扩展）
- 系统提示词读取、LLM 切换

依赖 AgentCore 实例属性：llm / tools / tool_timeout / _checkpointer /
_store / _extra_middleware / compaction_config / metrics / skill_manager /
auto_match_skills / _state_lock / agent_core_prompt / verbose。
"""
from __future__ import annotations

import logging

from langchain.agents import create_agent

from llm.llm_client import LLMClient
from skmng.middleware import SkillInjectionMW
from tools.tool_wrapper import wrap_tools_with_timeout

from .compaction import LCAgentCompactionMiddleware, LCAgentState
from .terminal_retry_cap_mw import TerminalRetryCapMW
from .tool_arg_validator_mw import ToolArgValidatorMW
from .tool_error_mw import ToolExecutionErrorMW
from .workspace_mw import WorkspaceSecurityMW

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Agent 构建 Mixin（供 AgentCore 多继承使用，自身不初始化状态）"""

    def _create_agent_executor(
        self,
        skill_block: str = "",
    ):
        """创建LangGraph ReAct Agent（仅在工具列表或 LLM 变化时调用）

        集成压缩中间件（before_model 自动触发增量摘要 + 工具输出 Prune）+
        技能注入中间件（awrap_model_call 从 state 读取 active_skills 并注入提示词）。

        关键设计：system_prompt 传入静态字符串（不再使用可变 SystemMessage），
        技能隔离由 SkillInjectionMW + LCAgentState.active_skills 保证
        （随 checkpoint per-thread 隔离），所有会话共享同一编译图。

        Args:
            skill_block: 构建时初始技能指引块（仅用于日志/调试，实际注入由中间件完成）
        """
        chat_model = self.llm.get_chat_model()

        # 静态系统提示词（技能注入由 SkillInjectionMW 在 model 调用时完成）
        system_prompt = self._get_system_prompt()

        # 压缩中间件：消息超阈值时自动增量摘要 + Prune 工具输出
        compaction_middleware = LCAgentCompactionMiddleware(
            model=chat_model,
            config=self.compaction_config,
            on_compaction=self.metrics.record_compaction,
        )

        # 技能注入中间件：从 state.active_skills 读取技能并注入 system prompt
        skill_middleware = SkillInjectionMW(
            skill_manager=self.skill_manager,
            auto_match=self.auto_match_skills,
        )
        self._skill_middleware = skill_middleware

        # 工作空间安全中间件：拦截文件/执行类工具，注入 workspace 路径 + 逃逸校验
        workspace_middleware = WorkspaceSecurityMW()

        # 工具参数校验中间件：拦截违反语义约束的参数组合（如 head/tail 互斥），
        # 冲突时返回 error ToolMessage，LLM 下一轮 ReAct 反思修正。
        # 放在 workspace 之前（更外层）：参数校验不依赖 workspace，纯参数检查
        tool_arg_validator_middleware = ToolArgValidatorMW()

        # 终端命令超时重试上限中间件：读 state 统计 exec 工具历史超时次数，
        # 达上限(3次)则拦截返回失败 ToolMessage，阻止主模型无限重试超时命令
        # （方案 B：主模型自行反思改命令重试，本中间件只做硬性 cap）
        # 放在 middleware 列表最前 = 最外层，最先拦截，包住 tool_error_mw
        terminal_retry_middleware = TerminalRetryCapMW()

        # 工具错误纠错中间件：捕获工具执行异常 → 转 ToolMessage(status="error")，
        # 附加异常类型 + workspace 提示 + 反思指令，使 LLM 能读到报错并修正重试
        tool_error_middleware = ToolExecutionErrorMW()

        # create_agent 直接返回可调用的agent
        wrapped_tools = wrap_tools_with_timeout(self.tools, self.tool_timeout)
        agent = create_agent(
            model=chat_model,
            tools=wrapped_tools,
            system_prompt=system_prompt,
            checkpointer=self._checkpointer,
            store=self._store,
            state_schema=LCAgentState,
            middleware=[
                terminal_retry_middleware,
                tool_error_middleware,
                compaction_middleware,
                skill_middleware,
                tool_arg_validator_middleware,
                workspace_middleware,
                *self._extra_middleware,
            ],
        )
        # 保存中间件引用，供手动压缩使用
        self._compaction_middleware = compaction_middleware
        # 记录工具签名，用于检测工具列表是否变化
        self._tools_signature = frozenset(t.name for t in self.tools)
        return agent

    async def _arebuild_agent_executor(self) -> None:
        """重建 Agent（仅在工具列表或 LLM 变化时使用）

        技能变化不需要重建——由 SkillInjectionMW 在 model 调用时
        从 state 动态读取，无需重建 Graph。
        此方法仅在以下场景调用：
        - MCP 工具列表变化（areload_mcp_tools 检测到工具签名不同）
        - LLM 切换（aswitch_llm，model 对象变化）

        所有会话共享同一编译图，重建后对所有会话即时生效。

        注意：调用方必须已持有 _state_lock（此方法不再自行加锁，
        避免在 areload_mcp_tools 内部调用时死锁）。
        """
        self.agent_executor = self._create_agent_executor()

    def _get_system_prompt(self) -> str:
        """获取系统提示词。

        技能注入由 SkillInjectionMW 在 model 调用时从 state 读取，
        历史对话摘要由压缩中间件作为 SystemMessage 放入 messages 列表头部，
        两者均与 system_prompt 分离，避免实例级共享状态污染。
        """
        return self.agent_core_prompt

    async def aswitch_llm(self, llm_client: LLMClient):
        """
        异步切换LLM提供商

        使用 _state_lock 保护共享状态。

        Args:
            llm_client: 新的LLM客户端实例
        """
        async with self._state_lock:
            self.llm = llm_client
            await self._arebuild_agent_executor()