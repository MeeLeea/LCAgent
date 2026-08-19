"""
工作流共享组件 - 节点进度跟踪、异步执行、技能注入、跨轮次记忆压缩与会话化

从 graph/simple.py 提取的公共逻辑，供所有工作流复用：
- NodeCallback 类型别名
- NodeTrackingHandler 节点级进度回调(TOKEN 级流式)
- SkillInjector 节点 prompt 层技能注入（复用 tools.skills.SkillManager）
- arun_compiled_workflow 通用异步工作流运行器（含跨轮次记忆压缩）
- compaction 基础设施：节点级压缩 wrapper（消息通道超阈值时增量摘要）

异步化说明：
    运行器使用 ``await graph.ainvoke(...)`` 而非同步 ``graph.invoke()``，
    避免在 asyncio 事件循环中阻塞。NodeTrackingHandler 设置 ``run_inline=True``，
    使同步回调方法在 ainvoke 期间直接运行于事件循环线程（而非线程池），
    从而保证回调中对 asyncio.Queue 等非线程安全对象的操作无需额外加锁。
"""
from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from functools import partial
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage

from utils.compaction import CompactionConfig, LCAgentCompactionMiddleware
from utils.events import AgentEvent
from utils.logging_config import TraceContext

logger = logging.getLogger(__name__)

# 节点进度回调类型:接收 AgentEvent（NODE_START / NODE_END / NODE_ERROR 事件）
NodeCallback = Callable[[AgentEvent], None]


def _extract_node_output(output: Any) -> str:
    """从节点返回值提取该节点的产出文本（供 NODE_END 事件携带）。

    节点函数统一返回 ``{"xxx": result, "messages": [AIMessage(content=result)]}``
    （见 graph/simple.py / graph/rtl_graph.py 等），本函数取 ``output["messages"]``
    最后一条消息的文本作为节点产出。

    Args:
        output: 节点返回值（LangGraph on_chain_end 的 output 参数）

    Returns:
        节点产出文本；无法提取时返回空串（静默降级，不阻塞节点事件流）
    """
    if not isinstance(output, dict):
        return ""
    try:
        messages = output.get("messages")
        if not messages:
            return ""
        last = messages[-1]
        # 仅提取节点追加的 AIMessage 产出（HumanMessage/ToolMessage 非节点产出）
        if not isinstance(last, AIMessage):
            return ""
        content = last.content
        if isinstance(content, str):
            return content
        # content blocks 形式（list[dict]）:仅拼接 text 类型块
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
            return "".join(parts)
        return str(content)
    except Exception as error:
        logger.debug("提取节点产出失败: %s", error)
        return ""


class NodeTrackingHandler(BaseCallbackHandler):
    """LangGraph 节点级进度跟踪回调处理器。

    通过 ``config["callbacks"]`` 注入 ``graph.invoke``，利用 LangGraph 节点执行时
    写入的 ``metadata["langgraph_node"]`` 字段识别业务节点（子图/内部 agent 的节点
    不在 known_nodes 中会被过滤），在节点开始/结束/异常时构造 AgentEvent
    （NODE_START / NODE_END / NODE_ERROR）并转发给外部回调。

    此外捕获 ``on_chat_model_stream``：节点内 TeamAgent.astream 透传 callbacks
    后，LLM token 增量事件到达本 handler，构造 AgentEvent.token 转发给外部回调，
    实现 workflow 节点执行期间的前端 TOKEN 级流式。
    """

    def __init__(
        self,
        known_nodes: set[str],
        on_node_start: NodeCallback | None = None,
        on_node_end: NodeCallback | None = None,
        on_node_error: NodeCallback | None = None,
        on_token: NodeCallback | None = None,
    ) -> None:
        self.known_nodes = known_nodes
        self.on_node_start = on_node_start
        self.on_node_end = on_node_end
        self.on_node_error = on_node_error
        self.on_token = on_token
        # run_id -> 节点名：on_chain_end/on_chain_error 无节点名参数，需按 run_id 反查
        self._active: dict[str, str] = {}
        # ainvoke 期间同步回调默认经 run_in_executor 在线程池执行；置为 True 使回调
        # 直接运行于事件循环线程，避免回调中对 asyncio.Queue 等非线程安全对象的跨线程访问。
        self.run_inline = True

    def on_chain_start(self, serialized: Any, inputs: Any, *, run_id: str, **kwargs: Any) -> None:
        metadata = kwargs.get("metadata") or {}
        node = metadata.get("langgraph_node")
        if node in self.known_nodes:
            self._active[run_id] = node
            if self.on_node_start:
                self.on_node_start(AgentEvent.node_start(node=node))

    def on_chain_end(self, output: Any, *, run_id: str, **kwargs: Any) -> None:
        node = self._active.pop(run_id, None)
        if node and self.on_node_end:
            # 携带节点产出（从返回值 messages 通道提取），供前端渲染节点结果块
            self.on_node_end(
                AgentEvent.node_end(node=node, content=_extract_node_output(output))
            )

    def on_chain_error(self, error: BaseException, *, run_id: str, **kwargs: Any) -> None:
        node = self._active.pop(run_id, None)
        if node and self.on_node_error:
            self.on_node_error(AgentEvent.node_error(node=node))

    def on_chat_model_stream(self, response: Any, **kwargs: Any) -> None:
        """LLM token 增量 → TOKEN 事件转发。

        节点内 TeamAgent.astream 透传 callbacks 后,LLM 的 on_chat_model_stream
        事件到达此处。仅转发非空文本块(空块/工具调用参数块无展示价值)。
        thread_id/trace_id 由外层闭包补充。
        """
        if not self.on_token:
            return
        content = getattr(response, "content", None)
        if isinstance(content, str) and content:
            self.on_token(AgentEvent.token(text=content))


class SkillInjector:
    """工作流技能注入器 - 在节点 prompt 层注入技能指引块。

    不走 AgentMiddleware（那是 create_agent 内部机制），而是复用
    tools.skills.SkillManager 的确定性打分匹配 + 指引块渲染：

    1. match_skills(task): 按任务文本与技能 name/description 的关键词重叠打分
    2. render_block(names): 把命中技能正文渲染为可注入 system prompt 的指引块

    工作流节点渲染 prompt 后，将 ``build_skill_block(task)`` 的结果追加到
    prompt 末尾（或作为 ``{skills}`` 占位符替换），实现技能注入。

    Args:
        skills_dir: 技能目录路径；为 None 时使用默认目录（<项目根>/.agents/skills）
        auto_match: 是否开启自动匹配（False 时始终返回空块，需手动指定技能名）
    """

    def __init__(
        self,
        skills_dir: str | None = None,
        auto_match: bool = True,
    ) -> None:
        from tools.skills import SkillManager, default_skills_dir

        self.skill_manager = SkillManager(skills_dir or default_skills_dir())
        self.auto_match = auto_match

    def build_skill_block(self, task: str) -> str:
        """根据任务匹配技能并渲染指引块。

        Args:
            task: 用户任务描述（用于技能匹配）

        Returns:
            技能指引块文本；未命中任何技能或未开启自动匹配时返回空串
        """
        if not self.auto_match or not task:
            return ""
        names = self.skill_manager.match_skills(task)
        if not names:
            return ""
        return self.skill_manager.render_block(sorted(names))

    def inject_into_prompt(self, prompt: str, task: str) -> str:
        """将技能指引块追加到 prompt 末尾（已含 skill 块时跳过）。

        Args:
            prompt: 渲染后的节点提示词
            task: 用户任务描述

        Returns:
            注入技能指引块后的提示词
        """
        skill_block = self.build_skill_block(task)
        if not skill_block:
            return prompt
        if "【已加载的技能指引" in prompt:
            return prompt
        return f"{prompt}\n\n{skill_block}"


def _build_compaction_middleware(
    agent: Any,
    config: CompactionConfig | None = None,
) -> LCAgentCompactionMiddleware | None:
    """从团队 Agent 构造 compaction 中间件（消息通道压缩）。

    从 ``agent.llm.get_chat_model()`` 获取摘要用 LLM；agent 无 llm 属性或
    构造失败时返回 None（该工作流不启用压缩，静默降级）。

    Args:
        agent: 任一团队 Agent（manager/worker/terminator 等，取其 llm）
        config: 压缩配置；为 None 时使用默认配置（阈值 50）

    Returns:
        压缩中间件实例；无法构造时返回 None
    """
    llm = getattr(agent, "llm", None)
    get_chat_model = getattr(llm, "get_chat_model", None)
    if not callable(get_chat_model):
        logger.debug(
            "Agent %s 无 llm.get_chat_model,compaction 中间件不启用",
            getattr(agent, "name", "?"),
        )
        return None
    try:
        return LCAgentCompactionMiddleware(
            model=llm.get_chat_model(),
            config=config or CompactionConfig(),
        )
    except Exception as error:
        logger.warning("compaction 中间件构造失败,已禁用: %s", error)
        return None


def wrap_node_with_compaction(node_fn: Callable, mw: LCAgentCompactionMiddleware | None) -> Callable:
    """包装工作流节点函数,使消息通道超阈值时自动压缩。

    mw 为 None 时原样返回节点（不启用压缩）。

    Args:
        node_fn: 待包装的异步节点函数
        mw: compaction 中间件；None 表示不压缩

    Returns:
        包装后的节点函数（或原节点）
    """
    if mw is None:
        return node_fn
    return partial(_compaction_wrapper, node_fn, mw)


async def _compaction_wrapper(
    node_fn: Callable,
    mw: LCAgentCompactionMiddleware,
    state: dict[str, Any],
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """节点执行后检查消息通道,超阈值时增量压缩（节点级 compaction）。

    累计消息数 = 节点执行前 state.messages + 节点本次追加的 messages。
    超过 mw.config.max_messages 时调用 ``mw.arun_compaction(force=True)``
    生成新摘要并清空旧消息（REMOVE_ALL + 摘要 SystemMessage + 保留近期
    消息），合并进节点返回 dict 由 LangGraph reducer 落盘。

    Args:
        node_fn: 被包装的原始节点函数
        mw: compaction 中间件（含阈值与摘要 LLM）
        state: 节点输入状态
        *args/**kwargs: 透传给节点函数的额外参数（agent/injector/config 等）

    Returns:
        节点返回 dict；若触发压缩则含更新后的 messages/summary
    """
    result = await node_fn(state, *args, **kwargs)
    new_messages = result.get("messages")
    if not new_messages:
        return result
    accumulated = [*state.get("messages", []), *new_messages]
    if len(accumulated) > mw.config.max_messages:
        update = await mw.arun_compaction(
            accumulated,
            state.get("summary", "") or "",
            force=True,
        )
        if update:
            result["messages"] = update["messages"]
            result["summary"] = update["summary"]
    return result


async def arun_compiled_workflow(
    graph,
    task: str,
    state_fields: dict[str, str] | None = None,
    raw_context: str = "",
    thread_id: str | None = None,
    workspace_path: str | None = None,
    on_node_start: NodeCallback | None = None,
    on_node_end: NodeCallback | None = None,
    on_node_error: NodeCallback | None = None,
    max_history_chars: int = 6000,
    memory: Any | None = None,
    memory_thread_id: str | None = None,
    is_run_mode: bool = False,
) -> dict:
    """
    通用异步工作流运行器 - 所有工作流共享的 ainvoke 逻辑

    各工作流只需提供自己特有的初始状态字段（state_fields），通用字段
    （task / raw_context / context_summary）由本函数自动填充。

    使用 ``await graph.ainvoke(...)`` 异步执行，不阻塞事件循环；
    节点进度回调通过 ``run_inline=True`` 的 NodeTrackingHandler 在事件循环线程触发。

    跨轮次记忆压缩：
        当 graph 编译时注入 checkpointer 且传入 thread_id 时，运行前读取
        该 thread 上一轮的工作流状态（task/plan/worker_result/final_answer），
        超长时截断为摘要并拼入 raw_context，实现多轮运行间的上下文延续
        与压缩（TeamAgent 自身无状态，记忆完全由工作流图 checkpointer 承载）。

    记忆接入（workflow 会话化）：
        传入 ``memory``（MemoryManager 实例）与 ``memory_thread_id`` 时：
        - 执行前调用 ``memory.recall_text(memory_thread_id)`` 召回长期记忆
          拼入 raw_context（与 checkpoint 摘要并存，维度互补）；
        - 执行后将 ``final_answer`` 构造 DONE AgentEvent 提交
          ``memory.consume_event``（结果级记忆沉淀）。
        两者缺一或为空时静默跳过，不影响运行。

    workspace 隔离：
        workspace_path 注入 config.configurable，节点经 config 透传给
        Worker 角色，其工具调用由 WorkspaceSecurityMiddleware 约束在
        workspace 内（路径解析 + 逃逸校验）。

    Args:
        graph: 编译好的 LangGraph StateGraph
        task: 用户任务
        state_fields: 工作流特有的初始状态字段（值为空串的占位）
            如 ``{"plan": "", "worker_result": "", "final_answer": ""}``
        raw_context: 原始记忆文本，为空则不注入记忆
        thread_id: 会话线程 ID。为 None 时自动生成（无持久化绑定）；
            传入显式值时配合 checkpointer 编译的图可实现状态持久化。
        workspace_path: 会话绑定的工作空间绝对路径。为 None 时不注入
            （工作流内工具调用不做 workspace 隔离，兼容旧场景）。
        on_node_start: 节点开始回调，接收节点名
        on_node_end: 节点结束回调，接收节点名
        on_node_error: 节点异常回调，接收节点名
        max_history_chars: 跨轮次记忆摘要的最大字符数（超长截断）
        memory: MemoryManager 实例（用于长期记忆召回与结果沉淀）；None 禁用
        memory_thread_id: 长期记忆使用的会话线程 ID（通常与 thread_id 相同）
        is_run_mode: 是否运行模式（决定 DONE 事件是否标记为重要记忆）

    Returns:
        工作流执行结果字典
    """
    tid = thread_id or f"workflow-{uuid.uuid4().hex[:8]}"

    # 用 utils 的统一日志体系注入 trace_id，使工作流执行与 agent/session 日志同源可追踪
    with TraceContext(trace_id=tid, thread_id=tid):
        return await _arun_with_trace(
            graph,
            task,
            state_fields,
            raw_context,
            tid,
            workspace_path,
            on_node_start,
            on_node_end,
            on_node_error,
            max_history_chars,
            memory,
            memory_thread_id,
            is_run_mode,
        )


async def _arun_with_trace(
    graph,
    task: str,
    state_fields: dict[str, str] | None,
    raw_context: str,
    tid: str,
    workspace_path: str | None,
    on_node_start: NodeCallback | None,
    on_node_end: NodeCallback | None,
    on_node_error: NodeCallback | None,
    max_history_chars: int,
    memory: Any | None,
    memory_thread_id: str | None,
    is_run_mode: bool,
) -> dict:
    """在 TraceContext 内执行工作流主体（原 arun_compiled_workflow 逻辑）。"""
    logger.info("工作流开始执行 [thread=%s]: %s", tid, task[:120])

    configurable: dict[str, Any] = {"thread_id": tid}
    if workspace_path is not None:
        configurable["workspace_path"] = workspace_path
    config: dict = {"configurable": configurable}

    # 跨轮次记忆压缩：从 checkpoint 读取上一轮工作流状态，截断为摘要注入 raw_context
    previous_summary = await _aget_previous_workflow_summary(graph, config, max_history_chars)
    if previous_summary:
        raw_context = (
            f"{raw_context}\n\n【上一轮工作流记录】\n{previous_summary}".strip()
            if raw_context
            else f"【上一轮工作流记录】\n{previous_summary}"
        )

    # 长期记忆召回注入（workflow 会话化记忆源；与 checkpoint 摘要并存,维度互补）
    if memory is not None and memory_thread_id:
        recalled = await memory.recall_text(memory_thread_id)
        if recalled:
            raw_context = (
                f"{raw_context}\n\n{recalled}".strip() if raw_context else recalled
            )

    initial_state: dict[str, str] = {
        "task": task,
        "raw_context": raw_context,
        "context_summary": "",
    }
    if state_fields:
        initial_state.update(state_fields)

    if on_node_start or on_node_end or on_node_error:
        known_nodes = {
            n.id for n in graph.get_graph().nodes.values() if not n.id.startswith("__")
        }
        config["callbacks"] = [
            NodeTrackingHandler(
                known_nodes,
                on_node_start=on_node_start,
                on_node_end=on_node_end,
                on_node_error=on_node_error,
            )
        ]

    result = await graph.ainvoke(initial_state, config=config)

    # 结果级记忆沉淀:final_answer 构造 DONE 事件提交 MemoryManager
    if memory is not None and memory_thread_id:
        final_answer = result.get("final_answer") or ""
        if final_answer:
            await memory.consume_event(
                AgentEvent.done(
                    content=final_answer,
                    thread_id=memory_thread_id,
                    role="assistant",
                    is_important=is_run_mode,
                )
            )

    logger.info("工作流执行完成 [thread=%s]", tid)
    return result


async def _aget_previous_workflow_summary(
    graph,
    config: dict[str, Any],
    max_chars: int = 6000,
) -> str:
    """读取指定 thread 上一轮工作流状态并压缩为摘要。

    .. deprecated::
        该函数（checkpoint 4 字段截断摘要）在 workflow 会话化后将由
        checkpoint messages + summary 通道取代（见 _compaction_wrapper 与
        WorkflowAdapter 的执行前消息注入）。当前保留以兼容 CLI/scheduler
        等既有调用路径，新接入方（WorkflowAdapter）不应再依赖它。

    仅当 graph 编译时带 checkpointer 且该 thread 已有历史时才返回非空；
    无 checkpointer / 无历史 / 读取失败时返回空串（静默降级）。

    Args:
        graph: 编译好的 LangGraph StateGraph
        config: 含 configurable.thread_id 的调用配置
        max_chars: 摘要最大字符数，超长截断

    Returns:
        上一轮工作流记录的摘要文本；无历史时返回空串
    """
    try:
        state = await graph.aget_state(config)
    except Exception as error:
        # 无 checkpointer 或读取失败：无跨轮次记忆，静默降级（仅记录 debug 便于排查）
        logger.debug("读取上一轮工作流状态失败 [thread=%s]: %s", config.get("configurable", {}).get("thread_id"), error)
        return ""
    if state is None:
        return ""

    values = getattr(state, "values", None) or {}
    # 提取可延续的上一轮字段（仅非空字段）
    fields = {k: v for k, v in values.items() if v}
    relevant = [f"{k}: {v}" for k, v in fields.items() if k in ("task", "plan", "worker_result", "final_answer")]
    if not relevant:
        return ""

    summary = "\n".join(relevant)
    if len(summary) > max_chars:
        summary = summary[:max_chars] + "\n...(上一轮工作流记录过长，已截断)"
    return summary
