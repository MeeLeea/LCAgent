"""
工作流共享组件 - 节点进度跟踪、异步执行、技能注入与跨轮次记忆压缩

从 graph/simple.py 提取的公共逻辑，供所有工作流复用：
- NodeCallback 类型别名
- NodeTrackingHandler 节点级进度回调
- ainvoke_team_agent 异步执行团队 Agent（TeamAgent 零改动）
- SkillInjector 节点 prompt 层技能注入（复用 tools.skills.SkillManager）
- arun_compiled_workflow 通用异步工作流运行器（含跨轮次记忆压缩）

异步化说明：
    运行器使用 ``await graph.ainvoke(...)`` 而非同步 ``graph.invoke()``，
    避免在 asyncio 事件循环中阻塞。NodeTrackingHandler 设置 ``run_inline=True``，
    使同步回调方法在 ainvoke 期间直接运行于事件循环线程（而非线程池），
    从而保证回调中对 asyncio.Queue 等非线程安全对象的操作无需额外加锁。
"""
from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

# 节点进度回调类型:接收节点名
NodeCallback = Callable[[str], None]


class NodeTrackingHandler(BaseCallbackHandler):
    """LangGraph 节点级进度跟踪回调处理器。

    通过 ``config["callbacks"]`` 注入 ``graph.invoke``，利用 LangGraph 节点执行时
    写入的 ``metadata["langgraph_node"]`` 字段识别业务节点（子图/内部 agent 的节点
    不在 known_nodes 中会被过滤），在节点开始/结束/异常时转发给外部回调。
    """

    def __init__(
        self,
        known_nodes: set[str],
        on_node_start: NodeCallback | None = None,
        on_node_end: NodeCallback | None = None,
        on_node_error: NodeCallback | None = None,
    ) -> None:
        self.known_nodes = known_nodes
        self.on_node_start = on_node_start
        self.on_node_end = on_node_end
        self.on_node_error = on_node_error
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
                self.on_node_start(node)

    def on_chain_end(self, output: Any, *, run_id: str, **kwargs: Any) -> None:
        node = self._active.pop(run_id, None)
        if node and self.on_node_end:
            self.on_node_end(node)

    def on_chain_error(self, error: BaseException, *, run_id: str, **kwargs: Any) -> None:
        node = self._active.pop(run_id, None)
        if node and self.on_node_error:
            self.on_node_error(node)


async def ainvoke_team_agent(agent, prompt: str) -> str:
    """异步执行团队 Agent（TeamAgent 零改动兼容）。

    优先调用 agent.ainvoke()（若未来 TeamAgent 提供异步接口则直接使用）；
    否则用 ``asyncio.to_thread`` 包装同步 ``invoke()``，避免阻塞事件循环，
    同时保证多 Agent 并行执行时不会互相阻塞。

    Args:
        agent: TeamAgent（或任何具备 invoke()/ainvoke() 接口的对象）
        prompt: 渲染后的提示词

    Returns:
        Agent 执行结果字符串
    """
    ainvoke = getattr(agent, "ainvoke", None)
    if callable(ainvoke):
        return await ainvoke(prompt)
    return await asyncio.to_thread(agent.invoke, prompt)


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


async def arun_compiled_workflow(
    graph,
    task: str,
    state_fields: dict[str, str] | None = None,
    raw_context: str = "",
    thread_id: str | None = None,
    on_node_start: NodeCallback | None = None,
    on_node_end: NodeCallback | None = None,
    on_node_error: NodeCallback | None = None,
    max_history_chars: int = 6000,
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

    Args:
        graph: 编译好的 LangGraph StateGraph
        task: 用户任务
        state_fields: 工作流特有的初始状态字段（值为空串的占位）
            如 ``{"plan": "", "worker_result": "", "final_answer": ""}``
        raw_context: 原始记忆文本，为空则不注入记忆
        thread_id: 会话线程 ID。为 None 时自动生成（无持久化绑定）；
            传入显式值时配合 checkpointer 编译的图可实现状态持久化。
        on_node_start: 节点开始回调，接收节点名
        on_node_end: 节点结束回调，接收节点名
        on_node_error: 节点异常回调，接收节点名
        max_history_chars: 跨轮次记忆摘要的最大字符数（超长截断）

    Returns:
        工作流执行结果字典
    """
    tid = thread_id or f"workflow-{uuid.uuid4().hex[:8]}"

    config: dict = {"configurable": {"thread_id": tid}}

    # 跨轮次记忆压缩：从 checkpoint 读取上一轮工作流状态，截断为摘要注入 raw_context
    previous_summary = await _aget_previous_workflow_summary(graph, config, max_history_chars)
    if previous_summary:
        raw_context = (
            f"{raw_context}\n\n【上一轮工作流记录】\n{previous_summary}".strip()
            if raw_context
            else f"【上一轮工作流记录】\n{previous_summary}"
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

    return await graph.ainvoke(initial_state, config=config)


async def _aget_previous_workflow_summary(
    graph,
    config: dict[str, Any],
    max_chars: int = 6000,
) -> str:
    """读取指定 thread 上一轮工作流状态并压缩为摘要。

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
    except Exception:
        # 无 checkpointer 或读取失败：无跨轮次记忆，静默降级
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
