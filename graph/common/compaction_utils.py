"""压缩中间件构造工具 — 从团队 Agent 的 LLM 构造 LCAgentCompactionMiddleware。

保留用于 workflow_adapter.manually_compact 等需要手动触发压缩的场景。
节点级压缩（_compaction_wrapper）已移除，统一由
LCAgentCompactionMiddleware.before_model 中间件在模型调用前自动触发。
"""
from __future__ import annotations

import logging
from typing import Any

from agent.compaction import CompactionConfig, LCAgentCompactionMiddleware

logger = logging.getLogger(__name__)


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
