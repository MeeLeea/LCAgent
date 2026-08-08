"""Memory 模块 — 独立记忆基础设施，与 agent 平级。

三层架构的 Memory 层，封装全部记忆职责：
- 事件过滤、记忆判定、LLM 压缩/摘要/去重
- 长期记忆入库（fact / lesson / entity / decision）
- 长期记忆召回注入 Agent prompt

模块结构：
    memory/
    ├── __init__.py        # 包导出
    ├── agent_memory.py    # AgentMemory（checkpointer + Store 基础设施）
    ├── config.py          # Memory 配置默认值
    ├── context.py         # MemoryContext（统一工厂，供入口程序调用）
    ├── lock_pool.py       # per-thread 并发锁池
    ├── manager.py         # MemoryManager（统一门面）
    ├── middleware.py      # 读写中间件（防抖 + Fact 抽取 + prompt 注入）
    ├── models.py          # 数据模型与事件分类判定
    └── store.py           # ThreadMemoryStore（Store 业务封装）

依赖关系：
    memory → agent.events  （消费 AgentEvent 事件流）
    memory ← session       （SessionManager 调用 MemoryManager）
    memory ← 入口程序      （main.py / api / scheduler 通过 MemoryContext 创建）
"""
from .context import MemoryContext
from .lock_pool import ThreadMemoryLockPool
from .manager import MemoryManager
from .middleware import (
    ThreadMemoryReadMiddleware,
    ThreadMemoryWriteMiddleware,
)
from .models import (
    MemoryCategory,
    MemoryInputEvent,
    ThreadFactItem,
    judge_long_term_memory,
)
from .store import ThreadMemoryStore

__all__ = [
    'MemoryCategory',
    'MemoryContext',
    'MemoryInputEvent',
    'MemoryManager',
    'ThreadFactItem',
    'ThreadMemoryLockPool',
    'ThreadMemoryReadMiddleware',
    'ThreadMemoryStore',
    'ThreadMemoryWriteMiddleware',
    'judge_long_term_memory',
]
