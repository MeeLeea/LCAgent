"""Memory 模块配置默认值。

三层架构 Memory 层的运行时参数，独立于 agent/config.py。
入口程序（main.py / api/server.py / scheduler/run.py）通过
``config.get(key, default)`` 方式读取，default 从本模块导入。
"""
from __future__ import annotations

# ── 防抖缓冲 ──
MEMORY_BUFFER_DELAY_SECONDS = 30
"""防抖缓冲窗口（秒）：同一 thread 的新事件重置计时"""

MEMORY_MAX_BUFFER_MESSAGES = 40
"""单 thread 缓冲区上限（防溢出）"""

# ── 存储容量 ──
MEMORY_MAX_FACTS_PER_THREAD = 60
"""单 thread 最大 fact 条数（超出 LRU 淘汰）"""

MEMORY_MAX_AGENT_FACTS = 200
"""agent 级长期记忆最大 fact 条数（超出 LRU 淘汰）"""

MEMORY_RECALL_LIMIT = 20
"""召回长期记忆时的默认条数上限"""

# ── Session 层开关 ──
SESSION_ENABLE_MEMORY = True
"""SessionManager 是否启用长期记忆处理"""

# 配置键名列表（供 agent_config.json 加载时透传）
CONFIG_KEYS = [
    "memory_buffer_delay_seconds",
    "memory_max_buffer_messages",
    "memory_max_facts_per_thread",
    "memory_max_agent_facts",
    "memory_recall_limit",
    "session_enable_memory",
]

# 默认值字典
DEFAULTS = {
    "memory_buffer_delay_seconds": MEMORY_BUFFER_DELAY_SECONDS,
    "memory_max_buffer_messages": MEMORY_MAX_BUFFER_MESSAGES,
    "memory_max_facts_per_thread": MEMORY_MAX_FACTS_PER_THREAD,
    "memory_max_agent_facts": MEMORY_MAX_AGENT_FACTS,
    "memory_recall_limit": MEMORY_RECALL_LIMIT,
    "session_enable_memory": SESSION_ENABLE_MEMORY,
}
