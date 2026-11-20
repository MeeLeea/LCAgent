"""Memory 模块配置默认值。

三层架构 Memory 层的运行时参数，独立于 agent/config.py。
由 memory/context.py、memory/middleware.py 直接按名导入使用
（不经 agent_config.json 配置透传；调整默认值即修改本文件常量）。
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

MEMORY_AGENT_KEY = "global"
"""agent 级长期记忆 namespace 标识：默认跨进程共享，可显式覆盖以隔离"""

MEMORY_RECALL_LIMIT = 20
"""召回长期记忆时的默认条数上限"""

# ── 写入质量门槛 ──
MEMORY_CONFIDENCE_FLOOR = 0.7
"""agent 级（跨会话共享）写入的置信度门槛：persistence=global 但 confidence
低于该值的 fact 降级写入 thread 级（会话隔离），防止低置信内容污染全局"""

MEMORY_LOW_CONFIDENCE_THRESHOLD = 0.6
"""读取注入时的低置信标注阈值：confidence 低于该值的条目附"（低置信）"标注"""
