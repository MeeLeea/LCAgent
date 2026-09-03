"""工具硬约束配置（唯一来源）

集中管理工具层的硬约束常量：
- DEFAULT_TIMEOUT: 全局默认超时（秒）
- TOOL_TIMEOUTS: 按工具名覆盖超时（优先于全局默认）
- NO_TIMEOUT_TOOLS: 完全排除超时的工具（无限等待）
- MAX_OUTPUT_CHARS: 工具输出截断字符数（避免回传 LLM 占用过多 token）

设计：
- 此文件是工具硬约束的唯一来源，agent_config.json 的 tool_timeout 字段
  置 0 后不再控制实际超时（回退到此处的 DEFAULT_TIMEOUT / TOOL_TIMEOUTS）
- terminal_tools.py / tool_wrapper.py 从此处导入，禁止再定义本地常量
- 保留 per-tool 覆盖能力（TOOL_TIMEOUTS），团队角色若需差异化超时
  仍可通过 agent_config.json 的 tool_timeout 显式覆盖（>0 生效）
"""
from __future__ import annotations

# 全局默认超时（秒）
DEFAULT_TIMEOUT: float = 60.0

# 按工具名覆盖超时（优先于全局默认）
TOOL_TIMEOUTS: dict[str, float] = {
    "ask_human": 600.0,       # 人工交互，给 10 分钟
    "schedule_task": 120.0,   # 调度器可能需要更长
    "search": 90.0,           # 搜索可能需要多次请求
    "run_shell": 600.0,       # shell 命令（vivado/xsim 批处理耗时 3-10 分钟）
}

# 完全排除超时的工具（无限等待）
# 目前为空，ask_human 通过 TOOL_TIMEOUTS 给了 600 秒上限
NO_TIMEOUT_TOOLS: set[str] = set()

# 工具输出截断字符数（超长输出截断，避免回传给 LLM 时占用过多 token）
MAX_OUTPUT_CHARS: int = 10000
