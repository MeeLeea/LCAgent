"""工具硬约束配置（唯一来源）

集中管理工具层的硬约束常量：
- DEFAULT_TIMEOUT: 全局默认超时（秒）
- TOOL_TIMEOUTS: 按工具名覆盖超时（优先于全局默认）
- NO_TIMEOUT_TOOLS: 完全排除超时的工具（无限等待）
- MAX_OUTPUT_CHARS: 工具输出截断字符数（避免回传 LLM 占用过多 token）
- SOFT_TIMEOUT_MARGIN: 软超时（内层）相对硬超时（外层）的安全余量（秒）

两层超时预算（重要）：
- 外层 / 硬超时：tool_wrapper.py 用 asyncio.wait_for 包裹工具 _arun，预算来自
  TOOL_TIMEOUTS / DEFAULT_TIMEOUT；触发后只返回裸 {"error": "tool_timeout"}，
  既无 partial 输出也无超时原因。
- 内层 / 软超时：终端工具函数自身 timeout 参数的默认值，驱动 Popen + ctrl+c
  软中断（等 grace period 收集缓冲输出后强杀进程树），触发后返回富结果
  {error_type, timeout_reason, partial_stdout, partial_stderr}，供主模型自纠。
- 软超时必须严格小于硬超时（相差至少 SOFT_TIMEOUT_MARGIN），否则外层先触发，
  模型拿到的只是无信息的裸错误；soft_timeout_for() 由硬超时统一派生软超时。

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
    "run_shell": 600.0,       # shell 命令
}

# 完全排除超时的工具（无限等待）
# 目前为空，ask_human 通过 TOOL_TIMEOUTS 给了 600 秒上限
NO_TIMEOUT_TOOLS: set[str] = set()

# 工具输出截断字符数（超长输出截断，避免回传给 LLM 时占用过多 token）
MAX_OUTPUT_CHARS: int = 10000

# 软超时（内层）相对硬超时（外层）的安全余量（秒）。
# 必须严格大于 terminal_tools.py 的 _GRACE_PERIOD（5 秒）：软超时触发后需要留出
# 发送 ctrl+c、等待 grace period 收集 partial 输出、再强杀进程树的时间，这些动作
# 都必须在外层 asyncio.wait_for 触发之前完成，否则富超时结果会被裸错误覆盖。
SOFT_TIMEOUT_MARGIN: float = 10.0

# 软超时的最小正值下限（秒）：硬超时配置过小时避免派生出非正超时。
MIN_SOFT_TIMEOUT: float = 1.0


def soft_timeout_for(tool_name: str) -> float:
    """终端工具函数级软超时 = 包装器硬超时 - margin。

    语义：hard = TOOL_TIMEOUTS.get(tool_name, DEFAULT_TIMEOUT)；
    返回 hard - SOFT_TIMEOUT_MARGIN，并 clamp 到 >= MIN_SOFT_TIMEOUT，
    防止硬超时配置过小时派生出非正值（此时软超时不再严格小于硬超时，
    属于配置错误，需人工修正 TOOL_TIMEOUTS）。

    Args:
        tool_name: 工具名（如 "run_shell" / "run_python" / "run_cmd"）

    Returns:
        软超时秒数；正常配置下严格小于对应硬超时，为 ctrl+c grace period 留出余量。
    """
    hard = TOOL_TIMEOUTS.get(tool_name, DEFAULT_TIMEOUT)
    return max(hard - SOFT_TIMEOUT_MARGIN, MIN_SOFT_TIMEOUT)
