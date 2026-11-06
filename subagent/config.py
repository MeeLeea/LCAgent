"""subagent 子程序配置总表（唯一事实源）。

集中声明 ``planner`` / ``executor`` 两个后台子程序的全部**可覆盖默认值**与
子程序专属运行时常量；各子程序 ``config.py`` 瘦壳中的 dataclass 字段默认值
与常量一律从本模块导入，禁止再写字面量默认值，防止两处漂移。

约定：
- 命名前缀即子程序名（``PLANNER_*`` / ``EXECUTOR_*``），值与
  ``subagent/<name>/README.md`` 的配置表一致；
- 本表**不**承载依赖路径的常量（``DEFAULT_CONFIG_FILE`` /
  ``DEFAULT_AGENT_PROMPT_FILE`` / ``DEFAULT_OUTPUT_ROOT`` 依赖 ``PROJECT_ROOT``
  计算，保留在各子程序 ``config.py`` 中）；
- ``provider`` / ``model`` 默认 ``None``（自动选择），属结构性语义而非调参值，
  直接在 dataclass 中写 ``None``，不入本表。
"""
from __future__ import annotations

# ---- planner（一次性方案设计：单次 LLM 调用）----

PLANNER_NAME: str = "planner"
PLANNER_TEMPERATURE: float = 0.4
PLANNER_MAX_TOKENS: int = 8192
PLANNER_TIMEOUT_SECONDS: float = 900.0

# 输出根目录的环境变量覆盖键（测试 / 多实例隔离用）
PLANNER_OUTPUT_ROOT_ENV: str = "LCAGENT_PLANNER_ROOT"

# ---- executor（多步 ReAct 工具循环）----

EXECUTOR_NAME: str = "executor"
EXECUTOR_TEMPERATURE: float = 0.2
EXECUTOR_MAX_TOKENS: int = 8192
# 整个任务的总预算（所有步骤共享的绝对 deadline，秒）
EXECUTOR_TIMEOUT_SECONDS: float = 1800.0
# 步数上限（每一步 = 一次 LLM 调用 + 本轮 tool_calls 执行）
EXECUTOR_MAX_STEPS: int = 24
# 单条工具观测（ToolMessage content）最大字符数，超出截断
EXECUTOR_MAX_OBSERVATION_CHARS: int = 6000
# read_file 单次读取的最大字节数
EXECUTOR_MAX_READ_BYTES: int = 262_144
# list_dir 返回条目数上限
EXECUTOR_MAX_LIST_ENTRIES: int = 500
# 单步 LLM 调用的最小剩余预算（秒）：小于它就不发起新一步，直接判定超时
EXECUTOR_MIN_STEP_SECONDS: float = 30.0

# 输出根目录的环境变量覆盖键（测试 / 多实例隔离用）
EXECUTOR_OUTPUT_ROOT_ENV: str = "LCAGENT_EXECUTOR_ROOT"

__all__ = [
    "EXECUTOR_MAX_LIST_ENTRIES",
    "EXECUTOR_MAX_OBSERVATION_CHARS",
    "EXECUTOR_MAX_READ_BYTES",
    "EXECUTOR_MAX_STEPS",
    "EXECUTOR_MAX_TOKENS",
    "EXECUTOR_MIN_STEP_SECONDS",
    "EXECUTOR_NAME",
    "EXECUTOR_OUTPUT_ROOT_ENV",
    "EXECUTOR_TEMPERATURE",
    "EXECUTOR_TIMEOUT_SECONDS",
    "PLANNER_MAX_TOKENS",
    "PLANNER_NAME",
    "PLANNER_OUTPUT_ROOT_ENV",
    "PLANNER_TEMPERATURE",
    "PLANNER_TIMEOUT_SECONDS",
]
