"""中间件统一包（midware/）

从 agent/ 迁出的 LangGraph/LangChain Agent 中间件集合，与 agent/ 平级：

- terminal_retry_cap_mw.py     终端命令超时重试上限中间件（TerminalRetryCapMW）
- tool_retry_cap_mw.py         重复调用熔断中间件（ToolRetryCapMW）
- tool_arg_validator_mw.py     工具参数语义校验中间件（ToolArgValidatorMW）
- tool_error_mw.py             工具错误反思纠错中间件（ToolExecutionErrorMW）
- workspace_mw.py              工作空间安全中间件（WorkspaceSecurityMW）
- compaction.py                长上下文压缩中间件（LCAgentCompactionMiddleware + LCAgentState）
- session_config_middleware.py 会话配置中间件（SessionConfigMW + SessionModelFactory）

注意：
1. 本包 ``__init__`` 刻意不做聚合导出，调用方一律使用完整子模块路径
   （如 ``from midware.tool_error_mw import ToolExecutionErrorMW``）。
   原因：``session_config_middleware`` 依赖 ``session.config``，而
   ``session/workflow_adapter`` 依赖 ``midware.compaction``，存在包级交叉依赖；
   聚合导出会使包初始化顺序对该交叉敏感，空 ``__init__`` 彻底免疫。
2. 各中间件互不依赖，仅引用 langchain/langgraph 框架与 ``llm/``、``session.config``
   基础层，是依赖图的叶子侧节点，禁止反向 import ``agent/``、``team/``、``graph/``。
"""
