"""通用工具模块

存放与业务解耦的通用能力，供 agent / graph / session 等模块共用：
- compaction：LangGraph 上下文压缩中间件（增量摘要 + 工具输出 Prune）
- exceptions：统一异常层次（LCAgentError）
- events：标准化执行事件模型（AgentEvent / EventType）
- logging_config：结构化日志（TraceContext / setup_logging）
- metrics：运行时指标收集（MetricsCollector）
"""
