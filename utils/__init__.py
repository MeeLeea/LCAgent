"""通用工具模块

存放与业务解耦的通用能力，供 agent / graph / session 等模块共用：
- exceptions：统一异常层次（LCAgentError）
- events：标准化执行事件模型（AgentEvent / EventType）
- logging_config：结构化日志（TraceContext / setup_logging）
- metrics：运行时指标收集（MetricsCollector）

注：上下文压缩中间件（compaction）已归位至 agent 包（agent/compaction.py），
因其实质为 LangGraph Agent 中间件而非通用工具。
"""
