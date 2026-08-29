"""LLM 模块包

统一大模型封装与运行时配置加载。从 agent/ 迁移至 llm/ 平级目录,
供 agent / team / api / scheduler / cli 等模块共享。

子模块:
    - config:        agent 运行时配置加载 (load_agent_config / resolve_path)
    - llm_client:    统一 LLM 客户端封装 (LLMClient / load_providers)
    - message_utils: LLM 异常提取与消息序列化 (extract_llm_error / stringify_content)
"""
