"""Core informational CLI commands."""

from __future__ import annotations

from .types import CommandContext, CommandOutcome, HANDLED


def show_ready(context: CommandContext, full_help: bool = False) -> None:
    info = context.agent.llm.get_info()
    context.print("\n" + "=" * 50)
    context.print("Agent 已就绪！")
    context.print("=" * 50)
    context.print(f"\n当前提供商: {info['provider_name']}")
    context.print(f"当前模型:   {info['model']}")
    context.print("框架:       LangChain")
    if not full_help:
        context.print("\n输入 'help' 查看所有命令说明")
        context.print()
        return
    context.print("\n命令说明:")
    for line in _HELP_LINES:
        context.print(line)
    context.print(f"\n本地工具: {', '.join(t.name for t in context.agent.local_tools)}")
    if context.agent.mcp_tools:
        context.print(f"MCP工具:  {', '.join(t.name for t in context.agent.mcp_tools)}")
    context.print()


def show_help(context: CommandContext) -> CommandOutcome:
    show_ready(context, full_help=True)
    return HANDLED


def show_info(context: CommandContext) -> CommandOutcome:
    info = context.agent.llm.get_info()
    context.print(f"\n当前提供商: {info['provider_name']}")
    context.print(f"当前模型:   {info['model']}")
    context.print(f"API地址:    {info['base_url']}")
    mem = context.agent.get_memory_summary()
    context.print("\n--- 记忆状态 ---")
    context.print(f"当前会话:   {mem['thread_id']}")
    context.print(f"Checkpoint: {mem['checkpoint_backend']} → {mem['checkpoint_file']}")
    context.print(f"已存消息:   {mem['checkpoint_messages']} 条")
    context.print(f"长期记忆:   {mem['long_term_count']} 条")
    context.print(f"总会话数:   {mem['total_threads']}")
    return HANDLED


def show_tools(context: CommandContext) -> CommandOutcome:
    context.print(f"\n可用工具: {', '.join(context.agent.get_available_tools())}")
    for tool in context.agent.tools:
        context.print(f"  - {tool.name}: {tool.description}")
    return HANDLED


_HELP_LINES = [
    "  - 输入 'quit' 或 'exit' 退出",
    "  - 输入 'react:任务' 使用Agent模式(自动调用工具)",
    "  - 输入 'cot:任务' 使用链式思考模式(纯推理)",
    "  - 输入 'switch' 方向键选择切换提供商(或 'switch:提供商名' 直接切换)",
    "  - 输入 'model' 查看当前提供商可用模型",
    "  - 输入 'model:<模型名>' 切换模型 (如 model:glm-4-flash)",
    "  - 输入 'info' 查看当前模型信息 + 记忆状态",
    "  - 输入 'tools' 查看可用工具",
    "  - 输入 'clear [long|short|all]' 清理记忆(默认 long)",
    "  - 输入 'compress' 压缩长期记忆(LLM摘要后替换原内容)",
    "  - 输入 'compact' 手动压缩当前会话上下文(增量摘要+工具输出裁剪)",
    "  - 输入 'thread' 查看所有会话(方向键选择切换,Ctrl+D 删除高亮会话)",
    "  - 输入 'thread:new' 开启新会话(原会话保留)",
    "  - 输入 'thread:delete <thread_id>' 删除指定会话",
    "  - 输入 'mcp' 查看 MCP Server 状态",
    "  - 输入 'mcp:reload [name]' 重新加载 MCP 工具(指定 name 只重连该 server)",
    "  - 输入 'mcp:add <name> <command> <arg1> [arg2...]' 添加 stdio MCP Server",
    "  - 输入 'mcp:add <name> <url>' 添加 sse/http MCP Server (前缀 sse: 或 http:)",
    "  - 输入 'mcp:remove <name>' 删除 MCP Server",
    "  - 输入 'mcp:toggle <name> <on|off>' 启用/禁用 MCP Server",
    "  - 输入 'skill' 查看所有本地可用技能",
    "  - 输入 'skill:<name>' 将某技能加载进当前会话(注入 system prompt)",
    "  - 输入 'skill:<name> <任务>' 加载技能并立即执行该任务(如 skill:git-commit 提交README)",
    "  - 输入 'skill:clear' 清空手动加载的技能",
    "  - 输入 'role' 查看团队角色(方向键选择切换)",
    "  - 输入 'role:<name>' 切换到指定团队角色 (如 role:manager)",
    "  - 输入 'role:<name> <任务>' 切换角色并立即执行该任务",
    "  - 输入 'safety' 查看当前安全策略(黑名单/白名单/确认)",
    "  - 输入 'safety:mode <blacklist|whitelist>' 切换模式",
    "  - 输入 'safety:confirm <on|off>' 开关危险命令确认",
    "  - 输入 'export' 或 'export:<thread_id> [路径]' 导出对话为 Markdown(默认存 exports/)",
    "  - 输入 'metrics' 查看运行时指标(LLM tokens/工具耗时/压缩统计)",
    "  - 输入 'metrics:reset' 重置所有指标",
    "  - 输入 'json:<任务>' 让 Agent 以 JSON 对象返回结果并解析展示",
    "  - 其他输入为普通对话模式",
    "  - 运行时配置见 agent/agent_config.json(迭代上限/技能目录/长上下文裁剪等)",
]
