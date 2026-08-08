"""
LangChain Agent 项目入口 - 支持多提供商(智谱/千问/DeepSeek/Kimi) + MCP工具 + 多Agent工作流
"""

from __future__ import annotations

import asyncio
import os

try:
    import readline  # noqa: F401 - 为交互式 CLI 提供历史/方向键支持
except ImportError:
    pass

from agent import AgentCore
from agent.config import load_agent_config, resolve_path
from agent.logging_config import setup_logging
from cli.cli_menu import select_menu
from cli.commands import CommandContext, dispatch_command
from cli.commands.core import show_ready
from cli.commands.provider import create_llm, select_provider
from cli.human_input import chat_until_completion, run_structured_until_completion
from memory import MemoryContext
from tools import safety as safety_module

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LLM_FILE = os.path.join(BASE_DIR, "config", "llm_config.json")
MCP_CONFIG_FILE = os.path.join(BASE_DIR, "config", "mcp_servers.json")
AGENT_CONFIG_FILE = os.path.join(BASE_DIR, "agent", "agent_config.json")
CHECKPOINT_FILE = os.path.join(BASE_DIR, "data", "checkpoints_async.sqlite")


def render_print(value: str = "") -> None:
    """输出命令处理器已经渲染好的文本。"""
    for line in str(value).split("\n"):
        print(line.strip())


async def build_agent(provider: str, process_type: str | None = None) -> tuple[AgentCore, object]:
    """根据提供商和运行时配置初始化 LLM 与 Agent。
    
    Args:
        provider: LLM 提供商名称
        process_type: 进程类型标识(feishu/None)，用于多进程隔离。
                      CLI 模式传 None(单进程不需要隔离)
    """
    from agent.llm_client import load_providers as list_providers
    
    print(f"\n初始化 {list_providers(LLM_FILE)[provider]['name']} 客户端...")
    llm = create_llm(provider, LLM_FILE)
    print("加载运行时配置...")
    config = load_agent_config(AGENT_CONFIG_FILE)
    agent_prompt_file = config.get("agent_prompt_file")
    # 配置中的相对路径统一锚定项目根，避免调用方工作目录影响资源加载。
    skills_dir = resolve_path(config["skills_dir"], BASE_DIR)
    mcp_config_file = resolve_path(config["mcp_config_file"], BASE_DIR)
    print("初始化Agent(含MCP工具加载 + Checkpoint 持久化)...")
    # 三层架构：先创建 MemoryContext（记忆基础设施），再创建 AgentCore（纯执行内核）
    memory_ctx = await MemoryContext.acreate(
        checkpoint_file=CHECKPOINT_FILE,
        short_term_size=config["memory_size"],
        use_sqlite=True,
        process_type=process_type,
        llm_getter=lambda: llm,
        buffer_delay_seconds=config.get("memory_buffer_delay_seconds", 20),
        max_buffer_messages=config.get("memory_max_buffer_messages", 30),
        max_facts_per_thread=config.get("memory_max_facts_per_thread", 50),
        recall_limit=config.get("memory_recall_limit", 10),
    )
    agent = await AgentCore.acreate(
        llm_client=llm,
        name=config["name"],
        max_iterations=config["max_iterations"],
        verbose=config["verbose"],
        mcp_config_file=mcp_config_file,
        enable_mcp=config["enable_mcp"],
        skills_dir=skills_dir,
        auto_match_skills=config["auto_match_skills"],
        max_context_messages=config["max_context_messages"],
        context_trim_keep=config["context_trim_keep"],
        process_type=process_type,
        agent_prompt_file=agent_prompt_file,
        max_execution_history=config.get("max_execution_history", 100),
        tool_timeout=config.get("tool_timeout", 120),
        checkpointer=memory_ctx.checkpointer,
        store=memory_ctx.store,
        extra_middleware=[memory_ctx.read_middleware],
        initial_thread_id=memory_ctx.thread_id,
        async_conn=memory_ctx.async_conn,
    )
    # 注入 MemoryManager → SessionManager 懒初始化时会自动接收
    agent.set_memory_manager(memory_ctx.memory_manager)
    agent._memory_context = memory_ctx  # 供 aclose 时关闭 SQLite 连接
    return agent, llm


def make_context(agent: AgentCore) -> CommandContext:
    """组装命令分发器所需的运行时依赖。"""
    from agent.llm_client import load_providers as list_providers
    
    # 命令模块只依赖该上下文，便于在测试中替换输入、菜单、LLM 和安全后端。
    return CommandContext(
        agent=agent,
        base_dir=BASE_DIR,
        config_file=LLM_FILE,
        mcp_config_file=MCP_CONFIG_FILE,
        print_fn=render_print,
        input_fn=input,
        select_menu=select_menu,
        create_llm=lambda provider: create_llm(provider, LLM_FILE),
        list_providers=lambda: list_providers(LLM_FILE),
        run_structured_until_completion=run_structured_until_completion,
        chat_until_completion=chat_until_completion,
        safety_backend=safety_module,
    )


async def main() -> None:
    """运行交互式命令行主循环。"""
    setup_logging()
    # 启动 banner（保持 print，面向用户）
    print("=" * 50)
    print("  LC Agent (基于LangChain框架)")
    print("=" * 50)
    provider = select_provider(LLM_FILE, select_menu)
    agent, _ = await build_agent(provider)
    context = make_context(agent)
    show_ready(context)

    try:
        while True:
            try:
                user_input = (await asyncio.to_thread(input, "\n你: ")).strip()
                if not user_input:
                    continue
                outcome = await dispatch_command(context, user_input)
                if outcome.should_break:
                    break
            except KeyboardInterrupt:
                print("\n\n程序被中断，再见!")
                break
            # CLI 最外层兜底只负责保持会话可用；业务模块仍应捕获具体异常。
            except Exception as error:
                print(f"\n错误: {error}")
                print("请重试...")
    finally:
        await agent.session_manager.aclose()
        # 关闭 MemoryContext（释放 SQLite 连接等底层资源）
        mem_ctx = getattr(agent, "_memory_context", None)
        if mem_ctx is not None:
            await mem_ctx.aclose()


if __name__ == "__main__":
    asyncio.run(main())
