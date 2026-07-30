"""
LangChain Agent 项目入口 - 支持多提供商(智谱/千问/DeepSeek/Kimi) + MCP工具
"""

from __future__ import annotations

import os

try:
    import readline
except ImportError:
    pass

from agent import AgentCore
from agent.config import load_agent_config, resolve_path
from llm_client import load_providers as list_providers
from tools import safety as safety_module
from cli.cli_menu import select_menu
from cli.commands import CommandContext, dispatch_command
from cli.commands.core import show_ready
from cli.commands.provider import create_llm, select_provider
from cli.human_input import chat_until_completion, run_structured_until_completion

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LLM_FILE = os.path.join(BASE_DIR, "config", "llm_config.json")
MCP_CONFIG_FILE = os.path.join(BASE_DIR, "config", "mcp_servers.json")
AGENT_CONFIG_FILE = os.path.join(BASE_DIR, "config", "agent_config.json")
MEMORY_FILE = os.path.join(BASE_DIR, "memory", "memory.json")
CHECKPOINT_FILE = os.path.join(BASE_DIR, "memory", "checkpoints.sqlite")


def render_print(value: str = "") -> None:
    """输出命令处理器已经渲染好的文本。"""
    for line in str(value).split("\n"):
        print(line.strip())


def build_agent(provider: str, process_type: str = None) -> tuple[AgentCore, object]:
    """根据提供商和运行时配置初始化 LLM 与 Agent。
    
    Args:
        provider: LLM 提供商名称
        process_type: 进程类型标识(feishu/None)，用于多进程隔离。
                      CLI 模式传 None(单进程不需要隔离)
    """
    print(f"\n初始化 {list_providers(LLM_FILE)[provider]['name']} 客户端...")
    llm = create_llm(provider, LLM_FILE)
    print("加载运行时配置...")
    config = load_agent_config(AGENT_CONFIG_FILE)
    # 配置中的相对路径统一锚定项目根，避免调用方工作目录影响资源加载。
    skills_dir = resolve_path(config["skills_dir"], BASE_DIR)
    mcp_config_file = resolve_path(config["mcp_config_file"], BASE_DIR)
    print("初始化Agent(含MCP工具加载 + Checkpoint 持久化)...")
    agent = AgentCore(
        llm_client=llm,
        memory_size=config["memory_size"],
        long_term_memory_file=MEMORY_FILE,
        checkpoint_file=CHECKPOINT_FILE,
        max_iterations=config["max_iterations"],
        verbose=config["verbose"],
        mcp_config_file=mcp_config_file,
        enable_mcp=config["enable_mcp"],
        skills_dir=skills_dir,
        auto_match_skills=config["auto_match_skills"],
        max_context_messages=config["max_context_messages"],
        context_trim_keep=config["context_trim_keep"],
        process_type=process_type
    )
    return agent, llm


def make_context(agent: AgentCore, llm: object) -> CommandContext:
    """组装命令分发器所需的运行时依赖。"""
    # 命令模块只依赖该上下文，便于在测试中替换输入、菜单、LLM 和安全后端。
    return CommandContext(
        agent=agent,
        llm=llm,
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


def main() -> None:
    """运行交互式命令行主循环。"""
    print("=" * 50)
    print("  LangChain Agent (基于LangChain框架)")
    print("=" * 50)
    provider = select_provider(LLM_FILE, select_menu)
    agent, llm = build_agent(provider)
    context = make_context(agent, llm)
    show_ready(context)

    while True:
        try:
            user_input = input("\n你: ").strip()
            if not user_input:
                continue
            outcome = dispatch_command(context, user_input)
            if outcome.should_break:
                break
        except KeyboardInterrupt:
            print("\n\n程序被中断，再见!")
            break
        # CLI 最外层兜底只负责保持会话可用；业务模块仍应捕获具体异常。
        except Exception as error:  # noqa: BROAD_EXCEPT_OK - CLI boundary keeps the session alive.
            print(f"\n错误: {error}")
            print("请重试...")


if __name__ == "__main__":
    main()
