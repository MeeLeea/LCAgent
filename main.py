"""
LangChain Agent 项目入口 - 支持多提供商(智谱/千问/DeepSeek/Kimi) + MCP工具
"""
import os
import sys
from llm_client import create_client, list_providers, LLMClient
from agent import AgentCore
from tools import mcp_loader

CONFIG_FILE = "llm_config.json"
MCP_CONFIG_FILE = "mcp_servers.json"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_FILE = os.path.join(BASE_DIR, "data", "memory.json")

def select_provider() -> str:
    """启动时选择提供商"""
    providers = list_providers()
    print("\n可用的大模型提供商:")
    print("-" * 50)
    for i, (key, config) in enumerate(providers.items(), 1):
        env_key = config["env_key"]
        has_key = bool(os.environ.get(env_key))
        status = "✓" if has_key else " "
        print(f"  {i}. [{status}] {key:10s} ({config['name']})")
        print(f"     模型: {', '.join(config['models'])}")
        print(f"     环境变量: {env_key}")
    print("-" * 50)
    print("  [✓] = 已检测到环境变量API密钥")

    # 检查配置文件
    if os.path.exists(CONFIG_FILE):
        try:
            import json
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
            for key in providers:
                if config.get(key, {}).get("api_key"):
                    print(f"  [✓] {key} 在 {CONFIG_FILE} 中已配置")
        except Exception:
            pass

    print()
    while True:
        choice = input("请选择提供商 (1-4) 或直接回车使用默认[智谱]: ").strip()
        if not choice:
            return "zhipu"
        try:
            idx = int(choice)
            if 1 <= idx <= len(providers):
                return list(providers.keys())[idx - 1]
        except ValueError:
            pass
        print("无效选择，请重试")


def create_llm(provider: str) -> LLMClient:
    """创建LLM客户端，自动从配置文件或环境变量获取密钥"""
    try:
        return create_client(
            provider=provider,
            config_file=CONFIG_FILE
        )
    except ValueError as e:
        print(f"\n错误: {e}")
        sys.exit(1)


def main():
    """主函数"""
    print("=" * 50)
    print("  LangChain Agent (基于LangChain框架)")
    print("=" * 50)

    # 选择提供商
    provider = select_provider()

    # 创建LLM客户端
    print(f"\n初始化 {list_providers()[provider]['name']} 客户端...")
    llm = create_llm(provider)

    # 创建Agent
    print("初始化Agent(含MCP工具加载)...")
    agent = AgentCore(
        llm_client=llm,
        memory_size=10,
        long_term_memory_file=MEMORY_FILE,
        max_iterations=5,
        verbose=True,
        mcp_config_file=MCP_CONFIG_FILE,
        enable_mcp=True
    )

    # 显示信息
    info = llm.get_info()
    print("\n" + "=" * 50)
    print("Agent 已就绪！")
    print("=" * 50)
    print(f"\n当前提供商: {info['provider_name']}")
    print(f"当前模型:   {info['model']}")
    print(f"框架:       LangChain")
    print("\n命令说明:")
    print("  - 输入 'quit' 或 'exit' 退出")
    print("  - 输入 'react:任务' 使用Agent模式(自动调用工具)")
    print("  - 输入 'cot:任务' 使用链式思考模式(纯推理)")
    print("  - 输入 'switch:提供商名' 切换模型 (zhipu/qwen/deepseek/kimi)")
    print("  - 输入 'info' 查看当前模型信息")
    print("  - 输入 'tools' 查看可用工具")
    print("  - 输入 'clear [long|short|all]' 清理记忆(默认 long)")
    print("  - 输入 'compress' 压缩长期记忆(LLM摘要后替换原内容)")
    print("  - 输入 'mcp' 查看 MCP Server 状态")
    print("  - 输入 'mcp:reload' 重新加载 MCP 工具")
    print("  - 输入 'mcp:add <name> <command> <arg1> [arg2...]' 添加 stdio MCP Server")
    print("  - 输入 'mcp:add <name> <url>' 添加 sse/http MCP Server (前缀 sse: 或 http:)")
    print("  - 输入 'mcp:remove <name>' 删除 MCP Server")
    print("  - 输入 'mcp:toggle <name> <on|off>' 启用/禁用 MCP Server")
    print("  - 其他输入为普通对话模式")
    print(f"\n本地工具: {', '.join(t.name for t in agent.local_tools)}")
    if agent.mcp_tools:
        print(f"MCP工具:  {', '.join(t.name for t in agent.mcp_tools)}")
    print()

    while True:
        try:
            user_input = input("\n你: ").strip()

            if not user_input:
                continue

            # 退出命令
            if user_input.lower() in ['quit', 'exit']:
                print("再见!")
                break

            # 查看当前信息
            if user_input.lower() == 'info':
                info = llm.get_info()
                print(f"\n当前提供商: {info['provider_name']}")
                print(f"当前模型:   {info['model']}")
                print(f"API地址:    {info['base_url']}")
                mem = agent.get_memory_summary()
                print(f"记忆状态:   短期{mem['short_term_count']}/{mem['short_term_capacity']}, 长期{mem['long_term_count']}")
                continue

            # 查看可用工具
            if user_input.lower() == 'tools':
                print(f"\n可用工具: {', '.join(agent.get_available_tools())}")
                for t in agent.tools:
                    print(f"  - {t.name}: {t.description}")
                continue

            # 清理记忆
            if user_input.lower().startswith('clear'):
                parts = user_input.split(None, 1)
                target = parts[1].strip().lower() if len(parts) > 1 else "long"
                if target in ("long", "长期"):
                    agent.memory.clear_long_term()
                    print("\n已清空长期记忆(并删除 memory.json)")
                elif target in ("short", "短期"):
                    agent.memory.clear_short_term()
                    print("\n已清空短期记忆")
                elif target in ("all", "全部"):
                    agent.memory.clear_long_term()
                    agent.memory.clear_short_term()
                    print("\n已清空全部记忆(长期+短期)")
                else:
                    print("\n用法: clear [long|short|all]  (默认 long)")
                continue

            # 压缩长期记忆
            if user_input.lower() in ('compress', '压缩'):
                mem = agent.get_memory_summary()
                if mem['long_term_count'] == 0:
                    print("\n没有长期记忆可压缩")
                    continue
                print(f"\n开始压缩长期记忆 (共 {mem['long_term_count']} 条)...")
                result = agent.compress_memory()
                if result['success']:
                    print(f"压缩完成！")
                    print(f"  原记忆条数:   {result['original_count']} 条")
                    print(f"  原字符数:     {result['original_chars']} 字符")
                    print(f"  压缩后字符数: {result['compressed_chars']} 字符")
                    ratio = (1 - result['compressed_chars'] / max(result['original_chars'], 1)) * 100
                    print(f"  压缩率:       {ratio:.1f}%")
                    print(f"\n--- 摘要内容 ---")
                    print(result['summary'])
                    print(f"--- 已保存到 memory.json ---")
                else:
                    print(f"\n压缩失败: {result.get('error', '未知错误')}")
                continue


            # 切换提供商
            if user_input.lower().startswith('switch:'):
                new_provider = user_input[7:].strip().lower()
                try:
                    new_llm = create_llm(new_provider)
                    agent.switch_llm(new_llm)
                    llm = new_llm
                    info = llm.get_info()
                    print(f"\n已切换到: {info['provider_name']} ({info['model']})")
                except SystemExit:
                    pass
                except Exception as e:
                    print(f"\n切换失败: {e}")
                continue

            # ========= MCP 管理命令 =========
            # 查看 MCP 服务器状态
            if user_input.lower() == 'mcp':
                servers = mcp_loader.list_configured_servers(MCP_CONFIG_FILE)
                print("\nMCP Servers:")
                print("-" * 60)
                if not servers:
                    print("  (空) 使用 'mcp:add' 添加")
                for s in servers:
                    status = "✓启用" if s["enabled"] else "✗禁用"
                    print(f"  [{status}] {s['name']} ({s['transport']})")
                    print(f"           {s['detail']}")
                print("-" * 60)
                print(f"已加载 MCP 工具数: {len(agent.mcp_tools)}")
                if agent.mcp_tools:
                    print(f"工具: {', '.join(t.name for t in agent.mcp_tools)}")
                continue

            # 重新加载 MCP 工具
            if user_input.lower() == 'mcp:reload':
                print("\n重新加载 MCP 工具...")
                count = agent.reload_mcp_tools()
                print(f"完成: 已加载 {count} 个 MCP 工具")
                continue

            # 添加 MCP Server
            if user_input.lower().startswith('mcp:add'):
                parts = user_input.split(None, 2)
                if len(parts) < 3:
                    print("\n用法:")
                    print("  stdio: mcp:add <name> <command> <arg1> [arg2...]")
                    print("  sse:   mcp:add <name> sse:<url>")
                    print("  http:  mcp:add <name> http:<url>")
                    continue
                name = parts[1]
                rest = parts[2].strip()

                try:
                    if rest.startswith("sse:"):
                        mcp_loader.add_server(
                            name=name, transport="sse",
                            url=rest[4:], config_file=MCP_CONFIG_FILE
                        )
                        print(f"\n已添加 sse MCP Server: {name}")
                    elif rest.startswith("http:"):
                        mcp_loader.add_server(
                            name=name, transport="streamable_http",
                            url=rest[5:], config_file=MCP_CONFIG_FILE
                        )
                        print(f"\n已添加 http MCP Server: {name}")
                    else:
                        # stdio 模式: rest = "command arg1 arg2 ..."
                        tokens = rest.split()
                        if not tokens:
                            print("错误: 缺少 command")
                            continue
                        mcp_loader.add_server(
                            name=name, transport="stdio",
                            command=tokens[0], args=tokens[1:],
                            config_file=MCP_CONFIG_FILE
                        )
                        print(f"\n已添加 stdio MCP Server: {name}")

                    # 重新加载
                    count = agent.reload_mcp_tools()
                    print(f"已加载 {count} 个 MCP 工具")
                except Exception as e:
                    print(f"\n添加失败: {e}")
                continue

            # 删除 MCP Server
            if user_input.lower().startswith('mcp:remove'):
                parts = user_input.split(None, 1)
                if len(parts) < 2:
                    print("用法: mcp:remove <name>")
                    continue
                name = parts[1].strip()
                if mcp_loader.remove_server(name, MCP_CONFIG_FILE):
                    print(f"\n已删除 MCP Server: {name}")
                    count = agent.reload_mcp_tools()
                    print(f"已加载 {count} 个 MCP 工具")
                else:
                    print(f"\n未找到: {name}")
                continue

            # 启用/禁用 MCP Server
            if user_input.lower().startswith('mcp:toggle'):
                parts = user_input.split()
                if len(parts) < 3:
                    print("用法: mcp:toggle <name> <on|off>")
                    continue
                name = parts[1]
                enabled = parts[2].lower() in ("on", "true", "1", "启用")
                if mcp_loader.toggle_server(name, enabled, MCP_CONFIG_FILE):
                    state = "启用" if enabled else "禁用"
                    print(f"\n已{state} MCP Server: {name}")
                    count = agent.reload_mcp_tools()
                    print(f"已加载 {count} 个 MCP 工具")
                else:
                    print(f"\n未找到: {name}")
                continue

            # ReAct/Agent模式
            if user_input.lower().startswith('react:'):
                task = user_input[6:].strip()
                result = agent.run(task)

            # CoT模式
            elif user_input.lower().startswith('cot:'):
                task = user_input[4:].strip()
                result = agent.cot(task)

            # 普通对话模式
            else:
                result = agent.chat(user_input)

            print(f"\n助手: {result}")

        except KeyboardInterrupt:
            print("\n\n程序被中断，再见!")
            break
        except Exception as e:
            print(f"\n错误: {str(e)}")
            print("请重试...")


if __name__ == "__main__":
    main()
