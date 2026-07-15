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
CHECKPOINT_FILE = os.path.join(BASE_DIR, "data", "checkpoints.sqlite")


def select_menu(title: str, options, current=None):
    """
    方向键交互式选择菜单(Windows msvcrt, 无需额外依赖)

    Args:
        title: 提示标题
        options: 选项列表(list[str] 或 list[(label, value)])
        current: 默认选中项的 value 或 label(可选)

    Returns:
        选中的 value(若 options 是 str 列表则返回该 str);按 Esc 取消返回 None
    """
    import msvcrt

    # 统一为 (label, value) 形式
    normalized = []
    for opt in options:
        if isinstance(opt, (tuple, list)) and len(opt) == 2:
            normalized.append((str(opt[0]), opt[1]))
        else:
            normalized.append((str(opt), opt))

    if not normalized:
        return None

    # 确定默认选中索引
    idx = 0
    if current is not None:
        for i, (label, value) in enumerate(normalized):
            if value == current or label == str(current):
                idx = i
                break

    def render():
        # 先清掉上一次的输出(按行数回退)
        lines = len(normalized) + 3  # 标题2行 + 选项 + 底部提示
        # \033[A = 光标上移一行, \r 回到行首
        sys.stdout.write("\r" + "\033[A" * lines + "\033[J")
        print(f"\033[36m{title}\033[0m")
        print("  (↑↓ 选择, Enter 确认, Esc 取消)")
        for i, (label, _) in enumerate(normalized):
            if i == idx:
                print(f"  \033[32m❯ {label}\033[0m")
            else:
                print(f"    {label}")
        print("-" * 40)

    print()  # 预留空行便于回退
    render()

    while True:
        key = msvcrt.getch()
        # Enter
        if key in (b"\r", b"\n"):
            # 清掉菜单输出
            lines = len(normalized) + 3
            sys.stdout.write("\r" + "\033[A" * lines + "\033[J")
            label, value = normalized[idx]
            print(f"{title} \033[32m❯ {label}\033[0m")
            return value
        # Esc
        if key == b"\x1b":
            lines = len(normalized) + 3
            sys.stdout.write("\r" + "\033[A" * lines + "\033[J")
            print(f"{title} \033[90m(已取消)\033[0m")
            return None
        # 方向键: 前缀 \xe0 或 \x00, 后跟 H=↑ P=↓ K=← M=→
        if key in (b"\xe0", b"\x00"):
            k2 = msvcrt.getch()
            if k2 == b"H":  # 上
                idx = (idx - 1) % len(normalized)
                render()
            elif k2 == b"P":  # 下
                idx = (idx + 1) % len(normalized)
                render()
        # 数字快捷键: 1-9 直接选第 N 项
        if key.isdigit():
            n = int(key)
            if 1 <= n <= len(normalized):
                idx = n - 1
                render()

def select_provider() -> str:
    """启动时选择提供商(方向键选择)"""
    providers = list_providers()

    # 检查配置文件中已配置的 key
    configured_keys = set()
    if os.path.exists(CONFIG_FILE):
        try:
            import json
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
            for key in providers:
                if config.get(key, {}).get("api_key"):
                    configured_keys.add(key)
        except Exception:
            pass

    # 构建选项: label 带 ✓ 标记和模型列表
    options = []
    for key, config in providers.items():
        env_key = config["env_key"]
        has_key = bool(os.environ.get(env_key)) or key in configured_keys
        mark = "✓" if has_key else " "
        label = f"[{mark}] {key:10s} ({config['name']})  模型: {', '.join(config['models'])}"
        options.append((label, key))

    selected = select_menu(
        "选择大模型提供商 ([✓] = 已配置 API Key)",
        options,
        current="zhipu"
    )
    return selected if selected else "zhipu"


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
    print("初始化Agent(含MCP工具加载 + Checkpoint 持久化)...")
    agent = AgentCore(
        llm_client=llm,
        memory_size=10,
        long_term_memory_file=MEMORY_FILE,
        checkpoint_file=CHECKPOINT_FILE,
        max_iterations=5,
        verbose=True,
        mcp_config_file=MCP_CONFIG_FILE,
        enable_mcp=True,
        skills_dir=os.path.join(BASE_DIR, ".agents", "skills"),
        auto_match_skills=True
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
    print("  - 输入 'switch:提供商名' 切换提供商 (zhipu/qwen/deepseek/kimi)")
    print("  - 输入 'model' 查看当前提供商可用模型")
    print("  - 输入 'model:<模型名>' 切换模型 (如 model:glm-4-flash)")
    print("  - 输入 'info' 查看当前模型信息 + 记忆状态")
    print("  - 输入 'tools' 查看可用工具")
    print("  - 输入 'clear [long|short|all]' 清理记忆(默认 long)")
    print("  - 输入 'compress' 压缩长期记忆(LLM摘要后替换原内容)")
    print("  - 输入 'thread' 查看所有会话(方向键选择切换)")
    print("  - 输入 'thread:new' 开启新会话(原会话保留)")
    print("  - 输入 'thread:delete <thread_id>' 删除指定会话")
    print("  - 输入 'mcp' 查看 MCP Server 状态")
    print("  - 输入 'mcp:reload' 重新加载 MCP 工具")
    print("  - 输入 'mcp:add <name> <command> <arg1> [arg2...]' 添加 stdio MCP Server")
    print("  - 输入 'mcp:add <name> <url>' 添加 sse/http MCP Server (前缀 sse: 或 http:)")
    print("  - 输入 'mcp:remove <name>' 删除 MCP Server")
    print("  - 输入 'mcp:toggle <name> <on|off>' 启用/禁用 MCP Server")
    print("  - 输入 'skill' 查看所有本地可用技能")
    print("  - 输入 'skill:<name>' 将某技能加载进当前会话(注入 system prompt)")
    print("  - 输入 'skill:<name> <任务>' 加载技能并立即执行该任务(如 skill:git-commit 提交README)")
    print("  - 输入 'skill:clear' 清空手动加载的技能")
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
                print(f"\n--- 记忆状态 ---")
                print(f"当前会话:   {mem['thread_id']}")
                print(f"Checkpoint: {mem['checkpoint_backend']} → {mem['checkpoint_file']}")
                print(f"已存消息:   {mem['checkpoint_messages']} 条")
                print(f"长期记忆:   {mem['long_term_count']} 条")
                print(f"总会话数:   {mem['total_threads']}")
                continue

            # ========= 会话(Thread)管理 =========
            if user_input.lower() == 'thread' or user_input.lower() == 'threads':
                threads = agent.memory.list_threads()
                current = agent.memory.thread_id
                if not threads:
                    print("\n暂无会话记录")
                    continue

                # 构建 label:带消息数预览,value 是 thread_id
                options = []
                for tid in threads:
                    # 临时切过去读消息数,再切回当前(不实际修改状态)
                    try:
                        saved = agent.memory.thread_id
                        agent.memory.thread_id = tid
                        msg_count = len(agent.memory.get_messages() or [])
                        agent.memory.thread_id = saved
                    except Exception:
                        msg_count = 0
                    mark = " (当前)" if tid == current else ""
                    label = f"{tid}  [{msg_count} 条消息]{mark}"
                    options.append((label, tid))

                selected = select_menu(
                    f"选择会话 (共 {len(threads)} 个,↑↓ 选择,Enter 切换)",
                    options,
                    current=current
                )
                if selected is None:
                    continue  # Esc 取消
                if selected == current:
                    print(f"\n已在当前会话: {current}")
                    continue
                # 执行切换
                agent.memory.switch_thread(selected)
                msgs = agent.memory.get_messages()
                print(f"\n已切换到会话: {selected} (恢复 {len(msgs or [])} 条历史消息)")
                continue

            if user_input.lower() == 'thread:new':
                old = agent.memory.thread_id
                new = agent.memory.new_thread()
                print(f"\n已开启新会话: {new}")
                print(f"原会话 {old} 已保留,可用 'thread' 切回")
                continue

            if user_input.lower().startswith('thread:delete'):
                parts = user_input.split(None, 1)
                if len(parts) < 2:
                    print("用法: thread:delete <thread_id>")
                    continue
                tid = parts[1].strip()
                # 二次确认
                confirm = input(f"确认删除会话 '{tid}'? 此操作不可恢复 [y/N]: ").strip().lower()
                if confirm not in ("y", "yes"):
                    print("已取消")
                    continue
                was_current = tid == agent.memory.thread_id
                ok = agent.memory.delete_thread(tid)
                if ok:
                    print(f"\n已删除会话: {tid}")
                    if was_current:
                        print(f"当前会话已被删除,自动切换到: {agent.memory.thread_id}")
                else:
                    print(f"\n删除失败:会话 '{tid}' 不存在或数据库错误")
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

            # ========= 模型管理 =========
            low = user_input.lower()

            # 查看可用模型(方向键选择)
            if low == 'model' or low == 'models':
                info = llm.get_info()
                models = llm.list_models()
                options = [(m, m) for m in models]
                selected = select_menu(
                    f"选择模型 [{info['provider_name']}]",
                    options,
                    current=llm.model
                )
                if selected is None:
                    continue  # Esc 取消
                if selected == llm.model:
                    print(f"\n模型未变: {llm.model}")
                    continue
                try:
                    llm.switch_model(selected)
                    # 重建 Agent 以应用新模型
                    agent.switch_llm(llm)
                    info = llm.get_info()
                    print(f"\n已切换模型: {info['model']} (提供商: {info['provider_name']})")
                except Exception as e:
                    print(f"\n切换失败: {e}")
                continue

            # 直接切换模型: 'model:<name>' 或 'model <name>'
            if low.startswith('model:') or (low.startswith('model ') and not low.startswith('model:')):
                if low.startswith('model:'):
                    new_model = user_input[6:].strip()
                else:
                    parts = user_input.split(None, 1)
                    new_model = parts[1].strip() if len(parts) > 1 else ""

                if not new_model:
                    print("用法: model:<模型名>  或  model <模型名>")
                    print("示例: model:glm-4-flash")
                    continue

                try:
                    llm.switch_model(new_model)
                    agent.switch_llm(llm)
                    info = llm.get_info()
                    print(f"\n已切换模型: {info['model']} (提供商: {info['provider_name']})")
                except ValueError as e:
                    print(f"\n切换失败: {e}")
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

            # ========= 技能阅读(Skills)管理 =========
            low_skill = user_input.lower()
            if low_skill == 'skill' or low_skill == 'skills':
                skills = agent.list_skills()
                if not skills:
                    print("\n当前没有可用技能(目录 .agents/skills 为空)")
                else:
                    print("\n可用技能:")
                    print("-" * 50)
                    for s in skills:
                        print(f"  • {s['name']}")
                        if s['description']:
                            desc = s['description']
                            if len(desc) > 80:
                                desc = desc[:80] + "..."
                            print(f"      {desc}")
                    print("-" * 50)
                    print("使用 'skill:<name>' 加载技能到当前会话, 'skill:clear' 清空已加载技能")
                continue

            if low_skill.startswith('skill:'):
                rest = user_input[6:].strip()
                # 拆成: 技能名(首个词) + 可选任务(剩余文本)
                parts = rest.split(None, 1)
                sub = parts[0].strip().lower()
                task_text = parts[1].strip() if len(parts) > 1 else ""
                if sub in ('clear', '清空', 'reset'):
                    agent.clear_skills()
                    print("\n已清空手动加载的技能")
                elif not sub:
                    print("\n用法: skill:<name> [任务]  或  skill (列出所有)")
                else:
                    matched = None
                    for s in agent.list_skills():
                        if s['name'].lower() == sub:
                            matched = s['name']
                            break
                    if matched is None:
                        for s in agent.list_skills():
                            if sub in s['name'].lower():
                                matched = s['name']
                                break
                    if matched is None:
                        available = [s['name'] for s in agent.list_skills()]
                        print(f"\n未找到技能: {sub}")
                        print(f"可用: {', '.join(available) or '(无)'}")
                    elif agent.load_skill(matched):
                        print(f"\n已加载技能: {matched} (将注入后续对话的 system prompt)")
                        if not agent.auto_match_skills:
                            print("提示: 自动匹配已关闭,本技能仅手动加载生效")
                        # 若附带任务,加载后直接以 Agent 模式执行
                        if task_text:
                            result = agent.run(task_text)
                            print(f"\n助手: {result}")
                    else:
                        print(f"\n加载失败: {matched}")
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
