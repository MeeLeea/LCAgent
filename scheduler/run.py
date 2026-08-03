# -*- coding: utf-8 -*-
"""
独立调度器进程入口 - 逻辑 B 的启动点

用法::

    # 使用 config/scheduler_config.json 配置启动
    python -m scheduler.run

配置值优先级：config/scheduler_config.json > 环境变量 > 内置默认值

配置文件路径写死在模块顶部 SCHEDULER_CONFIG_FILE 常量中，如需修改直接改该常量。

设计要点：
    - 调度器与 Agent 对话进程分离，可独立部署/重启
    - agent_factory 每次执行任务时创建一个 AgentCore 实例（加载工具 + LLM）
    - 启动时自动从数据库同步未完成的周期任务（程序重启不丢失）
"""
import json
import os
import sys

# 确保项目根在 sys.path 中（支持 python -m 和直接运行）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from agent.config import load_agent_config, resolve_path
from agent.llm_client import LLMClient, load_providers
from agent import AgentCore
from scheduler import TaskStore, SchedulerEngine


# ---- 默认路径 ----

BASE_DIR = _PROJECT_ROOT
LLM_CONFIG_FILE = os.path.join(BASE_DIR, "config", "llm_config.json")
AGENT_CONFIG_FILE = os.path.join(BASE_DIR, "agent", "agent_config.json")
SCHEDULER_CONFIG_FILE = os.path.join(BASE_DIR, "config", "scheduler_config.json")
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "memory", "scheduled_tasks.sqlite")
MEMORY_FILE = os.path.join(BASE_DIR, "memory", "memory.json")
CHECKPOINT_FILE = os.path.join(BASE_DIR, "memory", "checkpoints.sqlite")


# ---- 调度器配置加载（与 config.load_agent_config 同模式） ----

SCHEDULER_DEFAULTS = {
    "db_path": "memory/scheduled_tasks.sqlite",
    "poll_interval": 30,
    "timezone": None,
    "max_retries": 3,
    "max_workers": 5,
    "provider": None,
    "blocking": True,
}


def load_scheduler_config(config_file: str) -> dict:
    """
    加载调度器运行时配置，与默认值合并。

    文件不存在时使用全部默认值，不报错。
    相对路径的 db_path 会在使用时锚定项目根。
    """
    cfg = dict(SCHEDULER_DEFAULTS)
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for key in SCHEDULER_DEFAULTS:
                if key in data:
                    cfg[key] = data[key]
        except (json.JSONDecodeError, IOError):
            pass
    return cfg


# ---- Agent 工厂 ----

def make_agent_factory(provider: str):
    """
    构造 agent_factory：每次调用返回一个新的 AgentCore 实例。

    AgentCore 初始化会加载 LLM + 本地工具 + MCP 工具 + 技能，
    较重，但保证每次定时任务执行在干净上下文中（互不干扰）。
    """
    # 预加载配置（避免每次创建 agent 都读文件）
    agent_config = load_agent_config(AGENT_CONFIG_FILE)
    skills_dir = resolve_path(agent_config["skills_dir"], BASE_DIR)
    mcp_config_file = resolve_path(agent_config["mcp_config_file"], BASE_DIR)

    # 预创建 LLM 客户端（创建 chat model 是最重的部分，只做一次）
    llm = LLMClient(provider=provider, config_file=LLM_CONFIG_FILE)
    print(f"[Scheduler] LLM 已就绪: {llm.get_info()['provider_name']} / {llm.model}")

    def factory() -> AgentCore:
        return AgentCore(
            llm_client=llm,
            name=agent_config["name"],
            memory_size=agent_config["memory_size"],
            long_term_memory_file=MEMORY_FILE,
            checkpoint_file=CHECKPOINT_FILE,
            max_iterations=agent_config["max_iterations"],
            verbose=True,
            mcp_config_file=mcp_config_file,
            enable_mcp=agent_config["enable_mcp"],
            skills_dir=skills_dir,
            auto_match_skills=agent_config["auto_match_skills"],
            max_context_messages=agent_config["max_context_messages"],
            context_trim_keep=agent_config["context_trim_keep"],
            process_type="scheduler",
            agent_core_prompt=agent_config["agent_core_prompt"]
        )

    return factory


def pick_provider(config_provider):
    """
    选择 LLM 提供商，优先级：配置文件 > 环境变量 > 第一个可用提供商。
    """
    # 配置文件指定
    if config_provider:
        return config_provider

    # 环境变量
    env_provider = os.environ.get("AGENT_LLM_PROVIDER")
    if env_provider:
        return env_provider

    # 自动选择第一个已配置 API Key 的提供商
    providers = load_providers(LLM_CONFIG_FILE)
    if not providers:
        print("错误: config/llm_config.json 中未配置任何 LLM 提供商")
        sys.exit(1)

    for key, cfg in providers.items():
        if cfg.get("api_key") or os.environ.get(cfg.get("env_key", "")):
            return key

    return next(iter(providers))


def main():
    # 1. 加载配置文件（config/scheduler_config.json）
    config = load_scheduler_config(SCHEDULER_CONFIG_FILE)

    print(f"[Scheduler] 使用配置文件: {SCHEDULER_CONFIG_FILE}")

    # 从配置文件取值
    provider = pick_provider(config.get("provider"))
    db_path = resolve_path(config["db_path"], BASE_DIR)
    poll_interval = config["poll_interval"]
    timezone = config.get("timezone")
    max_workers = config["max_workers"]

    print("=" * 60)
    print("  定时任务调度器 (Scheduler Engine)")
    print("=" * 60)

    # 2. 构造 agent_factory
    print(f"\n[Scheduler] 使用 LLM 提供商: {provider}")
    agent_factory = make_agent_factory(provider)

    # 2.5 预加载 team 模块,触发 @register_agent 装饰器注册工作流角色
    #     使 executor 中的 workflow: 任务能正确构建所需 Agent
    try:
        import team  # noqa: F401  触发各 agent 模块的 @register_agent
        from graph.registry import list_workflows, AGENT_REGISTRY
        wf_list = list_workflows()
        print(f"[Scheduler] 已注册 Agent: {', '.join(sorted(AGENT_REGISTRY.keys()))}")
        print(f"[Scheduler] 可用工作流: {', '.join(name for name, _ in wf_list)}")
    except Exception as exc:
        print(f"[Scheduler] 警告: 工作流模块加载失败({exc}),workflow: 任务将不可用")

    # 3. 初始化 TaskStore
    task_store = TaskStore(db_path)
    print(f"[Scheduler] 数据库: {db_path}")

    # 4. 创建调度引擎（阻塞模式）
    engine = SchedulerEngine(
        task_store=task_store,
        agent_factory=agent_factory,
        poll_interval=poll_interval,
        blocking=True,
        timezone=timezone,
        max_workers=max_workers,
    )

    # 5. 启动（阻塞直到 Ctrl+C）
    print(f"\n[Scheduler] 轮询间隔: {poll_interval}s")
    print(f"[Scheduler] 并发线程数: {max_workers}")
    print(f"[Scheduler] 时区: {timezone or '系统默认'}")
    print()

    try:
        engine.start()  # BlockingScheduler 会阻塞在这里
    except KeyboardInterrupt:
        print("\n[Scheduler] 收到中断信号，正在停止...")
        engine.stop()
        print("[Scheduler] 已安全退出")


if __name__ == "__main__":
    main()