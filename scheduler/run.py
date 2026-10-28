"""
独立调度器进程入口 - 逻辑 B 的启动点

用法::

    # 使用 config/scheduler_config.json 配置启动
    python -m scheduler.run

配置值优先级：config/scheduler_config.json > 环境变量 > 内置默认值

配置文件路径写死在模块顶部 SCHEDULER_CONFIG_FILE 常量中，如需修改直接改该常量。

设计要点：
    - 调度器与 Agent 对话进程分离，可独立部署/重启
    - agent_factory 每次执行任务时创建一个 AgentCore 实例（加载工具 + LLM），
      执行时通过 SessionManager.arun() 走三层架构（Agent → Session → Memory）
    - 启动时自动从数据库同步未完成的周期任务（程序重启不丢失）
"""
import asyncio
import json
import logging
import os
import sys
import threading

# 确保项目根在 sys.path 中（支持 python -m 和直接运行）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from agent import AgentCore
from agent.memory_llm import build_memory_llm_resolver
from llm.config import load_agent_config, resolve_path
from llm.llm_client import LLMClient, load_providers
from memory import MemoryContext
from scheduler import SchedulerEngine, TaskStore
from utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


# ---- 默认路径 ----

BASE_DIR = _PROJECT_ROOT
LLM_CONFIG_FILE = os.path.join(BASE_DIR, "config", "llm_config.json")
AGENT_CONFIG_FILE = os.path.join(BASE_DIR, "agent", "agent_config.json")
SCHEDULER_CONFIG_FILE = os.path.join(BASE_DIR, "config", "scheduler_config.json")
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "data", "scheduled_tasks.sqlite")
MEMORY_FILE = os.path.join(BASE_DIR, "data", "memory.json")
CHECKPOINT_FILE = os.path.join(BASE_DIR, "data", "checkpoints_async.sqlite")


# ---- 调度器配置加载（与 config.load_agent_config 同模式） ----

SCHEDULER_DEFAULTS = {
    "db_path": "data/scheduled_tasks.sqlite",
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
        except (OSError, json.JSONDecodeError):
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
    agent_prompt_file = resolve_path(agent_config["agent_prompt_file"], BASE_DIR)
    skills_dir = resolve_path(agent_config["skills_dir"], BASE_DIR)
    mcp_config_file = resolve_path(agent_config["mcp_config_file"], BASE_DIR)

    # 预创建 LLM 客户端（创建 chat model 是最重的部分，只做一次）
    # 采样参数由 LLMClient 内部从全局 agent_config.json 读取，无需外部传参
    llm = LLMClient(provider=provider, config_file=LLM_CONFIG_FILE)
    logger.info("LLM 已就绪: %s / %s", llm.get_info()["provider_name"], llm.model)

    async def factory() -> AgentCore:
        # 三层架构：先创建 MemoryContext（记忆基础设施），再创建 AgentCore（纯执行内核）
        memory_ctx = await MemoryContext.acreate(
            checkpoint_file=CHECKPOINT_FILE,
            use_sqlite=True,
            process_type="scheduler",
            llm_getter=lambda: llm,
            # 记忆链路参数由 memory/config.py 统一管理，不再经 agent_config.json 配置
            # 注意：本 factory 由 executor 经 asyncio.run 在**临时 loop** 调用，
            # 严禁在此创建周期协程（随任务结束即被销毁）；后台 purge 由
            # _start_purge_host 的专属守护线程 loop 承载（见 main）。
        )
        agent = await AgentCore.acreate(
            llm_client=llm,
            name=agent_config["name"],
            max_iterations=agent_config["max_iterations"],
            verbose=True,
            mcp_config_file=mcp_config_file,
            enable_mcp=agent_config["enable_mcp"],
            skills_dir=skills_dir,
            auto_match_skills=agent_config["auto_match_skills"],
            max_context_tokens=agent_config["max_context_tokens"],
            context_trim_keep=agent_config["context_trim_keep"],
            process_type="scheduler",
            agent_prompt_file=agent_prompt_file,
            max_execution_history=agent_config.get("max_execution_history", 100),
            tool_timeout=agent_config.get("tool_timeout", 120),
            short_term_size=agent_config.get("latest_msg_cnt", 10),
            checkpointer=memory_ctx.checkpointer,
            store=memory_ctx.store,
            extra_middleware=[memory_ctx.read_middleware],
            initial_thread_id=memory_ctx.thread_id,
            async_conn=memory_ctx.async_conn,
        )
        # 注入 MemoryManager → SessionManager 懒初始化时会自动接收
        agent.set_memory_manager(memory_ctx.memory_manager)
        agent._memory_context = memory_ctx  # 供 executor aclose 时关闭 SQLite 连接
        # 动态绑定：记忆组件按会话解析 LLM（回落 agent 当前 LLM），
        # 切换 provider 后自动同步，且会话级 provider/model 正确生效
        memory_ctx.bind_llm(build_memory_llm_resolver(agent.session, lambda: agent.llm))
        # 注入 forget_memory 工具的会话依赖（工具无法自取 Agent，模块级注入模式）
        from tools.forget_memory import configure_for_agent as configure_forget_tool

        configure_forget_tool(agent)
        return agent

    return factory


def _purge_host_main(
    checkpoint_file: str,
    interval: float,
    forgotten_retention_days: float,
    superseded_retention_days: float,
    ready: threading.Event,
    stop: threading.Event,
) -> None:
    """purge 守护线程的循环体：在专属事件循环内存活长驻 MemoryContext。

    该 loop 独立于 executor 的 per-task ``asyncio.run`` loop，purge 周期协程
    随线程存活直到进程退出（``stop`` 置位或线程被 daemon 回收）。
    """

    async def _run() -> None:
        # 专用 purge 上下文：仅承载后台清理，不写任何 thread。
        # llm_getter 传 None（purge 不需 LLM）；不传 process_type（避免无谓的
        # thread_id 前缀——本上下文不生成 thread，不影响共享 DB 路径）。
        ctx = await MemoryContext.acreate(
            checkpoint_file=checkpoint_file,
            use_sqlite=True,
            llm_getter=None,
            enable_purge_task=True,
            purge_interval_seconds=interval,
        )
        ready.set()  # 通知调用方：purge 已启动
        try:
            # 保持 loop 存活直到收到停止信号（周期性 purge 协程与之并行运行）。
            # threading.Event 同步可等待，改用短轮询避免阻塞事件循环。
            while not stop.is_set():
                await asyncio.sleep(1.0)
        finally:
            await ctx.aclose()

    asyncio.run(_run())


def _start_purge_host(
    checkpoint_file: str,
    interval: float = 86400,
    forgotten_retention_days: float = 7,
    superseded_retention_days: float = 90,
):
    """启动承载后台 purge 的守护线程（专属事件循环），返回可停止的 host 句柄。

    为什么用守护线程而非主 loop：scheduler 的任务经 executor 在**临时
    ``asyncio.run`` loop** 执行，周期协程若在该 loop 创建会随任务结束被销毁
    （计划 §3.1 明令禁止）。因此为 purge 单开一个长驻 daemon 线程，在其中
    ``asyncio.run`` 一个长驻 MemoryContext，purge 周期协程随线程存活。

    Args:
        checkpoint_file: 与 factory 相同的 SQLite 文件（共享同一长期记忆库）
        interval: purge 周期（默认 86400s = 24h）
        forgotten_retention_days: forgotten 保留天数阈值
        superseded_retention_days: superseded 保留天数阈值

    Returns:
        含 ``thread``（守护线程）与 ``stop()``（幂等停止）的 host 对象。
    """
    ready = threading.Event()
    stop = threading.Event()

    def _target() -> None:
        try:
            _purge_host_main(
                checkpoint_file,
                interval,
                forgotten_retention_days,
                superseded_retention_days,
                ready,
                stop,
            )
        except Exception as error:  # pragma: no cover - 守护线程兜底
            logger.warning("purge 守护线程异常退出: %s", error)

    thread = threading.Thread(target=_target, name="memory-purge-host", daemon=True)
    thread.start()
    # 等待 purge 上下文就绪（有界）；超时不阻塞主流程（线程仍会继续）
    ready.wait(timeout=10.0)

    class _PurgeHost:
        """purge 守护线程句柄。"""

        def __init__(self) -> None:
            self.thread = thread
            self._stop = stop
            self._stopped = False

        def stop(self) -> None:
            """幂等停止：置位停止信号并短暂等待线程退出。"""
            if self._stopped:
                return
            self._stopped = True
            self._stop.set()
            thread.join(timeout=5.0)

    return _PurgeHost()


def pick_provider(config_provider):
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
        logger.error("config/llm_config.json 中未配置任何 LLM 提供商")
        sys.exit(1)

    for key, cfg in providers.items():
        if cfg.get("api_key") or os.environ.get(cfg.get("env_key", "")):
            return key

    return next(iter(providers))


def main():
    # 0. 初始化结构化日志
    setup_logging()

    # 1. 加载配置文件（config/scheduler_config.json）
    config = load_scheduler_config(SCHEDULER_CONFIG_FILE)

    logger.info("使用配置文件: %s", SCHEDULER_CONFIG_FILE)

    # 从配置文件取值
    provider = pick_provider(config.get("provider"))
    db_path = resolve_path(config["db_path"], BASE_DIR)
    poll_interval = config["poll_interval"]
    timezone = config.get("timezone")
    max_workers = config["max_workers"]

    # 启动 banner（保持 print，面向用户）
    print("=" * 60)
    print("  定时任务调度器 (Scheduler Engine)")
    print("=" * 60)

    # 2. 构造 agent_factory
    logger.info("使用 LLM 提供商: %s", provider)
    agent_factory = make_agent_factory(provider)

    # 2.4 启动后台 purge 守护线程：专属事件循环承载长驻 MemoryContext 的
    #     周期清理（executor 的临时 asyncio.run loop 严禁承载周期协程）
    purge_host = _start_purge_host(CHECKPOINT_FILE)
    logger.info("后台 purge 守护线程已启动（间隔 86400s）")

    # 2.5 预加载 team 模块,触发 @register_agent 装饰器注册工作流角色
    #     使 executor 中的 workflow: 任务能正确构建所需 Agent
    try:
        import team  # noqa: F401  触发各 agent 模块的 @register_agent
        from graph.common import AGENT_REGISTRY, list_workflows
        wf_list = list_workflows()
        logger.info("已注册 Agent: %s", ", ".join(sorted(AGENT_REGISTRY.keys())))
        logger.info("可用工作流: %s", ", ".join(name for name, _ in wf_list))
    except Exception as exc:
        logger.warning("工作流模块加载失败(%s),workflow: 任务将不可用", exc)

    # 3. 初始化 TaskStore
    task_store = TaskStore(db_path)
    logger.info("数据库: %s", db_path)

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
    logger.info("轮询间隔: %ds", poll_interval)
    logger.info("并发线程数: %d", max_workers)
    logger.info("时区: %s", timezone or "系统默认")

    try:
        engine.start()  # BlockingScheduler 会阻塞在这里
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在停止...")
        engine.stop()
        purge_host.stop()
        logger.info("已安全退出")


if __name__ == "__main__":
    main()