"""
运行时配置加载 - 将原本硬编码在 main.py 的参数外置到 agent/agent_config.json

支持的键(均带默认值,缺省不报错):
    name                  str   Agent 名称
    max_iterations        int   单次 invoke 最大推理步数(recursion_limit)
    skills_dir            str    技能目录(相对项目根或绝对路径)
    auto_match_skills     bool   任务自动匹配并注入技能
    enable_mcp            bool   是否加载 MCP 工具
    memory_size           int    兼容旧 API 的记忆容量
    verbose               bool   是否打印详细过程
    mcp_config_file       str    MCP 配置文件(相对项目根或绝对路径)
    max_context_messages  int    长上下文裁剪阈值(0=关闭)
    context_trim_keep     int    裁剪时保留的最近消息条数

Agent 核心提示词加载顺序：
    1. agent/AGENT.md（优先）
    2. 内置默认提示词（fallback）
"""
import json
import os
from typing import Any

_DEFAULT_AGENT_CORE_PROMPT = (
    "你是一个智能助手，配备了多种工具（文件读写、目录管理、搜索、计算、定时任务等）。\n"
    "\n"
    "【重要规则】\n"
    "1. 你有能力调用各种工具在用户本地真正执行操作。当用户要求操作文件/搜索/测试/创建目录等时，你【必须】"
    "调用相应工具完成，不要回复'我无法访问文件系统'、'我没有权限'、'请你自己保存'之类的话。"
    "只有当用户进行纯知识性问答（不需要操作文件/搜索/计算）时才直接回答，不调用工具。\n"
    "2. 创建文件、脚本、文件夹的默认位置是项目根目录下的 'tests/' 目录（即 main.py 所在目录）。\n"
    "3. 多步任务（如先创建目录再写文件）请依次调用多个工具。"
    "如果用户要求跑一下或测试一下，直接执行相应工具或测试文件。\n"
    "4. 如果用户要求生成新的工具文件，直接在 tools 目录下使用create_tools.py进行创建。\n"
    "5. 危险命令（如 rm -rf、format、shutdown 等）会被安全策略拦截或要求用户确认，"
    "不要尝试使用破坏性命令；删除/移动文件时优先使用专门的文件工具。\n"
    "6. 当任务涉及专业领域（如提交 git、生成 pptx、查找技能等）时，"
    "优先用 read_skill 工具读取对应技能的详细指引并按指引完成。\n"
    "7. 当需要人工确认、选择或补充信息才能继续时，调用 ask_human 工具并提供结构化 choices，"
    "等待返回的结构化选择后再继续；不要用普通文本假装等待人工输入。\n"
    "8. 当用户要求在某个时间点（如'2分钟后'、'明天下午3点'、'下周一'）或按周期"
    "（如'每天9点'、'每周一'、'工作日下午5点半'）执行任务时，【必须】按以下流程操作，"
    "不要立即执行任务本身：\n"
    "    ① 调用 get_local_time 获取当前精确时间\n"
    "    ② 计算出 execute_time（ISO 8601，如 '2026-07-29T17:36:00'）或 cron 表达式\n"
    "    ③ 调用 schedule_task 登记任务，完成后回复'任务已登记，将于[时间]自动执行'\n"
    "    要点：\n"
    "    - task_text 只写任务本身（自然语言，去掉时间），【不要写代码或函数调用】\n"
    "    - 一次性 → task_type='one_time' + execute_time；周期 → task_type='periodic' + cron_expr\n"
    "    - cron 示例：'0 9 * * *'=每天9点，'30 8 * * 1-5'=工作日8:30，'0 17 * * 5'=每周五17点\n"
    "    - 查询/管理任务 → list_scheduled_tasks / cancel_scheduled_task\n"
    "    - 清理历史任务 → delete_scheduled_task（删单个）/ cleanup_finished_tasks（批量清理已完成/失败/取消的）\n"
    "\n"
    "请用中文回答。"
)

DEFAULTS: dict[str, Any] = {
    "name": "LCAgent",
    "max_iterations": 15,
    "skills_dir": ".agents/skills",
    "auto_match_skills": True,
    "enable_mcp": True,
    "memory_size": 10,
    "verbose": True,
    "mcp_config_file": "config/mcp_servers.json",
    "max_context_messages": 0,
    "context_trim_keep": 12,
    "agent_prompt_file": "agent/AGENT.md",
    "provider": "zhipu",
    "model": None
}


def load_agent_config(config_file: str, base_dir: str = None) -> dict[str, Any]:
    """
    加载 agent 运行时配置,与默认值合并

    Args:
        config_file: agent/agent_config.json 路径(相对或绝对)
        base_dir: 项目根目录,用于锚定相对路径(为 None 时使用当前工作目录)

    Returns:
        合并后的配置字典
    """
    cfg = dict(DEFAULTS)
    
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for key in DEFAULTS:
                if key in data:
                    cfg[key] = data[key]
        except (OSError, json.JSONDecodeError):
            pass
    
    # 加载 Agent 核心提示词（优先从 AGENT.md 读取）
    prompt_file = cfg.get("agent_prompt_file", "agent/AGENT.md")
    if base_dir:
        prompt_file = resolve_path(prompt_file, base_dir)
    cfg["agent_core_prompt"] = _load_agent_prompt(prompt_file)
    
    return cfg


def _load_agent_prompt(prompt_file: str) -> str:
    """
    加载 Agent 核心提示词

    加载顺序：
        1. agent_prompt_file 指定的文件（默认 agent/AGENT.md）
        2. 内置默认提示词（fallback）

    Args:
        prompt_file: 提示词文件路径（相对或绝对）

    Returns:
        Agent 核心提示词字符串
    """
    if prompt_file and os.path.exists(prompt_file):
        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    return content
        except (OSError, UnicodeDecodeError):
            pass
    
    # Fallback 到默认提示词
    return _DEFAULT_AGENT_CORE_PROMPT


def resolve_path(path: str, base_dir: str) -> str:
    """将配置中的相对路径解析为基于项目根的绝对路径(并规范化分隔符)"""
    if not path:
        return path
    resolved = path if os.path.isabs(path) else os.path.join(base_dir, path)
    return os.path.normpath(resolved)
