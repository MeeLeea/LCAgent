"""
运行时配置加载 - 将原本硬编码在 main.py 的参数外置到 config/agent_config.json

支持的键(均带默认值,缺省不报错):
    max_iterations        int   单次 invoke 最大推理步数(recursion_limit)
    skills_dir            str    技能目录(相对项目根或绝对路径)
    auto_match_skills     bool   任务自动匹配并注入技能
    enable_mcp            bool   是否加载 MCP 工具
    memory_size           int    兼容旧 API 的记忆容量
    verbose               bool   是否打印详细过程
    mcp_config_file       str    MCP 配置文件(相对项目根或绝对路径)
    max_context_messages  int    长上下文裁剪阈值(0=关闭)
    context_trim_keep     int    裁剪时保留的最近消息条数
"""
import os
import json
from typing import Dict, Any

DEFAULTS: Dict[str, Any] = {
    "max_iterations": 15,
    "skills_dir": ".agents/skills",
    "auto_match_skills": True,
    "enable_mcp": True,
    "memory_size": 10,
    "verbose": True,
    "mcp_config_file": "config/mcp_servers.json",
    "max_context_messages": 0,
    "context_trim_keep": 12,
}


def load_agent_config(config_file: str) -> Dict[str, Any]:
    """
    加载 agent 运行时配置,与默认值合并

    Args:
        config_file: config/agent_config.json 路径(相对或绝对)

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
        except (json.JSONDecodeError, IOError):
            pass
    return cfg


def resolve_path(path: str, base_dir: str) -> str:
    """将配置中的相对路径解析为基于项目根的绝对路径(并规范化分隔符)"""
    if not path:
        return path
    resolved = path if os.path.isabs(path) else os.path.join(base_dir, path)
    return os.path.normpath(resolved)
