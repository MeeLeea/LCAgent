"""
运行时配置加载 - 将原本硬编码在 main.py 的参数外置到 agent/agent_config.json

支持的键(均带默认值,缺省不报错):
    name                       str   Agent 名称
    max_iterations             int   单次 invoke 最大推理步数(recursion_limit)
    skills_dir                 str    技能目录(相对项目根或绝对路径)
    auto_match_skills          bool   任务自动匹配并注入技能
    enable_mcp                 bool   是否加载 MCP 工具
    latest_msg_cnt             int    短期上下文窗口消息条数（最近 N 条消息，传递给 SessionRegistry.aget_short_term）
    verbose                    bool   是否打印详细过程
    mcp_config_file            str    MCP 配置文件(相对项目根或绝对路径)
    max_context_messages       int    长上下文裁剪阈值(0=关闭)
    context_trim_keep          int    裁剪时保留的最近消息条数
    max_execution_history      int    执行历史最大条数
    tool_timeout               int    工具调用超时(秒)
    temperature                float  LLM 采样温度(全局默认，非团队场景生效；
                                      团队角色分层配置于自身 agent_config.json，
                                      缺省回退 DEFAULTS，不读取全局自定义值)
    max_tokens                 int    LLM 最大生成 token 数(来源规则同 temperature)
    stream_chunk_timeout       float  LLM 流式响应 chunk 间隔超时(秒；来源规则同
                                      temperature。云雾网关思考型模型可能长时间零字节，
                                      默认 300.0 以免 langchain-openai 默认 120s 误触发)

Memory 层配置（由 memory/config.py 统一管理，不写入 agent_config.json）:
    memory_buffer_delay_seconds   int  记忆写入防抖延迟(秒)
    memory_max_buffer_messages    int  防抖 buffer 最大消息数(超出强制刷新)
    memory_max_facts_per_thread   int  单线程最大 fact 条数(超出 LRU 淘汰)
    memory_recall_limit           int  召回长期记忆时的默认条数上限
    session_enable_memory         bool SessionManager 是否启用长期记忆处理

Agent 核心提示词加载顺序：
    1. agent/AGENT.md（优先）
    2. 内置默认提示词（fallback）
"""
import json
import os
from typing import Any
# 内置默认提示词（fallback）：内容与 agent/AGENT.md 保持一致（隐式拼接，值逐字符相同），
# 使文件缺失/为空时主对话 Agent 与工作流节点仍能拿到同一套行为规则。
# 必须保留 `## 重要规则` / `## 工具规则` 小节标题——`load_agent_rules` 依赖标题按角色能力提取。
_DEFAULT_AGENT_CORE_PROMPT: str = (
    "# Agent 核心提示词\n"
    "\n"
    "你是一个智能助手，配备了多种工具（文件读写、目录管理、搜索、计算、定时任务等）。\n"
    "\n"
    "## 重要规则\n"
    "\n"
    "1. 当任务涉及专业领域（如提交 git、生成 pptx、查找技能等）时，优先用 read_skill 工具读取对应技能的详细指引并按指引完成。\n"
    "2. 文件路径始终以【当前最新用户消息】为准：若此前调用文件工具所用的路径与最新用户消息中给出的路径不一致，必须以最新消息重新解析，不要沿用历史失败的路径写法。路径基于当前工作空间（workspace）根目录按相对路径解析；若相对路径的首段与工作空间目录名重复（例如工作空间为 document 时写成 document\\xxx），会导致路径重复拼接，应去除该前缀。当工具报\"目录/文件不存在\"错误时，先核对路径是否重复拼接或基于错误的工作空间，再修正重试；文件类工具不允许越出工作空间，执行类工具的工作目录固定为工作空间。\n"
    "\n"
    "请用中文回答。\n"
    "\n"
    "## 工具规则\n"
    "\n"
    "1. 你有能力调用各种工具在用户本地真正执行操作。当用户要求操作文件/搜索/测试/创建目录等时，你【必须】调用相应工具完成，不要回复'我无法访问文件系统'、'我没有权限'、'请你自己保存'之类的话。只有当用户进行纯知识性问答（不需要操作文件/搜索/计算）时才直接回答，不调用工具。\n"
    "2. 创建文件、脚本、文件夹的默认位置根据工作目录动态确定，默认临时文件放在'.drafts/'目录下，markdown文件放在'docs/'目录下；目录不存在时先创建目录再写入。\n"
    "3. 多步任务（如先创建目录再写文件）请依次调用多个工具。如果用户要求跑一下或测试一下，直接执行相应工具或测试文件。\n"
    "4. 如果用户要求生成新的工具文件，直接使用工具 create_tool 进行创建。\n"
    "5. 危险命令（如 rm -rf、format、shutdown 等）会被安全策略拦截或要求用户确认，不要尝试使用破坏性命令；删除/移动文件时优先使用专门的文件工具。\n"
    "6. 当需要人工确认、选择或补充信息才能继续时，调用 ask_human 工具并提供结构化 choices，等待返回的结构化选择后再继续；不要用普通文本假装等待人工输入。\n"
    "7. 当用户要求在某个时间点（如'2分钟后'、'明天下午3点'、'下周一'）或按周期（如'每天9点'、'每周一'、'工作日下午5点半'）执行任务时，【必须】按以下流程操作，不要立即执行任务本身：\n"
    "\n"
    "   - ① 调用 get_local_time 获取当前精确时间\n"
    "   - ② 计算出 execute_time（ISO 8601，如 '2026-07-29T17:36:00'）或 cron 表达式\n"
    "   - ③ 调用 schedule_task 登记任务，完成后回复'任务已登记，将于[时间]自动执行'\n"
    "\n"
    "   要点：\n"
    "\n"
    "   - task_text 只写任务本身（自然语言，去掉时间），【不要写代码或函数调用】\n"
    "   - 一次性 → task_type='one_time' + execute_time；周期 → task_type='periodic' + cron_expr\n"
    "   - cron 示例：'0 9 * * *'=每天9点，'30 8 * * 1-5'=工作日8:30，'0 17 * * 5'=每周五17点\n"
    "   - 查询/管理任务 → list_scheduled_tasks / cancel_scheduled_task\n"
    "   - 清理历史任务 → delete_scheduled_task（删单个）/ cleanup_finished_tasks（批量清理已完成/失败/取消的）\n"
    "8. 避免交互式命令；长任务显式传 timeout；超时后读 timeout_reason/partial_stdout 修正命令再重试，而不是原样重发。\n"
    "9. 同一工具同类失败 ≥2 次必须改变策略或询问用户，不要重复相同参数。\n"
    "10. 执行/写文件后应核对真实返回值再汇报，禁止编造执行结果。\n"
    "11. 工具列表可能包含动态加载的 MCP 工具（如文件读写、网页抓取）；请按各工具的 schema 描述传参，不要假设某个工具一定存在。"
)

DEFAULTS: dict[str, Any] = {
    "name": "LCAgent",
    "max_iterations": 15,
    "skills_dir": ".agents/skills",
    "auto_match_skills": True,
    "enable_mcp": True,
    "latest_msg_cnt": 10,
    "verbose": True,
    "mcp_config_file": "config/mcp_servers.json",
    "max_context_messages": 0,
    "context_trim_keep": 12,
    "max_execution_history": 100,
    "agent_prompt_file": "agent/AGENT.md",
    "tool_timeout": 120,
    "temperature": 0.7,
    "max_tokens": 8192,
    "stream_chunk_timeout": 300.0,
}


# 项目根目录(本文件位于 <root>/llm/config.py)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 全局 agent 配置路径(供 LLMClient 等模块内部读取采样参数默认值)
DEFAULT_AGENT_CONFIG_FILE = os.path.join(_PROJECT_ROOT, "agent", "agent_config.json")

# 全局 Agent 核心提示词路径(供工作流节点等非主对话模块复用同一份 agent/AGENT.md)
DEFAULT_AGENT_PROMPT_FILE = os.path.join(_PROJECT_ROOT, "agent", "AGENT.md")

# agent/AGENT.md 的规则小节标题(按角色能力划分,供工作流节点按需继承):
#   - 重要规则: 通用行为规则,所有角色均继承
#   - 工具规则: 依赖具体工具的规则,仅持有工具的角色继承
AGENT_RULES_HEADING = "## 重要规则"
AGENT_TOOL_RULES_HEADING = "## 工具规则"


def load_agent_config(config_file: str) -> dict[str, Any]:
    """
    加载 agent 运行时配置,与默认值合并

    Args:
        config_file: agent/agent_config.json 路径(相对或绝对)

    Returns:
        合并后的配置字典
    """
    cfg = dict(DEFAULTS)
    
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 透传 JSON 中所有键，默认值仅覆盖 DEFAULTS 中定义的键
            cfg.update(data)
        except (OSError, json.JSONDecodeError):
            pass
    
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


def load_agent_rules(
    prompt_file: str = DEFAULT_AGENT_PROMPT_FILE,
    include_tool_rules: bool = False,
) -> str:
    """
    加载 agent/AGENT.md 的行为规则小节（供工作流节点按角色能力继承）

    仅提取规则小节，剔除文件标题、说明等其余内容，并按角色是否持有工具过滤：

    - `## 重要规则`：通用行为规则，所有角色均继承
    - `## 工具规则`：依赖具体工具的规则（调用工具、长流程等），仅当
      `include_tool_rules=True`（角色持有工具）时追加

    这样可避免「必须调用工具」等条款落在 Manager / Terminator 等不持有工具的
    纯文本角色节点上，既省 token 也避免误导模型去调不存在的工具。

    文件缺失时 `_load_agent_prompt` 回退内置默认提示词，其内容与 agent/AGENT.md 一致（含 `## 重要规则` / `## 工具规则` 标题），
    因此规则小节仍会被提取，不再等同于「不注入」。

    Args:
        prompt_file: 提示词文件路径（默认 agent/AGENT.md）
        include_tool_rules: 是否附带「工具规则」小节（持有工具的角色传 True）

    Returns:
        规则文本（多小节以空行分隔）；文件缺失、内容为空或无对应小节时返回空串
    """
    lines = _load_agent_prompt(prompt_file).splitlines()
    headings = [AGENT_RULES_HEADING]
    if include_tool_rules:
        headings.append(AGENT_TOOL_RULES_HEADING)

    sections: list[str] = []
    for heading in headings:
        start = next((i for i, ln in enumerate(lines) if ln.strip() == heading), None)
        if start is None:
            continue
        # 从标题行起收集，遇下一个二级标题即截止
        end = next(
            (
                i
                for i in range(start + 1, len(lines))
                if lines[i].strip().startswith("## ")
            ),
            len(lines),
        )
        sections.append("\n".join(lines[start:end]).strip())
    return "\n\n".join(section for section in sections if section)


def load_team_agent_config(agent_name: str, base_dir: str) -> dict[str, Any]:
    """
    从 team/team_agents.json 加载团队角色配置(default + 角色覆盖)

    Args:
        agent_name: 角色名称(如 "architect", "worker")
        base_dir: 项目根目录

    Returns:
        合并后的配置字典;文件不存在或角色不存在时返回空字典
    """
    team_config_path = os.path.join(base_dir, "team", "team_agents.json")
    if not os.path.exists(team_config_path):
        return {}

    try:
        with open(team_config_path, "r", encoding="utf-8") as f:
            team_config = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}

    result = dict(team_config.get("default", {}))
    if agent_name in team_config:
        result.update(team_config[agent_name])
    return result


def resolve_path(path: str, base_dir: str) -> str:
    """将配置中的相对路径解析为基于项目根的绝对路径(并规范化分隔符)"""
    if not path:
        return path
    resolved = path if os.path.isabs(path) else os.path.join(base_dir, path)
    return os.path.normpath(resolved)
