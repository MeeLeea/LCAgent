"""
安全护栏模块 - 终端/危险命令的黑名单、白名单与交互确认

设计目标:
- 防止 Agent 自动执行灾难性命令(rm -rf /、format、fork bomb 等)
- 对"危险但有时需要"的命令(rm、sudo、chmod、kill 等)进行交互式确认
- 支持黑名单(默认) / 白名单两种模式,可通过 config/safety.json 配置

决策函数:
    check_command(command) -> (status, reason)
        status ∈ {"allow", "deny", "confirm"}
    check_exec()          -> (status, reason)   # 执行脚本(.py/.ps1)时的判定
    check_path(path)      -> (allowed, reason)  # 删除/操作路径时的保护

 配置 (config/safety.json):
    {
        "mode": "blacklist",        # blacklist | whitelist
        "confirm_dangerous": true,   # 是否对危险命令交互确认
        "blacklist": [],            # 追加的拒绝正则(与内置合并)
        "whitelist": ["echo", "dir", "ls", "python", "pip"]  # 白名单模式下允许的首命令
    }

交互确认: confirm(prompt) 读取终端输入,默认拒绝(空/超时/EOF 均视为拒绝)
"""
import os
import re
import json
from typing import Tuple, Dict, Any, List, Optional, Callable

# ============ 内置规则 ============

# 始终拒绝的灾难性模式(命中即 deny,不询问)
BUILTIN_BLOCKLIST = [
    r"\brm\s+-rf\b", r"\brm\s+-fr\b", r"\brm\b.*--recursive",
    r"\brd\s+/[sq]", r"\bdeltree\b",
    # 覆盖 Windows/PowerShell 的递归删除和编码命令，避免只拦截 Unix 写法。
    r"\bremove-item\b(?=[^\r\n]*\s-(?:recurse|r)\b)(?=[^\r\n]*\s-(?:force|f)\b)",
    r"\b(?:del|erase)\b[^\r\n]*\s/s\b",
    r"\b(?:rd|rmdir)\b[^\r\n]*\s/s\b",
    r"\b(?:powershell|pwsh)(?:\.exe)?\b[^\r\n]*\s-(?:e|enc|encodedcommand)\b",
    r"\bformat\b", r"\bmkfs", r"\bdd\b.*\bif=",
    r"\bshutdown\b", r"\breboot\b",
    r":\(\)\s*\{.*\}\s*;",                      # fork bomb :(){:|:&};:
    r"curl\b[^\n|]*\|\s*(sh|bash)", r"wget\b[^\n|]*\|\s*(sh|bash)",
    r">\s*/dev/(sd|hd|nvme|sda)",
    r"\breg\s+(delete|add)\b", r"\bnetsh\b",
]

# 危险但非灾难性模式(命中且开启确认时 -> confirm)
BUILTIN_CONFIRM = [
    r"\bsudo\b", r"\brm\b", r"\bchmod\b", r"\bchown\b",
    r"\bmv\b", r"\bkill\b", r"\btaskkill\b", r"\bschtasks\b",
    r"\bremove-item\b", r"\b(?:del|erase|rd|rmdir)\b",
    # 解释器可隐藏任意副作用，脚本、内联代码和命令包装器统一要求人工确认。
    r"\b(?:python(?:3)?|py)(?:\.exe)?\b\s+(?:-[cmo]\b|[^\s;&|]+\.py\b)",
    r"\b(?:powershell|pwsh)(?:\.exe)?\b[^\r\n]*\s-(?:command|file)\b",
    r"\bcmd(?:\.exe)?\b\s+/(?:c|k)\b",
    r"\b(?:bash|sh)\b\s+-c\b",
]

# 默认配置
DEFAULT_CONFIG = {
    "mode": "blacklist",        # blacklist | whitelist
    "confirm_dangerous": True,
    "blacklist": [],
    "whitelist": ["echo", "dir", "ls", "python", "pip", "git", "cat", "type"],
}

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "safety.json"
)

# 模块级缓存配置
_config_cache: Optional[Dict[str, Any]] = None

# 交互确认后端：None 时使用终端 input()；server 等非交互环境可替换为 interrupt 后端。
_confirm_backend: Optional[Callable[[str], bool]] = None


def load_config() -> Dict[str, Any]:
    """加载安全配置(合并默认值与 safety.json)"""
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    cfg = dict(DEFAULT_CONFIG)
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k in DEFAULT_CONFIG:
                if k in data:
                    cfg[k] = data[k]
        except (json.JSONDecodeError, IOError):
            pass
    _config_cache = cfg
    return cfg


def reload_config() -> Dict[str, Any]:
    """强制重新加载配置(修改配置后调用)"""
    global _config_cache
    _config_cache = None
    return load_config()


def save_config(cfg: Dict[str, Any]) -> bool:
    """保存配置到 safety.json"""
    try:
        parent = os.path.dirname(CONFIG_PATH)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        reload_config()
        return True
    except Exception:
        return False


# ============ 编译正则 ============

def _compile(patterns: List[str]) -> List[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


def _blocklist() -> List[re.Pattern]:
    cfg = load_config()
    return _compile(BUILTIN_BLOCKLIST + list(cfg.get("blacklist", [])))


def _confirm_list() -> List[re.Pattern]:
    return _compile(BUILTIN_CONFIRM)


def _first_token(command: str) -> str:
    """取命令的首个 token(用于白名单匹配)"""
    tokens = command.strip().split()
    # 去掉前导的调用器(powershell -Command / bash -c 等)以取到真正的命令
    skip = {"powershell", "pwsh", "bash", "sh", "cmd", "python", "python3", "py"}
    for t in tokens:
        t_clean = t.strip('"\'-')
        if t_clean and t_clean.lower() not in skip:
            return t_clean
    return tokens[0] if tokens else ""


# ============ 决策函数 ============

def check_command(command: str) -> Tuple[str, str]:
    """
    判断命令是否允许执行

    Returns:
        (status, reason)
        status: "allow" | "deny" | "confirm"
    """
    cfg = load_config()

    # 1. 黑名单(始终拒绝)
    for pat in _blocklist():
        m = pat.search(command or "")
        if m:
            return "deny", f"匹配禁止规则: {pat.pattern}"

    # 2. 白名单模式:首命令必须在白名单
    if cfg.get("mode") == "whitelist":
        allowed = [t.lower() for t in cfg.get("whitelist", [])]
        if _first_token(command).lower() not in allowed:
            return "deny", f"白名单模式禁止命令: {_first_token(command)}"

    # 3. 需确认的危险模式
    if cfg.get("confirm_dangerous", True):
        for pat in _confirm_list():
            m = pat.search(command or "")
            if m:
                return "confirm", f"匹配危险模式: {pat.pattern}"

    return "allow", ""


def check_exec() -> Tuple[str, str]:
    """
    执行脚本文件(.py/.ps1/.bat)时的判定
    运行任意脚本本质危险,默认需确认(除非关闭 confirm_dangerous)
    """
    cfg = load_config()
    if cfg.get("confirm_dangerous", True):
        return "confirm", "执行脚本文件属于危险操作"
    return "allow", ""


def check_path(path: str) -> Tuple[bool, str]:
    """
    路径保护(用于删除/移动等操作):禁止操作系统关键目录与项目根

    Returns:
        (allowed, reason)
    """
    if not path:
        return False, "路径为空"
    abs_path = os.path.abspath(path)
    # 根目录 / 盘符根
    parent = os.path.dirname(abs_path)
    if abs_path == parent or abs_path.endswith(":\\") or abs_path in ("/", "\\"):
        return False, "禁止操作系统根目录"
    # 系统目录
    sys_root = os.environ.get("SystemRoot", "C:\\Windows")
    windir = os.environ.get("windir", "C:\\Windows")
    protected = [
        os.path.abspath(sys_root),
        os.path.abspath(windir),
        os.path.abspath(os.path.expanduser("~")),
    ]
    # 项目根
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    protected.append(os.path.abspath(project_root))
    # Windows 路径大小写不敏感，比较前统一规范化以防大小写绕过。
    comparable_path = os.path.normcase(abs_path)
    for p in protected:
        comparable_root = os.path.normcase(p)
        if comparable_path == comparable_root or comparable_path.startswith(comparable_root + os.sep):
            return False, f"禁止操作系统/项目关键目录: {p}"
    return True, ""


def set_confirm_backend(backend: Optional[Callable[[str], bool]]) -> None:
    """替换危险命令的交互确认后端。

    - None（默认）：使用终端 input()，CLI 场景。
    - interrupt_confirm：通过 LangGraph interrupt 把确认抛给前端（server 场景）。
    """
    global _confirm_backend
    _confirm_backend = backend


def interrupt_confirm(prompt: str) -> bool:
    """
    server 模式的确认后端：用 LangGraph interrupt 暂停图执行，把危险命令发给前端确认。

    resume payload 约定（与 ask_human 一致）：
        {"choice_id": "approve"} -> True
        {"choice_id": "deny"} / {"cancelled": True} / 其他 -> False
    """
    from langgraph.types import interrupt

    answer = interrupt(
        {
            "kind": "dangerous_command",
            "prompt": prompt,
            "choices": [
                {"id": "approve", "label": "确认执行"},
                {"id": "deny", "label": "拒绝执行"},
            ],
        }
    )
    if isinstance(answer, dict):
        if answer.get("cancelled"):
            return False
        return answer.get("choice_id") == "approve"
    return False


def confirm(prompt: str) -> bool:
    """
    交互式确认:默认读取终端输入(默认拒绝)
    空输入 / 'n' / EOF(非交互) / 异常 均视为拒绝

    若已通过 set_confirm_backend 注册后端(如 server 的 interrupt 后端),则委托给它。
    """
    if _confirm_backend is not None:
        return _confirm_backend(prompt)
    try:
        ans = input(prompt).strip().lower()
        return ans in ("y", "yes", "是")
    except (EOFError, KeyboardInterrupt, OSError):
        return False
