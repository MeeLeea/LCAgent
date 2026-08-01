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
    r":\(\)\s*\{.*\}\s*;",                      # fork bomb :(){:|:&};:
    r"\bformat\b", r"\bmkfs\b",                  # 磁盘格式化
    r"\bdd\b.*\bif=",                            # 直接写入硬盘
    r"curl\b[^\n|]*\|\s*(sh|bash)", r"wget\b[^\n|]*\|\s*(sh|bash)",  # 管道执行远程脚本
    r">\s*/dev/(sd|hd|nvme|sda)",                # 直接写入设备文件
    r"\bshutdown\b", r"\breboot\b",              # 系统关机/重启
    r"\breg\s+(delete|add)\b", r"\bnetsh\b",     # 注册表和网络配置危险操作
    r"\b(?:powershell|pwsh)(?:\.exe)?\b[^\r\n]*\s-(?:e|enc|encodedcommand)\b",  # 编码命令执行
]

# 危险但非灾难性模式(命中且开启确认时 -> confirm)
BUILTIN_CONFIRM = [
    # 删除操作（包含递归删除）
    r"\brm\b", r"\brm\s+-f\b", r"\brm\s+-rf\b", r"\brm\s+-fr\b", r"\brm\b.*--recursive",
    r"\bremove-item\b", 
    r"\bremove-item\b(?=[^\r\n]*\s-(?:recurse|r)\b)(?=[^\r\n]*\s-(?:force|f)\b)",  # Remove-Item -Recurse -Force
    r"\b(?:del|erase)\b", r"\b(?:del|erase)\b[^\r\n]*\s/s\b",  # del/erase 包括递归
    r"\b(?:rd|rmdir)\b", r"\b(?:rd|rmdir)\b[^\r\n]*\s/s\b",     # rd/rmdir 包括递归
    r"\brd\s+/[sq]", r"\bdeltree\b",
    # 权限和进程管理
    r"\bsudo\b", r"\bchmod\b", r"\bchown\b",
    r"\bmv\b", r"\bkill\b", r"\btaskkill\b", r"\bschtasks\b",
    # 解释器可隐藏任意副作用，脚本、内联代码和命令包装器统一要求人工确认
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
    "path_protection": {
        "enabled": True,
        "protected_paths": [
            "{system_root}", "{user_home}",
            "{project_root}",
            "{project_root}/agent", "{project_root}/api", "{project_root}/cli",
            "{project_root}/config", "{project_root}/memory", "{project_root}/scheduler",
            "{project_root}/web", "{project_root}/remote", "{project_root}/.agents",
            "{project_root}/main.py", "{project_root}/llm_client.py",
            "{project_root}/requirements.txt", "{project_root}/pyproject.toml",
            "{project_root}/uv.lock", "{project_root}/skills-lock.json",
        ],
        "confirm_paths": [
            "{project_root}/tests", "{project_root}/tools", "{project_root}/docs",
            "{project_root}/README.md", "{project_root}/.venv",
            "{project_root}/__pycache__", "{project_root}/.pytest_cache",
            "{project_root}/.ruff_cache",
        ],
    },
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
    # 深拷贝嵌套字典
    if "path_protection" in cfg:
        cfg["path_protection"] = dict(cfg["path_protection"])
        cfg["path_protection"]["protected_paths"] = list(cfg["path_protection"]["protected_paths"])
        cfg["path_protection"]["confirm_paths"] = list(cfg["path_protection"]["confirm_paths"])
    
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 合并所有键（不只是 DEFAULT_CONFIG 中的键）
            for k, v in data.items():
                if k == "path_protection" and isinstance(v, dict):
                    # 深度合并 path_protection
                    if "path_protection" not in cfg:
                        cfg["path_protection"] = {}
                    cfg["path_protection"].update(v)
                else:
                    cfg[k] = v
        except (json.JSONDecodeError, IOError):
            pass
    _config_cache = cfg
    return cfg


def reload_config() -> Dict[str, Any]:
    """强制重新加载配置(修改配置后调用)"""
    global _config_cache, _protected_paths_cache, _confirm_paths_cache
    _config_cache = None
    _protected_paths_cache = None
    _confirm_paths_cache = None
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


# ============ 路径分类系统 ============

def _resolve_placeholder(path_template: str) -> List[str]:
    """
    解析路径模板中的占位符
    
    占位符:
        {project_root} - 项目根目录
        {system_root}  - 系统根目录 (Windows: C:\\Windows)
        {user_home}    - 用户主目录
        {drive_roots}  - 所有盘符根 (Windows: C:\\, D:\\, etc.)
    
    Returns:
        解析后的路径列表（{drive_roots} 会展开为多个路径）
    """
    # {project_root}
    if "{project_root}" in path_template:
        return [path_template.replace("{project_root}", PROJECT_ROOT)]
    
    # {system_root}
    if "{system_root}" in path_template:
        sys_root = os.environ.get("SystemRoot", os.environ.get("windir", "C:\\Windows"))
        return [path_template.replace("{system_root}", sys_root)]
    
    # {user_home}
    if "{user_home}" in path_template:
        return [path_template.replace("{user_home}", os.path.expanduser("~"))]
    
    # {drive_roots} - Windows 盘符根
    if "{drive_roots}" in path_template:
        import string
        drives = []
        for letter in string.ascii_uppercase:
            drive = f"{letter}:\\"
            if os.path.exists(drive):
                drives.append(path_template.replace("{drive_roots}", drive))
        return drives if drives else [path_template]
    
    # 无占位符，直接返回
    return [path_template]


def _normalize_path(path: str) -> str:
    """
    规范化路径：转为绝对路径 + Windows 大小写统一
    
    处理:
        - 相对路径 -> 绝对路径
        - Windows 大小写不敏感 (normcase)
        - 符号链接解析 (realpath)
    """
    try:
        # 转为绝对路径
        abs_path = os.path.abspath(path)
        # 解析符号链接
        real_path = os.path.realpath(abs_path)
        # Windows 大小写统一
        return os.path.normcase(real_path)
    except (OSError, ValueError):
        # 路径不存在或无效，按字面值处理
        return os.path.normcase(os.path.abspath(path))


def _path_matches(target: str, rule: str) -> bool:
    """
    判断目标路径是否匹配规则
    
    匹配规则:
        - 精确匹配: target == rule
        - 前缀匹配: target 是 rule 的子路径
    
    Args:
        target: 已规范化的目标路径
        rule: 已规范化的规则路径
    
    Returns:
        是否匹配
    """
    # 精确匹配
    if target == rule:
        return True
    
    # 前缀匹配（rule 是目录，target 在其下）
    # 确保 rule 以分隔符结尾，避免误匹配 (如 /home 匹配到 /home2)
    rule_with_sep = rule if rule.endswith(os.sep) else rule + os.sep
    return target.startswith(rule_with_sep)


def _get_protected_paths() -> List[str]:
    """获取并缓存保护级路径列表（已解析占位符和规范化）"""
    global _protected_paths_cache
    
    if _protected_paths_cache is not None:
        return _protected_paths_cache
    
    cfg = load_config()
    path_protection = cfg.get("path_protection", {})
    
    if not path_protection.get("enabled", True):
        _protected_paths_cache = []
        return []
    
    templates = path_protection.get("protected_paths", [])
    paths = []
    for template in templates:
        resolved = _resolve_placeholder(template)
        for p in resolved:
            normalized = _normalize_path(p)
            if normalized not in paths:
                paths.append(normalized)
    
    _protected_paths_cache = paths
    return paths


def _get_confirm_paths() -> List[str]:
    """获取并缓存询问级路径列表（已解析占位符和规范化）"""
    global _confirm_paths_cache
    
    if _confirm_paths_cache is not None:
        return _confirm_paths_cache
    
    cfg = load_config()
    path_protection = cfg.get("path_protection", {})
    
    if not path_protection.get("enabled", True):
        _confirm_paths_cache = []
        return []
    
    templates = path_protection.get("confirm_paths", [])
    paths = []
    for template in templates:
        resolved = _resolve_placeholder(template)
        for p in resolved:
            normalized = _normalize_path(p)
            if normalized not in paths:
                paths.append(normalized)
    
    _confirm_paths_cache = paths
    return paths


def _classify_path(path: str) -> str:
    """
    分类路径为保护级、询问级或普通级别（最长匹配原则）
    
    决策逻辑:
        1. 收集所有匹配的规则（保护级和询问级）
        2. 选择最长（最具体）的匹配规则
        3. 如果没有匹配，返回 normal
    
    Args:
        path: 待分类的路径
    
    Returns:
        "protected" - 保护级，禁止任何修改
        "confirm"   - 询问级，需要用户确认
        "normal"    - 普通路径
    """
    if not path:
        return "normal"
    
    normalized = _normalize_path(path)
    
    # 收集所有匹配的规则及其长度
    matches = []
    
    # 检查保护级
    for protected in _get_protected_paths():
        if _path_matches(normalized, protected):
            matches.append(("protected", len(protected)))
    
    # 检查询问级
    for confirm in _get_confirm_paths():
        if _path_matches(normalized, confirm):
            matches.append(("confirm", len(confirm)))
    
    # 返回最长匹配（最具体的规则）
    if matches:
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[0][0]
    
    return "normal"


# ============ 常量 ============

# 项目根目录(用于保护)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 文件操作命令集合（用于判断是否需要路径分类保护）
FILE_OPERATION_COMMANDS = {
    'rm', 'rmdir', 'del', 'erase', 'rd',
    'mv', 'move', 'cp', 'copy',
    'chmod', 'chown', 'remove-item'
}

# 模块级缓存：解析后的保护级和询问级路径
_protected_paths_cache: Optional[List[str]] = None
_confirm_paths_cache: Optional[List[str]] = None

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


def _is_file_operation(command: str) -> bool:
    """
    判断命令是否为文件操作命令
    
    文件操作命令需要进行路径分类保护
    """
    first = _first_token(command).lower()
    return first in FILE_OPERATION_COMMANDS


def _extract_paths_from_command(command: str) -> List[str]:
    """
    从命令中提取所有路径参数（通用版本）
    
    支持的命令:
        - 删除: rm, del, erase, rd, rmdir, Remove-Item
        - 移动: mv, move
        - 复制: cp, copy
        - 权限: chmod, chown
    
    Returns:
        路径列表（已转为绝对路径）
    """
    if not command:
        return []
    
    cmd_lower = command.lower()
    first = _first_token(command).lower()
    
    # 不是文件操作命令，返回空
    if first not in FILE_OPERATION_COMMANDS:
        return []
    
    # Unix-like 命令使用简单分割
    is_unix_like = first in {'rm', 'rmdir', 'mv', 'cp', 'chmod', 'chown'}
    
    try:
        if is_unix_like:
            tokens = command.split()
        else:
            # Windows 命令使用 shlex 处理引号，但要注意 Windows 路径中的反斜杠
            # 在 Windows 上，shlex 会把反斜杠当作转义字符，导致路径损坏
            # 所以对于 Windows 命令，也使用简单分割
            tokens = command.split()
    except ValueError:
        tokens = command.split()
    
    # 提取路径参数（跳过命令本身和选项）
    paths = []
    skip_next = False
    for i, token in enumerate(tokens):
        if i == 0:  # 跳过命令本身
            continue
        
        if skip_next:
            skip_next = False
            continue
        
        # 跳过选项
        if token.startswith('-') or token.startswith('/'):
            # 某些选项后面跟参数，需要跳过
            if token in {'-o', '-t', '--output', '--target'}:
                skip_next = True
            continue
        
        # 清理引号
        cleaned = token.strip('"\'-')
        if cleaned:
            paths.append(cleaned)
    
    return paths


# ============ 决策函数 ============

def check_command(command: str) -> Tuple[str, str]:
    """
    判断命令是否允许执行（基于两级路径分类保护）

    决策矩阵:
        命令类型          保护级路径        询问级路径        普通路径
        ─────────────────────────────────────────────────────
        BLOCKLIST         deny            deny             deny
        CONFIRM           deny            confirm          confirm
        普通命令          allow           allow            allow

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

    # 3. 检查是否命中危险命令模式
    confirm_dangerous = cfg.get("confirm_dangerous", True)
    is_confirm_cmd = False
    matched_pattern = None
    
    if confirm_dangerous:
        for pat in _confirm_list():
            m = pat.search(command or "")
            if m:
                is_confirm_cmd = True
                matched_pattern = pat.pattern
                break
    
    # 4. 如果是危险命令且是文件操作，检查路径分类
    if is_confirm_cmd and _is_file_operation(command):
        paths = _extract_paths_from_command(command)
        
        if paths:
            # 检查所有路径，取最严格的保护级别
            has_protected = False
            has_confirm = False
            
            for path in paths:
                classification = _classify_path(path)
                if classification == "protected":
                    has_protected = True
                    # 保护级路径 + CONFIRM 命令 -> deny
                    return "deny", f"禁止操作保护级路径: {path}"
                elif classification == "confirm":
                    has_confirm = True
            
            # 所有路径都是询问级或普通级 + CONFIRM 命令 -> confirm
            if has_confirm:
                return "confirm", f"操作询问级路径，需要确认"
            
            # 所有路径都是普通级 + CONFIRM 命令 -> confirm
            return "confirm", f"匹配危险模式: {matched_pattern}"
    
    # 5. 非文件操作的危险命令 -> confirm
    if is_confirm_cmd:
        return "confirm", f"匹配危险模式: {matched_pattern}"

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


def check_path(path: str, is_delete: bool = False) -> Tuple[bool, str]:
    """
    路径保护(用于删除/移动等操作)
    
    使用新的两级路径分类系统进行判断

    Args:
        path: 待检查路径
        is_delete: 是否为删除操作

    Returns:
        (allowed, reason)
    """
    if not path:
        return False, "路径为空"
    
    # 使用新的路径分类系统
    classification = _classify_path(path)
    
    # 保护级路径：禁止操作
    if classification == "protected":
        return False, f"禁止操作保护级路径: {path}"
    
    # 询问级路径：需要上层调用者处理确认逻辑
    # 这里返回 True，由调用者决定是否需要确认
    if classification == "confirm":
        return True, f"询问级路径，需要确认: {path}"
    
    # 普通路径：允许操作
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
