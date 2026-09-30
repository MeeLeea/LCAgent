"""
技能阅读管理器 - 扫描并解析本地 .agents/skills/ 下的 SKILL.md 文件

技能目录结构(与 open agent skills 规范一致):
    .agents/skills/
        <skill-name>/
            SKILL.md          # 含 YAML frontmatter (name / description) + 正文指引
            ... 其他资源文件

本模块提供:
- 列出所有技能(名称 + 描述)
- 读取指定技能的完整内容
- 根据任务描述自动匹配相关技能(确定性关键词打分,不调用 LLM)
  - 中文按 2-gram(bigram)分词,避免单字分词让无关中文文本因共用汉字而误命中
  - 中英停用词在打分前剔除,消除泛化词造成的噪声命中
  - 重叠系数打分 + 命中阈值(0.25)过滤噪声
  - 技能名显著 token 豁免:任务直接点名技能时无视阈值注入
- 将若干技能内容渲染为可注入 system prompt 的指引块
"""
import os
import re
from typing import ClassVar

# 中文停用词: 高频泛化 bigram(语义噪声的主要来源),在打分前剔除
_STOPWORDS_ZH: frozenset[str] = frozenset({
    "技能", "文件", "运行", "流程", "使用", "如何", "帮我", "一下", "可以", "需要",
    "项目", "目录", "功能", "实现", "方法", "工具", "查看", "报告", "内容", "支持",
    "提供", "包含", "包括", "以及", "相关", "进行", "通过", "这个", "那个", "什么",
    "怎么", "哪些", "一个", "我们", "就是", "还是", "因为", "所以", "但是", "如果",
    "然后", "现在", "已经", "应该", "可能", "主要", "重要", "基本", "具体", "一般",
    "通常", "生成", "创建", "的", "了", "是", "在", "和", "与", "或", "上", "下", "里",
})

# 英文停用词: 仅剔除功能词,保留 pptx/skill/file/presentation 等有信息量的词
_STOPWORDS_EN: frozenset[str] = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with", "is",
    "are", "this", "that", "it", "use", "using", "used", "any", "time", "when",
    "how", "what", "you", "your", "can", "will", "be", "by", "from", "as", "at",
    "do", "does", "not", "no", "all", "has", "have",
})

# 命中阈值: 重叠系数低于此值视为噪声命中(0.25 经真实技能目录标定:
# 正样本最低 0.25/0.286,负样本最高 0.2)
_MIN_MATCH_SCORE: float = 0.25


def default_skills_dir() -> str:
    """返回默认技能目录(<项目根>/.agents/skills)的绝对路径"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, ".agents", "skills")


class SkillManager:
    """本地技能管理器(只读 .agents/skills 目录)"""

    def __init__(self, skills_dir: str):
        """
        Args:
            skills_dir: 技能根目录(通常指向 <项目>/.agents/skills)
        """
        self.skills_dir = skills_dir

    # ============ 扫描与解析 ============

    def list_skills(self) -> list[dict[str, str]]:
        """
        列出目录下所有技能

        Returns:
            [{"name":..., "description":..., "path":...}, ...]
        """
        result = []
        if not os.path.isdir(self.skills_dir):
            return result

        for entry in sorted(os.listdir(self.skills_dir)):
            skill_path = os.path.join(self.skills_dir, entry)
            if not os.path.isdir(skill_path):
                continue
            skill_md = os.path.join(skill_path, "SKILL.md")
            if not os.path.isfile(skill_md):
                continue
            meta = self._parse_frontmatter(skill_md)
            result.append({
                "name": meta.get("name") or entry,
                "description": meta.get("description") or "",
                "path": skill_md,
            })
        return result

    def get_skill(self, name: str) -> str | None:
        """
        读取指定技能的完整 SKILL.md 内容

        Args:
            name: 技能名(目录名或 frontmatter 中的 name)

        Returns:
            文件全文;不存在返回 None
        """
        skill_md = self._resolve_skill_path(name)
        if not skill_md or not os.path.isfile(skill_md):
            return None
        try:
            with open(skill_md, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return None

    # ============ 自动匹配 ============

    def match_skills(self, task: str, top_k: int = 3) -> list[str]:
        """
        根据任务描述匹配相关技能(确定性打分,不调用 LLM)

        算法:
        - 任务文本先经中文关键词→英文扩展(如 提交→commit/git),再与技能
          name + description 一并按 2-gram/词 分词并剔除中英停用词
        - 用重叠系数 |A∩B| / min(|A|,|B|) 打分,分数 >= _MIN_MATCH_SCORE(0.25)
          才视为命中,过滤泛化词造成的噪声
        - 技能名豁免:任务直接点名技能名中的显著 token(如 vivado/pptx)时,
          即使分数低于阈值也命中,保证用户点名技能必被注入

        Args:
            task: 用户任务描述
            top_k: 最多返回的技能数

        Returns:
            命中的技能名列表(按分数降序)
        """
        if not task or not task.strip():
            return []

        # 中文关键词扩展为英文,提升中文任务的命中率
        expanded_task = self._expand_text(task)
        task_tokens = self._tokenize(expanded_task)
        if not task_tokens:
            return []

        scored = []
        for skill in self.list_skills():
            desc = f"{skill['name']} {skill['description']}"
            skill_tokens = self._tokenize(desc)
            if not skill_tokens:
                continue
            score = self._overlap_score(task_tokens, skill_tokens)
            # 命中条件: 分数达阈值,或任务直接点名了技能名中的显著 token
            name_hit = bool(self._name_tokens(skill["name"]) & task_tokens)
            if score >= _MIN_MATCH_SCORE or name_hit:
                scored.append((score, skill["name"]))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [name for _, name in scored[:top_k]]

    # ============ 渲染 ============

    def render_block(self, names: list[str]) -> str:
        """
        将若干技能内容渲染为可注入 system prompt 的指引块

        Args:
            names: 技能名列表

        Returns:
            拼接后的技能指引文本(空列表返回空字符串)
        """
        if not names:
            return ""

        blocks = []
        for name in names:
            content = self.get_skill(name)
            if not content:
                continue
            # 去掉 frontmatter,只保留正文指引
            body = self._strip_frontmatter(content)
            blocks.append(f"### 技能: {name}\n\n{body}")

        if not blocks:
            return ""

        return (
            "\n\n【已加载的技能指引(请在处理任务时遵循)】\n"
            + "\n\n---\n\n".join(blocks)
            + "\n"
        )

    # ============ 内部辅助 ============

    def _resolve_skill_path(self, name: str) -> str | None:
        """根据技能名(目录名或 frontmatter name)解析 SKILL.md 路径"""
        # 1. 直接匹配目录名
        direct = os.path.join(self.skills_dir, name, "SKILL.md")
        if os.path.isfile(direct):
            return direct

        # 2. 遍历匹配 frontmatter 中的 name
        for skill in self.list_skills():
            if skill["name"] == name:
                return skill["path"]
        return None

    @staticmethod
    def _parse_frontmatter(skill_md: str) -> dict[str, str]:
        """解析 SKILL.md 顶部的 YAML frontmatter,提取 name / description"""
        try:
            with open(skill_md, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception:
            return {}

        if not text.startswith("---"):
            return {}

        # 取第一个 --- 与下一个 --- 之间的内容
        m = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
        if not m:
            return {}

        meta = {}
        for line in m.group(1).splitlines():
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key in ("name", "description"):
                meta[key] = value
        return meta

    @staticmethod
    def _strip_frontmatter(content: str) -> str:
        """去掉 frontmatter,返回正文"""
        if content.startswith("---"):
            m = re.match(r"^---\s*\n.*?\n---\s*\n", content, re.DOTALL)
            if m:
                return content[m.end():].strip()
        return content.strip()

    @staticmethod
    def _tokenize(text: str) -> set:
        """分词: 英文/数字按词,中文按 2-gram(bigram),并剔除停用词

        中文按单字分词会让"科技/技能"这类无关词因共用汉字而误命中
        (见模块 docstring 的 bug 记录),故改用 2-gram:
        - 连续中文串切为相邻两字组合(如"提交代码" → 提交/交代/代码)
        - 长度为 1 的中文串保留单字(避免短词丢失)
        - 英文/数字仍按词切分
        - 中英停用词在返回前剔除,消除泛化词造成的噪声命中
        """
        text = text.lower()
        tokens = set(re.findall(r"[a-z0-9]+", text))
        for run in re.findall(r"[\u4e00-\u9fff]+", text):
            if len(run) == 1:
                tokens.add(run)
                continue
            for i in range(len(run) - 1):
                tokens.add(run[i : i + 2])
        return {
            t for t in tokens
            if t not in _STOPWORDS_ZH and t not in _STOPWORDS_EN
        }

    @staticmethod
    def _name_tokens(name: str) -> set:
        """提取技能名的"显著"token: 仅英文/数字词,长度≥3 且非纯数字

        技能名是最强的意图信号(用户直接点名 vivado/pptx/gitmcp 时应注入),
        故 match_skills 用它做阈值豁免。过滤掉 vivado-2025-2 里的 2025/2
        这类版本号噪声,避免"2分钟后提醒我"之类的任务误命中。
        """
        return {
            w for w in re.findall(r"[a-z0-9]+", name.lower())
            if len(w) >= 3 and not w.isdigit()
        }

    # 中文关键词 → 英文扩展词(仅用于匹配打分,不改变原任务)
    _ALIASES: ClassVar[dict[str, list[str]]] = {
        "提交": ["commit", "git"],
        "推送": ["push", "git"],
        "拉取": ["pull", "git"],
        "分支": ["branch", "git"],
        "技能": ["skill"],
        "查找": ["find", "search"],
        "搜索": ["find", "search"],
        "发现": ["find", "search"],
        "安装": ["install", "add"],
    }

    @classmethod
    def _expand_text(cls, text: str) -> str:
        """将任务中的中文关键词替换为对应英文词(便于与英文描述匹配)"""
        result = text
        for zh, en_words in cls._ALIASES.items():
            if zh in result:
                result = result.replace(zh, " " + " ".join(en_words) + " ")
        return result

    @staticmethod
    def _overlap_score(a: set, b: set) -> float:
        """重叠系数打分: |A∩B| / min(|A|,|B|)

        用 min 归一化而非并集(Jaccard):技能描述通常远长于任务文本,
        Jaccard 会被长描述稀释,导致真实命中得分过低。
        """
        if not a or not b:
            return 0.0
        inter = a & b
        if not inter:
            return 0.0
        return len(inter) / min(len(a), len(b))
