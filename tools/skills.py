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
- 将若干技能内容渲染为可注入 system prompt 的指引块
"""
import os
import re
from typing import Dict, Any, List, Optional


class SkillManager:
    """本地技能管理器(只读 .agents/skills 目录)"""

    def __init__(self, skills_dir: str):
        """
        Args:
            skills_dir: 技能根目录(通常指向 <项目>/.agents/skills)
        """
        self.skills_dir = skills_dir

    # ============ 扫描与解析 ============

    def list_skills(self) -> List[Dict[str, str]]:
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

    def get_skill(self, name: str) -> Optional[str]:
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

    def match_skills(self, task: str, top_k: int = 3) -> List[str]:
        """
        根据任务描述匹配相关技能(确定性打分,不调用 LLM)

        算法: 对任务文本与每个技能的 name + description 做关键词重叠度打分,
        取分数 > 0 的前 top_k 个技能(按分数降序)。
        任务中的中文关键词会先扩展为对应英文词(如 提交→commit/git),
        以解决技能描述多为英文导致的中文任务无法命中问题。

        Args:
            task: 用户任务描述
            top_k: 最多返回的技能数

        Returns:
            命中的技能名列表(降序)
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
            if score > 0:
                scored.append((score, skill["name"]))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [name for _, name in scored[:top_k]]

    # ============ 渲染 ============

    def render_block(self, names: List[str]) -> str:
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

    def _resolve_skill_path(self, name: str) -> Optional[str]:
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
    def _parse_frontmatter(skill_md: str) -> Dict[str, str]:
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
        """简单分词: 中文按字/词粗分,英文按单词;统一小写去标点"""
        text = text.lower()
        # 提取英文/数字词
        en = set(re.findall(r"[a-z0-9]+", text))
        # 提取中文字符(按单字,够用即可)
        zh = set(re.findall(r"[\u4e00-\u9fff]", text))
        return en | zh

    # 中文关键词 → 英文扩展词(仅用于匹配打分,不改变原任务)
    _ALIASES = {
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
        """重叠度打分: 基于 Jaccard 相似度 + 命中词数"""
        if not a or not b:
            return 0.0
        inter = a & b
        if not inter:
            return 0.0
        union = a | b
        jaccard = len(inter) / len(union)
        # 命中词数加权,避免长描述天然占优
        return jaccard * (1 + len(inter) * 0.1)
