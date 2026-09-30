"""skmng.manager 自动匹配算法单元测试 - 2-gram 分词 / 停用词 / 重叠系数阈值 / 技能名豁免

回归锁定已修复的 bug: 中文按单字分词导致语义无关的中文任务因共用汉字
(技/行/流/动/文) 误命中 vivado-2025-2 技能(实测分数 0.0743),把 FPGA 指引
注入到模型价格调研任务中。

全部用例 hermetic: 技能在 tmp_path 中构造,不依赖真实 .agents/skills/。
"""
import pytest

from skmng.manager import SkillManager

# 复刻真实 vivado-2025-2 技能的长中文描述(原误命中的来源)
_VIVADO_DESC = (
    "Vivado 2025.2 FPGA 工程自动化构建技能，覆盖综合、布局布线、比特流生成全流程。"
    "当用户需要创建 Vivado 工程、运行 FPGA 编译流程、生成 bit 文件或查看综合/实现报告时调用。"
)

# 报告的真实误命中任务(模型中转站价格表查询)
_BUG_QUERY = (
    "帮我收集一下模型中转站的模型价格表，价格单位为（元/百万token），"
    "列表的行为模型（），列为不同厂家（火山方舟、并行科技、云雾、硅基流动、"
    "中国移动 MoMA）的价格，包括缓存输入输出价格，文本输入输出、图片输入输出、上下文长度"
)


def _make_manager(
    tmp_path, name: str = "vivado-2025-2", description: str = _VIVADO_DESC
) -> SkillManager:
    """在 tmp_path 下构造含单个技能的 SkillManager"""
    skills_root = tmp_path / "skills"
    skill_dir = skills_root / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n# 技能正文\n指引内容",
        encoding="utf-8",
    )
    return SkillManager(str(skills_root))


def test_reported_bug_query_matches_nothing(tmp_path):
    """主回归锁: 报告的真实任务不得再误命中 vivado 技能"""
    sm = _make_manager(tmp_path)
    assert sm.match_skills(_BUG_QUERY) == []


def test_generic_chinese_word_does_not_match(tmp_path):
    """仅由泛化词构成的查询不命中长中文描述技能"""
    sm = _make_manager(tmp_path)
    assert sm.match_skills("这个项目的目录结构是怎样的") == []
    assert sm.match_skills("查看一下文件内容") == []


def test_positive_chinese_match_still_works(tmp_path):
    """真实相关的中文任务仍能命中(阈值不应误杀正样本)"""
    sm = _make_manager(tmp_path)
    assert sm.match_skills("运行 FPGA 综合流程") == ["vivado-2025-2"]
    assert sm.match_skills("跑一下布局布线") == ["vivado-2025-2"]


def test_skill_name_token_overrides_threshold(tmp_path):
    """技能名 token 豁免: 分数低于阈值但任务点名了技能名中的显著 token"""
    sm = _make_manager(tmp_path)
    assert sm.match_skills("vivado 怎么设置器件型号") == ["vivado-2025-2"]


def test_version_number_in_name_is_not_a_name_token():
    """技能名里的版本号(2025/2)不作为显著 token,避免版本号噪声误命中"""
    assert SkillManager._name_tokens("vivado-2025-2") == {"vivado"}


def test_bigram_tokenization_splits_adjacent_pairs():
    """中文按相邻两字组合切分,不再产生单字 token"""
    tokens = SkillManager._tokenize("提交代码")
    assert {"提交", "交代", "代码"} <= tokens
    assert "提" not in tokens
    assert "码" not in tokens


def test_stopwords_are_filtered():
    """中英停用词在分词结果中被剔除

    注意: 2-gram 会跨越词边界("生成文件" 产生非停用 bigram "成文"),故纯停用词
    串需以分隔符断开才能得到空集;同一连续串内的停用词本身仍不会进入结果。
    """
    assert SkillManager._tokenize("生成 文件") == set()
    assert SkillManager._tokenize("技能") == set()
    tokens = SkillManager._tokenize("生成文件")
    assert "生成" not in tokens
    assert "文件" not in tokens


def test_name_hit_does_not_require_chinese_overlap(tmp_path):
    """技能名豁免不依赖中文重叠: 英文描述技能也能被点名任务命中"""
    sm = _make_manager(
        tmp_path, name="pptx", description="Create and edit presentation slides"
    )
    assert sm.match_skills("帮我做一个pptx演示文稿") == ["pptx"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
