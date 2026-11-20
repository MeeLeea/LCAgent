"""``subagent.common`` 提示词解析与占位符渲染的单元测试。

覆盖范围：
    - parse_workflow_section：成功切分（系统提示词在上 / 模板在下、止于下一个
      ``## `` 二级标题、``###`` 不截断）；文件缺失/为空、缺 workflow 小节、
      系统提示词为空、模板为空四类 ValueError；
    - render_placeholders：按插入序链式替换的语义锁（后位替换作用于前位注入值）、
      ``{task}`` 与 ``{task_id}`` 因闭合花括号互不误替换、JSON 花括号原样保留。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from subagent.common.prompt import parse_workflow_section, render_placeholders

__all__ = ["test_parse_workflow_section_splits_correctly"]


def _write(tmp_path: Path, content: str) -> str:
    """写入临时提示词文件并返回路径字符串。"""
    path = tmp_path / "AGENT.md"
    path.write_text(content, encoding="utf-8")
    return str(path)


# ====================================================== parse_workflow_section ===


def test_parse_workflow_section_splits_correctly(tmp_path: Path) -> None:
    """系统提示词在上、模板在下；模板止于下一个二级标题，``###`` 不截断。"""
    prompt_file = _write(
        tmp_path,
        "# 角色说明\n\n你是规划者。\n"
        "## workflow:design\n"
        "任务：{requirement}\n"
        "### 子小节保持\n"
        "细节行\n"
        "## 下一个大节会被排除\n内容\n",
    )

    system_prompt, template = parse_workflow_section(prompt_file, missing_file_label="L")

    assert system_prompt == "# 角色说明\n\n你是规划者。"
    assert template == "任务：{requirement}\n### 子小节保持\n细节行"


def test_parse_workflow_section_missing_or_empty_file(tmp_path: Path) -> None:
    """文件缺失/为空：错误消息恰为 ``f"{missing_file_label}: {path}"``。"""
    missing = str(tmp_path / "missing.md")

    with pytest.raises(ValueError) as exc_info:
        parse_workflow_section(missing, missing_file_label="规划者提示词文件缺失或为空")
    assert str(exc_info.value) == "规划者提示词文件缺失或为空: " + missing

    empty = _write(tmp_path, "   \n")
    with pytest.raises(ValueError) as exc_info:
        parse_workflow_section(empty, missing_file_label="执行者提示词文件缺失或为空")
    assert str(exc_info.value) == "执行者提示词文件缺失或为空: " + empty


def test_parse_workflow_section_missing_workflow(tmp_path: Path) -> None:
    """没有 workflow 小节时报缺小节错误（含文件路径）。"""
    prompt_file = _write(tmp_path, "# 标题\n\n## 职责\n内容\n")

    with pytest.raises(ValueError, match="缺少 ## workflow: 小节"):
        parse_workflow_section(prompt_file, missing_file_label="L")


def test_parse_workflow_section_empty_system_prompt(tmp_path: Path) -> None:
    """文件以 workflow 行开头（系统提示词层为空）时报错。"""
    prompt_file = _write(tmp_path, "## workflow:design\n模板内容\n")

    with pytest.raises(ValueError, match="之前缺少系统提示词"):
        parse_workflow_section(prompt_file, missing_file_label="L")


def test_parse_workflow_section_empty_template(tmp_path: Path) -> None:
    """workflow 小节后无模板正文（仅紧跟下一个二级标题）时报错。"""
    prompt_file = _write(tmp_path, "# 系统\n\n## workflow:design\n\n## 下一节\n")

    with pytest.raises(ValueError, match="小节缺少模板内容"):
        parse_workflow_section(prompt_file, missing_file_label="L")


# ======================================================== render_placeholders ===


def test_render_placeholders_replaces_in_insertion_order() -> None:
    """严格按 mapping 插入序做链式替换：后位替换会作用于前位注入的值。"""
    rendered = render_placeholders(
        "{task} 与尾注", {"{task}": "值含 {b}", "{b}": "X"}
    )
    # {task} 的替换值里的 {b} 会被后续替换项再次处理（链式替换语义）
    assert rendered == "值含 X 与尾注"

    reversed_result = render_placeholders(
        "{task} 与尾注", {"{b}": "X", "{task}": "值含 {b}"}
    )
    # 反转插入序后，{b} 在替换值注入前已处理完，替换值原样保留
    assert reversed_result == "值含 {b} 与尾注"


def test_render_placeholders_no_prefix_pollution() -> None:
    """``{task}`` 与 ``{task_id}`` 因闭合花括号互不误替换（前缀不污染）。"""
    rendered = render_placeholders(
        "{task}/{task_id}", {"{task}": "A", "{task_id}": "B"}
    )
    assert rendered == "A/B"

    # 仅替换 {task} 时，{task_id} 原样保留（不应产生 "A_id" 类污染）
    single = render_placeholders("{task}/{task_id}", {"{task}": "A"})
    assert single == "A/{task_id}"


def test_render_placeholders_json_braces_safe() -> None:
    """占位符外的 JSON 花括号原样保留（刻意不用 ``str.format`` 的原因）。"""
    rendered = render_placeholders(
        '{"标题": "{plan}", "其他": {"a": 1}}', {"{plan}": "方案句子"}
    )
    assert rendered == '{"标题": "方案句子", "其他": {"a": 1}}'
