"""提示词文件解析与模板占位符渲染（planner / executor 共享）。

两个后台子程序的 ``worker`` 都用 ``AGENT.md`` 驱动运行时：文件被唯一一个
``## workflow:`` 小节切成两半——其上为系统提示词，其下为工作流模板正文；模板
中的 ``{xxx}`` 占位符再用纯 ``str.replace`` 链式渲染（刻意不用 ``str.format``，
避免模板里的 JSON 花括号被误解析）。两处实现此前在 ``planner/worker.py`` 与
``executor/worker.py`` 中逐字重复，唯一差异是「文件缺失/为空」文案里的通称
（规划者 / 执行者），故把该文案交由调用方传入；本模块只依赖标准库。
"""
from __future__ import annotations

__all__ = ["parse_workflow_section", "render_placeholders"]


def parse_workflow_section(prompt_file: str, *, missing_file_label: str) -> tuple[str, str]:
    """解析提示词文件，返回 ``(系统提示词, 工作流小节模板)``。

    契约（与 planner/executor 原实现一致）：文件被唯一一个 ``## workflow:`` 小节
    切成两半——其上为系统提示词，其下为该小节模板正文（从该行下一行起直到下一个
    ``## `` 二级标题（不含）；模板内 ``###`` 不截断）。提示词与模板任一为空即抛
    ``ValueError``。

    Args:
        prompt_file: 提示词文件路径（如 ``AGENT.md``）。
        missing_file_label: 文件缺失/为空时的整句文案（如「规划者提示词文件缺失
            或为空」）；子程序文案在测试中以子串断言锁死，故由调用方传入自身通称，
            错误消息为 ``f"{missing_file_label}: {prompt_file}"``。

    Raises:
        ValueError: 文件缺失/为空、缺 ``## workflow:`` 小节、
            或系统提示词 / 模板任一侧为空。
    """
    try:
        with open(prompt_file, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        raise ValueError(f"{missing_file_label}: {prompt_file}") from None
    if not text.strip():
        raise ValueError(f"{missing_file_label}: {prompt_file}")

    lines = text.splitlines()
    workflow_index: int | None = None
    for index, line in enumerate(lines):
        if line.strip().startswith("## workflow:"):
            workflow_index = index
            break
    if workflow_index is None:
        raise ValueError(f"提示词文件缺少 ## workflow: 小节: {prompt_file}")

    system_prompt = "\n".join(lines[:workflow_index]).strip()
    template_lines: list[str] = []
    for line in lines[workflow_index + 1:]:
        if line.strip().startswith("## "):
            break
        template_lines.append(line)
    template = "\n".join(template_lines).strip()

    if not system_prompt:
        raise ValueError(f"提示词文件在 ## workflow: 之前缺少系统提示词: {prompt_file}")
    if not template:
        raise ValueError(f"提示词文件的 ## workflow: 小节缺少模板内容: {prompt_file}")
    return system_prompt, template


def render_placeholders(template: str, mapping: dict[str, str]) -> str:
    """按 ``mapping`` 的插入顺序对模板做链式 ``str.replace`` 占位符替换。

    刻意不用 ``str.format``：模板内可能出现 JSON 花括号等会被 ``format`` 误解析
    的内容；调用方需保证占位符之间无前缀包含关系（如 ``{task}`` 与 ``{task_id}``
    不可同用，或把长占位符排在前面）。

    Args:
        template: 含 ``{xxx}`` 占位符的模板正文。
        mapping: ``占位符（含花括号） → 替换值`` 的有序映射。

    Returns:
        渲染后的文本。
    """
    rendered = template
    for placeholder, value in mapping.items():
        rendered = rendered.replace(placeholder, value)
    return rendered
