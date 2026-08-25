import ast
import importlib.util

import pytest

import tools.create_tools as create_tool_module
from tools.create_tools import create_tool

VALID_ARGS = {
    "tool_name": "read_markdown_file",
    "tool_description": "读取Markdown文件内容",
    "args_spec": "file_path:str=本地文件路径;encoding:str=utf-8文件编码，可选",
    "tool_logic": "with open(file_path, 'r', encoding=encoding) as f:\n    result = f.read()",
}

INIT_STUB = """from skmng.tool import read_skill
from .scheduler_tool import schedule_task

all_tools = [
    read_skill,
    schedule_task,
]

__all__ = [
    'read_skill',
    'schedule_task',
    'all_tools',
]
"""


def _invoke_with_tools_dir(
    monkeypatch,
    tmp_path,
    args: dict[str, str | bool | None],
) -> dict[str, object]:
    monkeypatch.setattr(create_tool_module, "DEFAULT_TOOL_DIR", str(tmp_path))
    return create_tool.invoke(args)


def test_create_tool_default_path_saves_to_tools_dir(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(create_tool_module, "DEFAULT_TOOL_DIR", str(tmp_path))
    result = create_tool.invoke(VALID_ARGS)

    assert result["success"] is True
    expected = tmp_path / "read_markdown_file.py"
    assert result["file_path"] == str(expected)
    assert expected.exists()
    assert expected.read_text(encoding="utf-8") == result["source_code"]


@pytest.mark.parametrize(
    ("tool_path",),
    [
        ("sub",),
        ("custom/my_tool.py",),
    ],
)
def test_create_tool_writes_expected_file(monkeypatch, tmp_path, tool_path: str) -> None:
    target = tmp_path / tool_path
    if target.suffix:
        args = {**VALID_ARGS, "tool_path": str(target)}
        expected = target
    else:
        target.mkdir()
        args = {**VALID_ARGS, "tool_path": str(target)}
        expected = target / "read_markdown_file.py"

    result = _invoke_with_tools_dir(monkeypatch, tmp_path, args)

    assert result["success"] is True
    assert result["file_path"] == str(expected)
    assert expected.exists()


@pytest.mark.parametrize(
    ("tool_path",),
    [
        ("valid_tool.py",),
        ("self_contained.py",),
    ],
)
def test_generated_artifacts_are_valid(monkeypatch, tmp_path, tool_path: str) -> None:
    target = tmp_path / tool_path
    result = _invoke_with_tools_dir(monkeypatch, tmp_path, {**VALID_ARGS, "tool_path": str(target)})

    assert result["success"] is True
    ast.parse(target.read_text(encoding="utf-8"))
    assert "from langchain_core.tools import tool" in result["source_code"]
    assert "from typing import Dict, Any" in result["source_code"]


def test_generated_tool_is_importable_and_runs(monkeypatch, tmp_path) -> None:
    md_file = tmp_path / "sample.md"
    md_file.write_text("hello", encoding="utf-8")
    target = tmp_path / "importable_tool.py"
    result = _invoke_with_tools_dir(monkeypatch, tmp_path, {**VALID_ARGS, "tool_path": str(target)})
    assert result["success"] is True

    spec = importlib.util.spec_from_file_location("generated_importable_tool", target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    out = module.read_markdown_file.invoke({"file_path": str(md_file), "encoding": "utf-8"})
    assert out["success"] is True
    assert out["result"] == "hello"


def _load_generated(tmp_path: str, target: str, name: str):
    spec = importlib.util.spec_from_file_location(name, target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_multiline_fstring_content_is_preserved(monkeypatch, tmp_path) -> None:
    target = tmp_path / "greet.py"
    args = {
        "tool_name": "greet",
        "tool_description": "多行打招呼",
        "args_spec": "name:str=姓名",
        "tool_logic": 'result = f"""你好, {name}\n今天是周五"""',
        "tool_path": str(target),
    }

    result = _invoke_with_tools_dir(monkeypatch, tmp_path, args)

    assert result["success"] is True
    module = _load_generated(str(tmp_path), str(target), "generated_greet")
    out = module.greet.invoke({"name": "张三"})
    assert out["success"] is True
    assert out["result"] == "你好, 张三\n今天是周五"


def test_tool_logic_with_dict_literal_and_fstring(monkeypatch, tmp_path) -> None:
    target = tmp_path / "fmt_dict.py"
    args = {
        "tool_name": "fmt_dict",
        "tool_description": "格式化字典",
        "args_spec": "key:str=键;value:int=值",
        "tool_logic": 'data = {"k": 1}\nresult = f"值: {data}"',
        "tool_path": str(target),
    }

    result = _invoke_with_tools_dir(monkeypatch, tmp_path, args)

    assert result["success"] is True
    module = _load_generated(str(tmp_path), str(target), "generated_fmt_dict")
    out = module.fmt_dict.invoke({"key": "a", "value": 1})
    assert out["success"] is True
    assert out["result"] == "值: {'k': 1}"


def test_description_with_braces_does_not_break_template(monkeypatch, tmp_path) -> None:
    target = tmp_path / "brace_desc.py"
    args = {
        **VALID_ARGS,
        "tool_description": "解析形如 {key: value} 的字典",
        "tool_path": str(target),
    }

    result = _invoke_with_tools_dir(monkeypatch, tmp_path, args)

    assert result["success"] is True
    assert "形如 {key: value} 的字典" in result["source_code"]


def test_nested_logic_keeps_relative_indent(monkeypatch, tmp_path) -> None:
    target = tmp_path / "sum_list.py"
    args = {
        "tool_name": "sum_list",
        "tool_description": "求和列表",
        "args_spec": "items:list=数值列表",
        "tool_logic": "total = 0\nfor i in items:\n    total += i\nresult = total",
        "tool_path": str(target),
    }

    result = _invoke_with_tools_dir(monkeypatch, tmp_path, args)

    assert result["success"] is True
    module = _load_generated(str(tmp_path), str(target), "generated_sum_list")
    out = module.sum_list.invoke({"items": [1, 2, 3]})
    assert out["success"] is True
    assert out["result"] == 6


def test_create_tool_registers_in_tools_init(monkeypatch, tmp_path) -> None:
    init_path = tmp_path / "__init__.py"
    init_path.write_text(INIT_STUB, encoding="utf-8")
    monkeypatch.setattr(create_tool_module, "DEFAULT_TOOL_DIR", str(tmp_path))

    result = create_tool.invoke({**VALID_ARGS, "tool_path": str(tmp_path)})

    assert result["success"] is True
    assert result["registered"] is True
    content = init_path.read_text(encoding="utf-8")
    assert "from .read_markdown_file import read_markdown_file" in content
    assert "    read_markdown_file,\n" in content
    assert "    'read_markdown_file',\n" in content


def test_create_tool_register_is_idempotent(monkeypatch, tmp_path) -> None:
    init_path = tmp_path / "__init__.py"
    init_path.write_text(INIT_STUB, encoding="utf-8")
    monkeypatch.setattr(create_tool_module, "DEFAULT_TOOL_DIR", str(tmp_path))

    first = create_tool.invoke({**VALID_ARGS, "tool_path": str(tmp_path)})
    second = create_tool.invoke({**VALID_ARGS, "tool_path": str(tmp_path)})

    assert first["registered"] is True
    assert second["registered"] is False
    content = init_path.read_text(encoding="utf-8")
    assert content.count("from .read_markdown_file import read_markdown_file") == 1
    assert content.count("'read_markdown_file'") == 1


def test_create_tool_does_not_register_outside_tools_dir(monkeypatch, tmp_path) -> None:
    target_dir = tmp_path / "external"
    target_dir.mkdir()

    result = _invoke_with_tools_dir(monkeypatch, tmp_path, {**VALID_ARGS, "tool_path": str(target_dir)})

    assert result["success"] is True
    assert result["registered"] is False


def test_full_pipeline_registers_and_imports(monkeypatch, tmp_path) -> None:
    pkg = tmp_path / "genpkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(
        "from ._dummy import dummy\n\nall_tools = [\n    dummy,\n]\n\n__all__ = [\n    'dummy',\n    'all_tools',\n]\n",
        encoding="utf-8",
    )
    (pkg / "_dummy.py").write_text(
        "from langchain_core.tools import tool\n\n@tool\ndef dummy(x: str) -> str:\n    \"\"\"占位工具\"\"\"\n    return x\n",
        encoding="utf-8",
    )
    md_file = pkg / "sample.md"
    md_file.write_text("hello", encoding="utf-8")
    monkeypatch.setattr(create_tool_module, "DEFAULT_TOOL_DIR", str(pkg))
    monkeypatch.syspath_prepend(str(tmp_path))

    result = create_tool.invoke({**VALID_ARGS, "tool_path": str(pkg)})
    assert result["success"] is True
    assert result["registered"] is True

    import genpkg

    tool = next(t for t in genpkg.all_tools if t.name == "read_markdown_file")
    out = tool.invoke({"file_path": str(md_file), "encoding": "utf-8"})
    assert out["success"] is True
    assert out["result"] == "hello"


def test_create_tool_rejects_invalid_syntax(monkeypatch, tmp_path) -> None:
    target = tmp_path / "bad.py"
    args = {
        **VALID_ARGS,
        "tool_logic": "def broken(",
        "tool_path": str(target),
    }

    result = _invoke_with_tools_dir(monkeypatch, tmp_path, args)

    assert result["success"] is False
    assert result["file_path"] is None
    assert not target.exists()


