import pytest

import tools.create_tools as create_tool_module
from tools.create_tools import create_tool

VALID_ARGS = {
    "tool_name": "read_markdown_file",
    "tool_description": "读取Markdown文件内容",
    "args_spec": "file_path:str=本地文件路径;encoding:str=utf-8文件编码，可选",
    "tool_logic": "with open(file_path, 'r', encoding=encoding) as f:\n    result = f.read()",
}


def _invoke_with_tools_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    args: dict[str, str | bool | None],
) -> dict[str, object]:
    monkeypatch.setattr(create_tool_module, "DEFAULT_TOOL_DIR", str(tmp_path))
    return create_tool.invoke(args)


@pytest.mark.parametrize(
    ("tool_name",),
    [
        ("my tool",),
        ("1tool",),
        ("_hidden",),
        ("",),
    ],
)
def test_create_tool_rejects_invalid_tool_names(monkeypatch, tmp_path, tool_name: str) -> None:
    target = tmp_path / "bad.py"
    args = {**VALID_ARGS, "tool_name": tool_name, "tool_path": str(target)}

    result = _invoke_with_tools_dir(monkeypatch, tmp_path, args)

    assert result["success"] is False
    assert "合法 Python 标识符" in result["error"]
    assert not target.exists()


@pytest.mark.parametrize(
    ("tool_path",),
    [
        ("../escape.py",),
    ],
)
def test_create_tool_rejects_path_escape(monkeypatch, tmp_path, tool_path: str) -> None:
    args = {**VALID_ARGS, "tool_path": str(tmp_path / tool_path)}

    result = _invoke_with_tools_dir(monkeypatch, tmp_path, args)

    assert result["success"] is False
    assert "路径逃逸被禁止" in result["error"]


def test_create_tool_rejects_absolute_path_escape(monkeypatch, tmp_path) -> None:
    args = {**VALID_ARGS, "tool_path": str(tmp_path.parent / "escape.py")}

    result = _invoke_with_tools_dir(monkeypatch, tmp_path, args)

    assert result["success"] is False
    assert "路径逃逸被禁止" in result["error"]


def test_create_tool_rejects_existing_file_by_default(monkeypatch, tmp_path) -> None:
    target = tmp_path / "occupied.py"
    target.write_text("existing", encoding="utf-8")

    result = _invoke_with_tools_dir(monkeypatch, tmp_path, {**VALID_ARGS, "tool_path": str(target)})

    assert result["success"] is False
    assert "目标文件已存在" in result["error"]
    assert target.read_text(encoding="utf-8") == "existing"


def test_create_tool_allows_overwrite_when_forced(monkeypatch, tmp_path) -> None:
    target = tmp_path / "occupied.py"
    target.write_text("existing", encoding="utf-8")

    result = _invoke_with_tools_dir(
        monkeypatch,
        tmp_path,
        {**VALID_ARGS, "tool_path": str(target), "force": True},
    )

    assert result["success"] is True
    assert target.read_text(encoding="utf-8") == result["source_code"]


@pytest.mark.parametrize(
    ("tool_logic", "expected_import"),
    [
        ("import os\nresult = 1", "os"),
        ("import subprocess\nresult = 1", "subprocess"),
        ("from pathlib import Path\nresult = 1", "pathlib"),
        ("import importlib\nresult = 1", "importlib"),
        ("import requests\nresult = 1", "requests"),
        ("import socket\nresult = 1", "socket"),
    ],
)
def test_create_tool_rejects_high_risk_imports(
    monkeypatch,
    tmp_path,
    tool_logic: str,
    expected_import: str,
) -> None:
    target = tmp_path / f"{expected_import}.py"
    args = {**VALID_ARGS, "tool_logic": tool_logic, "tool_path": str(target)}

    result = _invoke_with_tools_dir(monkeypatch, tmp_path, args)

    assert result["success"] is False
    assert expected_import in result["error"]
    assert not target.exists()
