"""Executor worker 受限工具集（read_file / write_file / list_dir）的单元测试。

覆盖范围：
    - build_executor_tools：恰好 3 个工具、名称与非空描述
    - write_file：工作区内写盘（含子目录自动创建）、逃逸拒绝（相对/绝对）
    - read_file：正常读取、超大文件拒绝并提示分段、分段读取
    - list_dir：条目前缀、limit 截断、空目录、逃逸拒绝

全部同步驱动（StructuredTool.invoke），只用 ``tmp_path`` 工作区，不触网。
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from subagent.executor.tools import build_executor_tools

TOOLS = ("read_file", "write_file", "list_dir")


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """独立的临时工作区目录。"""
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws


@pytest.fixture
def tools(workspace: Path) -> list:
    """按小上限构建的工具集（100 字节 / 2 条目）。"""
    config = SimpleNamespace(max_read_bytes=100, max_list_entries=2)
    return build_executor_tools(str(workspace), config)


def _by_name(tools: list, name: str):
    """按名称取工具。"""
    matched = [t for t in tools if t.name == name]
    assert matched, f"缺少工具: {name}"
    return matched[0]


def test_exactly_three_tools_with_names(tools: list) -> None:
    """恰好 3 个受限工具，名称固定且描述非空。"""
    assert [t.name for t in tools] == list(TOOLS)
    for t in tools:
        assert t.description.strip()


def test_write_file_inside_workspace(tools: list, workspace: Path) -> None:
    """工作区内写入成功：返回摘要、内容一致、父目录自动创建。"""
    result = str(_by_name(tools, "write_file").invoke({
        "path": "sub/dir/hello.txt",
        "content": "你好",
    }))

    assert not result.startswith("[错误]")
    assert (workspace / "sub" / "dir" / "hello.txt").read_text(encoding="utf-8") == "你好"


def test_write_file_relative_escape_rejected(tools: list, workspace: Path) -> None:
    """``../`` 相对逃逸被拒绝，目标未创建。"""
    outside = workspace.parent / "outside.txt"
    assert not outside.exists()

    result = str(_by_name(tools, "write_file").invoke({
        "path": "../outside.txt",
        "content": "x",
    }))

    assert result.startswith("[错误]")
    assert not outside.exists()


def test_write_file_absolute_escape_rejected(tools: list, workspace: Path) -> None:
    """绝对路径逃逸（指向工作区外）被拒绝。"""
    outside = workspace.parent / "abs.txt"

    result = str(_by_name(tools, "write_file").invoke({
        "path": str(outside),
        "content": "x",
    }))

    assert result.startswith("[错误]")
    assert not outside.exists()


def test_read_file_inside(tools: list, workspace: Path) -> None:
    """正常读取工作区内文件。"""
    (workspace / "a.txt").write_text("内容甲乙丙", encoding="utf-8")

    result = str(_by_name(tools, "read_file").invoke({"path": "a.txt"}))

    assert result == "内容甲乙丙"


def test_read_file_too_large_requires_window(tools: list, workspace: Path) -> None:
    """超过 max_read_bytes 且未指定分段时报错并提示分段读取。"""
    (workspace / "big.txt").write_text("y" * 300, encoding="utf-8")

    result = str(_by_name(tools, "read_file").invoke({"path": "big.txt"}))

    assert result.startswith("[错误]")
    assert "分段" in result


def test_read_file_windowed_read(tools: list, workspace: Path) -> None:
    """指定 start/length 后可分段读取超大文件。"""
    text = "".join(str(i % 10) for i in range(300))
    (workspace / "big.txt").write_text(text, encoding="utf-8")

    result = str(_by_name(tools, "read_file").invoke({
        "path": "big.txt", "start": 0, "length": 50,
    }))

    assert result == text[:50]


def test_read_file_missing_returns_error(tools: list) -> None:
    """读取不存在的文件返回 [错误] 提示而非抛异常。"""
    result = str(_by_name(tools, "read_file").invoke({"path": "nope.txt"}))

    assert result.startswith("[错误]")


def test_list_dir_entries_with_prefix(tools: list, workspace: Path) -> None:
    """list_dir 输出 [文件]/[目录] 前缀。"""
    (workspace / "file_a.txt").write_text("", encoding="utf-8")
    (workspace / "sub").mkdir()

    result = str(_by_name(tools, "list_dir").invoke({"path": "."}))

    assert "[文件] file_a.txt" in result
    assert "[目录] sub" in result


def test_list_dir_truncates_to_limit(tools: list, workspace: Path) -> None:
    """条目数超过 max_list_entries 时截断并提示剩余数。"""
    for i in range(4):
        (workspace / f"f{i}.txt").write_text("", encoding="utf-8")

    result = str(_by_name(tools, "list_dir").invoke({"path": "."}))

    lines = result.splitlines()
    assert len(lines) == 2 + 1  # 2 个条目 + 1 行截断提示
    assert "剩余 2 条未显示" in result


def test_list_dir_empty_directory(tools: list, workspace: Path) -> None:
    """空目录返回占位提示。"""
    result = str(_by_name(tools, "list_dir").invoke({"path": "."}))

    assert result == "（空目录）"


def test_list_dir_escape_rejected(tools: list, workspace: Path) -> None:
    """list_dir 的相对逃逸被拒绝。"""
    result = str(_by_name(tools, "list_dir").invoke({"path": "../"}))

    assert result.startswith("[错误]")
