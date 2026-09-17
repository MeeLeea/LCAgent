"""creat_tool_tests：为动态生成的工具自动生成 pytest 单元测试（tests/tools/test_<tool>.py）。"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

import tools.create_tools as create_tool_module
from tools.create_tools import creat_tool_tests, create_tool

VALID_ARGS = {
    "tool_name": "read_markdown_file",
    "tool_description": "读取Markdown文件内容",
    "args_spec": "file_path:str=本地文件路径;encoding:str=utf-8文件编码，可选",
    "tool_logic": "with open(file_path, 'r', encoding=encoding) as f:\n    result = f.read()",
}

# creat_tool_tests 不需要 tool_logic
TEST_ARGS = {key: VALID_ARGS[key] for key in ("tool_name", "tool_description", "args_spec")}


def _patch_dirs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path]:
    """把工具目录与测试目录重定向到 tmp_path 下，返回 (tool_dir, test_dir)。"""
    tool_dir = tmp_path / "tools"
    test_dir = tmp_path / "tests" / "tools"
    tool_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(create_tool_module, "DEFAULT_TOOL_DIR", str(tool_dir))
    monkeypatch.setattr(create_tool_module, "DEFAULT_TEST_DIR", str(test_dir))
    return tool_dir, test_dir


def _run_pytest_on(test_dir: Path, run_dir: Path) -> subprocess.CompletedProcess[str]:
    """用真实 pytest 进程运行生成目录下的测试文件（隔离运行，不受外层会话影响）。"""
    return subprocess.run(
        [sys.executable, "-m", "pytest", str(test_dir), "-q"],
        cwd=run_dir,
        capture_output=True,
        text=True,
        check=False,
    )


def test_creat_tool_tests_saves_to_default_test_dir(monkeypatch, tmp_path) -> None:
    _, test_dir = _patch_dirs(monkeypatch, tmp_path)
    create_tool.invoke({**VALID_ARGS, "with_test": False})

    result = creat_tool_tests(**TEST_ARGS)

    assert result["success"] is True
    expected = test_dir / "test_read_markdown_file.py"
    assert result["test_file_path"] == str(expected)
    assert expected.read_text(encoding="utf-8") == result["test_source_code"]


def test_create_tool_generates_test_automatically(monkeypatch, tmp_path) -> None:
    _, test_dir = _patch_dirs(monkeypatch, tmp_path)

    result = create_tool.invoke(VALID_ARGS)

    assert result["success"] is True
    test_file = test_dir / "test_read_markdown_file.py"
    assert result["test_file_path"] == str(test_file)
    assert test_file.exists()
    assert str(test_file) in result["message"]


def test_create_tool_can_skip_test_generation(monkeypatch, tmp_path) -> None:
    _, test_dir = _patch_dirs(monkeypatch, tmp_path)

    result = create_tool.invoke({**VALID_ARGS, "with_test": False})

    assert result["success"] is True
    assert result["test_file_path"] is None
    assert not (test_dir / "test_read_markdown_file.py").exists()


def test_generated_test_source_is_valid_python(monkeypatch, tmp_path) -> None:
    _patch_dirs(monkeypatch, tmp_path)

    result = creat_tool_tests(**TEST_ARGS)

    assert result["success"] is True
    ast.parse(result["test_source_code"])
    assert "read_markdown_file" in result["test_source_code"]
    assert "EXPECTED_PARAMS = ['file_path', 'encoding']" in result["test_source_code"]
    assert "'file_path': \"test\"" in result["test_source_code"]


def test_generated_sample_args_follow_declared_types(monkeypatch, tmp_path) -> None:
    _patch_dirs(monkeypatch, tmp_path)

    result = creat_tool_tests(
        tool_name="sum_items",
        tool_description="数值求和",
        args_spec="items:list[int]=数值列表;flag:bool=开关;ratio:float=比例",
    )

    source = result["test_source_code"]
    assert result["success"] is True
    assert "'items': [1]" in source
    assert "'flag': True" in source
    assert "'ratio': 1.0" in source


def test_generated_tests_are_green_under_pytest(monkeypatch, tmp_path) -> None:
    _, test_dir = _patch_dirs(monkeypatch, tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert create_tool.invoke(VALID_ARGS)["success"] is True
    logic_result = create_tool.invoke(
        {
            "tool_name": "sum_items",
            "tool_description": "数值求和",
            "args_spec": "items:list[int]=数值列表",
            "tool_logic": "result = sum(items)",
        }
    )
    assert logic_result["success"] is True

    completed = _run_pytest_on(test_dir, run_dir)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "8 passed" in completed.stdout


def test_creat_tool_tests_rejects_existing_file_by_default(monkeypatch, tmp_path) -> None:
    _, test_dir = _patch_dirs(monkeypatch, tmp_path)
    test_dir.mkdir(parents=True, exist_ok=True)
    target = test_dir / "test_read_markdown_file.py"
    target.write_text("existing", encoding="utf-8")

    result = creat_tool_tests(**TEST_ARGS)

    assert result["success"] is False
    assert "目标测试文件已存在" in result["error"]
    assert target.read_text(encoding="utf-8") == "existing"


def test_creat_tool_tests_allows_overwrite_when_forced(monkeypatch, tmp_path) -> None:
    _, test_dir = _patch_dirs(monkeypatch, tmp_path)
    test_dir.mkdir(parents=True, exist_ok=True)
    target = test_dir / "test_read_markdown_file.py"
    target.write_text("existing", encoding="utf-8")

    result = creat_tool_tests(**TEST_ARGS, force=True)

    assert result["success"] is True
    assert target.read_text(encoding="utf-8") == result["test_source_code"]


def test_creat_tool_tests_rejects_path_escape(monkeypatch, tmp_path) -> None:
    _patch_dirs(monkeypatch, tmp_path)
    escape = tmp_path / "escape" / "test_read_markdown_file.py"

    result = creat_tool_tests(**TEST_ARGS, test_path=str(escape))

    assert result["success"] is False
    assert "测试路径逃逸被禁止" in result["error"]
    assert not escape.exists()


def test_creat_tool_tests_rejects_invalid_tool_name(monkeypatch, tmp_path) -> None:
    _patch_dirs(monkeypatch, tmp_path)

    result = creat_tool_tests(**{**TEST_ARGS, "tool_name": "_hidden"})

    assert result["success"] is False
    assert "合法 Python 标识符" in result["error"]


@pytest.mark.parametrize(("custom", "expected_name"), [("dir", "test_read_markdown_file.py"), ("file", "custom_name.py")])
def test_creat_tool_tests_supports_custom_test_path(
    monkeypatch, tmp_path, custom: str, expected_name: str
) -> None:
    _, test_dir = _patch_dirs(monkeypatch, tmp_path)
    custom_dir = test_dir / "generated"
    custom_dir.mkdir(parents=True)
    test_path = str(custom_dir) if custom == "dir" else str(custom_dir / expected_name)

    result = creat_tool_tests(**TEST_ARGS, test_path=test_path)

    assert result["success"] is True
    assert result["test_file_path"] == str(custom_dir / expected_name)
    assert (custom_dir / expected_name).exists()
