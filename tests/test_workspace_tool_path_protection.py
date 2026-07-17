import os

import pytest

from tools import workspace_tool


def test_path_protection_is_case_insensitive_on_windows():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(workspace_tool.__file__)))

    blocked = workspace_tool._is_protected_path(project_root.swapcase())

    assert blocked is not None


@pytest.fixture(name="protected_root")
def fixture_protected_root(monkeypatch: pytest.MonkeyPatch, tmp_path):
    protected = tmp_path / "protected"
    protected.mkdir()
    monkeypatch.setattr(workspace_tool, "_is_protected_path", lambda path: "blocked" if os.path.abspath(path).startswith(str(protected)) else None)
    return protected


@pytest.mark.parametrize("operation", [workspace_tool.move_workspace, workspace_tool.copy_workspace])
def test_workspace_operation_rejects_protected_source_before_mutation(operation, protected_root, tmp_path):
    source = protected_root / "source"
    source.mkdir()
    (source / "data.txt").write_text("keep", encoding="utf-8")
    destination = tmp_path / "destination"

    result = operation(str(source), str(destination))

    assert result == {"success": False, "error": "blocked"}
    assert source.exists()
    assert not destination.exists()


@pytest.mark.parametrize("operation", [workspace_tool.move_workspace, workspace_tool.copy_workspace])
def test_workspace_operation_rejects_protected_destination_before_mutation(operation, protected_root, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "data.txt").write_text("keep", encoding="utf-8")
    destination = protected_root / "new-destination"

    result = operation(str(source), str(destination))

    assert result == {"success": False, "error": "blocked"}
    assert source.exists()
    assert not destination.exists()


def test_move_workspace_allows_safe_tmp_path_operation(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace_tool, "_is_protected_path", lambda path: None)
    source = tmp_path / "source"
    source.mkdir()
    (source / "data.txt").write_text("move", encoding="utf-8")
    destination = tmp_path / "destination"

    result = workspace_tool.move_workspace(str(source), str(destination))


    assert result["success"] is True
    assert not source.exists()
    assert (destination / "data.txt").read_text(encoding="utf-8") == "move"


def test_copy_workspace_allows_safe_tmp_path_operation(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace_tool, "_is_protected_path", lambda path: None)
    source = tmp_path / "source"
    source.mkdir()
    (source / "data.txt").write_text("copy", encoding="utf-8")
    destination = tmp_path / "destination"

    result = workspace_tool.copy_workspace(str(source), str(destination))

    assert result["success"] is True
    assert source.exists()
    assert (destination / "data.txt").read_text(encoding="utf-8") == "copy"
