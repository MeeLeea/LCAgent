import json
import subprocess
from pathlib import Path
from typing import TypeAlias


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
EXPECTED_TEMPLATES = {
    "llm_config.json.example",
    "mcp_servers.json.example",
    "safety.json.example",
}
JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


def _load_json(path: Path) -> JsonValue:
    with path.open(encoding="utf-8") as template_file:
        loaded: JsonValue = json.load(template_file)
    return loaded


def _keys(value: JsonValue) -> list[str]:
    match value:
        case dict():
            nested_keys: list[str] = []
            for key, child in value.items():
                nested_keys.append(key)
                nested_keys.extend(_keys(child))
            return nested_keys
        case list():
            nested_keys = []
            for child in value:
                nested_keys.extend(_keys(child))
            return nested_keys
        case str() | int() | float() | bool() | None:
            return []


def test_expected_config_templates_parse_when_checked_out() -> None:
    # Given: a fresh checkout needs sanitized example config files.
    paths = [CONFIG_DIR / name for name in sorted(EXPECTED_TEMPLATES)]

    # When: each expected template is loaded as JSON.
    parsed = [_load_json(path) for path in paths]

    # Then: every expected file exists and parses into a JSON object.
    assert {path.name for path in paths} == EXPECTED_TEMPLATES
    assert all(isinstance(value, dict) for value in parsed)


def test_config_templates_omit_secret_and_plural_model_keys() -> None:
    # Given: example templates are safe to commit.
    templates = [CONFIG_DIR / name for name in sorted(EXPECTED_TEMPLATES)]

    # When: their nested keys are inspected.
    all_keys = [key for template in templates for key in _keys(_load_json(template))]

    # Then: sensitive api_key keys and plural models keys are absent.
    assert "models" not in all_keys
    assert all(key.casefold() != "api_key" for key in all_keys)


def test_config_templates_are_not_ignored_by_git() -> None:
    # Given: config/ is ignored for runtime secrets.
    template_paths = [f"config/{name}" for name in sorted(EXPECTED_TEMPLATES)]

    # When: git checks whether the example templates are ignored.
    result = subprocess.run(
        ["git", "check-ignore", *template_paths],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    # Then: narrow .gitignore exceptions keep the templates visible to git.
    assert result.returncode == 1, result.stdout
