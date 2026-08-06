import pytest

from tools.calculator import calculate


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 3 * 4", 14),
        ("(10 + 5) / 3", 5.0),
        ("-2.5 * +4", -10.0),
    ],
)
def test_calculate_supports_bounded_arithmetic(expression: str, expected: float) -> None:
    result = calculate.invoke({"expression": expression})
    assert result["success"] is True
    assert result["result"] == expected


@pytest.mark.parametrize(
    "expression",
    [
        "",
        "__import__('os')",
        "2 ** 100",
        "9" * 65,
        "1 + " * 60 + "1",
        "(" * 40 + "1" + ")" * 40,
        "999999999 * 999999999",
        "1e309",
    ],
)
def test_calculate_rejects_unsafe_or_unbounded_expression(expression: str) -> None:
    result = calculate.invoke({"expression": expression})
    assert result["success"] is False
    assert result["result"] is None


def test_calculate_reports_division_by_zero() -> None:
    result = calculate.invoke({"expression": "1 / 0"})
    assert result == {
        "success": False,
        "error": "除零错误",
        "expression": "1/0",
        "result": None,
    }
