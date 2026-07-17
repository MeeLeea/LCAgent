"""
计算器工具 - 使用LangChain @tool装饰器
"""
import ast
import math
import operator
from collections.abc import Callable

from langchain_core.tools import tool
from typing import Dict, Any


MAX_EXPRESSION_LENGTH = 200
MAX_AST_NODES = 64
MAX_AST_DEPTH = 20
MAX_ABS_VALUE = 1_000_000_000
# 四层上限分别约束输入、语法树规模、递归深度和中间结果，避免计算型拒绝服务。
BinaryOperator = Callable[[int | float, int | float], int | float]
BINARY_OPERATORS: dict[type[ast.operator], BinaryOperator] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
UNARY_OPERATORS: dict[type[ast.unaryop], BinaryOperator] = {
    ast.UAdd: lambda value, unused: value,
    ast.USub: lambda value, unused: -value,
}


class InvalidExpressionError(ValueError):
    """Raised when an expression exceeds the calculator's safe subset."""


def _bounded(value: int | float) -> int | float:
    # 每一步运算都检查中间值，防止最终结果虽小但中途已产生超大数。
    if isinstance(value, bool) or not math.isfinite(value) or abs(value) > MAX_ABS_VALUE:
        raise InvalidExpressionError("数值超出允许范围")
    return value


def _evaluate(node: ast.AST, depth: int = 0) -> int | float:
    # 仅解释明确允许的算术节点，禁止调用、名称解析和指数运算。
    if depth > MAX_AST_DEPTH:
        raise InvalidExpressionError("表达式嵌套过深")
    if isinstance(node, ast.Constant):
        if type(node.value) not in (int, float):
            raise InvalidExpressionError("只允许数字")
        return _bounded(node.value)
    if isinstance(node, ast.BinOp):
        operation = BINARY_OPERATORS.get(type(node.op))
        if operation is None:
            raise InvalidExpressionError("不支持的运算符")
        return _bounded(operation(_evaluate(node.left, depth + 1), _evaluate(node.right, depth + 1)))
    if isinstance(node, ast.UnaryOp):
        operation = UNARY_OPERATORS.get(type(node.op))
        if operation is None:
            raise InvalidExpressionError("不支持的运算符")
        return _bounded(operation(_evaluate(node.operand, depth + 1), 0))
    raise InvalidExpressionError("表达式包含不允许的语法")


@tool
def calculate(expression: str) -> Dict[str, Any]:
    """
    数学计算工具。计算给定的数学表达式。
    支持加(+)、减(-)、乘(*)、除(/)、括号()。

    Args:
        expression: 数学表达式，如 "2 + 3 * 4" 或 "(10 + 5) / 3"

    Returns:
        包含计算结果的字典，成功时包含result字段
    """
    normalized = expression.replace(" ", "")
    try:
        # 在递归求值前限制文本、括号和 AST 规模，快速拒绝资源耗尽输入。
        if not normalized or len(normalized) > MAX_EXPRESSION_LENGTH:
            raise InvalidExpressionError("表达式为空或过长")
        if normalized.count("(") > MAX_AST_DEPTH:
            raise InvalidExpressionError("表达式嵌套过深")
        tree = ast.parse(normalized, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
            raise InvalidExpressionError("表达式过于复杂")
        result = _evaluate(tree.body)

        return {
            "success": True,
            "expression": normalized,
            "result": result,
            "message": f"计算结果: {normalized} = {result}"
        }
    except ZeroDivisionError:
        return {
            "success": False,
            "error": "除零错误",
            "expression": normalized,
            "result": None
        }
    except (InvalidExpressionError, SyntaxError, TypeError, ValueError) as error:
        return {
            "success": False,
            "error": f"计算错误: {error}",
            "expression": normalized,
            "result": None
        }
