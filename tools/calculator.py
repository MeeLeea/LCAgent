"""
计算器工具 - 使用LangChain @tool装饰器
"""
from langchain_core.tools import tool
from typing import Dict, Any
import re


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
    try:
        # 安全检查：只允许数字、运算符、括号和空格
        if not re.match(r'^[\d+\-*/().\s]+$', expression):
            return {
                "success": False,
                "error": "表达式包含不允许的字符",
                "expression": expression,
                "result": None
            }

        expression = expression.replace(" ", "")
        result = eval(expression)

        return {
            "success": True,
            "expression": expression,
            "result": result,
            "message": f"计算结果: {expression} = {result}"
        }
    except ZeroDivisionError:
        return {
            "success": False,
            "error": "除零错误",
            "expression": expression,
            "result": None
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"计算错误: {str(e)}",
            "expression": expression,
            "result": None
        }
