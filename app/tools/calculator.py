"""calculator tool: safe arithmetic without eval().

The expression is parsed with `ast` and only number literals, + - * / // % **,
unary +/- and parentheses are evaluated. Everything else, and anything that
would be too large to compute, returns a message starting with "Error".
"""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable

from langchain_core.tools import tool

MAX_EXPRESSION_LENGTH = 200
MAX_EXPONENT = 1000
MAX_RESULT_DIGITS = 1000

Number = int | float

_BINARY: dict[type[ast.operator], Callable[[Number, Number], Number]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY: dict[type[ast.unaryop], Callable[[Number], Number]] = {ast.UAdd: operator.pos, ast.USub: operator.neg}


class CalculatorError(ValueError):
    pass


def _check_size(value: Number) -> Number:
    if not isinstance(value, int | float):  # e.g. (-8) ** 0.5 is complex in Python
        raise CalculatorError("result is not a real number")
    if isinstance(value, float) and not math.isfinite(value):
        raise CalculatorError("result is not a finite number")
    if isinstance(value, int) and value != 0 and math.log10(abs(value)) >= MAX_RESULT_DIGITS:
        raise CalculatorError(f"result has more than {MAX_RESULT_DIGITS} digits")
    return value


def _check_power(base: Number, exponent: Number) -> None:
    """Refuse powers whose result would be too large, before computing them."""
    if abs(exponent) > MAX_EXPONENT:
        raise CalculatorError(f"exponent larger than {MAX_EXPONENT}")
    if abs(base) > 1 and exponent > 0 and exponent * math.log10(abs(base)) >= MAX_RESULT_DIGITS:
        raise CalculatorError(f"result would have more than {MAX_RESULT_DIGITS} digits")


def _evaluate(node: ast.expr) -> Number:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float) and not isinstance(node.value, bool):
        return _check_size(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_evaluate(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow):
            _check_power(left, right)
        return _check_size(_BINARY[type(node.op)](left, right))
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitXor):
        raise CalculatorError("use ** for powers, not ^")
    raise CalculatorError("only numbers, + - * / // % ** and parentheses are allowed")


def _format(value: Number) -> str:
    return f"{value:.12g}" if isinstance(value, float) else str(value)


def calculate(expression: str) -> str:
    """Evaluate `expression`; return the result, or a message starting with "Error"."""
    if len(expression) > MAX_EXPRESSION_LENGTH:
        return f"Error: expression longer than {MAX_EXPRESSION_LENGTH} characters"
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        return _format(_evaluate(tree.body))
    except CalculatorError as exc:
        return f"Error: {exc}"
    except SyntaxError:
        return "Error: not a valid arithmetic expression"
    except ZeroDivisionError:
        return "Error: division by zero"
    except (OverflowError, RecursionError, ValueError):
        return "Error: the result is too large or the expression too complex"


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression and return the exact result.

    Supports numbers, + - * / // % ** and parentheses, e.g. "(24 - 5) * 1.5" or "2 ** 10".
    Use it for every calculation instead of working the numbers out yourself.
    Returns a message starting with "Error" if the expression is not plain arithmetic.
    """
    return calculate(expression)
