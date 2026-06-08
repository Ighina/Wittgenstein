"""Deterministic, sandbox-free evaluation of pure-numeric arithmetic expressions.

Used by the statistical verifier to recompute a reported quantity from the
numbers stated in the text and compare it against the reported value. Unlike the
SymPy sandbox (which runs a subprocess for symbolic algebra), this evaluates a
*closed* arithmetic expression — only literal numbers and a whitelist of math
functions, no names, no attribute access, no calls to anything else. It raises
``ArithmeticEvalError`` for anything outside that grammar, so a malformed or
non-numeric expression can never silently produce a wrong "contradiction".
"""

from __future__ import annotations

import ast
import math
from typing import Union

Number = Union[int, float]


class ArithmeticEvalError(Exception):
    """Raised when an expression is outside the safe numeric grammar."""


# Whitelisted zero-argument-or-more math helpers. All operate on plain numbers.
_FUNCS = {
    "sqrt": math.sqrt,
    "log": math.log,          # natural log (1 arg) or log(x, base) (2 args)
    "ln": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": lambda *a: sum(a[0]) if len(a) == 1 and isinstance(a[0], (list, tuple)) else sum(a),
    "mean": lambda *a: (sum(a[0]) / len(a[0])) if len(a) == 1 and isinstance(a[0], (list, tuple)) else (sum(a) / len(a)),
    "floor": math.floor,
    "ceil": math.ceil,
    "pow": pow,
}

_CONSTS = {"pi": math.pi, "e": math.e}

_BIN_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a ** b,
}

_UNARY_OPS = {
    ast.UAdd: lambda a: +a,
    ast.USub: lambda a: -a,
}


def safe_eval(expr: str) -> float:
    """Evaluate a closed numeric arithmetic expression and return a float.

    Args:
        expr: e.g. "100 - (33.3 + 33.3 + 33.3)", "log(1/1e-9)", "5.35 * log(2)".

    Returns:
        The numeric result as a float.

    Raises:
        ArithmeticEvalError: if the expression uses anything outside the
            numbers-and-whitelisted-functions grammar, or fails to evaluate.
    """
    if not isinstance(expr, str) or not expr.strip():
        raise ArithmeticEvalError("empty expression")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ArithmeticEvalError(f"syntax error: {exc}") from exc

    try:
        value = _eval_node(tree.body)
    except ArithmeticEvalError:
        raise
    except Exception as exc:  # ZeroDivisionError, math domain, etc.
        raise ArithmeticEvalError(f"evaluation failed: {exc}") from exc

    if isinstance(value, bool):  # bool is an int subclass; reject explicitly
        raise ArithmeticEvalError("boolean result is not numeric")
    if not isinstance(value, (int, float)):
        raise ArithmeticEvalError(f"non-numeric result: {type(value).__name__}")
    return float(value)


def _eval_node(node: ast.AST):  # noqa: C901 - small dispatch
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ArithmeticEvalError(f"non-numeric literal: {node.value!r}")
        return node.value
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ArithmeticEvalError(f"operator not allowed: {type(node.op).__name__}")
        return op(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ArithmeticEvalError(f"unary operator not allowed: {type(node.op).__name__}")
        return op(_eval_node(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            raise ArithmeticEvalError("only whitelisted math functions may be called")
        if node.keywords:
            raise ArithmeticEvalError("keyword arguments are not allowed")
        args = [_eval_node(a) for a in node.args]
        return _FUNCS[node.func.id](*args)
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval_node(e) for e in node.elts]
    if isinstance(node, ast.Name):
        if node.id in _CONSTS:
            return _CONSTS[node.id]
        raise ArithmeticEvalError(f"name not allowed: {node.id!r}")
    raise ArithmeticEvalError(f"disallowed syntax: {type(node).__name__}")
