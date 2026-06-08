"""Verification layers for the progressive math pipeline.

Each layer takes an equation (LHS, RHS as SymPy expression strings) and the
accumulated ``SymbolContext``, and returns a ``LayerResult``.

Layers are called in order; each short-circuits if a prior layer definitively
proved or disproved the claim:

  1. symbolic equivalence   →  simplify(LHS - RHS) == 0?
  2. dimensional consistency →  (optional) sympy.physics.units
  3. side-condition check   →  resolve pending obligations
  4. numeric falsification  →  sample with domain-aware constraints
"""

from __future__ import annotations

import json
import re
from typing import Optional

from loguru import logger

from src.utils.sandbox import (
    SandboxError,
    SandboxTimeoutError,
    run_sympy_sandbox,
)
from src.verifiers.progressive.context_graph import build_all_symbol_declarations
from src.verifiers.progressive.models import LayerResult, SymbolContext


# ---------------------------------------------------------------------------
# Shared sandbox harness (reusable mini-harness for layer execution)
# ---------------------------------------------------------------------------

_BASE_HARNESS = r'''
import json as _json
from sympy import *


def _emit_layer_result(layer, status, **extra):
    payload = {"layer": layer, "status": status}
    payload.update(extra)
    print("LAYER_RESULT:" + _json.dumps(payload, default=str))
'''


def _run_layer_code(
    layer_name: str,
    python_code: str,
    python_executable: str = "python3",
    timeout_seconds: int = 10,
    max_output_bytes: int = 65536,
) -> LayerResult:
    """Execute a layer's SymPy code in the sandbox and parse the result."""
    full_code = _BASE_HARNESS + "\n" + python_code

    try:
        stdout, stderr, returncode = run_sympy_sandbox(
            code=full_code,
            python_executable=python_executable,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
        )
    except SandboxTimeoutError:
        return LayerResult(
            layer=layer_name,
            status="unknown",
            details="Layer execution timed out.",
        )
    except SandboxError as exc:
        return LayerResult(
            layer=layer_name,
            status="unknown",
            details=f"Sandbox error: {exc}",
        )

    if returncode != 0:
        detail = stderr[:300] if stderr else f"exit code {returncode}"
        return LayerResult(
            layer=layer_name,
            status="unknown",
            details=f"Execution error: {detail}",
        )

    # Parse LAYER_RESULT: line
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("LAYER_RESULT:"):
            try:
                data = json.loads(line[len("LAYER_RESULT:"):])
                return LayerResult(
                    layer=data.get("layer", layer_name),
                    status=data.get("status", "unknown"),
                    confidence=float(data.get("confidence", 0.0)),
                    condition=data.get("condition"),
                    details=data.get("details", ""),
                    residual=data.get("residual"),
                    free_symbols=data.get("free_symbols", []),
                )
            except (json.JSONDecodeError, ValueError):
                continue

    return LayerResult(
        layer=layer_name,
        status="unknown",
        details="No result emitted by layer code.",
    )


# ---------------------------------------------------------------------------
# Layer 1 — Symbolic equivalence
# ---------------------------------------------------------------------------

_SYMBOLIC_LAYER_CODE = r'''
# -- User declarations (from context) --
{context_declarations}

# -- Equation under test --
{user_code}

# -- Check --
try:
    residual = simplify(lhs - rhs)
    free = list(residual.free_symbols) if hasattr(residual, 'free_symbols') else []
    if residual == 0 or residual == S.Zero:
        _emit_layer_result("symbolic", "proved",
                           details="LHS - RHS simplifies to zero.",
                           residual=str(residual),
                           free_symbols=[str(s) for s in free])
    else:
        try:
            is_zero_bool = bool(residual == 0)
        except Exception:
            is_zero_bool = False
        if is_zero_bool:
            _emit_layer_result("symbolic", "proved",
                               details="Residual evaluates to zero.",
                               residual=str(residual),
                               free_symbols=[str(s) for s in free])
        elif not free:
            _emit_layer_result("symbolic", "disproved",
                               details=f"Non-zero numeric residual: {residual}",
                               residual=str(residual),
                               confidence=0.9)
        else:
            _emit_layer_result("symbolic", "unknown",
                               details=f"Residual has free symbols: {free}",
                               residual=str(residual),
                               free_symbols=[str(s) for s in free])
except Exception as exc:
    _emit_layer_result("symbolic", "unknown",
                       details=f"Simplify failed: {exc}")
'''


def check_symbolic_equivalence(
    lhs_expr: str,
    rhs_expr: str,
    ctx: SymbolContext,
    python_executable: str = "python3",
    timeout_seconds: int = 10,
) -> LayerResult:
    """Check whether LHS ≡ RHS by simplifying LHS - RHS.

    This is the fastest and most definitive check — if the residual simplifies
    to exactly zero, the equation is proven; if it reduces to a non-zero number
    with no free symbols, it's disproven; otherwise it's unknown.
    """
    context_declarations = build_all_symbol_declarations(ctx)
    user_code = f"lhs = {lhs_expr}\nrhs = {rhs_expr}"
    code = _SYMBOLIC_LAYER_CODE.format(
        context_declarations=context_declarations,
        user_code=user_code,
    )

    result = _run_layer_code(
        "symbolic", code,
        python_executable=python_executable,
        timeout_seconds=timeout_seconds,
    )
    logger.debug(f"Symbolic layer: {result.status} — {result.details}")
    return result


# ---------------------------------------------------------------------------
# Layer 2 — Dimensional consistency
# ---------------------------------------------------------------------------

_DIMENSIONAL_LAYER_CODE = r'''
# -- User declarations (from context) --
{context_declarations}

# -- Equation under test --
{user_code}

try:
    from sympy.physics.units import Dimension
    from sympy.physics.units import quantity_simplify

    # Try to determine the dimension of each side.
    # If the expression has no units, this is not applicable.
    try:
        lhs_dim = quantity_simplify(lhs).dimension if hasattr(quantity_simplify(lhs), 'dimension') else None
    except Exception:
        lhs_dim = None
    try:
        rhs_dim = quantity_simplify(rhs).dimension if hasattr(quantity_simplify(rhs), 'dimension') else None
    except Exception:
        rhs_dim = None

    if lhs_dim is None and rhs_dim is None:
        _emit_layer_result("dimensional", "not_applicable",
                           details="No dimensional information found in expressions.")
    elif lhs_dim is not None and rhs_dim is not None:
        if lhs_dim == rhs_dim:
            _emit_layer_result("dimensional", "proved",
                               details=f"Dimensions match: {{lhs_dim}} == {{rhs_dim}}",
                               confidence=0.85)
        else:
            _emit_layer_result("dimensional", "disproved",
                               details=f"Dimension mismatch: {{lhs_dim}} != {{rhs_dim}}",
                               confidence=0.9)
    else:
        _emit_layer_result("dimensional", "unknown",
                           details="Only one side had dimensional information.")
except ImportError:
    _emit_layer_result("dimensional", "not_applicable",
                       details="sympy.physics.units not available.")
except Exception as exc:
    _emit_layer_result("dimensional", "unknown",
                       details=f"Dimensional check error: {{exc}}")
'''


def check_dimensional_consistency(
    lhs_expr: str,
    rhs_expr: str,
    ctx: SymbolContext,
    python_executable: str = "python3",
    timeout_seconds: int = 10,
) -> LayerResult:
    """Check whether LHS and RHS have consistent dimensions.

    Requires ``sympy.physics.units``. Silently returns ``not_applicable``
    when the module is unavailable or the expressions carry no unit information.
    """
    context_declarations = build_all_symbol_declarations(ctx)
    user_code = f"lhs = {lhs_expr}\nrhs = {rhs_expr}"
    code = _DIMENSIONAL_LAYER_CODE.format(
        context_declarations=context_declarations,
        user_code=user_code,
    )

    result = _run_layer_code(
        "dimensional", code,
        python_executable=python_executable,
        timeout_seconds=timeout_seconds,
    )
    logger.debug(f"Dimensional layer: {result.status} — {result.details}")
    return result


# ---------------------------------------------------------------------------
# Layer 3 — Side-condition / obligation check
# ---------------------------------------------------------------------------

def check_side_conditions(
    residual: str,
    free_symbols: list[str],
    ctx: SymbolContext,
) -> LayerResult:
    """Check whether the residual's free symbols are constrained by context
    assumptions that would force the residual to zero.

    This is computed in-process (no sandbox needed) by examining the context's
    declarations and assumptions against the free symbols in the residual.
    """
    from src.verifiers.progressive.context_graph import build_sympy_assumptions

    if not free_symbols:
        return LayerResult(
            layer="side_condition",
            status="unknown",
            details="No free symbols to check conditions for.",
        )

    # Build a picture of what's known about each free symbol
    known: dict[str, dict[str, bool]] = {}
    for sym in free_symbols:
        decl = ctx.declarations.get(sym)
        if decl is not None:
            known[sym] = build_sympy_assumptions(decl)
        else:
            known[sym] = {}

    completely_unknown = all(not v for v in known.values())
    if completely_unknown:
        return LayerResult(
            layer="side_condition",
            status="unknown",
            details=f"No context assumptions for free symbols {free_symbols}.",
            free_symbols=free_symbols,
        )

    # If every free symbol has a declaration, the equation is at least
    # well-formed; whether the residual being non-zero is a problem depends
    # on the statement class (handled by the caller).
    fully_declared = all(
        known.get(s, {}).get("real") or known.get(s, {}).get("complex")
        for s in free_symbols
    )
    if fully_declared:
        return LayerResult(
            layer="side_condition",
            status="unknown",
            details=(
                f"All free symbols {free_symbols} are declared in context, "
                f"but residual is non-zero — may be a conditional or definition."
            ),
            free_symbols=free_symbols,
        )

    return LayerResult(
        layer="side_condition",
        status="unknown",
        details=f"Some free symbols {free_symbols} lack domain declarations.",
        free_symbols=free_symbols,
    )


# ---------------------------------------------------------------------------
# Layer 4 — Numeric falsification (context-aware sampling)
# ---------------------------------------------------------------------------

_NUMERIC_LAYER_CODE = r'''
# -- User declarations (from context) --
{context_declarations}

# -- Equation under test --
{user_code}

# -- Numeric sampling --
_NUM_SAMPLES = 30
_free_syms = sorted(list(lhs.free_symbols | rhs.free_symbols), key=lambda s: s.name)

if not _free_syms:
    residual = simplify(lhs - rhs)
    try:
        val = complex(residual.evalf()) if hasattr(residual, 'evalf') else complex(residual)
    except Exception:
        val = None
    if val is not None and abs(val) < 1e-12:
        _emit_layer_result("numeric", "proved",
                           details="Numeric residual is effectively zero (constant).",
                           confidence=0.85,
                           residual=str(residual))
    elif val is not None:
        _emit_layer_result("numeric", "disproved",
                           details=f"Constant non-zero residual: {{residual}} = {{val}}",
                           confidence=0.9,
                           residual=str(residual))
    else:
        _emit_layer_result("numeric", "unknown",
                           details="Could not evaluate constant residual.")
else:
    # Build sample points
    base_points = [Rational(p, q) for p, q in (
        (2, 1), (3, 2), (5, 3), (7, 4), (-2, 1), (-3, 2), (11, 5), (-7, 4),
        (4, 3), (-5, 3), (13, 6), (9, 7), (-11, 5), (6, 5), (-13, 6), (8, 5),
    )]
    n_eval = 0
    n_nonzero = 0
    max_abs = 0.0

    for i in range(_NUM_SAMPLES):
        subs = {{}}
        skip = False
        for j, sym in enumerate(_free_syms):
            val = base_points[(i + j) % len(base_points)]
            # Domain-aware: if symbol is declared positive, constrain to positive samples
            if sym.is_positive:
                val = abs(val)
            if sym.is_nonnegative:
                val = abs(val)
            if sym.is_integer:
                val = Rational(int(val) if abs(val) >= 1 else 1, 1)
            subs[sym] = val
        try:
            residual = simplify((lhs - rhs).subs(subs))
            val = complex(residual.evalf()) if hasattr(residual, 'evalf') else complex(residual)
        except Exception:
            continue
        if val != val or abs(val) == float("inf"):
            continue
        n_eval += 1
        mag = abs(val)
        max_abs = max(max_abs, mag)
        if mag > 1e-9:
            n_nonzero += 1

    frac = (n_nonzero / n_eval) if n_eval else 0.0

    if n_eval == 0:
        _emit_layer_result("numeric", "unknown",
                           details="All numeric samples hit singularities.")
    elif frac >= 0.999:
        _emit_layer_result("numeric", "disproved",
                           details=f"Non-zero at all {{n_eval}} sampled points (frac={{frac:.3f}}, max_abs={{max_abs:.3g}}).",
                           confidence=0.75,
                           free_symbols=[str(s) for s in _free_syms])
    elif frac <= 0.03:
        _emit_layer_result("numeric", "proved",
                           details=f"Zero at {{n_eval - n_nonzero}}/{{n_eval}} of sampled points.",
                           confidence=0.6,
                           free_symbols=[str(s) for s in _free_syms])
    else:
        _emit_layer_result("numeric", "unknown",
                           details=f"Non-zero at {{n_nonzero}}/{{n_eval}} points (frac={{frac:.3f}}).",
                           free_symbols=[str(s) for s in _free_syms])
'''


def check_numeric_falsification(
    lhs_expr: str,
    rhs_expr: str,
    ctx: SymbolContext,
    python_executable: str = "python3",
    timeout_seconds: int = 15,
) -> LayerResult:
    """Numerically sample LHS - RHS using domain-aware sample points.

    If symbols are declared positive/nonnegative/integer in the context,
    the sample points are constrained to those domains — this prevents
    false-positive "counterexamples" from sampling outside the domain.
    """
    context_declarations = build_all_symbol_declarations(ctx)
    user_code = f"lhs = {lhs_expr}\nrhs = {rhs_expr}"
    code = _NUMERIC_LAYER_CODE.format(
        context_declarations=context_declarations,
        user_code=user_code,
    )

    result = _run_layer_code(
        "numeric", code,
        python_executable=python_executable,
        timeout_seconds=timeout_seconds,
    )
    logger.debug(f"Numeric layer: {result.status} — {result.details}")
    return result


# ---------------------------------------------------------------------------
# Orchestration — run layers in sequence with short-circuit
# ---------------------------------------------------------------------------


def run_verification_layers(
    sympy_code: str,
    ctx: SymbolContext,
    layers: tuple[str, ...] = ("symbolic", "dimensional", "side_condition", "numeric"),
    python_executable: str = "python3",
) -> list[LayerResult]:
    """Run all active verification layers in order, short-circuiting when one
    definitively proves or disproves the equation.

    ``sympy_code`` must define ``lhs`` and ``rhs`` as SymPy expressions.
    """
    results: list[LayerResult] = []

    # Extract LHS/RHS expression strings from the sympy_code.
    # The LLM-generated code will have `report(residual)` calls — we extract
    # `lhs = ...` and `rhs = ...` lines and use them directly.
    lhs_match = re.search(r"lhs\s*=\s*(.+?)(?:\n|$)", sympy_code)
    rhs_match = re.search(r"rhs\s*=\s*(.+?)(?:\n|$)", sympy_code)

    if not lhs_match or not rhs_match:
        return [LayerResult(
            layer="pipeline",
            status="unknown",
            details="Could not extract LHS/RHS from generated code.",
        )]

    lhs_expr = lhs_match.group(1).strip()
    rhs_expr = rhs_match.group(1).strip()

    for layer_name in layers:
        if layer_name == "symbolic":
            result = check_symbolic_equivalence(
                lhs_expr, rhs_expr, ctx,
                python_executable=python_executable,
            )
        elif layer_name == "dimensional":
            result = check_dimensional_consistency(
                lhs_expr, rhs_expr, ctx,
                python_executable=python_executable,
            )
        elif layer_name == "side_condition":
            # Side-condition check operates on the residual from symbolic
            prev = results[-1] if results else None
            residual = prev.residual if prev else None
            free_syms = list(set(
                prev.free_symbols if prev else []
            ))
            result = check_side_conditions(
                residual or "", free_syms, ctx,
            )
        elif layer_name == "numeric":
            result = check_numeric_falsification(
                lhs_expr, rhs_expr, ctx,
                python_executable=python_executable,
            )
        else:
            continue

        results.append(result)

        # Short-circuit on definitive verdict
        if result.status in ("proved", "disproved"):
            break

    return results
