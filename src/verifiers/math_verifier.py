"""Phase 7: Mathematical equation verification using SymPy.

Sends equations to an LLM for conversion to SymPy code, executes the code
in a sandboxed subprocess, and interprets the results.

Design note — why this verifier is *conservative* about INVALID
----------------------------------------------------------------
Symbolic identity checking (``simplify(LHS - RHS) == 0``) is only a sound
falsification test for a narrow class of equations. Most equations in real
papers are NOT free-standing algebraic identities:

* **Definitions / assignments** — ``B = Q/N``, ``α = 15``, ``φ = conj(h)``.
  These introduce a name; ``LHS - RHS`` has free symbols and never simplifies
  to 0, but that is *not* an error — there is nothing to falsify.
* **Conditional / constrained equations** — ``s² + t² = 1`` (a point on a
  circle), relations that hold only under assumptions the LaTeX does not carry.
* **Operator / non-commutative algebra** — symbols that are matrices or
  operators (``I``, ``J``) which SymPy treats as commuting scalars, producing
  a spurious non-zero residual.

For all of these the only safe verdict is **UNVERIFIABLE**. We therefore only
emit **INVALID** when the residual is *deterministically* contradictory:

1. the residual reduces to a **non-zero number** (no free symbols), or
2. the LLM asserts the equation is an unconditional **identity** AND a numeric
   sampling cross-check finds the residual non-zero at *every* sampled point.

Everything else — residual with free symbols, definitions, conditionals,
matrices/operators that don't reduce, anything ambiguous — is UNVERIFIABLE.
The VALID / INVALID / UNVERIFIABLE decision is made deterministically inside
the sandbox by an injected harness (``_VERDICT_HARNESS``), not by string-
matching the raw SymPy output, so zero-matrices and boolean ``True`` results
are recognised correctly.
"""

from __future__ import annotations

import time
from typing import Optional

from loguru import logger

from src.models import (
    EquationVerificationResult,
    VerificationSnippet,
    VerificationStatus,
)
from src.utils.llm import parse_json_response
from src.utils.sandbox import (
    SandboxError,
    SandboxTimeoutError,
    run_sympy_sandbox,
)
from src.verifiers.base import BaseVerifier


# ---------------------------------------------------------------------------
# Verdict harness
# ---------------------------------------------------------------------------
# Prepended to every LLM-generated snippet before execution. The LLM is asked
# to compute the residual ``LHS - RHS`` and pass it to ``report(...)`` (or call
# ``report_unverifiable(...)``). The harness then classifies the residual
# deterministically and prints a single machine-readable ``VERDICT:`` line that
# ``_interpret_output`` parses. This removes all fragile string parsing of
# free-form SymPy output (zero matrices, booleans, etc.).
_VERDICT_HARNESS = r'''
import json as _json
from sympy import simplify, nsimplify, Matrix, Basic, S, Rational, zoo, oo, nan
from sympy.core.numbers import Number
from sympy.logic.boolalg import BooleanTrue, BooleanFalse

_NUM_SAMPLES = 30


def _emit(verdict, **extra):
    payload = {"verdict": verdict}
    payload.update(extra)
    print("VERDICT:" + _json.dumps(payload, default=str))


def _is_zero_obj(expr):
    """Robust zero check across scalars, matrices, relationals and booleans."""
    if expr is True or isinstance(expr, BooleanTrue):
        return True
    if expr is False or isinstance(expr, BooleanFalse):
        return False
    if isinstance(expr, Matrix):
        try:
            return bool(expr.is_zero_matrix) or all(_is_zero_obj(e) for e in expr)
        except Exception:
            return all(simplify(e) == 0 for e in expr)
    try:
        s = simplify(expr)
    except Exception:
        s = expr
    if isinstance(s, Matrix):
        return bool(s.is_zero_matrix)
    return s == 0 or s == S.Zero


def _free_symbols(expr):
    try:
        return getattr(expr, "free_symbols", set()) or set()
    except Exception:
        return set()


def _sample_nonzero_fraction(expr, syms):
    """Numerically sample the residual over free symbols.

    Returns (fraction_nonzero, n_evaluated, max_abs) using a deterministic set
    of rational sample points (no RNG — keeps the pipeline reproducible).
    Points that hit singularities / non-finite values are skipped.
    """
    syms = sorted(syms, key=lambda s: s.name)
    # Deterministic, well-spread rational sample values avoiding 0 and 1.
    base_points = [Rational(p, q) for p, q in (
        (2, 1), (3, 2), (5, 3), (7, 4), (-2, 1), (-3, 2), (11, 5), (-7, 4),
        (4, 3), (-5, 3), (13, 6), (9, 7), (-11, 5), (6, 5), (-13, 6), (8, 5),
    )]
    n_eval = 0
    n_nonzero = 0
    max_abs = 0.0
    for i in range(_NUM_SAMPLES):
        subs = {}
        for j, sym in enumerate(syms):
            subs[sym] = base_points[(i + j) % len(base_points)]
        try:
            val = expr.subs(subs)
            val = complex(val.evalf()) if hasattr(val, "evalf") else complex(val)
        except Exception:
            continue
        if val != val or abs(val) in (float("inf"),):  # NaN / inf
            continue
        n_eval += 1
        mag = abs(val)
        max_abs = max(max_abs, mag)
        if mag > 1e-9:
            n_nonzero += 1
    frac = (n_nonzero / n_eval) if n_eval else 0.0
    return frac, n_eval, max_abs


def report(residual):
    """Classify a residual expression (expected to be LHS - RHS)."""
    try:
        if _is_zero_obj(residual):
            _emit("VALID")
            return
        # Matrix that is not all-zero → check if fully numeric.
        if isinstance(residual, Matrix):
            free = set()
            for e in residual:
                free |= _free_symbols(e)
            if not free:
                _emit("INVALID_NUMERIC", residual=str(residual))
            else:
                _emit("UNVERIFIABLE", reason="symbolic matrix residual",
                      residual=str(residual))
            return
        free = _free_symbols(residual)
        if not free:
            # Pure number, non-zero → deterministic contradiction.
            _emit("INVALID_NUMERIC", residual=str(simplify(residual)))
            return
        # Residual still has free symbols: do NOT conclude INVALID on symbolic
        # non-zero alone. Provide a numeric sampling signal for the caller.
        frac, n_eval, max_abs = _sample_nonzero_fraction(residual, free)
        _emit(
            "SYMBOLIC_NONZERO",
            residual=str(residual),
            free=sorted(s.name for s in free),
            nonzero_fraction=frac,
            n_evaluated=n_eval,
            max_abs=max_abs,
        )
    except Exception as exc:  # pragma: no cover - defensive
        _emit("HARNESS_ERROR", error=str(exc))


def report_unverifiable(reason="not symbolically decidable"):
    _emit("UNVERIFIABLE", reason=str(reason))


def report_valid(reason="verified"):
    _emit("VALID", reason=str(reason))
'''


MATH_SYSTEM_PROMPT = """You are a mathematical verification expert. You convert a LaTeX equation into SymPy Python code that lets a deterministic harness decide whether the equation is mathematically valid.

A helper module is ALREADY imported for you. You MUST finish by calling exactly ONE of:
  - report(residual)            # residual = simplify(LHS - RHS); the harness classifies it
  - report_unverifiable(reason) # use when the equation cannot be soundly checked symbolically
  - report_valid(reason)        # only when you have *proven* validity by a custom check

Do NOT call print() yourself for the verdict — always go through these helpers.

## How to classify the equation (choose `equation_type`)

- "numeric": both sides reduce to concrete numbers (e.g. "2.5 = 5/2", a computed
  constant, a unit conversion). These are fully checkable.
- "identity": a claim that holds for ALL real values of every free symbol with NO
  side conditions — i.e. an algebraic/trigonometric simplification rule such as
  (a+b)^2 = a^2 + 2ab + b^2. Only use this when the equation must hold universally.
- "definition": the equation DEFINES or names a quantity (introduces a new symbol),
  e.g. B = Q/N, alpha = 15, phi = conjugate(h). There is nothing to falsify.
- "conditional": the equation only holds under constraints, specific values, or
  assumptions not written in the LaTeX (e.g. s^2 + t^2 = 1, recurrences, equations
  involving operators/matrices/non-commuting symbols treated as scalars).
- "unverifiable": non-algebraic, relies on external semantics, too complex, involves
  limits/integrals/series that SymPy cannot settle, or you are unsure.

## What code to generate

- For "numeric" and "identity": define all symbols with `symbols(...)`, build LHS and
  RHS, then call `report(simplify(LHS - RHS))`.
- For "definition", "conditional", and "unverifiable": call
  `report_unverifiable("<short reason>")`. Do NOT try to force a residual to zero.
- Treat matrices/operators with `Matrix(...)` or non-commutative symbols ONLY when you
  are confident of their structure; otherwise classify as "conditional".

## Output Format

Return a JSON object with exactly these fields:
```json
{
  "sympy_code": "syms = symbols('a b')\\n...\\nreport(simplify(lhs - rhs))",
  "equation_type": "numeric | identity | definition | conditional | unverifiable",
  "explanation": "Brief explanation",
  "unverifiable": false
}
```

`sympy_code` must contain ONLY executable Python (no markdown, no import of the helper —
it is already available; you may still `from sympy import *` if convenient). Set
`unverifiable: true` only as a shortcut equivalent to calling report_unverifiable.
"""


# Numeric INVALID (residual is a non-zero number) is fully deterministic and
# trustworthy. A symbolic identity that fails numeric sampling at every point is
# strong but slightly less certain (missing assumptions are still possible).
_CONF_NUMERIC_INVALID = 0.9
_CONF_IDENTITY_INVALID = 0.75
_CONF_VALID = 0.85


class MathEquationVerifier(BaseVerifier):
    """Verifies mathematical equations using SymPy symbolic computation.

    Workflow:
    1. Send equation to LLM → generate SymPy code + an ``equation_type`` label.
    2. Prepend the deterministic verdict harness and execute in the sandbox.
    3. Parse the harness ``VERDICT:`` line and, combined with ``equation_type``,
       decide VALID / INVALID / UNVERIFIABLE / MALFORMED conservatively.
    """

    name: str = "math_equation"

    def verify(
        self,
        snippet: VerificationSnippet,
    ) -> EquationVerificationResult:
        """Verify a mathematical equation snippet."""
        start_time = time.monotonic()

        logger.debug(f"Verifying equation: {snippet.location} ({snippet.snippet_id})")

        if not self.can_verify(snippet):
            return EquationVerificationResult(
                snippet_id=snippet.snippet_id,
                verifier_name=self.name,
                status=VerificationStatus.SKIPPED,
                execution_time_ms=(time.monotonic() - start_time) * 1000,
            )

        # Step 1: Get SymPy code + classification from LLM
        latex = snippet.metadata.get("latex", snippet.content)
        prompt = self._build_prompt(latex, snippet)

        try:
            llm_response = self._call_llm(
                prompt=prompt,
                system_prompt=MATH_SYSTEM_PROMPT,
            )
            parsed = parse_json_response(llm_response)

            equation_type = str(parsed.get("equation_type", "")).strip().lower()

            # The LLM can short-circuit to UNVERIFIABLE for definitions /
            # conditionals / non-algebraic content.
            if parsed.get("unverifiable", False) or equation_type in {
                "definition",
                "conditional",
                "unverifiable",
            }:
                return EquationVerificationResult(
                    snippet_id=snippet.snippet_id,
                    verifier_name=self.name,
                    status=VerificationStatus.UNVERIFIABLE,
                    reasoning=parsed.get(
                        "explanation",
                        f"Equation classified as '{equation_type or 'unverifiable'}'; "
                        "not a falsifiable symbolic identity.",
                    ),
                    sympy_code=parsed.get("sympy_code"),
                    execution_time_ms=(time.monotonic() - start_time) * 1000,
                )

            sympy_code = parsed.get("sympy_code", "")
            if not sympy_code:
                return EquationVerificationResult(
                    snippet_id=snippet.snippet_id,
                    verifier_name=self.name,
                    status=VerificationStatus.MALFORMED,
                    reasoning="LLM did not produce SymPy code",
                    execution_time_ms=(time.monotonic() - start_time) * 1000,
                )

        except Exception as exc:
            logger.error(f"LLM call failed for {snippet.snippet_id}: {exc}")
            return EquationVerificationResult(
                snippet_id=snippet.snippet_id,
                verifier_name=self.name,
                status=VerificationStatus.UNVERIFIABLE,
                reasoning=f"LLM call failed: {exc}",
                execution_time_ms=(time.monotonic() - start_time) * 1000,
            )

        # Step 2: Execute (harness + LLM code) in sandbox
        full_code = _VERDICT_HARNESS + "\n\n" + sympy_code
        try:
            stdout, stderr, returncode = run_sympy_sandbox(
                code=full_code,
                python_executable=self.config.sandbox.python_executable,
                timeout_seconds=self.config.sandbox.timeout_seconds,
                max_output_bytes=self.config.sandbox.max_output_bytes,
            )
        except SandboxTimeoutError:
            return EquationVerificationResult(
                snippet_id=snippet.snippet_id,
                verifier_name=self.name,
                status=VerificationStatus.UNVERIFIABLE,
                reasoning="SymPy execution timed out",
                sympy_code=sympy_code,
                execution_time_ms=(time.monotonic() - start_time) * 1000,
            )
        except SandboxError as exc:
            return EquationVerificationResult(
                snippet_id=snippet.snippet_id,
                verifier_name=self.name,
                status=VerificationStatus.MALFORMED,
                reasoning=f"Sandbox execution error: {exc}",
                sympy_code=sympy_code,
                execution_time_ms=(time.monotonic() - start_time) * 1000,
            )

        # Step 3: Interpret results
        status, reasoning, confidence = self._interpret_output(
            stdout, stderr, returncode, equation_type
        )

        return EquationVerificationResult(
            snippet_id=snippet.snippet_id,
            verifier_name=self.name,
            status=status,
            error_detected=(status == VerificationStatus.INVALID),
            confidence=confidence,
            reasoning=reasoning,
            predicted_error_category="Equation / proof" if status == VerificationStatus.INVALID else None,
            sympy_code=sympy_code,
            execution_output=stdout.strip(),
            execution_error=stderr.strip() if stderr else None,
            return_code=returncode,
            execution_time_ms=(time.monotonic() - start_time) * 1000,
        )

    def can_verify(self, snippet: VerificationSnippet) -> bool:
        """Check if the snippet contains a verifiable equation."""
        if not self.verifier_config.enabled:
            return False
        latex = snippet.metadata.get("latex", "")
        return bool(latex and len(latex) > 2)

    def _build_prompt(self, latex: str, snippet: VerificationSnippet) -> str:
        """Build the prompt for the LLM."""
        parts = [
            "Convert the following LaTeX equation into SymPy verification code.",
            "First decide its `equation_type` (numeric / identity / definition / "
            "conditional / unverifiable), then emit the matching code.",
            "",
            "Equation:",
            latex,
        ]

        context = snippet.content
        if context:
            parts.extend([
                "",
                "Additional context (use it to judge whether symbols are defined "
                "here or constrained — definitions and conditionals are UNVERIFIABLE):",
                context[:2000],
            ])

        parts.extend([
            "",
            "Generate ONLY the JSON response with the sympy_code and equation_type fields.",
        ])

        return "\n".join(parts)

    @staticmethod
    def _parse_verdict(stdout: str) -> Optional[dict]:
        """Extract the last ``VERDICT:{...}`` JSON payload from stdout."""
        import json

        verdict = None
        for line in stdout.splitlines():
            line = line.strip()
            if line.startswith("VERDICT:"):
                try:
                    verdict = json.loads(line[len("VERDICT:"):])
                except json.JSONDecodeError:
                    continue
        return verdict

    @classmethod
    def _interpret_output(
        cls,
        stdout: str,
        stderr: str,
        returncode: int,
        equation_type: str,
    ) -> tuple[VerificationStatus, str, float]:
        """Interpret the harness verdict conservatively.

        Returns (status, reasoning, confidence).
        """
        # Execution failure → malformed, not an error in the paper.
        if returncode != 0:
            detail = stderr[:500] if stderr else f"exit code {returncode}"
            return (
                VerificationStatus.MALFORMED,
                f"Execution error: {detail}",
                0.0,
            )

        verdict = cls._parse_verdict(stdout)
        if verdict is None:
            # No machine-readable verdict (LLM bypassed the helpers). We refuse
            # to guess INVALID from raw output — that is what produced the old
            # false positives. Treat as unverifiable.
            return (
                VerificationStatus.UNVERIFIABLE,
                "No deterministic verdict produced; cannot soundly decide.",
                0.0,
            )

        v = verdict.get("verdict")

        if v == "VALID":
            return (
                VerificationStatus.VALID,
                "Residual reduces to zero (identity verified).",
                _CONF_VALID,
            )

        if v == "INVALID_NUMERIC":
            return (
                VerificationStatus.INVALID,
                f"Equation is numerically false: LHS - RHS = "
                f"{verdict.get('residual')} (no free symbols).",
                _CONF_NUMERIC_INVALID,
            )

        if v == "SYMBOLIC_NONZERO":
            frac = float(verdict.get("nonzero_fraction", 0.0) or 0.0)
            n_eval = int(verdict.get("n_evaluated", 0) or 0)
            residual = verdict.get("residual")
            free = verdict.get("free")
            # Only escalate to INVALID for *unconditional identities* that fail
            # numeric sampling at every point. Definitions/conditionals never
            # reach here (handled before sandbox) but we guard again.
            if (
                equation_type == "identity"
                and n_eval >= 5
                and frac >= 0.999
            ):
                return (
                    VerificationStatus.INVALID,
                    f"Claimed identity fails: residual {residual} is non-zero at "
                    f"all {n_eval} sampled points (free symbols {free}).",
                    _CONF_IDENTITY_INVALID,
                )
            # Everything else: symbolic non-zero with free symbols is NOT a
            # sound basis for flagging an error.
            return (
                VerificationStatus.UNVERIFIABLE,
                f"Residual {residual} has free symbols {free} and does not "
                f"reduce to zero; not a falsifiable identity "
                f"(type='{equation_type or 'unknown'}', "
                f"nonzero_fraction={frac:.2f}).",
                0.0,
            )

        if v == "UNVERIFIABLE":
            return (
                VerificationStatus.UNVERIFIABLE,
                verdict.get("reason", "Marked unverifiable by harness."),
                0.0,
            )

        # HARNESS_ERROR or unknown → malformed / unverifiable.
        if v == "HARNESS_ERROR":
            return (
                VerificationStatus.MALFORMED,
                f"Harness error: {verdict.get('error')}",
                0.0,
            )

        return (
            VerificationStatus.UNVERIFIABLE,
            f"Unrecognized verdict '{v}'.",
            0.0,
        )
