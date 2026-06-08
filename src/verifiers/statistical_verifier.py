"""Deterministic statistical / numeric verifier.

Targets the error classes SymPy cannot reach and prose review is unreliable on:
reported numbers that are internally inconsistent — percentages that don't sum,
a derived quantity that doesn't follow from the inputs stated alongside it, a
unit conversion that's off, a mean that doesn't match the values given.

Two-stage, with the *decision* kept deterministic:

1. An LLM EXTRACTS candidate numeric relationships from the excerpt. It does NOT
   judge correctness — it only restates, as a closed arithmetic expression, what
   the text claims should hold (e.g. "the three percentages sum to 100" →
   ``{"expr": "33.3 + 33.3 + 33.3", "expected": 100}``). Crucially, the expression
   may contain ONLY numbers taken from the text — no free variables.
2. Python RECOMPUTES each expression with a safe evaluator and compares it to the
   reported value within a tolerance. An error is flagged ONLY when a check
   deterministically fails. This mirrors the math verifier's philosophy: flag
   INVALID only on a provable numeric contradiction; otherwise UNVERIFIABLE.

Optional unit conversions use ``pint`` if installed; absent, unit checks are
skipped (never guessed).
"""

from __future__ import annotations

import time
from typing import Optional

from loguru import logger

from src.models import (
    StatisticalVerificationResult,
    VerificationSnippet,
    VerificationStatus,
)
from src.utils.llm import parse_json_response
from src.utils.safe_arithmetic import ArithmeticEvalError, safe_eval
from src.verifiers.base import BaseVerifier

try:  # optional unit support
    import pint  # type: ignore

    _UREG = pint.UnitRegistry()
except Exception:  # pragma: no cover - pint is optional
    _UREG = None


STAT_SYSTEM_PROMPT = """You extract checkable NUMERIC relationships from one excerpt of a scientific paper. You do NOT decide whether anything is correct. You only restate, as closed arithmetic, the numeric claims the text makes so a calculator can re-check them.

A "check" is something the stated numbers must satisfy if the paper is internally consistent. Examples:
- Percentages/proportions that should sum to a total ("groups of 33%, 33%, 34%" → expr "33 + 33 + 34", expected 100).
- A derived value computed from stated inputs ("mean of 2, 4, 6 is 4.0" → expr "mean([2,4,6])", expected 4.0).
- An arithmetic identity stated in prose ("12 of 50, i.e. 30%" → expr "12/50*100", expected 30).
- A unit conversion ("5 km = 5500 m") → provide a unit_check instead.

STRICT RULES for `expr`:
- It may contain ONLY literal numbers taken from THIS excerpt and the operators + - * / ** % and functions sqrt, log, ln, log10, log2, exp, abs, round, min, max, sum, mean, floor, ceil, pow, and the constants pi, e.
- NO variable names, NO symbols, NO units inside expr. If a relationship needs an unknown or symbolic quantity, DO NOT emit it.
- Only emit a check when BOTH sides are fully determined by numbers written in the excerpt. When in doubt, omit it.
- Set a sensible relative `tolerance` (default 0.01 = 1%); use a larger tolerance when the text rounds.

For unit conversions, emit a `unit_check`: {"description","value","from_unit","to_unit","expected"}.

## Output Format

Return ONLY JSON:
```json
{
  "checks": [
    {"description": "...", "expr": "33 + 33 + 34", "expected": 100, "tolerance": 0.01}
  ],
  "unit_checks": [],
  "note": "short note; empty list if no purely-numeric claim is present"
}
```
If the excerpt contains no fully-numeric checkable claim, return {"checks": [], "unit_checks": []}.
"""


class StatisticalVerifier(BaseVerifier):
    """Recomputes reported numeric claims and flags deterministic contradictions."""

    name: str = "statistical"

    def verify(self, snippet: VerificationSnippet) -> StatisticalVerificationResult:
        start_time = time.monotonic()

        if not self.can_verify(snippet):
            return StatisticalVerificationResult(
                snippet_id=snippet.snippet_id,
                verifier_name=self.name,
                status=VerificationStatus.SKIPPED,
                execution_time_ms=(time.monotonic() - start_time) * 1000,
            )

        try:
            parsed = parse_json_response(
                self._call_llm(
                    prompt=self._build_prompt(snippet),
                    system_prompt=STAT_SYSTEM_PROMPT,
                )
            )
        except Exception as exc:
            logger.warning(f"Stat extraction failed for {snippet.snippet_id}: {exc}")
            return StatisticalVerificationResult(
                snippet_id=snippet.snippet_id,
                verifier_name=self.name,
                status=VerificationStatus.UNVERIFIABLE,
                reasoning=f"Extraction failed: {exc}",
                execution_time_ms=(time.monotonic() - start_time) * 1000,
            )

        checks_out: list[dict] = []
        failures: list[dict] = []

        for raw in parsed.get("checks", []) or []:
            evaluated = self._run_arithmetic_check(raw)
            checks_out.append(evaluated)
            if evaluated.get("passed") is False:
                failures.append(evaluated)

        for raw in parsed.get("unit_checks", []) or []:
            evaluated = self._run_unit_check(raw)
            checks_out.append(evaluated)
            if evaluated.get("passed") is False:
                failures.append(evaluated)

        elapsed = (time.monotonic() - start_time) * 1000

        # No purely-numeric checkable claim → unverifiable (NOT an error).
        evaluable = [c for c in checks_out if c.get("passed") is not None]
        if not evaluable:
            return StatisticalVerificationResult(
                snippet_id=snippet.snippet_id,
                verifier_name=self.name,
                status=VerificationStatus.UNVERIFIABLE,
                reasoning="No fully-numeric, self-contained claim to recompute.",
                checks=checks_out,
                execution_time_ms=elapsed,
            )

        if failures:
            reason = "; ".join(
                f"{f['description']}: computed {f['computed']:.4g} vs reported "
                f"{f['expected']:.4g} (>{f['tolerance']:.1%})"
                for f in failures[:3]
            )
            return StatisticalVerificationResult(
                snippet_id=snippet.snippet_id,
                verifier_name=self.name,
                status=VerificationStatus.INVALID,
                error_detected=True,
                confidence=0.9,
                reasoning=f"Numeric inconsistency: {reason}",
                predicted_error_category="Statistical reporting",
                checks=checks_out,
                execution_time_ms=elapsed,
            )

        return StatisticalVerificationResult(
            snippet_id=snippet.snippet_id,
            verifier_name=self.name,
            status=VerificationStatus.VALID,
            confidence=0.85,
            reasoning=f"All {len(evaluable)} numeric check(s) consistent.",
            checks=checks_out,
            execution_time_ms=elapsed,
        )

    def can_verify(self, snippet: VerificationSnippet) -> bool:
        if not self.verifier_config.enabled:
            return False
        text = (snippet.content or "") + " " + (snippet.metadata.get("latex", "") or "")
        # Cheap gate: must contain at least one digit to have anything to check.
        return bool(text.strip()) and any(ch.isdigit() for ch in text)

    # ------------------------------------------------------------------
    def _run_arithmetic_check(self, raw: dict) -> dict:
        out = {
            "description": str(raw.get("description", ""))[:200],
            "expr": str(raw.get("expr", "")),
            "expected": raw.get("expected"),
            "tolerance": self._tolerance(raw.get("tolerance")),
            "computed": None,
            "passed": None,
            "error": None,
        }
        try:
            expected = float(out["expected"])
        except (TypeError, ValueError):
            out["error"] = "non-numeric expected value"
            return out
        out["expected"] = expected
        try:
            computed = safe_eval(out["expr"])
        except ArithmeticEvalError as exc:
            # Not safely computable → leave passed=None (does not count as error).
            out["error"] = str(exc)
            return out
        out["computed"] = computed
        out["passed"] = self._within(computed, expected, out["tolerance"])
        return out

    def _run_unit_check(self, raw: dict) -> dict:
        out = {
            "description": str(raw.get("description", ""))[:200],
            "expr": f"{raw.get('value')} {raw.get('from_unit')} -> {raw.get('to_unit')}",
            "expected": raw.get("expected"),
            "tolerance": self._tolerance(raw.get("tolerance")),
            "computed": None,
            "passed": None,
            "error": None,
        }
        if _UREG is None:
            out["error"] = "pint not installed; unit check skipped"
            return out
        try:
            expected = float(out["expected"])
            qty = float(raw["value"]) * _UREG(str(raw["from_unit"]))
            computed = qty.to(str(raw["to_unit"])).magnitude
        except Exception as exc:  # pint errors, bad units, etc.
            out["error"] = f"unit conversion failed: {exc}"
            return out
        out["expected"] = expected
        out["computed"] = computed
        out["passed"] = self._within(computed, expected, out["tolerance"])
        return out

    @staticmethod
    def _tolerance(value) -> float:
        try:
            t = float(value)
            return t if t > 0 else 0.01
        except (TypeError, ValueError):
            return 0.01

    @staticmethod
    def _within(computed: float, expected: float, tol: float) -> bool:
        denom = max(abs(expected), 1e-12)
        return abs(computed - expected) / denom <= tol

    def _build_prompt(self, snippet: VerificationSnippet) -> str:
        return (
            f"Excerpt location: {snippet.location}\n\n"
            f"Content:\n{snippet.content[:3000]}\n\n"
            "Extract ONLY fully-numeric, self-contained checks. Return ONLY the JSON object."
        )
