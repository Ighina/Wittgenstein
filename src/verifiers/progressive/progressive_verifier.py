"""Progressive math equation verifier with context accumulation.

Unlike ``MathEquationVerifier``, which treats each equation in isolation, this
verifier builds a ``SymbolContext`` as it processes a paper's equations in
order.  Domain assumptions, definitions, and constraints recorded from earlier
equations are available to verify later ones (e.g., "x∈R" → "x>0" → "√(x²)=x").

The verifier uses four verification layers (symbolic, dimensional, side-condition,
numeric) and classifies each statement as a declaration, constraint, or derivation.

Thread safety: the orchestrator may call ``verify()`` from multiple threads for
the same paper.  A ``threading.Lock`` per paper serialises access so the context
is updated in a safe, deterministic order.  ``num_workers=1`` is recommended for
fully deterministic results.
"""

from __future__ import annotations

import json
import threading
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
from src.verifiers.progressive.context_graph import (
    add_assumption_to_context,
    add_declaration_to_context,
    add_definition_to_context,
    build_sympy_assumptions,
    resolve_pending_obligations,
    summarize_context,
)
from src.verifiers.progressive.layers import run_verification_layers
from src.verifiers.progressive.models import (
    Assumption,
    LayerResult,
    ProofObligation,
    StatementClass,
    SymbolContext,
    SymbolDeclaration,
    SymbolDomain,
)


# ---------------------------------------------------------------------------
# Extended verdict harness
# ---------------------------------------------------------------------------
# Same core logic as MathEquationVerifier._VERDICT_HARNESS, with added helpers
# for assumption recording and conditional validity.

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
    syms = sorted(syms, key=lambda s: s.name)
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
        if val != val or abs(val) in (float("inf"),):
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
            _emit("INVALID_NUMERIC", residual=str(simplify(residual)))
            return
        frac, n_eval, max_abs = _sample_nonzero_fraction(residual, free)
        _emit(
            "SYMBOLIC_NONZERO",
            residual=str(residual),
            free=sorted(s.name for s in free),
            nonzero_fraction=frac,
            n_evaluated=n_eval,
            max_abs=max_abs,
        )
    except Exception as exc:
        _emit("HARNESS_ERROR", error=str(exc))


def report_unverifiable(reason="not symbolically decidable"):
    _emit("UNVERIFIABLE", reason=str(reason))


def report_valid(reason="verified"):
    _emit("VALID", reason=str(reason))


def report_assumption_added(symbols, description=""):
    """Record that non-verifiable assumptions/declarations were extracted."""
    _emit("ASSUMPTION_ADDED", symbols=list(symbols) if isinstance(symbols, (list, tuple)) else [str(symbols)],
          description=str(description))


def report_definition_added(symbol, definition=""):
    """Record that a definition (e.g. B = Q/N) was registered."""
    _emit("DEFINITION_ADDED", symbol=str(symbol), definition=str(definition))


def report_conditional_valid(condition=""):
    """Report that the equation is valid provided a condition holds."""
    _emit("CONDITIONAL_VALID", condition=str(condition))
'''


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_PROGRESSIVE_SYSTEM_PROMPT = """You are a mathematical verification expert. You classify a LaTeX equation and generate SymPy code that integrates with a deterministic harness.

A helper module is ALREADY imported for you. You MUST finish by calling exactly ONE of:
  - report(residual)            — for checkable identities/derivations
  - report_assumption_added(symbols, description) — for uncheckable declarations like "let x∈R"
  - report_definition_added(symbol, definition)   — for symbol definitions like "B = Q/N"
  - report_conditional_valid(condition)            — for identities valid only under a condition
  - report_unverifiable(reason)                     — when the equation cannot be checked

## Statement classification (choose ONE)

1. **uncheckable_declaration**: The equation introduces a symbol, declares a domain
   (e.g., x∈R, x>0), or defines a quantity (B = Q/N). There is NO truth value to
   check — it's a statement of context. Call report_assumption_added() or
   report_definition_added().

2. **checkable_constraint**: The equation states a relation that holds under
   specific conditions (e.g., s²+t²=1 on the unit circle). Call report(residual)
   as usual; the harness will classify it. Set symbolic_constraint=true in the
   JSON output.

3. **checkable_derivation**: The equation claims an identity derivable from prior
   definitions/assumptions. Call report(simplify(LHS - RHS)).

## Output Format

Return a JSON object with exactly these fields:
```json
{
  "statement_class": "uncheckable_declaration | checkable_constraint | checkable_derivation",
  "sympy_code": "from sympy import *\\nx = Symbol('x', real=True)\\nlhs = ...\\nrhs = ...\\nreport(simplify(lhs - rhs))",
  "symbols_introduced": ["x"],
  "symbol_domains": {"x": "real"},
  "defines_symbols": {},
  "assumptions_added": ["x ∈ ℝ"],
  "depends_on_symbols": ["a", "b"],
  "explanation": "Brief explanation of classification"
}
```

Important:
- `sympy_code` must contain ONLY executable Python (the harness is already imported).
  It must define `lhs` and `rhs` and call one of the report helpers.
- `symbols_introduced`: new symbols this equation declares/defines.
- `symbol_domains`: map symbol → domain hint (real, integer, positive_real, complex, matrix, unknown).
- `defines_symbols`: map symbol → RHS expression for definitions like B = Q/N.
- `depends_on_symbols`: symbols this equation references that were presumably declared earlier.
- `assumptions_added`: list of human-readable assumption descriptions.
"""


# Confidence constants (same as MathEquationVerifier)
_CONF_NUMERIC_INVALID = 0.9
_CONF_IDENTITY_INVALID = 0.75
_CONF_VALID = 0.85
_CONF_LAYER_PROVED = 0.75
_CONF_LAYER_DISPROVED = 0.7


class ProgressiveMathVerifier(BaseVerifier):
    """Verifies mathematical equations with progressive context accumulation.

    Builds a SymbolContext per paper as equations are processed in order.
    Uses four verification layers (symbolic → dimensional → side-condition →
    numeric) to check checkable derivations and constraints.
    """

    name: str = "progressive_math"

    # Per-paper mutable state
    _contexts: dict[str, SymbolContext] = {}
    _context_locks: dict[str, threading.Lock] = {}

    # ------------------------------------------------------------------
    # verify() — main entry point
    # ------------------------------------------------------------------

    def verify(
        self,
        snippet: VerificationSnippet,
    ) -> EquationVerificationResult:
        start_time = time.monotonic()

        if not self.can_verify(snippet):
            return EquationVerificationResult(
                snippet_id=snippet.snippet_id,
                verifier_name=self.name,
                status=VerificationStatus.SKIPPED,
                execution_time_ms=(time.monotonic() - start_time) * 1000,
            )

        paper_id = snippet.paper_id
        lock = self._get_lock(paper_id)

        with lock:
            ctx = self._get_context(paper_id)
            ctx.record_order(snippet.snippet_id)

            latex = snippet.metadata.get("latex", snippet.content)

            # Step 1 — classify + generate code via LLM (context-aware)
            prompt = self._build_prompt(latex, snippet, ctx)

            try:
                llm_response = self._call_llm(
                    prompt=prompt,
                    system_prompt=_PROGRESSIVE_SYSTEM_PROMPT,
                )
                parsed = parse_json_response(llm_response)
            except Exception as exc:
                logger.error(f"LLM call failed for {snippet.snippet_id}: {exc}")
                return EquationVerificationResult(
                    snippet_id=snippet.snippet_id,
                    verifier_name=self.name,
                    status=VerificationStatus.UNVERIFIABLE,
                    reasoning=f"LLM call failed: {exc}",
                    execution_time_ms=(time.monotonic() - start_time) * 1000,
                )

            statement_class = str(
                parsed.get("statement_class", "checkable_derivation")
            ).strip().lower()
            sympy_code = parsed.get("sympy_code", "")
            explanation = parsed.get("explanation", "")

            # Step 2 — record symbol information from the LLM classification
            self._record_symbol_info(parsed, snippet, ctx)

            # Step 3 — dispatch by statement class
            if statement_class == StatementClass.UNCHECKABLE_DECLARATION.value:
                result = self._handle_declaration(
                    snippet, parsed, ctx, start_time,
                )
            elif statement_class == StatementClass.CHECKABLE_CONSTRAINT.value:
                result = self._handle_derivation_or_constraint(
                    snippet, parsed, sympy_code, ctx, start_time,
                    is_constraint=True,
                )
            else:
                result = self._handle_derivation_or_constraint(
                    snippet, parsed, sympy_code, ctx, start_time,
                    is_constraint=False,
                )

            # Step 4 — resolve any pending obligations that are now satisfiable
            newly_resolved = resolve_pending_obligations(ctx)
            if newly_resolved:
                logger.debug(
                    f"Resolved {len(newly_resolved)} pending obligations "
                    f"for paper {paper_id}"
                )

            return result

    # ------------------------------------------------------------------
    # Statement-class handlers
    # ------------------------------------------------------------------

    def _handle_declaration(
        self,
        snippet: VerificationSnippet,
        parsed: dict,
        ctx: SymbolContext,
        start_time: float,
    ) -> EquationVerificationResult:
        """Handle an uncheckable declaration — record it in the context."""
        symbols = parsed.get("symbols_introduced", [])
        domains = parsed.get("symbol_domains", {})
        defines = parsed.get("defines_symbols", {})
        assumptions = parsed.get("assumptions_added", [])
        explanation = parsed.get("explanation", "")

        # Record each symbol
        for sym in symbols:
            domain_name = domains.get(sym, "unknown")
            try:
                domain = SymbolDomain(domain_name)
            except ValueError:
                domain = SymbolDomain.UNKNOWN

            decl = SymbolDeclaration(
                symbol=sym,
                domain=domain,
                source_snippet_id=snippet.snippet_id,
                source_location=snippet.location,
                first_mentioned_at=ctx.equation_order.get(snippet.snippet_id, 0),
            )

            # Build SymPy assumptions from domain
            from src.verifiers.progressive.context_graph import DOMAIN_IMPLICATIONS
            implied = DOMAIN_IMPLICATIONS.get(domain, {})
            decl.sympy_assumptions = dict(implied)

            add_declaration_to_context(ctx, decl)
            logger.debug(f"Recorded symbol: {sym} (domain={domain.value})")

        # Record definitions
        for sym, rhs in defines.items():
            add_definition_to_context(ctx, sym, rhs, snippet.snippet_id)
            logger.debug(f"Recorded definition: {sym} = {rhs}")

        # Record assumptions
        for desc in assumptions:
            assumption = Assumption(
                snippet_id=snippet.snippet_id,
                latex=snippet.metadata.get("latex", ""),
                description=str(desc),
                symbols=symbols,
                assumption_type="domain",
            )
            add_assumption_to_context(ctx, assumption)

        detail_parts = []
        if symbols:
            detail_parts.append(f"declared symbols: {', '.join(symbols)}")
        if defines:
            detail_parts.append(
                f"definitions: {', '.join(f'{k}={v}' for k, v in defines.items())}"
            )

        return EquationVerificationResult(
            snippet_id=snippet.snippet_id,
            verifier_name=self.name,
            status=VerificationStatus.VALID,
            error_detected=False,
            confidence=1.0,
            reasoning=(
                f"Uncheckable declaration — {explanation}"
                if explanation
                else f"Recorded context: {'; '.join(detail_parts)}"
            ),
            statement_class=StatementClass.UNCHECKABLE_DECLARATION.value,
            sympy_code=parsed.get("sympy_code"),
            verification_layer="context",
            context_snapshot={"symbols_declared": len(ctx.declarations)},
            execution_time_ms=(time.monotonic() - start_time) * 1000,
        )

    def _handle_derivation_or_constraint(
        self,
        snippet: VerificationSnippet,
        parsed: dict,
        sympy_code: str,
        ctx: SymbolContext,
        start_time: float,
        is_constraint: bool = False,
    ) -> EquationVerificationResult:
        """Handle a checkable derivation or constraint → run sandbox + layers."""

        if not sympy_code:
            return EquationVerificationResult(
                snippet_id=snippet.snippet_id,
                verifier_name=self.name,
                status=VerificationStatus.MALFORMED,
                reasoning="LLM did not produce SymPy code for checkable statement.",
                statement_class=(
                    StatementClass.CHECKABLE_CONSTRAINT.value if is_constraint
                    else StatementClass.CHECKABLE_DERIVATION.value
                ),
                execution_time_ms=(time.monotonic() - start_time) * 1000,
            )

        stmt_class = (
            StatementClass.CHECKABLE_CONSTRAINT.value if is_constraint
            else StatementClass.CHECKABLE_DERIVATION.value
        )

        # Step A — run the harness in the sandbox
        harness_verdict, harness_stdout, harness_stderr, harness_rc = (
            self._run_harness(sympy_code)
        )

        # Step B — decide from the harness verdict
        if harness_verdict is not None:
            verdict = harness_verdict.get("verdict")

            if verdict == "VALID":
                return EquationVerificationResult(
                    snippet_id=snippet.snippet_id,
                    verifier_name=self.name,
                    status=VerificationStatus.VALID,
                    error_detected=False,
                    confidence=_CONF_VALID,
                    reasoning="Residual simplifies to zero — identity verified.",
                    statement_class=stmt_class,
                    sympy_code=sympy_code,
                    execution_output=harness_stdout.strip(),
                    execution_error=harness_stderr.strip() if harness_stderr else None,
                    return_code=harness_rc,
                    verification_layer="symbolic",
                    context_snapshot={"symbols_declared": len(ctx.declarations)},
                    execution_time_ms=(time.monotonic() - start_time) * 1000,
                )

            if verdict == "INVALID_NUMERIC":
                return EquationVerificationResult(
                    snippet_id=snippet.snippet_id,
                    verifier_name=self.name,
                    status=VerificationStatus.INVALID,
                    error_detected=True,
                    confidence=_CONF_NUMERIC_INVALID,
                    reasoning=(
                        f"Equation is numerically false: "
                        f"LHS - RHS = {harness_verdict.get('residual')} (no free symbols)."
                    ),
                    predicted_error_category="Equation / proof",
                    statement_class=stmt_class,
                    sympy_code=sympy_code,
                    execution_output=harness_stdout.strip(),
                    execution_error=harness_stderr.strip() if harness_stderr else None,
                    return_code=harness_rc,
                    verification_layer="symbolic",
                    context_snapshot={"symbols_declared": len(ctx.declarations)},
                    execution_time_ms=(time.monotonic() - start_time) * 1000,
                )

            if verdict == "CONDITIONAL_VALID":
                condition = harness_verdict.get("condition", "")
                ob = ProofObligation(
                    obligation_id=f"{snippet.snippet_id}_cond",
                    condition=condition,
                    source_snippet_id=snippet.snippet_id,
                    depends_on_symbols=parsed.get("depends_on_symbols", []),
                )
                ctx.add_obligation(ob)
                return EquationVerificationResult(
                    snippet_id=snippet.snippet_id,
                    verifier_name=self.name,
                    status=VerificationStatus.VALID,
                    error_detected=False,
                    confidence=0.7,
                    reasoning=f"Valid provided condition holds: {condition}",
                    statement_class=stmt_class,
                    proof_obligations=[ob.model_dump()],
                    sympy_code=sympy_code,
                    execution_output=harness_stdout.strip(),
                    return_code=harness_rc,
                    verification_layer="symbolic",
                    execution_time_ms=(time.monotonic() - start_time) * 1000,
                )

            if verdict == "SYMBOLIC_NONZERO":
                frac = float(harness_verdict.get("nonzero_fraction", 0.0) or 0.0)
                n_eval = int(harness_verdict.get("n_evaluated", 0) or 0)
                residual = harness_verdict.get("residual")
                free = harness_verdict.get("free")

                # Only escalate to INVALID for unconditional identities that fail
                # numeric sampling at every point. Constraints may be non-zero
                # everywhere and still correct.
                if (
                    not is_constraint
                    and n_eval >= 5
                    and frac >= 0.999
                ):
                    return EquationVerificationResult(
                        snippet_id=snippet.snippet_id,
                        verifier_name=self.name,
                        status=VerificationStatus.INVALID,
                        error_detected=True,
                        confidence=_CONF_IDENTITY_INVALID,
                        reasoning=(
                            f"Claimed identity fails: residual {residual} is non-zero "
                            f"at all {n_eval} sampled points (free symbols {free})."
                        ),
                        predicted_error_category="Equation / proof",
                        statement_class=stmt_class,
                        sympy_code=sympy_code,
                        execution_output=harness_stdout.strip(),
                        execution_error=harness_stderr.strip() if harness_stderr else None,
                        return_code=harness_rc,
                        verification_layer="numeric",
                        context_snapshot={"symbols_declared": len(ctx.declarations)},
                        execution_time_ms=(time.monotonic() - start_time) * 1000,
                    )

                # Not a definitive falsification → fall through to layers
                pass

            if verdict == "UNVERIFIABLE":
                # Fall through to layers
                pass

            if verdict == "HARNESS_ERROR":
                return EquationVerificationResult(
                    snippet_id=snippet.snippet_id,
                    verifier_name=self.name,
                    status=VerificationStatus.MALFORMED,
                    reasoning=f"Harness error: {harness_verdict.get('error')}",
                    sympy_code=sympy_code,
                    execution_time_ms=(time.monotonic() - start_time) * 1000,
                )

        # Step C — harness was inconclusive; run additional layers
        return self._run_layers_and_decide(
            snippet, parsed, sympy_code, ctx, start_time, is_constraint,
        )

    def _run_layers_and_decide(
        self,
        snippet: VerificationSnippet,
        parsed: dict,
        sympy_code: str,
        ctx: SymbolContext,
        start_time: float,
        is_constraint: bool,
    ) -> EquationVerificationResult:
        """Run verification layers and decide the final status."""

        stmt_class = (
            StatementClass.CHECKABLE_CONSTRAINT.value if is_constraint
            else StatementClass.CHECKABLE_DERIVATION.value
        )

        try:
            layer_results = run_verification_layers(
                sympy_code=sympy_code,
                ctx=ctx,
                python_executable=self.config.sandbox.python_executable,
            )
        except Exception as exc:
            logger.error(f"Layer execution failed: {exc}")
            return EquationVerificationResult(
                snippet_id=snippet.snippet_id,
                verifier_name=self.name,
                status=VerificationStatus.UNVERIFIABLE,
                reasoning=f"Verification layer error: {exc}",
                statement_class=stmt_class,
                sympy_code=sympy_code,
                execution_time_ms=(time.monotonic() - start_time) * 1000,
            )

        # Check if any layer produced a definitive result
        proved_layer = None
        disproved_layer = None
        layer_details: list[str] = []

        proof_obligations: list[dict] = []

        for lr in layer_results:
            layer_details.append(f"[{lr.layer}] {lr.status}: {lr.details}")
            if lr.status == "proved" and proved_layer is None:
                proved_layer = lr
            if lr.status == "disproved" and disproved_layer is None:
                disproved_layer = lr
            if lr.condition:
                ob = ProofObligation(
                    obligation_id=f"{snippet.snippet_id}_{lr.layer}",
                    condition=lr.condition,
                    source_snippet_id=snippet.snippet_id,
                    depends_on_symbols=lr.free_symbols,
                )
                ctx.add_obligation(ob)
                proof_obligations.append(ob.model_dump())

        # If both dimensional and numeric say it's false, mark INVALID
        if disproved_layer is not None:
            return EquationVerificationResult(
                snippet_id=snippet.snippet_id,
                verifier_name=self.name,
                status=VerificationStatus.INVALID,
                error_detected=True,
                confidence=_CONF_LAYER_DISPROVED,
                reasoning=f"Disproved by {disproved_layer.layer} layer: {disproved_layer.details}",
                predicted_error_category="Equation / proof",
                statement_class=stmt_class,
                proof_obligations=proof_obligations,
                sympy_code=sympy_code,
                verification_layer=disproved_layer.layer,
                context_snapshot={"symbols_declared": len(ctx.declarations)},
                execution_time_ms=(time.monotonic() - start_time) * 1000,
            )

        # If a layer proved it while harness was inconclusive
        if proved_layer is not None:
            return EquationVerificationResult(
                snippet_id=snippet.snippet_id,
                verifier_name=self.name,
                status=VerificationStatus.VALID,
                error_detected=False,
                confidence=_CONF_LAYER_PROVED,
                reasoning=f"Proved by {proved_layer.layer} layer: {proved_layer.details}",
                statement_class=stmt_class,
                proof_obligations=proof_obligations,
                sympy_code=sympy_code,
                verification_layer=proved_layer.layer,
                context_snapshot={"symbols_declared": len(ctx.declarations)},
                execution_time_ms=(time.monotonic() - start_time) * 1000,
            )

        # Fully inconclusive
        return EquationVerificationResult(
            snippet_id=snippet.snippet_id,
            verifier_name=self.name,
            status=VerificationStatus.UNVERIFIABLE,
            reasoning=f"Inconclusive after {len(layer_results)} layer(s): {'; '.join(layer_details)}",
            statement_class=stmt_class,
            proof_obligations=proof_obligations,
            sympy_code=sympy_code,
            verification_layer=",".join(lr.layer for lr in layer_results),
            context_snapshot={"symbols_declared": len(ctx.declarations)},
            execution_time_ms=(time.monotonic() - start_time) * 1000,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def can_verify(self, snippet: VerificationSnippet) -> bool:
        """Check if the snippet contains a verifiable equation."""
        if not self.verifier_config.enabled:
            return False
        latex = snippet.metadata.get("latex", "")
        return bool(latex and len(latex) > 2)

    def _build_prompt(
        self, latex: str, snippet: VerificationSnippet, ctx: SymbolContext,
    ) -> str:
        """Build the context-aware LLM prompt."""
        context_summary = summarize_context(ctx)
        parts = [
            "Classify the following LaTeX equation and generate SymPy verification code.",
            "Use the accumulated context below to inform your classification.",
            "",
            "## Accumulated Paper Context",
            context_summary,
            "",
            "## Equation to Process",
            latex,
        ]

        surrounding = snippet.content
        if surrounding:
            parts.extend([
                "",
                "## Surrounding Text (for context)",
                surrounding[:1500],
            ])

        parts.extend([
            "",
            "Generate ONLY the JSON response with statement_class, sympy_code, "
            "symbols_introduced, symbol_domains, defines_symbols, "
            "assumptions_added, depends_on_symbols, and explanation fields.",
        ])

        return "\n".join(parts)

    def _record_symbol_info(
        self,
        parsed: dict,
        snippet: VerificationSnippet,
        ctx: SymbolContext,
    ) -> None:
        """Record symbol declarations, definitions, and assumptions from the
        LLM classification into the context, even for derivations (which may
        reference previously-undeclared symbols)."""
        symbols = parsed.get("symbols_introduced", [])
        domains = parsed.get("symbol_domains", {})
        defines = parsed.get("defines_symbols", {})
        assumptions = parsed.get("assumptions_added", [])
        depends_on = parsed.get("depends_on_symbols", [])

        for sym in symbols:
            if sym in ctx.declarations:
                continue  # already known
            domain_name = domains.get(sym, "unknown")
            try:
                domain = SymbolDomain(domain_name)
            except ValueError:
                domain = SymbolDomain.UNKNOWN

            from src.verifiers.progressive.context_graph import DOMAIN_IMPLICATIONS
            decl = SymbolDeclaration(
                symbol=sym,
                domain=domain,
                sympy_assumptions=dict(DOMAIN_IMPLICATIONS.get(domain, {})),
                source_snippet_id=snippet.snippet_id,
                source_location=snippet.location,
                first_mentioned_at=ctx.equation_order.get(snippet.snippet_id, 0),
            )
            add_declaration_to_context(ctx, decl)

        for sym, rhs in defines.items():
            if sym not in ctx.definitions:
                add_definition_to_context(ctx, sym, rhs, snippet.snippet_id)

        for desc in assumptions:
            assumption = Assumption(
                snippet_id=snippet.snippet_id,
                latex=snippet.metadata.get("latex", ""),
                description=str(desc),
                symbols=symbols,
                assumption_type="domain",
            )
            add_assumption_to_context(ctx, assumption)

        # Record dependency edges
        for dep_sym in depends_on:
            source = trace_symbol_to_source(ctx, dep_sym)
            if source:
                ctx.add_edge(snippet.snippet_id, source, relation="uses_symbol")

    def _run_harness(self, sympy_code: str) -> tuple[Optional[dict], str, str, int]:
        """Run the sandbox with the harness + user code.

        Returns (verdict_dict | None, stdout, stderr, returncode).
        """
        full_code = _VERDICT_HARNESS + "\n\n" + sympy_code
        try:
            stdout, stderr, returncode = run_sympy_sandbox(
                code=full_code,
                python_executable=self.config.sandbox.python_executable,
                timeout_seconds=self.config.sandbox.timeout_seconds,
                max_output_bytes=self.config.sandbox.max_output_bytes,
            )
        except SandboxTimeoutError:
            return None, "", "Sandbox timeout", -1
        except SandboxError as exc:
            return None, "", str(exc), -1

        if returncode != 0:
            return None, stdout, stderr, returncode

        # Parse the VERDICT: line
        for line in stdout.splitlines():
            line = line.strip()
            if line.startswith("VERDICT:"):
                try:
                    return json.loads(line[len("VERDICT:"):]), stdout, stderr, returncode
                except json.JSONDecodeError:
                    continue

        return None, stdout, stderr, returncode

    def _get_context(self, paper_id: str) -> SymbolContext:
        """Get or create the per-paper SymbolContext."""
        if paper_id not in self._contexts:
            self._contexts[paper_id] = SymbolContext(paper_id=paper_id)
        return self._contexts[paper_id]

    def _get_lock(self, paper_id: str) -> threading.Lock:
        """Get or create the per-paper lock."""
        if paper_id not in self._context_locks:
            self._context_locks[paper_id] = threading.Lock()
        return self._context_locks[paper_id]

    def cleanup_paper(self, paper_id: str) -> None:
        """Release per-paper state after verification completes."""
        self._contexts.pop(paper_id, None)
        self._context_locks.pop(paper_id, None)
        logger.debug(f"Cleaned up context for paper {paper_id}")


# Re-export for convenience
def trace_symbol_to_source(ctx: SymbolContext, symbol: str) -> Optional[str]:
    """Find the snippet where a symbol was first declared."""
    from src.verifiers.progressive.context_graph import trace_symbol_to_source as _ts
    return _ts(ctx, symbol)
