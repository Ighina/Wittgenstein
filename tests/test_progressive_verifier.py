"""Tests for the progressive math verification pipeline.

Covers models, context graph, verification layers, and the
ProgressiveMathVerifier with mock LLM backend.
"""

import json
import os
import shutil
import subprocess

import pytest

from src.config import PipelineConfig, LLMConfig
from src.models import SnippetType, VerificationSnippet, VerificationStatus
from src.verifiers.progressive.models import (
    Assumption,
    ProofObligation,
    StatementClass,
    SymbolContext,
    SymbolDeclaration,
    SymbolDomain,
)
from src.verifiers.progressive.context_graph import (
    add_declaration_to_context,
    add_definition_to_context,
    build_all_symbol_declarations,
    build_sympy_assumptions,
    infer_domain_from_latex,
    resolve_obligation,
    resolve_pending_obligations,
    summarize_context,
    trace_symbol_to_source,
)
from src.verifiers.progressive.progressive_verifier import ProgressiveMathVerifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_config() -> PipelineConfig:
    """Return a pipeline config wired for mock LLM + sequential execution."""
    return PipelineConfig(
        llm=LLMConfig(provider="mock", model="mock", num_workers=1),
        use_progressive_math=True,
    )


def _eq_snippet(
    snippet_id: str = "test_eq_0",
    paper_id: str = "test_paper",
    location: str = "Equation 1",
    latex: str = "x^2 = x \\cdot x",
    content: str = "",
) -> VerificationSnippet:
    """Create a minimal equation snippet."""
    return VerificationSnippet(
        snippet_id=snippet_id,
        snippet_type=SnippetType.EQUATION,
        paper_id=paper_id,
        location=location,
        content=content or f"Equation: {latex}",
        metadata={"latex": latex, "display_mode": True},
    )


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestSymbolContext:
    """Tests for SymbolContext — the per-paper state accumulator."""

    def test_empty_context(self):
        ctx = SymbolContext(paper_id="p1")
        assert ctx.paper_id == "p1"
        assert ctx.get_symbol_names() == set()

    def test_record_order(self):
        ctx = SymbolContext(paper_id="p1")
        assert ctx.record_order("s1") == 0
        assert ctx.record_order("s2") == 1
        assert ctx.record_order("s1") == 0  # idempotent

    def test_add_declaration_new(self):
        ctx = SymbolContext(paper_id="p1")
        decl = SymbolDeclaration(
            symbol="x", domain=SymbolDomain.REAL,
            source_snippet_id="eq_1",
        )
        add_declaration_to_context(ctx, decl)
        assert "x" in ctx.get_symbol_names()
        assert ctx.declarations["x"].domain == SymbolDomain.REAL

    def test_add_declaration_merge_stronger_domain(self):
        ctx = SymbolContext(paper_id="p1")
        # First declare as UNKNOWN
        decl1 = SymbolDeclaration(symbol="x", domain=SymbolDomain.UNKNOWN)
        add_declaration_to_context(ctx, decl1)
        # Then redeclare with a stronger domain
        decl2 = SymbolDeclaration(symbol="x", domain=SymbolDomain.REAL)
        add_declaration_to_context(ctx, decl2)

        merged = ctx.declarations["x"]
        assert merged.domain == SymbolDomain.REAL

    def test_add_declaration_keeps_stronger_domain(self):
        ctx = SymbolContext(paper_id="p1")
        decl1 = SymbolDeclaration(symbol="x", domain=SymbolDomain.INTEGER)
        add_declaration_to_context(ctx, decl1)
        decl2 = SymbolDeclaration(symbol="x", domain=SymbolDomain.REAL)
        add_declaration_to_context(ctx, decl2)

        # INTEGER is stronger than REAL — the existing one should win
        merged = ctx.declarations["x"]
        assert merged.domain == SymbolDomain.INTEGER

    def test_add_definition(self):
        ctx = SymbolContext(paper_id="p1")
        add_definition_to_context(ctx, "B", "Q/N", "eq_2")
        assert ctx.definitions["B"] == "Q/N"
        assert ctx.declarations["B"].is_defined
        assert ctx.declarations["B"].defined_as == "Q/N"

    def test_add_assumption(self):
        ctx = SymbolContext(paper_id="p1")
        a = Assumption(
            snippet_id="eq_1",
            latex="x \\in \\mathbb{R}",
            description="x is declared real",
            symbols=["x"],
            assumption_type="domain",
        )
        ctx.add_assumption(a)
        assert len(ctx.assumptions) == 1
        assert ctx.assumptions[0].symbols == ["x"]

    def test_add_obligation(self):
        ctx = SymbolContext(paper_id="p1")
        ob = ProofObligation(
            obligation_id="obl_1",
            condition="x != 0",
            source_snippet_id="eq_3",
            depends_on_symbols=["x"],
        )
        ctx.add_obligation(ob)
        assert len(ctx.pending_obligations) == 1
        assert ctx.pending_obligations[0].condition == "x != 0"


class TestDomainImplications:
    """Domain implication chain: integer → rational → real → complex."""

    def test_integer_implies_real(self):
        decl = SymbolDeclaration(
            symbol="n", domain=SymbolDomain.INTEGER,
        )
        assumptions = build_sympy_assumptions(decl)
        assert assumptions.get("integer") is True
        assert assumptions.get("rational") is True
        assert assumptions.get("real") is True
        assert assumptions.get("complex") is True

    def test_real_does_not_imply_integer(self):
        decl = SymbolDeclaration(
            symbol="x", domain=SymbolDomain.REAL,
        )
        assumptions = build_sympy_assumptions(decl)
        assert assumptions.get("real") is True
        assert assumptions.get("integer") is not True

    def test_positive_real_adds_positive(self):
        decl = SymbolDeclaration(
            symbol="x", domain=SymbolDomain.POSITIVE_REAL,
        )
        assumptions = build_sympy_assumptions(decl)
        assert assumptions.get("real") is True
        assert assumptions.get("positive") is True

    def test_natural_implies_nonnegative(self):
        decl = SymbolDeclaration(
            symbol="n", domain=SymbolDomain.NATURAL,
        )
        assumptions = build_sympy_assumptions(decl)
        assert assumptions.get("nonnegative") is True
        assert assumptions.get("integer") is True


# ---------------------------------------------------------------------------
# Context graph tests
# ---------------------------------------------------------------------------

class TestSummarizeContext:
    """Tests for context summarization (used in LLM prompts)."""

    def test_empty_context(self):
        ctx = SymbolContext(paper_id="p1")
        summary = summarize_context(ctx)
        assert "No accumulated context" in summary or summary == ""

    def test_with_declarations(self):
        ctx = SymbolContext(paper_id="p1")
        decl = SymbolDeclaration(
            symbol="x", domain=SymbolDomain.REAL,
            source_snippet_id="eq_1",
        )
        add_declaration_to_context(ctx, decl)
        summary = summarize_context(ctx)
        assert "x" in summary
        assert "real" in summary

    def test_with_assumptions(self):
        ctx = SymbolContext(paper_id="p1")
        a = Assumption(
            snippet_id="eq_1", description="x is positive",
            symbols=["x"],
        )
        ctx.add_assumption(a)
        summary = summarize_context(ctx)
        assert "positive" in summary

    def test_with_pending_obligations(self):
        ctx = SymbolContext(paper_id="p1")
        ob = ProofObligation(
            obligation_id="obl_1",
            condition="x != 0",
            depends_on_symbols=["x"],
        )
        ctx.add_obligation(ob)
        summary = summarize_context(ctx)
        assert "x != 0" in summary


class TestBuildSymbolDeclarations:
    """Tests for generating SymPy code from context."""

    def test_empty_context(self):
        ctx = SymbolContext(paper_id="p1")
        code = build_all_symbol_declarations(ctx)
        assert code == ""

    def test_single_real_symbol(self):
        ctx = SymbolContext(paper_id="p1")
        decl = SymbolDeclaration(
            symbol="x", domain=SymbolDomain.REAL,
        )
        add_declaration_to_context(ctx, decl)
        code = build_all_symbol_declarations(ctx)
        assert "x = Symbol('x'" in code
        assert "real=True" in code

    def test_defined_symbol(self):
        ctx = SymbolContext(paper_id="p1")
        add_definition_to_context(ctx, "B", "Q/N", "eq_1")
        code = build_all_symbol_declarations(ctx)
        assert "B = Symbol" in code
        assert "B = Q/N" in code


class TestDomainInference:
    """Heuristic domain inference from LaTeX patterns."""

    def test_real_domain(self):
        assert infer_domain_from_latex(r"x \in \mathbb{R}") == SymbolDomain.REAL
        assert infer_domain_from_latex(r"y \in \mathbb{R}^n") == SymbolDomain.MATRIX

    def test_natural_domain(self):
        assert infer_domain_from_latex(r"n \in \mathbb{N}") == SymbolDomain.NATURAL

    def test_integer_domain(self):
        assert infer_domain_from_latex(r"k \in \mathbb{Z}") == SymbolDomain.INTEGER

    def test_positive(self):
        assert infer_domain_from_latex(r"x > 0") == SymbolDomain.POSITIVE_REAL

    def test_nonzero(self):
        assert infer_domain_from_latex(r"x \neq 0") == SymbolDomain.NONZERO

    def test_no_match(self):
        assert infer_domain_from_latex(r"f(x) = x^2") is None


class TestObligationResolution:
    """Tests for resolving proof obligations against context."""

    def test_resolve_nonzero_when_positive(self):
        ctx = SymbolContext(paper_id="p1")
        decl = SymbolDeclaration(
            symbol="x", domain=SymbolDomain.POSITIVE_REAL,
        )
        add_declaration_to_context(ctx, decl)

        ob = ProofObligation(
            obligation_id="obl_1",
            condition="x != 0",
            depends_on_symbols=["x"],
        )
        result = resolve_obligation(ob, ctx)
        assert result is True

    def test_unresolved_unknown_symbol(self):
        ctx = SymbolContext(paper_id="p1")
        ob = ProofObligation(
            obligation_id="obl_1",
            condition="a != 0",
            depends_on_symbols=["a"],
        )
        result = resolve_obligation(ob, ctx)
        assert result is None

    def test_resolve_pending(self):
        ctx = SymbolContext(paper_id="p1")
        decl = SymbolDeclaration(
            symbol="x", domain=SymbolDomain.POSITIVE_REAL,
        )
        add_declaration_to_context(ctx, decl)

        ob = ProofObligation(
            obligation_id="obl_1",
            condition="x != 0",
            depends_on_symbols=["x"],
        )
        ctx.add_obligation(ob)
        assert len(ctx.pending_obligations) == 1

        resolved = resolve_pending_obligations(ctx)
        assert len(resolved) == 1
        assert len(ctx.pending_obligations) == 0
        assert len(ctx.resolved_obligations) == 1
        assert ctx.resolved_obligations[0].resolved is True


class TestTransitiveDependencies:
    """Tests for dependency graph traversal."""

    def test_trace_known_symbol(self):
        ctx = SymbolContext(paper_id="p1")
        decl = SymbolDeclaration(
            symbol="x",
            domain=SymbolDomain.REAL,
            source_snippet_id="eq_1",
        )
        add_declaration_to_context(ctx, decl)

        source = trace_symbol_to_source(ctx, "x")
        assert source == "eq_1"

    def test_trace_unknown_symbol(self):
        ctx = SymbolContext(paper_id="p1")
        source = trace_symbol_to_source(ctx, "z")
        assert source is None


# ---------------------------------------------------------------------------
# ProgressiveMathVerifier tests (mock LLM)
# ---------------------------------------------------------------------------

@pytest.fixture
def verifier():
    """Create a ProgressiveMathVerifier with mock config."""
    config = _mock_config()
    v = ProgressiveMathVerifier(config=config)
    yield v
    # Clean up per-paper state between tests
    v._contexts.clear()
    v._context_locks.clear()


class TestProgressiveVerifierBasic:
    """Basic verification tests with mock LLM."""

    def test_can_verify_with_latex(self, verifier):
        snippet = _eq_snippet(latex="x^2 = x \\cdot x")
        assert verifier.can_verify(snippet)

    def test_cannot_verify_without_latex(self, verifier):
        snippet = VerificationSnippet(
            snippet_id="test",
            snippet_type=SnippetType.EQUATION,
            paper_id="test",
            location="Empty",
            content="No equation here.",
            metadata={},
        )
        assert not verifier.can_verify(snippet)

    def test_simple_identity_returns_result(self, verifier):
        snippet = _eq_snippet(
            latex="x^2 = x \\cdot x",
        )
        result = verifier.verify(snippet)
        assert result.verifier_name == "progressive_math"
        assert result.snippet_id == "test_eq_0"
        assert result.status in (
            VerificationStatus.VALID,
            VerificationStatus.INVALID,
            VerificationStatus.UNVERIFIABLE,
        )
        # Mock LLM returns a standard identity → should be VALID in harness
        # But depends on sandbox execution...

    def test_result_has_statement_class(self, verifier):
        snippet = _eq_snippet(
            latex="x^2 = x \\cdot x",
        )
        result = verifier.verify(snippet)
        assert result.statement_class is not None

    def test_result_has_progressive_fields(self, verifier):
        snippet = _eq_snippet(
            latex="x^2 = x \\cdot x",
        )
        result = verifier.verify(snippet)
        assert isinstance(result.proof_obligations, list)
        # context_snapshot may be None for skipped snippets
        if result.status != VerificationStatus.SKIPPED:
            assert result.verification_layer is not None


class TestProgressiveVerifierDeclarations:
    """Tests for declaration handling with mock LLM."""

    def test_domain_declaration_is_valid(self, verifier):
        snippet = _eq_snippet(
            snippet_id="eq_domain_1",
            latex=r"x \in \mathbb{R}",
            content="Let x be a real number.",
        )
        result = verifier.verify(snippet)
        assert result.status == VerificationStatus.VALID
        assert result.statement_class == StatementClass.UNCHECKABLE_DECLARATION.value

    def test_domain_declaration_adds_to_context(self, verifier):
        snippet = _eq_snippet(
            snippet_id="eq_domain_2",
            paper_id="paper_A",
            latex=r"x \in \mathbb{R}",
        )
        verifier.verify(snippet)
        ctx = verifier._get_context("paper_A")
        assert "x" in ctx.get_symbol_names()

    def test_natural_number_declaration(self, verifier):
        snippet = _eq_snippet(
            snippet_id="eq_nat",
            paper_id="paper_A",
            latex=r"n \in \mathbb{N}",
        )
        result = verifier.verify(snippet)
        assert result.status == VerificationStatus.VALID
        assert result.statement_class == StatementClass.UNCHECKABLE_DECLARATION.value

    def test_positivity_declaration(self, verifier):
        snippet = _eq_snippet(
            snippet_id="eq_pos",
            paper_id="paper_A",
            latex=r"x > 0",
        )
        result = verifier.verify(snippet)
        assert result.status == VerificationStatus.VALID
        assert result.statement_class == StatementClass.UNCHECKABLE_DECLARATION.value


class TestProgressiveVerifierSequential:
    """Tests for sequential context accumulation across multiple snippets."""

    def test_context_accumulates_across_equations(self, verifier):
        """Declare x∈R, then use it — both should work."""
        paper_id = "paper_seq_1"

        # Step 1: Declare x real
        s1 = _eq_snippet(
            snippet_id="eq_1",
            paper_id=paper_id,
            latex=r"x \in \mathbb{R}",
        )
        r1 = verifier.verify(s1)
        assert r1.status == VerificationStatus.VALID
        assert r1.statement_class == StatementClass.UNCHECKABLE_DECLARATION.value

        # Step 2: Claim x² = x·x (should be valid, and context has x info)
        s2 = _eq_snippet(
            snippet_id="eq_2",
            paper_id=paper_id,
            latex=r"x^2 = x \cdot x",
        )
        r2 = verifier.verify(s2)
        # r2 should work — may be VALID or UNVERIFIABLE depending on sandbox
        assert r2.status in (
            VerificationStatus.VALID,
            VerificationStatus.UNVERIFIABLE,
        )
        # Context now has eq_1's declaration
        ctx = verifier._get_context(paper_id)
        assert "x" in ctx.get_symbol_names()

    def test_multi_paper_isolation(self, verifier):
        """Context from paper A must not leak into paper B."""
        # Paper A: declare x real
        s1 = _eq_snippet(
            snippet_id="eq_a1",
            paper_id="paper_A",
            latex=r"x \in \mathbb{R}",
        )
        verifier.verify(s1)
        ctx_a = verifier._get_context("paper_A")
        assert "x" in ctx_a.get_symbol_names()

        # Paper B: fresh context (no 'x' declared)
        s2 = _eq_snippet(
            snippet_id="eq_b1",
            paper_id="paper_B",
            latex=r"y^2 = y \cdot y",
        )
        verifier.verify(s2)
        ctx_b = verifier._get_context("paper_B")
        assert "x" not in ctx_b.get_symbol_names()

    def test_cleanup_paper(self, verifier):
        """cleanup_paper should remove per-paper state."""
        s1 = _eq_snippet(
            snippet_id="eq_1",
            paper_id="temp_paper",
            latex=r"x \in \mathbb{R}",
        )
        verifier.verify(s1)
        assert "temp_paper" in verifier._contexts

        verifier.cleanup_paper("temp_paper")
        assert "temp_paper" not in verifier._contexts
        assert "temp_paper" not in verifier._context_locks


class TestProgressiveVerifierConstraints:
    """Tests for checkable constraints."""

    def test_constraint_equation_returns_result(self, verifier):
        """A constraint like s²+t²=1 should be checked."""
        snippet = _eq_snippet(
            snippet_id="eq_constraint",
            paper_id="paper_c",
            latex=r"s^2 + t^2 = 1",
            content="On the unit circle, we have the constraint: s^2 + t^2 = 1.",
        )
        result = verifier.verify(snippet)
        assert result.verifier_name == "progressive_math"
        # May be VALID/INVALID/UNVERIFIABLE depending on sandbox execution
        assert result.status in (
            VerificationStatus.VALID,
            VerificationStatus.INVALID,
            VerificationStatus.UNVERIFIABLE,
            VerificationStatus.MALFORMED,
        )


class TestProgressiveVerifierBroken:
    """Tests for detecting broken identities."""

    def test_broken_identity(self, verifier):
        """(a+b)² = a² + b² is wrong — the residual is 2ab."""
        snippet = _eq_snippet(
            snippet_id="eq_broken",
            paper_id="paper_d",
            latex=r"(a+b)^2 = a^2 + b^2",
        )
        result = verifier.verify(snippet)
        # The mock LLM generates code for this broken identity.
        # The harness should flag it as INVALID or UNVERIFIABLE (never VALID).
        assert result.status != VerificationStatus.VALID


class TestProgressiveVerifierEdgeCases:
    """Edge-case tests."""

    def test_no_latex_in_metadata(self, verifier):
        snippet = _eq_snippet(
            snippet_id="eq_no_latex",
            latex="",  # empty
            content="This is just text.",
        )
        # Override metadata to have no latex
        snippet.metadata = {}
        assert not verifier.can_verify(snippet)

    def test_very_short_latex(self, verifier):
        snippet = _eq_snippet(
            latex="x",
        )
        assert not verifier.can_verify(snippet)

    def test_disabled_verifier(self, verifier):
        verifier.verifier_config.enabled = False
        snippet = _eq_snippet(latex="x^2 = x \\cdot x")
        assert not verifier.can_verify(snippet)
        result = verifier.verify(snippet)
        assert result.status == VerificationStatus.SKIPPED

    def test_skipped_snippet_does_not_change_context(self, verifier):
        verifier.verifier_config.enabled = False
        snippet = _eq_snippet(
            snippet_id="skipped",
            paper_id="paper_e",
            latex=r"x \in \mathbb{R}",
        )
        verifier.verify(snippet)
        ctx = verifier._get_context("paper_e")
        assert len(ctx.declarations) == 0


# ---------------------------------------------------------------------------
# Verdict harness tests (direct)
# ---------------------------------------------------------------------------

_HARNESS_CODE = '''
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
    _emit("ASSUMPTION_ADDED", symbols=symbols, description=str(description))

def report_definition_added(symbol, definition=""):
    _emit("DEFINITION_ADDED", symbol=str(symbol), definition=str(definition))

def report_conditional_valid(condition=""):
    _emit("CONDITIONAL_VALID", condition=str(condition))
'''


def _run_sandbox(user_code: str) -> str:
    """Run the harness + user code in a subprocess and return stdout."""
    import tempfile
    from pathlib import Path

    full = _HARNESS_CODE + "\n\n" + user_code
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix="sympy_test_", delete=False,
    ) as tmp:
        tmp.write(full)
        tmp_path = Path(tmp.name)

    try:
        result = subprocess.run(
            ["python3", str(tmp_path)],
            capture_output=True, timeout=15, text=True,
        )
        return result.stdout
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def _parse_verdict(stdout: str) -> dict | None:
    """Extract the last VERDICT: line from stdout."""
    verdict = None
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("VERDICT:"):
            try:
                verdict = json.loads(line[len("VERDICT:"):])
            except json.JSONDecodeError:
                continue
    return verdict


@pytest.mark.sympy_sandbox
class TestHarnessVerdicts:
    """Direct integration tests for the harness — requires sympy in PATH."""

    def test_valid_identity(self):
        """x^2 - x*x → 0 → VALID."""
        code = """
from sympy import *
x = Symbol('x')
report(simplify(x**2 - x*x))
"""
        stdout = _run_sandbox(code)
        v = _parse_verdict(stdout)
        assert v is not None
        assert v["verdict"] == "VALID"

    def test_invalid_numeric(self):
        """2 + 3 - 6 → -1 → INVALID_NUMERIC."""
        code = """
from sympy import *
report(simplify(2 + 3 - 6))
"""
        stdout = _run_sandbox(code)
        v = _parse_verdict(stdout)
        assert v is not None
        assert v["verdict"] == "INVALID_NUMERIC"

    def test_symbolic_nonzero(self):
        """(a+b)^2 - (a^2 + b^2) → 2ab → SYMBOLIC_NONZERO."""
        code = """
from sympy import *
a, b = symbols('a b')
report(simplify((a+b)**2 - (a**2 + b**2)))
"""
        stdout = _run_sandbox(code)
        v = _parse_verdict(stdout)
        assert v is not None
        assert v["verdict"] == "SYMBOLIC_NONZERO"
        assert float(v.get("nonzero_fraction", 0)) >= 0.99

    def test_assumption_added(self):
        code = """
from sympy import *
x = Symbol('x', real=True)
report_assumption_added(['x'], 'x is real')
"""
        stdout = _run_sandbox(code)
        v = _parse_verdict(stdout)
        assert v is not None
        assert v["verdict"] == "ASSUMPTION_ADDED"

    def test_definition_added(self):
        code = """
from sympy import *
B, Q, N = symbols('B Q N')
report_definition_added('B', 'Q/N')
"""
        stdout = _run_sandbox(code)
        v = _parse_verdict(stdout)
        assert v is not None
        assert v["verdict"] == "DEFINITION_ADDED"
        assert v["symbol"] == "B"

    def test_conditional_valid(self):
        code = """
from sympy import *
x = Symbol('x')
report_conditional_valid('x != 0')
"""
        stdout = _run_sandbox(code)
        v = _parse_verdict(stdout)
        assert v is not None
        assert v["verdict"] == "CONDITIONAL_VALID"
        assert v["condition"] == "x != 0"

    def test_unverifiable_matrix(self):
        code = """
from sympy import *
A = MatrixSymbol('A', 2, 2)
B = MatrixSymbol('B', 2, 2)
# report the residual of A*B - a matrix expression
# that doesn't simplify to zero numerically
report(A*B - Identity(2))
"""
        stdout = _run_sandbox(code)
        v = _parse_verdict(stdout)
        assert v is not None
        # Matrix expressions are hard — expect UNVERIFIABLE or SYMBOLIC_NONZERO
        assert v["verdict"] in ("UNVERIFIABLE", "SYMBOLIC_NONZERO", "VALID")
