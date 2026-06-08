"""Tests for the verifier framework."""

import pytest
from src.models import SnippetType, VerificationSnippet, VerificationStatus
from src.verifiers.base import BaseVerifier
from src.verifiers.registry import VerifierRegistry
from src.verifiers.math_verifier import MathEquationVerifier
from src.verifiers.text_verifier import TextVerifier
from src.verifiers.vision_verifier import VisionVerifier


class TestVerifierRegistry:
    """Tests for the verifier plugin registry."""

    def test_register_and_get(self):
        """Test basic register/get workflow."""
        registry = VerifierRegistry()
        registry.register("math_equation", MathEquationVerifier)
        assert "math_equation" in registry
        assert registry.get("math_equation") is MathEquationVerifier

    def test_register_duplicate_same_class(self):
        """Test re-registering the same class is fine."""
        registry = VerifierRegistry()
        registry.register("math_equation", MathEquationVerifier)
        registry.register("math_equation", MathEquationVerifier)  # Should not raise

    def test_register_duplicate_different_class_raises(self):
        """Test that re-registering with a different class raises ValueError."""
        registry = VerifierRegistry()
        registry.register("math_equation", MathEquationVerifier)
        with pytest.raises(ValueError):
            registry.register("math_equation", TextVerifier)

    def test_get_nonexistent_raises_keyerror(self):
        """Test that getting a non-existent verifier raises KeyError."""
        registry = VerifierRegistry()
        with pytest.raises(KeyError):
            registry.get("nonexistent")

    def test_unregister(self):
        """Test unregistering a verifier."""
        registry = VerifierRegistry()
        registry.register("math_equation", MathEquationVerifier)
        registry.unregister("math_equation")
        assert "math_equation" not in registry

    def test_list_verifiers(self):
        """Test listing registered verifiers."""
        registry = VerifierRegistry()
        registry.register("math_equation", MathEquationVerifier)
        registry.register("text", TextVerifier)
        assert set(registry.list_verifiers()) == {"math_equation", "text"}

    def test_has(self):
        """Test the has() method."""
        registry = VerifierRegistry()
        assert not registry.has("math_equation")
        registry.register("math_equation", MathEquationVerifier)
        assert registry.has("math_equation")


@pytest.fixture
def equation_snippet():
    """Create a sample equation snippet."""
    return VerificationSnippet(
        snippet_id="test_eq_1",
        snippet_type=SnippetType.EQUATION,
        paper_id="test_paper",
        location="Equation 1",
        content="Equation: x^2 + y^2 = z^2\nContext: Pythagorean theorem.",
        metadata={"latex": "x^2 + y^2 = z^2", "display_mode": True},
    )


@pytest.fixture
def text_snippet():
    """Create a sample text snippet."""
    return VerificationSnippet(
        snippet_id="test_sec_1",
        snippet_type=SnippetType.SECTION,
        paper_id="test_paper",
        location="Section Introduction",
        content="The theorem states that all swans are white. "
        "However, black swans have been observed in Australia. "
        "This contradiction suggests the theorem may be false.",
    )


@pytest.fixture
def figure_snippet():
    """Create a sample figure snippet."""
    return VerificationSnippet(
        snippet_id="test_fig_1",
        snippet_type=SnippetType.FIGURE,
        paper_id="test_paper",
        location="Figure 1",
        content="Caption: Figure 1: Test figure.\nContext: As shown in the figure...",
        image_path="/tmp/test.jpg",
    )


class TestMathEquationVerifier:
    """Tests for the math equation verifier."""

    def test_verify_equation(self, equation_snippet):
        """Test that equation verification runs and returns a result."""
        verifier = MathEquationVerifier()
        result = verifier.verify(equation_snippet)
        assert result.verifier_name == "math_equation"
        assert result.snippet_id == "test_eq_1"
        assert result.status in (
            VerificationStatus.VALID,
            VerificationStatus.INVALID,
            VerificationStatus.UNVERIFIABLE,
            VerificationStatus.MALFORMED,
        )

    def test_can_verify_with_latex(self, equation_snippet):
        """Test that equations with LaTeX can be verified."""
        verifier = MathEquationVerifier()
        assert verifier.can_verify(equation_snippet)

    def test_cannot_verify_without_latex(self):
        """Test that snippets without LaTeX cannot be verified."""
        snippet = VerificationSnippet(
            snippet_id="test",
            snippet_type=SnippetType.EQUATION,
            paper_id="test",
            location="Empty",
            content="No equation here.",
            metadata={},
        )
        verifier = MathEquationVerifier()
        assert not verifier.can_verify(snippet)

    def test_result_includes_sympy_code(self, equation_snippet):
        """Test that the result includes generated SymPy code."""
        verifier = MathEquationVerifier()
        result = verifier.verify(equation_snippet)
        if result.status not in (VerificationStatus.UNVERIFIABLE,):
            assert result.sympy_code is not None


class TestMathInterpretConservatism:
    """The verifier must only emit INVALID for deterministic contradictions.

    Regression guard for the false-positive flood where definitions and
    constrained equations (residual has free symbols) were flagged INVALID
    with "Equation does not hold symbolically".
    """

    import json as _json

    @staticmethod
    def _verdict_line(verdict, **kw):
        import json
        kw["verdict"] = verdict
        return "VERDICT:" + json.dumps(kw)

    def test_numeric_contradiction_is_invalid(self):
        out = self._verdict_line("INVALID_NUMERIC", residual="-1/2")
        status, _, conf = MathEquationVerifier._interpret_output(out, "", 0, "numeric")
        assert status == VerificationStatus.INVALID
        assert conf >= 0.85

    def test_zero_residual_is_valid(self):
        out = self._verdict_line("VALID")
        status, _, _ = MathEquationVerifier._interpret_output(out, "", 0, "identity")
        assert status == VerificationStatus.VALID

    def test_definition_with_free_symbols_is_unverifiable(self):
        # B - Q/N style residual: free symbols, not an identity.
        out = self._verdict_line(
            "SYMBOLIC_NONZERO", residual="B - Q/N",
            free=["B", "N", "Q"], nonzero_fraction=1.0, n_evaluated=30,
        )
        status, _, _ = MathEquationVerifier._interpret_output(out, "", 0, "numeric")
        assert status == VerificationStatus.UNVERIFIABLE

    def test_constraint_equation_is_unverifiable(self):
        # s^2 + t^2 - 1: holds only on the unit circle -> conditional.
        out = self._verdict_line(
            "SYMBOLIC_NONZERO", residual="s**2 + t**2 - 1",
            free=["s", "t"], nonzero_fraction=1.0, n_evaluated=30,
        )
        status, _, _ = MathEquationVerifier._interpret_output(out, "", 0, "conditional")
        assert status == VerificationStatus.UNVERIFIABLE

    def test_broken_identity_is_invalid(self):
        # (a+b)^2 = a^2 + b^2 is wrong: residual 2ab nonzero everywhere AND
        # explicitly classified as an unconditional identity.
        out = self._verdict_line(
            "SYMBOLIC_NONZERO", residual="2*a*b",
            free=["a", "b"], nonzero_fraction=1.0, n_evaluated=30,
        )
        status, _, conf = MathEquationVerifier._interpret_output(out, "", 0, "identity")
        assert status == VerificationStatus.INVALID
        assert conf < 0.85  # lower confidence than a numeric contradiction

    def test_no_verdict_line_is_unverifiable_not_invalid(self):
        # Old behavior would read raw output and flag INVALID; now we refuse.
        status, _, _ = MathEquationVerifier._interpret_output(
            "output=s**2 + t**2 - 1", "", 0, "identity"
        )
        assert status == VerificationStatus.UNVERIFIABLE

    def test_runtime_error_is_malformed(self):
        status, _, _ = MathEquationVerifier._interpret_output(
            "", "NameError: name 'foo' is not defined", 1, "identity"
        )
        assert status == VerificationStatus.MALFORMED


class TestTextVerifier:
    """Tests for the text verifier."""

    def test_verify_text_section(self, text_snippet):
        """Test that text verification runs."""
        verifier = TextVerifier()
        result = verifier.verify(text_snippet)
        assert result.verifier_name == "text"
        assert result.status in (
            VerificationStatus.ERROR_DETECTED,
            VerificationStatus.NO_ERROR,
            VerificationStatus.UNVERIFIABLE,
        )

    def test_can_verify_section_type(self, text_snippet):
        """Test that text verifier handles SECTION type."""
        verifier = TextVerifier()
        assert verifier.can_verify(text_snippet)

    def test_can_verify_theorem_type(self):
        """Test that text verifier handles THEOREM type."""
        snippet = VerificationSnippet(
            snippet_id="test",
            snippet_type=SnippetType.THEOREM,
            paper_id="test",
            location="Theorem 1",
            content="A theorem statement and its proof.",
        )
        verifier = TextVerifier()
        assert verifier.can_verify(snippet)

    def test_cannot_verify_empty_content(self):
        """Test that empty content cannot be verified."""
        snippet = VerificationSnippet(
            snippet_id="test",
            snippet_type=SnippetType.SECTION,
            paper_id="test",
            location="Empty",
            content="   ",
        )
        verifier = TextVerifier()
        assert not verifier.can_verify(snippet)


class TestVisionVerifier:
    """Tests for the vision verifier."""

    def test_verify_figure(self, figure_snippet):
        """Test that figure verification runs."""
        verifier = VisionVerifier()
        result = verifier.verify(figure_snippet)
        assert result.verifier_name == "vision"
        assert result.status in (
            VerificationStatus.ERROR_DETECTED,
            VerificationStatus.NO_ERROR,
            VerificationStatus.UNVERIFIABLE,
        )

    def test_can_verify_figure_type(self, figure_snippet):
        """Test that vision verifier handles FIGURE type."""
        verifier = VisionVerifier()
        assert verifier.can_verify(figure_snippet)

    def test_can_verify_table_type(self):
        """Test that vision verifier handles TABLE type."""
        snippet = VerificationSnippet(
            snippet_id="test",
            snippet_type=SnippetType.TABLE,
            paper_id="test",
            location="Table 1",
            content="| A | B |\n|---|---|\n| 1 | 2 |",
        )
        verifier = VisionVerifier()
        assert verifier.can_verify(snippet)

    def test_cannot_verify_text_type(self, text_snippet):
        """Test that vision verifier rejects non-visual snippets."""
        verifier = VisionVerifier()
        assert not verifier.can_verify(text_snippet)
