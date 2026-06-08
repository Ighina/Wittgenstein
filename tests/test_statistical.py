"""Tests for the deterministic statistical verifier and safe arithmetic."""

import pytest

from src.config import PipelineConfig
from src.models import SnippetType, VerificationSnippet, VerificationStatus
from src.utils.safe_arithmetic import ArithmeticEvalError, safe_eval
from src.verifiers.citation_verifier import CitationVerifier
from src.verifiers.statistical_verifier import StatisticalVerifier


@pytest.fixture
def mock_config():
    cfg = PipelineConfig()
    cfg.llm.provider = "mock"
    cfg.llm.num_workers = 1
    return cfg


class TestSafeArithmetic:
    def test_basic(self):
        assert safe_eval("1 + 2 * 3") == 7.0
        assert safe_eval("100 - (60 + 30)") == 10.0

    def test_functions_and_consts(self):
        assert safe_eval("log10(1000)") == pytest.approx(3.0)
        assert safe_eval("mean([2, 4, 6])") == pytest.approx(4.0)
        assert safe_eval("sqrt(16)") == 4.0

    @pytest.mark.parametrize("bad", [
        "__import__('os')", "x + 1", "open('f')", "True and 1", "lambda: 1", "",
    ])
    def test_rejects_non_numeric(self, bad):
        with pytest.raises(ArithmeticEvalError):
            safe_eval(bad)


class TestStatisticalVerifier:
    def test_flags_inconsistent_percentages(self, mock_config):
        snip = VerificationSnippet(
            snippet_id="p_sec_0", snippet_type=SnippetType.SECTION, paper_id="p",
            location="Results",
            content="Of participants, 60% chose A and 30% chose B, accounting for 100% of the cohort.",
        )
        r = StatisticalVerifier(config=mock_config).verify(snip)
        assert r.status == VerificationStatus.INVALID
        assert r.error_detected is True
        assert r.predicted_error_category == "Statistical reporting"
        assert any(c["passed"] is False for c in r.checks)

    def test_no_numeric_claim_is_unverifiable(self, mock_config):
        snip = VerificationSnippet(
            snippet_id="p_sec_1", snippet_type=SnippetType.SECTION, paper_id="p",
            location="Intro", content="We study operator systems with no specific figures here 1.",
        )
        r = StatisticalVerifier(config=mock_config).verify(snip)
        assert r.status == VerificationStatus.UNVERIFIABLE
        assert r.error_detected is False

    def test_skips_text_without_digits(self, mock_config):
        snip = VerificationSnippet(
            snippet_id="p_sec_2", snippet_type=SnippetType.SECTION, paper_id="p",
            location="Intro", content="Purely qualitative discussion with no numbers.",
        )
        r = StatisticalVerifier(config=mock_config).verify(snip)
        assert r.status == VerificationStatus.SKIPPED


class TestCitationVerifier:
    def test_flags_novelty_overclaim(self, mock_config):
        snip = VerificationSnippet(
            snippet_id="p_sec_0", snippet_type=SnippetType.SECTION, paper_id="p",
            location="Intro",
            content="Our method is novel and original; it was previously established by Smith et al.",
        )
        r = CitationVerifier(config=mock_config).verify(snip)
        assert r.status == VerificationStatus.ERROR_DETECTED
        assert r.error_detected is True

    def test_clean_text_no_error(self, mock_config):
        snip = VerificationSnippet(
            snippet_id="p_sec_1", snippet_type=SnippetType.SECTION, paper_id="p",
            location="Intro", content="We build on standard techniques from the literature.",
        )
        r = CitationVerifier(config=mock_config).verify(snip)
        assert r.status == VerificationStatus.NO_ERROR
        assert r.error_detected is False
