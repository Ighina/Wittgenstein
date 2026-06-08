"""Tests for the single-call baseline."""

import pytest

from src.config import PipelineConfig
from src.models import NormalizedPaper, PaperPrediction, PaperSection
from src.baseline.single_call_baseline import SingleCallBaseline


@pytest.fixture
def mock_config():
    cfg = PipelineConfig()
    cfg.llm.provider = "mock"
    return cfg


def _paper(content: str) -> NormalizedPaper:
    return NormalizedPaper(
        paper_id="demo", title="T", paper_category="Math",
        sections=[PaperSection(id="s0", section_title="Intro", section_level=1, content=content)],
        tagged_full_text=content,
    )


class TestSingleCallBaseline:
    def test_output_is_orchestrator_compatible(self, mock_config):
        pred = SingleCallBaseline(config=mock_config).run(_paper("The result is incorrect."))
        assert isinstance(pred, PaperPrediction)
        assert pred.paper_id == "demo"
        assert pred.errors_detected == len(pred.predicted_errors)
        assert pred.predicted_errors  # mock flags on "incorrect"
        e = pred.predicted_errors[0]
        assert e.verifier_name == "single_call_baseline"
        assert 0.0 <= e.confidence <= 1.0
        assert e.error_category and e.error_location

    def test_clean_paper_yields_no_errors(self, mock_config):
        pred = SingleCallBaseline(config=mock_config).run(
            _paper("A perfectly fine, routine paragraph of background.")
        )
        assert pred.errors_detected == 0
        assert pred.predicted_errors == []

    def test_truncation_is_bounded(self, mock_config):
        big = "word " * 50000  # ~250k chars
        b = SingleCallBaseline(config=mock_config, max_input_chars=1000)
        # Should not raise; truncation is internal.
        pred = b.run(_paper(big))
        assert isinstance(pred, PaperPrediction)

    def test_fulltext_fallback_without_tagged_text(self, mock_config):
        paper = NormalizedPaper(
            paper_id="d2", title="T", paper_category="Math",
            sections=[PaperSection(id="s0", section_title="S", section_level=1,
                                   content="some content with incorrect claim")],
            tagged_full_text="",  # force the reconstruction fallback
        )
        text = SingleCallBaseline(config=mock_config)._full_text(paper)
        assert "some content" in text
