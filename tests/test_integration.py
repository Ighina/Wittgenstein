"""Integration tests for the full pipeline."""

import pytest
import pandas as pd

from src.parser.content_parser import parse_paper_content
from src.parser.location_parser import parse_error_location
from src.segmentation.segmenter import segment_paper
from src.orchestrator.orchestrator import VerificationOrchestrator
from src.orchestrator.router import create_default_registry


@pytest.fixture
def sample_df():
    """Load the first paper from the dataset."""
    return pd.read_parquet("data/train-00000-of-00001.parquet").head(1)


@pytest.fixture
def sample_row(sample_df):
    """Get the first row."""
    return sample_df.iloc[0]


class TestFullPipeline:
    """End-to-end integration tests."""

    def test_parse_to_segment(self, sample_row):
        """Test parsing followed by segmentation."""
        paper = parse_paper_content(
            paper_id=str(sample_row["doi/arxiv_id"]),
            title=sample_row["title"],
            paper_category=sample_row["paper_category"],
            paper_content=sample_row["paper_content"],
            decode_images=False,
        )
        assert paper.paper_id

        snippets = segment_paper(paper)
        assert len(snippets) > 0

        # Verify snippet structure
        for s in snippets:
            assert s.snippet_id
            assert s.content

    def test_parse_to_verify(self, sample_row):
        """Test full parse → segment → verify pipeline."""
        paper = parse_paper_content(
            paper_id=str(sample_row["doi/arxiv_id"]),
            title=sample_row["title"],
            paper_category=sample_row["paper_category"],
            paper_content=sample_row["paper_content"],
            decode_images=False,
        )

        orchestrator = VerificationOrchestrator()
        prediction = orchestrator.run(paper)

        assert prediction.paper_id == str(sample_row["doi/arxiv_id"])
        assert prediction.total_snippets > 0
        assert prediction.snippets_verified > 0
        assert isinstance(prediction.verifier_usage, dict)

    def test_pipeline_with_multiple_papers(self, sample_df):
        """Test that the pipeline handles multiple papers."""
        predictions = []
        orchestrator = VerificationOrchestrator()

        for _, row in sample_df.iterrows():
            paper = parse_paper_content(
                paper_id=str(row["doi/arxiv_id"]),
                title=row["title"],
                paper_category=row["paper_category"],
                paper_content=row["paper_content"],
                decode_images=False,
            )
            prediction = orchestrator.run(paper)
            predictions.append(prediction)

        assert len(predictions) == 1

    def test_location_parsing_on_all_formats(self, sample_df):
        """Test that all error_location values in the dataset can be parsed."""
        df = pd.read_parquet("data/train-00000-of-00001.parquet")
        for _, row in df.iterrows():
            loc = row["error_location"]
            ref = parse_error_location(loc)
            assert ref.raw == loc.strip()
            assert ref.location_type is not None
            assert ref.identifier

    def test_orchestrator_produces_verifier_usage(self, sample_row):
        """Test that the orchestrator tracks verifier usage."""
        paper = parse_paper_content(
            paper_id=str(sample_row["doi/arxiv_id"]),
            title=sample_row["title"],
            paper_category=sample_row["paper_category"],
            paper_content=sample_row["paper_content"],
            decode_images=False,
        )
        orchestrator = VerificationOrchestrator()
        prediction = orchestrator.run(paper)

        assert len(prediction.verifier_usage) > 0
        total = sum(prediction.verifier_usage.values())
        assert total == prediction.snippets_verified
