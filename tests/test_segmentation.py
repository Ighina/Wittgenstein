"""Tests for the segmentation module."""

import pytest
from src.models import (
    EquationBlock,
    ImageBlock,
    NormalizedPaper,
    PaperSection,
    SnippetType,
    TheoremBlock,
)
from src.segmentation.segmenter import segment_paper
from src.config import SegmentationConfig


@pytest.fixture
def sample_paper():
    """Create a minimal NormalizedPaper for testing."""
    return NormalizedPaper(
        paper_id="test_paper_1",
        title="Test Paper",
        paper_category="Mathematics",
        sections=[
            PaperSection(
                id="s1",
                section_title="Introduction",
                section_level=1,
                content="This is the introduction. It contains some text about the problem.",
                start_index=0,
                end_index=60,
            ),
            PaperSection(
                id="s2",
                section_title="2. Main Result",
                section_level=2,
                content="The main result of this paper is Theorem 2.1.",
                start_index=60,
                end_index=110,
            ),
        ],
        equations=[
            EquationBlock(
                id="eq1",
                equation_label="Equation 1",
                latex="x^2 + y^2 = z^2",
                display_mode=True,
                context_before="Pythagorean theorem:",
                context_after="This is a well-known result.",
            ),
        ],
        images=[
            ImageBlock(
                id="img1",
                caption="Figure 1: Test figure",
                image_path="/tmp/test_fig.jpg",
                context_before="See the figure below:",
                context_after="The figure shows...",
            ),
        ],
        theorems=[
            TheoremBlock(
                id="thm1",
                theorem_type="theorem",
                label="**Theorem 2.1.**",
                statement="All swans are white.",
                proof="Assume not. Then there exists a non-white swan.",
            ),
        ],
    )


class TestSegmentation:
    """Tests for paper segmentation."""

    def test_segment_produces_snippets(self, sample_paper):
        """Test that segmentation produces a non-empty list of snippets."""
        snippets = segment_paper(sample_paper)
        assert len(snippets) > 0

    def test_segment_includes_sections(self, sample_paper):
        """Test that sections are segmented."""
        snippets = segment_paper(sample_paper)
        section_snippets = [s for s in snippets if s.snippet_type == SnippetType.SECTION]
        assert len(section_snippets) >= 1

    def test_segment_includes_equations(self, sample_paper):
        """Test that equations are segmented."""
        snippets = segment_paper(sample_paper)
        eq_snippets = [s for s in snippets if s.snippet_type == SnippetType.EQUATION]
        assert len(eq_snippets) == 1
        eq = eq_snippets[0]
        assert "x^2 + y^2 = z^2" in eq.content

    def test_segment_includes_figures(self, sample_paper):
        """Test that figures are segmented."""
        snippets = segment_paper(sample_paper)
        fig_snippets = [s for s in snippets if s.snippet_type == SnippetType.FIGURE]
        assert len(fig_snippets) == 1
        fig = fig_snippets[0]
        assert fig.image_path == "/tmp/test_fig.jpg"

    def test_segment_includes_theorems(self, sample_paper):
        """Test that theorems are segmented."""
        snippets = segment_paper(sample_paper)
        thm_snippets = [s for s in snippets if s.snippet_type == SnippetType.THEOREM]
        assert len(thm_snippets) == 1
        thm = thm_snippets[0]
        assert "All swans are white" in thm.content

    def test_snippet_has_required_fields(self, sample_paper):
        """Test that each snippet has all required fields."""
        snippets = segment_paper(sample_paper)
        for snippet in snippets:
            assert snippet.snippet_id
            assert snippet.snippet_type
            assert snippet.paper_id == "test_paper_1"
            assert snippet.content_length > 0
            assert snippet.estimated_tokens > 0

    def test_long_section_chunking(self):
        """Test that long sections are split into multiple chunks."""
        config = SegmentationConfig(max_section_chars=100, max_snippet_chars=80, overlap_chars=10)
        paper = NormalizedPaper(
            paper_id="test",
            title="Test",
            paper_category="Math",
            sections=[
                PaperSection(
                    id="s1",
                    section_title="Long Section",
                    section_level=1,
                    content="A" * 500,
                    start_index=0,
                    end_index=500,
                ),
            ],
        )
        snippets = segment_paper(paper, config=config)
        section_snippets = [s for s in snippets if s.snippet_type == SnippetType.SECTION]
        assert len(section_snippets) > 1  # Should be chunked
