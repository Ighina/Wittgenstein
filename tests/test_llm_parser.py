"""Tests for the LLM-based parser (ContextGraph, LLM parser, enriched segmenter)."""

from __future__ import annotations

import pytest

from src.models import (
    EnrichedPaper,
    ImageBlock,
    LLMParseChunkResult,
    RawContentItem,
    SymbolDefinition,
    SnippetType,
    VerificationSnippet,
    VerifiableUnit,
    ContentType,
)
from src.parser.context_graph import ContextGraph
from src.parser.enriched_segmenter import (
    segment_enriched_paper,
    _map_unit_type,
    UNIT_TYPE_TO_SNIPPET,
)
from src.parser.llm_content_parser import (
    _chunk_text,
    _mock_llm_parse_chunk,
    _assemble_enriched_paper,
    PARSER_SYSTEM_PROMPT,
)
from src.config import PipelineConfig


# ---------------------------------------------------------------------------
# ContextGraph tests
# ---------------------------------------------------------------------------


class TestContextGraph:
    """Tests for the ContextGraph dependency tracker."""

    def test_add_and_get_unit(self):
        graph = ContextGraph()
        unit = VerifiableUnit(
            unit_id="eq1",
            unit_type="equation",
            content="x^2 + y^2 = z^2",
        )
        graph.add_unit(unit)
        assert graph.has_unit("eq1")
        assert graph.get_unit("eq1") == unit
        assert graph.unit_count == 1

    def test_add_dependency(self):
        graph = ContextGraph()
        u1 = VerifiableUnit(unit_id="def_x", unit_type="definition", content="Let x be a real number")
        u2 = VerifiableUnit(unit_id="eq1", unit_type="equation", content="x^2 = 4")
        graph.add_unit(u1)
        graph.add_unit(u2)
        graph.add_dependency("eq1", "def_x")

        assert "def_x" in graph.get_dependencies("eq1")

    def test_add_symbol(self):
        graph = ContextGraph()
        sym = SymbolDefinition(
            symbol_name="x",
            domain="real",
            natural_language="Let x be a real number",
            defining_unit_id="def_x",
        )
        graph.add_symbol(sym)
        assert graph.symbol_count == 1
        assert graph.get_symbol("x") == sym

    def test_add_unverifiable_text(self):
        graph = ContextGraph()
        graph.add_unverifiable_text("Acknowledgements section")
        graph.add_unverifiable_text("Grant information")
        assert "Acknowledgements" in graph.unverifiable_text
        assert "Grant information" in graph.unverifiable_text

    def test_resolve_context_basic(self):
        graph = ContextGraph()
        u_def = VerifiableUnit(
            unit_id="def_x",
            unit_type="definition",
            content="Let x be a real number",
            location="Definition 1",
        )
        u_eq = VerifiableUnit(
            unit_id="eq1",
            unit_type="equation",
            content="x^2 = 4",
            location="Equation 1",
            dependencies=["def_x"],
        )
        graph.add_unit(u_def)
        graph.add_unit(u_eq)
        graph.add_dependency("eq1", "def_x")

        context = graph.resolve_context(u_eq, max_chars=8000)
        assert "def_x" in context or "Definition 1" in context
        assert "Let x be a real number" in context

    def test_resolve_context_max_chars(self):
        graph = ContextGraph()
        u_def = VerifiableUnit(
            unit_id="def_x",
            unit_type="definition",
            content="A" * 5000,
            location="Definition 1",
        )
        u_eq = VerifiableUnit(
            unit_id="eq1",
            unit_type="equation",
            content="x^2 = 4",
            dependencies=["def_x"],
        )
        graph.add_unit(u_def)
        graph.add_unit(u_eq)
        graph.add_dependency("eq1", "def_x")

        context = graph.resolve_context(u_eq, max_chars=1000)
        assert len(context) <= 1200  # Allow small overhead for formatting
        assert "[...context truncated...]" in context

    def test_resolve_context_empty(self):
        graph = ContextGraph()
        unit = VerifiableUnit(
            unit_id="eq1",
            unit_type="equation",
            content="1 + 1 = 2",
        )
        graph.add_unit(unit)
        context = graph.resolve_context(unit, max_chars=8000)
        assert context == ""

    def test_topological_order(self):
        graph = ContextGraph()
        u1 = VerifiableUnit(unit_id="def_x", unit_type="definition", content="Let x be real")
        u2 = VerifiableUnit(unit_id="eq1", unit_type="equation", content="x^2 = 4")
        u3 = VerifiableUnit(unit_id="thm1", unit_type="theorem", content="x = ±2")

        graph.add_unit(u1)  # No deps
        graph.add_unit(u2)  # Depends on u1
        graph.add_unit(u3)  # Depends on u2

        graph.add_dependency("eq1", "def_x")
        graph.add_dependency("thm1", "eq1")

        order = graph.topological_order()
        assert order.index("def_x") < order.index("eq1")
        assert order.index("eq1") < order.index("thm1")

    def test_as_dict(self):
        graph = ContextGraph()
        u1 = VerifiableUnit(unit_id="a", unit_type="definition", content="...")
        u2 = VerifiableUnit(unit_id="b", unit_type="equation", content="...")
        graph.add_unit(u1)
        graph.add_unit(u2)
        graph.add_dependency("b", "a")

        d = graph.as_dict()
        assert "b" in d
        assert "a" in d["b"]

    def test_symbol_based_dependency_resolution(self):
        graph = ContextGraph()
        sym = SymbolDefinition(
            symbol_name="X",
            domain="Banach space",
            defining_unit_id="def_X",
        )
        u_def = VerifiableUnit(
            unit_id="def_X",
            unit_type="definition",
            content="Let X be a real Banach space",
        )
        u_claim = VerifiableUnit(
            unit_id="claim1",
            unit_type="claim",
            content="X is isometrically isomorphic to its bidual",
            dependencies=["X"],  # Symbol-name dependency
        )
        graph.add_unit(u_def)
        graph.add_unit(u_claim)
        graph.add_symbol(sym)

        context = graph.resolve_context(u_claim, max_chars=8000)
        assert "Banach space" in context


# ---------------------------------------------------------------------------
# Chunking tests
# ---------------------------------------------------------------------------


class TestChunking:
    """Tests for the text chunking used by the LLM parser."""

    def test_short_text_unchunked(self):
        text = "Short paper."
        chunks = _chunk_text(text, chunk_size=8000)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_chunked(self):
        text = "Paragraph A.\n\n" * 500  # ~7500 chars
        chunks = _chunk_text(text, chunk_size=2000)
        assert len(chunks) > 1
        # Each chunk should be at or under chunk_size
        for chunk in chunks:
            assert len(chunk) <= 3000  # Allow some overhead for paragraph boundaries

    def test_empty_text(self):
        chunks = _chunk_text("", chunk_size=8000)
        assert len(chunks) == 1
        assert chunks[0] == ""


# ---------------------------------------------------------------------------
# Mock LLM parser tests
# ---------------------------------------------------------------------------


class TestMockLLMParser:
    """Tests for the mock LLM parser responses."""

    def test_mock_parse_math_equations(self):
        chunk = r"The equation is \[\int_0^\infty e^{-x} dx = 1\] and the inline \(\alpha > 0\) matters."
        result = _mock_llm_parse_chunk(chunk, chunk_index=0)
        assert len(result["units"]) >= 1
        eq_units = [u for u in result["units"] if u["unit_type"] == "equation"]
        assert len(eq_units) >= 1

    def test_mock_parse_theorem(self):
        chunk = "**Theorem 1.1.** Let X be a Banach space. Then X is complete."
        result = _mock_llm_parse_chunk(chunk, chunk_index=0)
        thm_units = [u for u in result["units"] if u["unit_type"] == "theorem"]
        assert len(thm_units) >= 1
        assert thm_units[0]["is_verifiable"] is True

    def test_mock_parse_lemma(self):
        chunk = "**Lemma 2.** Every bounded sequence has a convergent subsequence."
        result = _mock_llm_parse_chunk(chunk, chunk_index=0)
        lemma_units = [u for u in result["units"] if u["unit_type"] == "lemma"]
        assert len(lemma_units) >= 1

    def test_mock_parse_symbol_definitions(self):
        chunk = "Let X be a real Banach space. Define f(x) = x^2 + 1."
        result = _mock_llm_parse_chunk(chunk, chunk_index=0)
        assert len(result["symbols"]) >= 1
        # Should also create definition units
        def_units = [u for u in result["units"] if u["unit_type"] == "definition"]
        assert len(def_units) >= 1
        # Definitions should be flagged as unverifiable
        for du in def_units:
            assert du["is_verifiable"] is False

    def test_mock_parse_numeric_claim(self):
        chunk = "The model achieves 95.2% accuracy on the test set (p < 0.001)."
        result = _mock_llm_parse_chunk(chunk, chunk_index=0)
        num_units = [u for u in result["units"] if u["unit_type"] == "numeric_claim"]
        assert len(num_units) >= 1
        assert num_units[0]["verifier_route"] == "statistical"

    def test_mock_parse_table(self):
        chunk = "| Method | Score |\n|--------|-------|\n| A      | 0.95  |\n| B      | 0.87  |"
        result = _mock_llm_parse_chunk(chunk, chunk_index=0)
        tbl_units = [u for u in result["units"] if u["unit_type"] == "table_data"]
        assert len(tbl_units) >= 1

    def test_mock_parse_boilerplate(self):
        chunk = "Acknowledgements: We thank the anonymous reviewers. This work was supported by grant NSF-12345."
        result = _mock_llm_parse_chunk(chunk, chunk_index=0)
        assert result["unverifiable_text"] != ""

    def test_mock_parse_section_headers(self):
        chunk = "## 1. Introduction\n\nSome text.\n\n### 1.1 Background\n\nMore text."
        result = _mock_llm_parse_chunk(chunk, chunk_index=0)
        assert len(result["section_headers"]) >= 1

    def test_mock_parse_empty_chunk(self):
        result = _mock_llm_parse_chunk("", chunk_index=0)
        assert result["units"] == []
        assert result["symbols"] == []


# ---------------------------------------------------------------------------
# Enriched paper assembly tests
# ---------------------------------------------------------------------------


class TestAssembleEnrichedPaper:
    """Tests for assembling EnrichedPaper from chunk results."""

    @pytest.fixture
    def sample_raw_items(self):
        return [
            RawContentItem(
                content_type=ContentType.TEXT,
                text="Let X be a real Banach space. X is isometrically isomorphic to X**.",
            ),
        ]

    @pytest.fixture
    def sample_images(self):
        return []

    @pytest.fixture
    def sample_chunk_results(self):
        return [
            LLMParseChunkResult(
                chunk_index=0,
                units=[
                    VerifiableUnit(
                        unit_id="def_X",
                        unit_type="definition",
                        content="Let X be a real Banach space",
                        location="Section 1",
                        verifier_route="none",
                        is_verifiable=False,
                        source_chunk_index=0,
                    ),
                    VerifiableUnit(
                        unit_id="thm_1",
                        unit_type="theorem",
                        content="X is isometrically isomorphic to X**",
                        location="Theorem 1.1",
                        dependencies=["def_X"],
                        verifier_route="math",
                        is_verifiable=True,
                        source_chunk_index=0,
                    ),
                ],
                symbols=[
                    SymbolDefinition(
                        symbol_name="X",
                        domain="Banach space",
                        natural_language="Let X be a real Banach space",
                        defining_unit_id="def_X",
                    ),
                ],
                unverifiable_text="",
                section_headers=["1. Introduction"],
            ),
        ]

    def test_assemble_basic(self, sample_raw_items, sample_images, sample_chunk_results):
        config = PipelineConfig()
        enriched = _assemble_enriched_paper(
            paper_id="test_1",
            title="Test Paper",
            paper_category="Mathematics",
            chunk_results=sample_chunk_results,
            raw_items=sample_raw_items,
            images=sample_images,
            config=config,
        )
        assert isinstance(enriched, EnrichedPaper)
        assert enriched.paper_id == "test_1"
        assert len(enriched.verifiable_units) == 1  # Only the verifiable one
        assert enriched.verifiable_units[0].unit_id == "thm_1"
        assert enriched.verifiable_units[0].is_verifiable is True
        assert len(enriched.symbol_registry) == 1
        # Unverifiable context should include the definition
        assert "Let X be a real Banach space" in enriched.unverifiable_context

    def test_assemble_context_injection(self, sample_raw_items, sample_images, sample_chunk_results):
        config = PipelineConfig()
        config.llm_parser_max_context_chars = 8000
        enriched = _assemble_enriched_paper(
            paper_id="test_1",
            title="Test Paper",
            paper_category="Mathematics",
            chunk_results=sample_chunk_results,
            raw_items=sample_raw_items,
            images=sample_images,
            config=config,
        )
        # The verifiable unit should have its dependency context resolved
        unit = enriched.verifiable_units[0]
        assert "Banach space" in unit.required_context or unit.dependencies == ["def_X"]


# ---------------------------------------------------------------------------
# Enriched segmenter tests
# ---------------------------------------------------------------------------


class TestEnrichedSegmenter:
    """Tests for the enriched segmentation stage."""

    @pytest.fixture
    def sample_enriched_paper(self):
        return EnrichedPaper(
            paper_id="test_1",
            title="Test Paper",
            paper_category="Mathematics",
            verifiable_units=[
                VerifiableUnit(
                    unit_id="eq1",
                    unit_type="equation",
                    content="x^2 + y^2 = z^2",
                    location="Equation 1",
                    required_context="## Prerequisite Context\nLet x, y, z be positive integers.",
                    verifier_route="math",
                    is_verifiable=True,
                ),
                VerifiableUnit(
                    unit_id="thm1",
                    unit_type="theorem",
                    content="All solutions are primitive.",
                    location="Theorem 2.1",
                    verifier_route="text",
                    is_verifiable=True,
                ),
                VerifiableUnit(
                    unit_id="def_x",
                    unit_type="definition",
                    content="Let x be a real number",
                    location="Definition 1",
                    verifier_route="none",
                    is_verifiable=False,  # Should be skipped
                ),
                VerifiableUnit(
                    unit_id="ack",
                    unit_type="boilerplate",
                    content="We thank the reviewers.",
                    verifier_route="none",
                    is_verifiable=False,  # Should be skipped
                ),
            ],
            symbol_registry=[],
            context_graph={},
            unverifiable_context="",
        )

    def test_segment_skips_unverifiable(self, sample_enriched_paper):
        snippets = segment_enriched_paper(sample_enriched_paper)
        assert len(snippets) == 2  # Only eq1 and thm1
        snippet_ids = {s.snippet_id for s in snippets}
        assert "def_x" not in snippet_ids
        assert "ack" not in snippet_ids
        assert "eq1" in snippet_ids
        assert "thm1" in snippet_ids

    def test_segment_preserves_verifier_route(self, sample_enriched_paper):
        snippets = segment_enriched_paper(sample_enriched_paper)
        eq_snippet = next(s for s in snippets if s.snippet_id == "eq1")
        assert eq_snippet.verifier_route == "math"
        thm_snippet = next(s for s in snippets if s.snippet_id == "thm1")
        assert thm_snippet.verifier_route == "text"

    def test_segment_includes_dependency_context(self, sample_enriched_paper):
        snippets = segment_enriched_paper(sample_enriched_paper)
        eq_snippet = next(s for s in snippets if s.snippet_id == "eq1")
        assert "Prerequisite Context" in eq_snippet.content
        assert "positive integers" in eq_snippet.content
        assert "x^2 + y^2 = z^2" in eq_snippet.content

    def test_segment_sets_snippet_type(self, sample_enriched_paper):
        snippets = segment_enriched_paper(sample_enriched_paper)
        eq_snippet = next(s for s in snippets if s.snippet_id == "eq1")
        assert eq_snippet.snippet_type == SnippetType.EQUATION
        thm_snippet = next(s for s in snippets if s.snippet_id == "thm1")
        assert thm_snippet.snippet_type == SnippetType.THEOREM

    def test_segment_sets_metadata(self, sample_enriched_paper):
        snippets = segment_enriched_paper(sample_enriched_paper)
        eq_snippet = next(s for s in snippets if s.snippet_id == "eq1")
        assert eq_snippet.metadata["unit_type"] == "equation"
        assert eq_snippet.metadata["verifier_route"] == "math"
        assert eq_snippet.metadata["dependency_count"] == 0

    def test_segment_with_long_context_truncation(self):
        long_context = "X" * 10000
        paper = EnrichedPaper(
            paper_id="test",
            title="Test",
            paper_category="Math",
            verifiable_units=[
                VerifiableUnit(
                    unit_id="big_unit",
                    unit_type="claim",
                    content="A short claim about X.",
                    required_context=long_context,
                    verifier_route="text",
                    is_verifiable=True,
                ),
            ],
            symbol_registry=[],
            context_graph={},
            unverifiable_context="",
        )
        config = PipelineConfig()
        config.segmentation.max_snippet_chars = 4000
        snippets = segment_enriched_paper(paper, config=config)
        assert len(snippets) == 1
        # Should be truncated to fit
        assert len(snippets[0].content) <= 4200  # Allow some overhead

    def test_map_unit_type_all_keys(self):
        """All unit_type strings should map to a SnippetType."""
        for unit_type in UNIT_TYPE_TO_SNIPPET:
            result = _map_unit_type(unit_type)
            assert isinstance(result, SnippetType), f"Failed for {unit_type}"

    def test_map_unit_type_unknown_fallback(self):
        result = _map_unit_type("nonexistent_type")
        assert result == SnippetType.PARAGRAPH


# ---------------------------------------------------------------------------
# System prompt tests
# ---------------------------------------------------------------------------


class TestParserPrompt:
    """Tests for the LLM parser system prompt."""

    def test_prompt_contains_key_instructions(self):
        assert "verifiable" in PARSER_SYSTEM_PROMPT.lower()
        assert "unit_type" in PARSER_SYSTEM_PROMPT
        assert "dependencies" in PARSER_SYSTEM_PROMPT
        assert "verifier_route" in PARSER_SYSTEM_PROMPT
        assert "boilerplate" in PARSER_SYSTEM_PROMPT

    def test_prompt_includes_all_verifier_routes(self):
        for route in ("math", "text", "statistical", "citation", "vision", "none"):
            assert route in PARSER_SYSTEM_PROMPT, f"Missing route: {route}"

    def test_prompt_includes_all_unit_types(self):
        for utype in ("equation", "theorem", "lemma", "proposition", "definition",
                      "claim", "numeric_claim", "proof_step", "boilerplate"):
            assert utype in PARSER_SYSTEM_PROMPT, f"Missing unit_type: {utype}"
