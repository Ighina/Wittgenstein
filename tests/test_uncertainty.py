"""Tests for uncertainty-driven orchestration."""

import pytest

from src.config import PipelineConfig
from src.models import (
    EquationBlock,
    NormalizedPaper,
    PaperSection,
    SnippetType,
    VerificationSnippet,
)
from src.orchestrator.router import resolve_route_to_verifier
from src.orchestrator.uncertainty_orchestrator import UncertaintyOrchestrator
from src.verifiers.triage_verifier import TriageVerifier


@pytest.fixture
def mock_config():
    cfg = PipelineConfig(orchestration_mode="uncertainty", uncertainty_threshold=0.3)
    cfg.llm.provider = "mock"
    cfg.llm.num_workers = 1  # deterministic
    return cfg


@pytest.fixture
def demo_paper():
    return NormalizedPaper(
        paper_id="demo",
        title="Demo",
        paper_category="Math",
        sections=[
            PaperSection(id="s0", section_title="Introduction", section_level=1,
                         content="A routine introduction with standard background."),
            PaperSection(id="s1", section_title="Methods", section_level=1,
                         content="We claim the result is incorrect under conditions."),
        ],
        equations=[
            EquationBlock(id="e0", equation_label="1", latex="x^2 + y^2 = z^2",
                          display_mode=True, context_before="Pythagoras", context_after=""),
        ],
    )


class TestRouteResolution:
    def test_math_route_maps_to_math_equation(self, mock_config):
        snip = VerificationSnippet(
            snippet_id="p_eq_0", snippet_type=SnippetType.EQUATION, paper_id="p",
            location="Eq 1", content="x=y", metadata={"latex": "x = y"},
        )
        assert resolve_route_to_verifier("math", snip, mock_config) == "math_equation"

    def test_none_route_returns_none(self, mock_config):
        snip = VerificationSnippet(
            snippet_id="p_sec_0", snippet_type=SnippetType.SECTION, paper_id="p",
            location="Intro", content="text",
        )
        assert resolve_route_to_verifier("none", snip, mock_config) is None

    def test_unknown_route_falls_back_to_type(self, mock_config):
        snip = VerificationSnippet(
            snippet_id="p_sec_0", snippet_type=SnippetType.SECTION, paper_id="p",
            location="Intro", content="text",
        )
        # Unknown semantic route → structural fallback (SECTION → text).
        assert resolve_route_to_verifier("astrology", snip, mock_config) == "text"


class TestTriageVerifier:
    def test_triage_scores_equation_high(self, mock_config):
        v = TriageVerifier(config=mock_config)
        snip = VerificationSnippet(
            snippet_id="p_eq_0", snippet_type=SnippetType.EQUATION, paper_id="p",
            location="Eq 1", content="ctx", metadata={"latex": "x^2 + y^2 = z^2"},
        )
        t = v.triage(snip)
        assert 0.0 <= t.uncertainty <= 1.0
        assert t.uncertainty >= mock_config.uncertainty_threshold
        assert t.suggested_route == "math"

    def test_triage_scores_routine_low(self, mock_config):
        v = TriageVerifier(config=mock_config)
        snip = VerificationSnippet(
            snippet_id="p_sec_0", snippet_type=SnippetType.SECTION, paper_id="p",
            location="Intro", content="A perfectly routine paragraph of prose.",
        )
        t = v.triage(snip)
        assert t.uncertainty < mock_config.uncertainty_threshold


class TestUncertaintyOrchestrator:
    def test_routes_specialists_by_uncertainty(self, mock_config, demo_paper):
        orch = UncertaintyOrchestrator(config=mock_config)
        pred = orch.run(demo_paper)

        # Every snippet is triaged; the map is complete.
        assert len(pred.uncertainty_map) == pred.total_snippets == 3

        by_id = {e["snippet_id"]: e for e in pred.uncertainty_map}

        # The routine intro is below threshold and NOT escalated.
        intro = by_id["demo_sec_0"]
        assert intro["selected"] is False

        # The equation is escalated to the math verifier.
        eq = by_id["e0"]
        assert eq["selected"] is True
        assert eq["routed_to"] == "math_equation"

        # Specialist checks < total snippets — effort was concentrated.
        assert pred.snippets_verified < pred.total_snippets
        assert pred.verifier_usage.get("triage") == pred.total_snippets

    def test_budget_caps_specialist_calls(self, mock_config, demo_paper):
        mock_config.uncertainty_budget = 1  # only the single most-uncertain node
        orch = UncertaintyOrchestrator(config=mock_config)
        pred = orch.run(demo_paper)
        assert pred.snippets_verified == 1
