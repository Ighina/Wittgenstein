#!/usr/bin/env python3
"""End-to-end smoke test for the LLM parser pipeline (mock mode)."""
from src.config import PipelineConfig
from src.parser.llm_content_parser import llm_parse_paper
from src.parser.enriched_segmenter import segment_enriched_paper
from src.orchestrator.orchestrator import VerificationOrchestrator


def main():
    paper_content = [
        {
            "type": "text",
            "text": """## 1. Introduction

Let X be a real Banach space. Define the bidual X** = (X*)* as the dual of the dual space.

## 2. Main Results

**Theorem 2.1.** X is isometrically isomorphic to a subspace of X**.

**Lemma 2.2.** The canonical embedding J: X -> X** is continuous.

The proof follows from the Hahn-Banach theorem: for any x in X,
\\(\\|J(x)\\| = \\|x\\|\\) holds.

**Theorem 2.3.** The error in the approximation is bounded by
\\[\\frac{1}{\\sqrt{n}}\\] for n samples.

The model achieves 97.3% accuracy (p < 0.001) on the test set.

## Acknowledgments

We thank the anonymous reviewers. This work was supported by NSF grant DMS-12345.
""",
        },
    ]

    config = PipelineConfig()
    config.parser_mode = "llm"
    config.llm.provider = "mock"

    # Parse with LLM
    enriched = llm_parse_paper(
        paper_id="test_paper",
        title="Test Paper",
        paper_category="Mathematics",
        paper_content=paper_content,
        config=config,
        decode_images=False,
    )

    print(f"Verifiable units: {len(enriched.verifiable_units)}")
    n_verifiable = sum(1 for u in enriched.verifiable_units if u.is_verifiable)
    n_skipped = sum(1 for u in enriched.verifiable_units if not u.is_verifiable)
    print(f"  Verifiable: {n_verifiable}")
    print(f"  Unverifiable (skipped): {n_skipped}")
    print(f"  Symbols: {len(enriched.symbol_registry)}")

    for u in enriched.verifiable_units:
        status = "VERIFIABLE" if u.is_verifiable else "SKIPPED"
        route = u.verifier_route or "(none)"
        print(f"  [{status}] {u.unit_type:20s} -> {route:12s} | {u.unit_id}")

    # Segment
    snippets = segment_enriched_paper(enriched, config=config)
    print(f"\nSnippets produced: {len(snippets)}")
    for s in snippets:
        route_label = s.verifier_route or "(none)"
        print(f"  {s.snippet_id:30s} type={s.snippet_type.value:12s} route={route_label:12s}")

    # Verify
    orchestrator = VerificationOrchestrator(config=config)
    prediction = orchestrator.run(enriched)
    print(f"\nPrediction: {len(prediction.predicted_errors)} errors detected")
    print(f"  Snippets verified: {prediction.snippets_verified}")
    print(f"  Verifier usage: {prediction.verifier_usage}")
    print("\nDone!")


if __name__ == "__main__":
    main()
