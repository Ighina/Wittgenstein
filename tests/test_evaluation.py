"""Tests for evaluation modules."""

import pytest
import pandas as pd
from src.models import (
    AlignedPrediction,
    PaperPrediction,
    PredictedError,
)
from src.evaluation.alignment import match_predictions_to_ground_truth
from src.evaluation.metrics import evaluate_predictions


@pytest.fixture
def ground_truth_df():
    """Create a minimal ground truth DataFrame."""
    return pd.DataFrame([
        {
            "doi/arxiv_id": "paper1",
            "title": "Paper 1",
            "paper_category": "Mathematics",
            "error_category": "Equation / proof",
            "error_location": "Lemma 3,4",
            "error_severity": "retract",
            "error_annotation": "An error in the proof.",
        },
        {
            "doi/arxiv_id": "paper2",
            "title": "Paper 2",
            "paper_category": "Biology",
            "error_category": "Figure duplication",
            "error_location": "Fig 5",
            "error_severity": "errata",
            "error_annotation": "Figure 5 is a duplicate.",
        },
    ])


@pytest.fixture
def predictions():
    """Create sample paper predictions."""
    return [
        PaperPrediction(
            paper_id="paper1",
            title="Paper 1",
            paper_category="Mathematics",
            predicted_errors=[
                PredictedError(
                    error_category="Equation / proof",
                    error_location="Lemma 3",
                    confidence=0.85,
                    supporting_evidence="Mock finding.",
                    verifier_name="text",
                    snippet_id="s1",
                ),
            ],
            total_snippets=10,
            snippets_verified=10,
            errors_detected=1,
        ),
        PaperPrediction(
            paper_id="paper2",
            title="Paper 2",
            paper_category="Biology",
            predicted_errors=[
                PredictedError(
                    error_category="Figure duplication",
                    error_location="Fig 5",
                    confidence=0.90,
                    supporting_evidence="Mock finding.",
                    verifier_name="vision",
                    snippet_id="s2",
                ),
            ],
            total_snippets=8,
            snippets_verified=8,
            errors_detected=1,
        ),
    ]


class TestAlignment:
    """Tests for prediction-to-ground-truth alignment."""

    def test_align_predictions(self, predictions, ground_truth_df):
        """Test that predictions are aligned with ground truth."""
        aligned = match_predictions_to_ground_truth(predictions, ground_truth_df)
        assert len(aligned) > 0

    def test_exact_match_produces_true_positive(self, predictions, ground_truth_df):
        """Test that matching prediction+GT produces a true positive."""
        aligned = match_predictions_to_ground_truth(predictions, ground_truth_df)
        tp = [a for a in aligned if a.is_true_positive]
        # paper2 has perfect Fig 5 match
        assert len(tp) >= 1

    def test_partial_match_scores(self, predictions, ground_truth_df):
        """Test that partial matches have appropriate scores."""
        aligned = match_predictions_to_ground_truth(predictions, ground_truth_df)
        for a in aligned:
            if a.matched_ground_truth:
                assert 0.0 <= a.match_quality <= 1.0

    def test_correct_ground_truth_fields(self, predictions, ground_truth_df):
        """Test that aligned predictions have correct ground truth fields."""
        aligned = match_predictions_to_ground_truth(predictions, ground_truth_df)
        matched = [a for a in aligned if a.matched_ground_truth]
        for a in matched:
            assert a.ground_truth_category is not None
            assert a.ground_truth_location is not None
            assert a.ground_truth_severity is not None


class TestMetrics:
    """Tests for metrics computation."""

    def test_evaluate_returns_metrics(self, predictions, ground_truth_df):
        """Test that evaluation produces valid metrics."""
        aligned = match_predictions_to_ground_truth(predictions, ground_truth_df)
        metrics = evaluate_predictions(aligned, ground_truth_df)
        assert metrics.total_papers > 0
        assert metrics.total_ground_truth_errors == 2
        assert 0.0 <= metrics.accuracy <= 1.0
        assert 0.0 <= metrics.precision <= 1.0
        assert 0.0 <= metrics.recall <= 1.0
        assert 0.0 <= metrics.f1_score <= 1.0

    def test_metrics_have_category_breakdowns(self, predictions, ground_truth_df):
        """Test that per-category metrics are computed."""
        aligned = match_predictions_to_ground_truth(predictions, ground_truth_df)
        metrics = evaluate_predictions(aligned, ground_truth_df)
        assert len(metrics.by_error_category) > 0
        assert len(metrics.by_error_severity) > 0

    def test_category_metrics_are_valid(self, predictions, ground_truth_df):
        """Test that category metrics have valid ranges."""
        aligned = match_predictions_to_ground_truth(predictions, ground_truth_df)
        metrics = evaluate_predictions(aligned, ground_truth_df)
        for cm in metrics.by_error_category:
            assert 0.0 <= cm.precision <= 1.0
            assert 0.0 <= cm.recall <= 1.0
            assert 0.0 <= cm.f1_score <= 1.0
