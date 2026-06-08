"""Tests for the ProcessBench dataset loader and utilities.

These tests verify the data model, snippet conversion, and metrics
computation without requiring network access (mock data).
"""

import pytest

from src.datasets.processbench import (
    ProcessBenchCase,
    ProcessBenchMetrics,
    ProcessBenchResult,
    _extract_latex,
    case_to_snippets,
    compute_processbench_metrics,
)


# ---------------------------------------------------------------------------
# ProcessBenchCase tests
# ---------------------------------------------------------------------------

class TestProcessBenchCase:
    def test_has_error_true(self):
        case = ProcessBenchCase(
            id="test-0",
            generator="test-model",
            problem="What is 2+2?",
            steps=["Step 0: 2+2=4", "Step 1: 4+1=5"],
            final_answer_correct=False,
            label=1,
        )
        assert case.has_error is True
        assert case.error_step_text == "Step 1: 4+1=5"
        assert case.n_steps == 2

    def test_has_error_false(self):
        case = ProcessBenchCase(
            id="test-1",
            generator="test-model",
            problem="What is 2+2?",
            steps=["Step 0: 2+2=4"],
            final_answer_correct=True,
            label=-1,
        )
        assert case.has_error is False
        assert case.error_step_text is None

    def test_label_out_of_range(self):
        case = ProcessBenchCase(
            id="test-2",
            generator="test-model",
            problem="What is 2+2?",
            steps=["Step 0: 2+2=4"],
            final_answer_correct=True,
            label=5,
        )
        assert case.error_step_text is None


# ---------------------------------------------------------------------------
# LaTeX extraction tests
# ---------------------------------------------------------------------------

class TestLatexExtraction:
    def test_extract_paren_delimiters(self):
        text = r"We have \(x^2 + y^2 = z^2\) and then \(a+b\) is the result."
        result = _extract_latex(text)
        assert "x^2 + y^2 = z^2" in result
        assert "a+b" in result

    def test_extract_dollar_delimiters(self):
        text = r"The formula $E = mc^2$ is well known."
        result = _extract_latex(text)
        assert "E = mc^2" in result

    def test_no_latex(self):
        text = "This step has no mathematical notation."
        result = _extract_latex(text)
        assert result == ""


# ---------------------------------------------------------------------------
# Snippet conversion tests
# ---------------------------------------------------------------------------

class TestCaseToSnippets:
    def test_conversion(self):
        case = ProcessBenchCase(
            id="test-0",
            generator="test-model",
            problem="Find x if 2x + 3 = 7",
            steps=[
                r"Subtract 3: \(2x = 4\)",
                r"Divide by 2: \(x = 3\)",
            ],
            final_answer_correct=False,
            label=1,
        )
        snippets = case_to_snippets(case)
        assert len(snippets) == 2

        # First snippet
        s0 = snippets[0]
        assert s0["snippet_id"] == "test-0_step_0"
        assert s0["metadata"]["step_index"] == 0
        assert "Find x if 2x + 3 = 7" in s0["content"]
        assert s0["metadata"]["prior_steps"] == []

        # Second snippet includes prior step as context
        s1 = snippets[1]
        assert s1["snippet_id"] == "test-0_step_1"
        assert s1["metadata"]["step_index"] == 1
        assert "Subtract 3" in s1["content"]
        assert len(s1["metadata"]["prior_steps"]) == 1

    def test_latex_extraction_in_snippet(self):
        case = ProcessBenchCase(
            id="test-0",
            generator="test-model",
            problem="Solve for x",
            steps=[r"We compute \(a^2 + b^2\)"],
            final_answer_correct=True,
            label=-1,
        )
        snippets = case_to_snippets(case)
        assert "a^2 + b^2" in snippets[0]["metadata"]["latex"]


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

class TestProcessBenchMetrics:
    def test_all_correct(self):
        results = [
            ProcessBenchResult(
                case_id="0", problem="p", n_steps=3,
                true_label=-1, predicted_label=-1, is_correct=True,
            ),
            ProcessBenchResult(
                case_id="1", problem="q", n_steps=4,
                true_label=2, predicted_label=2, is_correct=True,
            ),
        ]
        metrics = compute_processbench_metrics(results, "test")
        assert metrics.total_cases == 2
        assert metrics.accuracy == 1.0
        assert metrics.correct_predictions == 2

    def test_all_wrong(self):
        results = [
            ProcessBenchResult(
                case_id="0", problem="p", n_steps=3,
                true_label=-1, predicted_label=0, is_correct=False,
            ),
            ProcessBenchResult(
                case_id="1", problem="q", n_steps=4,
                true_label=1, predicted_label=-1, is_correct=False,
            ),
        ]
        metrics = compute_processbench_metrics(results, "test")
        assert metrics.accuracy == 0.0
        assert metrics.false_positives == 1
        assert metrics.false_negatives == 1

    def test_mixed(self):
        results = [
            ProcessBenchResult(
                case_id="0", problem="p", n_steps=3,
                true_label=-1, predicted_label=-1, is_correct=True,
            ),
            ProcessBenchResult(
                case_id="1", problem="q", n_steps=4,
                true_label=2, predicted_label=2, is_correct=True,
            ),
            ProcessBenchResult(
                case_id="2", problem="r", n_steps=5,
                true_label=3, predicted_label=-1, is_correct=False,
            ),
        ]
        metrics = compute_processbench_metrics(results, "test")
        assert metrics.accuracy == 2 / 3
        assert metrics.correct_with_error == 1
        assert metrics.correct_all_correct == 1
        assert metrics.false_negatives == 1

    def test_position_accuracy(self):
        results = [
            ProcessBenchResult(
                case_id="0", problem="p", n_steps=3,
                true_label=1, predicted_label=1, is_correct=True,
            ),
            ProcessBenchResult(
                case_id="1", problem="q", n_steps=3,
                true_label=1, predicted_label=0, is_correct=False,
            ),
            ProcessBenchResult(
                case_id="2", problem="r", n_steps=3,
                true_label=-1, predicted_label=-1, is_correct=True,
            ),
        ]
        metrics = compute_processbench_metrics(results, "test")
        assert 1 in metrics.position_accuracy
        assert metrics.position_accuracy[1] == pytest.approx(0.5)

    def test_empty_results(self):
        metrics = compute_processbench_metrics([], "test")
        assert metrics.total_cases == 0
        assert metrics.accuracy == 0.0
