"""ProcessBench dataset loader — mathematical CoT error identification.

ProcessBench (Zheng et al., ACL 2025) is a benchmark for identifying
erroneous steps in chain-of-thought mathematical reasoning.  It contains
3,400 test cases across four splits (GSM8K, MATH, OlympiadBench, OmniMath),
each with a word problem, step-by-step solution, and a label indicating the
first erroneous step (or -1 if all steps are correct).

This module is completely independent of the Paperena paper-verification
pipeline — it loads, converts, and evaluates ProcessBench data without
touching any existing code.

Usage::

    from src.datasets.processbench import load_processbench

    cases = load_processbench("gsm8k")
    for case in cases:
        print(case.problem, case.label)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ProcessBenchCase:
    """A single ProcessBench test case."""

    id: str
    generator: str
    problem: str
    steps: list[str]
    final_answer_correct: bool
    label: int  # -1 = all correct; 0..N = first erroneous step index

    @property
    def has_error(self) -> bool:
        return self.label >= 0

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def error_step_text(self) -> Optional[str]:
        if self.label >= 0 and self.label < len(self.steps):
            return self.steps[self.label]
        return None


@dataclass
class ProcessBenchResult:
    """Result of verifying a single ProcessBench case."""

    case_id: str
    problem: str
    n_steps: int
    true_label: int
    predicted_label: int  # -1 if no error found
    is_correct: bool  # prediction matches label
    step_predictions: list[dict] = field(default_factory=list)
    # Each step_prediction: {step_index, status, confidence, reasoning}


@dataclass
class ProcessBenchMetrics:
    """Aggregate metrics over a ProcessBench split."""

    split_name: str
    total_cases: int
    correct_predictions: int
    accuracy: float
    # Breakdowns
    correct_with_error: int = 0   # correctly identified error-containing cases
    correct_all_correct: int = 0  # correctly identified all-correct cases
    false_positives: int = 0      # predicted error on all-correct case
    false_negatives: int = 0      # missed error on error-containing case
    # Per-step-position accuracy
    position_accuracy: dict[int, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_processbench(split: str = "gsm8k") -> list[ProcessBenchCase]:
    """Load a ProcessBench split from HuggingFace.

    Args:
        split: One of "gsm8k", "math", "olympiadbench", "omnimath".

    Returns:
        List of ProcessBenchCase objects.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets library is required for ProcessBench. "
            "Install with: pip install datasets"
        )

    valid_splits = ("gsm8k", "math", "olympiadbench", "omnimath")
    if split not in valid_splits:
        raise ValueError(
            f"Unknown split {split!r}. Valid splits: {valid_splits}"
        )

    ds = load_dataset("Qwen/ProcessBench", split=split)

    return [
        ProcessBenchCase(
            id=entry["id"],
            generator=entry["generator"],
            problem=entry["problem"],
            steps=list(entry["steps"]),
            final_answer_correct=entry["final_answer_correct"],
            label=entry["label"],
        )
        for entry in ds
    ]


# ---------------------------------------------------------------------------
# Conversion to verification snippets (for use with progressive verifier)
# ---------------------------------------------------------------------------


def case_to_snippets(case: ProcessBenchCase) -> list[dict]:
    """Convert a ProcessBench case into a list of step-level snippets.

    Each snippet wraps a single reasoning step with the problem statement
    as context.  The progressive math verifier can check each step against
    the accumulated context of prior steps.

    Returns a list of dicts compatible with ``VerificationSnippet``
    construction (or direct use by the progressive verifier).
    """
    snippets: list[dict] = []
    for i, step_text in enumerate(case.steps):
        # Build context: problem + all prior steps
        prior_steps = case.steps[:i]
        context_parts = [f"Problem: {case.problem}"]
        for j, prior in enumerate(prior_steps):
            context_parts.append(f"Step {j}: {prior}")
        context_parts.append(f"Current step {i}: {step_text}")
        full_content = "\n\n".join(context_parts)

        # Extract any LaTeX-like math from the step
        latex = _extract_latex(step_text)

        snippets.append({
            "snippet_id": f"{case.id}_step_{i}",
            "snippet_type": "PARAGRAPH",  # CoT steps are prose+math
            "paper_id": case.id,
            "location": f"Step {i}",
            "content": full_content,
            "metadata": {
                "latex": latex,
                "step_index": i,
                "problem": case.problem,
                "prior_steps": prior_steps,
            },
        })
    return snippets


def _extract_latex(text: str) -> str:
    r"""Extract inline LaTeX expressions from a CoT step.

    ProcessBench uses \\(...\\) delimiters for inline math.
    """
    import re

    fragments: list[str] = []
    for match in re.finditer(r"\\\((.*?)\\\)", text):
        fragments.append(match.group(1))
    for match in re.finditer(r"\$(.*?)\$", text):
        fragments.append(match.group(1))
    return " ; ".join(fragments) if fragments else ""


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_processbench_metrics(
    results: list[ProcessBenchResult],
    split_name: str = "",
) -> ProcessBenchMetrics:
    """Compute aggregate metrics from a list of case-level results.

    Args:
        results: List of verification results.
        split_name: Name of the split being evaluated.

    Returns:
        ProcessBenchMetrics with accuracy and breakdowns.
    """
    n = len(results)
    if n == 0:
        return ProcessBenchMetrics(
            split_name=split_name,
            total_cases=0,
            correct_predictions=0,
            accuracy=0.0,
        )

    correct = sum(1 for r in results if r.is_correct)
    correct_w_error = sum(
        1 for r in results if r.is_correct and r.true_label >= 0
    )
    correct_all_ok = sum(
        1 for r in results if r.is_correct and r.true_label == -1
    )
    fp_count = sum(
        1 for r in results if not r.is_correct and r.true_label == -1
    )
    fn_count = sum(
        1 for r in results if not r.is_correct and r.true_label >= 0
    )

    # Per-position accuracy
    pos_correct: dict[int, int] = {}
    pos_total: dict[int, int] = {}
    for r in results:
        lbl = r.true_label
        pos_total[lbl] = pos_total.get(lbl, 0) + 1
        if r.is_correct:
            pos_correct[lbl] = pos_correct.get(lbl, 0) + 1

    position_accuracy = {
        k: pos_correct.get(k, 0) / max(1, pos_total.get(k, 0))
        for k in sorted(pos_total.keys())
    }

    return ProcessBenchMetrics(
        split_name=split_name,
        total_cases=n,
        correct_predictions=correct,
        accuracy=correct / n,
        correct_with_error=correct_w_error,
        correct_all_correct=correct_all_ok,
        false_positives=fp_count,
        false_negatives=fn_count,
        position_accuracy=position_accuracy,
    )


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------


def describe_split(split: str) -> dict:
    """Return metadata about a ProcessBench split."""
    info = {
        "gsm8k": {
            "name": "GSM8K",
            "difficulty": "Grade-school math",
            "n_cases": 400,
            "generator": "Qwen2-7B-Instruct",
        },
        "math": {
            "name": "MATH",
            "difficulty": "Competition-level",
            "n_cases": 1000,
            "generator": "Qwen2-7B-Instruct",
        },
        "olympiadbench": {
            "name": "OlympiadBench",
            "difficulty": "Olympiad-level",
            "n_cases": 1000,
            "generator": "Qwen2-7B-Instruct",
        },
        "omnimath": {
            "name": "OmniMath",
            "difficulty": "Advanced",
            "n_cases": 1000,
            "generator": "Qwen2-7B-Instruct",
        },
    }
    return info.get(split, {})
