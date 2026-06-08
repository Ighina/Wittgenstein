"""Phase 12: Evaluation metrics computation.

Computes binary classification metrics, per-category breakdowns,
and generates sklearn-compatible reports.
"""

from __future__ import annotations

from collections import defaultdict

from loguru import logger
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.models import AlignedPrediction, CategoryMetrics, EvaluationMetrics


def evaluate_predictions(
    aligned: list[AlignedPrediction],
    ground_truth_df: pd.DataFrame,
) -> EvaluationMetrics:
    """Compute comprehensive evaluation metrics from aligned predictions.

    Args:
        aligned: List of aligned predictions from the alignment step.
        ground_truth_df: Original ground-truth DataFrame for context.

    Returns:
        EvaluationMetrics with binary, per-category, and per-severity breakdowns.
    """
    logger.info(f"Evaluating {len(aligned)} aligned predictions")

    # Separate into TP, FP, FN
    tp_predictions = [a for a in aligned if a.is_true_positive]
    fp_predictions = [a for a in aligned if a.is_false_positive]
    fn_predictions = [a for a in aligned if not a.is_true_positive and not a.is_false_positive]

    # True negatives: papers without errors where we predicted none
    # In this dataset every paper has at least one error, so TN is hard to compute
    # We approximate: papers where no ground truth errors and no predictions
    all_paper_ids = set(ground_truth_df["doi/arxiv_id"].astype(str))
    predicted_paper_ids = set(a.paper_id for a in aligned)
    # Papers with ground truth errors
    gt_paper_ids = set(ground_truth_df["doi/arxiv_id"].astype(str))
    tn_count = 0  # Every paper has errors in this dataset

    tp_count = len(tp_predictions)
    fp_count = len(fp_predictions)
    fn_count = len(fn_predictions)

    # Compute overall metrics
    y_true = []
    y_pred = []

    for a in aligned:
        # True label: 1 if there is a real error at this location
        true_label = 1 if a.ground_truth_category else 0
        y_true.append(true_label)

        # Predicted label: 1 if we predicted an error
        pred_label = 1 if a.predicted.error_category else 0
        y_pred.append(pred_label)

    # Only compute metrics if we have both classes
    if len(set(y_true)) > 1 and len(set(y_pred)) > 1:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
    else:
        accuracy = tp_count / max(1, tp_count + fp_count + fn_count)
        precision = tp_count / max(1, tp_count + fp_count)
        recall = tp_count / max(1, tp_count + fn_count)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)

    logger.info(
        f"Overall metrics: Accuracy={accuracy:.3f}, Precision={precision:.3f}, "
        f"Recall={recall:.3f}, F1={f1:.3f}"
    )

    # Per-category metrics
    by_error_category = _compute_category_metrics(
        aligned, "error_category", ground_truth_df
    )
    by_error_severity = _compute_category_metrics(
        aligned, "error_severity", ground_truth_df
    )
    by_paper_category = _compute_category_metrics(
        aligned, "paper_category", ground_truth_df
    )

    return EvaluationMetrics(
        true_positives=tp_count,
        true_negatives=tn_count,
        false_positives=fp_count,
        false_negatives=fn_count,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        by_error_category=by_error_category,
        by_error_severity=by_error_severity,
        by_paper_category=by_paper_category,
        total_papers=len(all_paper_ids),
        total_ground_truth_errors=len(ground_truth_df),
        total_predictions=len(aligned),
        matched_predictions=tp_count + fp_count,
    )


def _compute_category_metrics(
    aligned: list[AlignedPrediction],
    category_field: str,
    ground_truth_df: pd.DataFrame,
) -> list[CategoryMetrics]:
    """Compute per-category precision, recall, and F1.

    Args:
        aligned: Aligned predictions.
        category_field: Which field to group by ("error_category", "error_severity", or "paper_category").
        ground_truth_df: Ground truth DataFrame.

    Returns:
        List of CategoryMetrics for each unique category value.
    """
    # Build mapping of paper_id → category value
    paper_to_category: dict[str, str] = {}
    for _, row in ground_truth_df.iterrows():
        paper_id = str(row["doi/arxiv_id"])
        if category_field == "paper_category":
            paper_to_category[paper_id] = row["paper_category"]
        elif category_field == "error_severity":
            paper_to_category[paper_id] = row["error_severity"]
        # For error_category, it's per-aligned-prediction

    categories: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0}
    )

    for a in aligned:
        if category_field == "error_category":
            cat = a.ground_truth_category or a.predicted.error_category or "Unknown"
        elif category_field == "error_severity":
            cat = a.ground_truth_severity or "Unknown"
        elif category_field == "paper_category":
            cat = paper_to_category.get(a.paper_id, "Unknown")
        else:
            cat = "Unknown"

        if a.is_true_positive:
            categories[cat]["tp"] += 1
        elif a.is_false_positive:
            categories[cat]["fp"] += 1
        else:
            # False negative
            categories[cat]["fn"] += 1

    result: list[CategoryMetrics] = []
    for cat_name, counts in sorted(categories.items()):
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        support = tp + fn

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        accuracy = tp / max(1, tp + fp + fn)

        result.append(CategoryMetrics(
            category_name=cat_name,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            precision=precision,
            recall=recall,
            f1_score=f1,
            accuracy=accuracy,
            support=support,
        ))

    return result


def generate_classification_report(
    aligned: list[AlignedPrediction],
) -> str:
    """Generate a sklearn-style classification report string."""
    y_true = []
    y_pred = []

    for a in aligned:
        y_true.append(1 if a.ground_truth_category else 0)
        y_pred.append(1 if a.predicted.error_category else 0)

    if len(set(y_true)) <= 1:
        return "Cannot generate classification report: only one class present in ground truth."

    return classification_report(
        y_true,
        y_pred,
        target_names=["No Error", "Error"],
        zero_division=0,
    )


def generate_confusion_matrix(
    aligned: list[AlignedPrediction],
) -> list[list[int]]:
    """Generate a confusion matrix as a list of lists."""
    y_true = []
    y_pred = []

    for a in aligned:
        y_true.append(1 if a.ground_truth_category else 0)
        y_pred.append(1 if a.predicted.error_category else 0)

    if not y_true:
        return [[0, 0], [0, 0]]

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return cm.tolist()
