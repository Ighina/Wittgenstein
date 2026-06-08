"""Phase 13: Report generation.

Generates metrics.json, predictions.json, confusion_matrix.csv,
and a comprehensive run_summary.md markdown report.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from src.evaluation.metrics import (
    generate_classification_report,
    generate_confusion_matrix,
)
from src.models import (
    AlignedPrediction,
    EvaluationMetrics,
    PaperContentSchemaReport,
    PaperPrediction,
    PredictedError,
)


def generate_report(
    metrics: EvaluationMetrics,
    aligned_predictions: list[AlignedPrediction],
    predictions: list[PaperPrediction],
    schema_report: Optional[PaperContentSchemaReport],
    output_dir: str | Path,
    ground_truth_df: Optional[pd.DataFrame] = None,
) -> Path:
    """Generate all output files for a pipeline run.

    Writes:
        - metrics.json
        - predictions.json
        - confusion_matrix.csv
        - run_summary.md

    Args:
        metrics: Computed evaluation metrics.
        aligned_predictions: Aligned prediction results.
        predictions: Raw paper predictions.
        schema_report: Dataset schema analysis report.
        output_dir: Directory to write output files.
        ground_truth_df: Ground truth DataFrame (for statistics).

    Returns:
        Path to the output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Generating reports in: {output_dir}")

    # 1. metrics.json
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        metrics.model_dump_json(indent=2)
    )
    logger.info(f"Written: {metrics_path}")

    # 2. predictions.json
    predictions_path = output_dir / "predictions.json"
    pred_data = {
        "generated_at": datetime.now().isoformat(),
        "total_papers": len(predictions),
        "total_predictions": sum(len(p.predicted_errors) for p in predictions),
        "predictions": [
            p.model_dump() for p in predictions
        ],
        "aligned": [
            a.model_dump() for a in aligned_predictions
        ],
    }
    predictions_path.write_text(
        json.dumps(pred_data, indent=2, default=str)
    )
    logger.info(f"Written: {predictions_path}")

    # 3. confusion_matrix.csv
    cm = generate_confusion_matrix(aligned_predictions)
    cm_df = pd.DataFrame(
        cm,
        columns=["Predicted: No Error", "Predicted: Error"],
        index=["Actual: No Error", "Actual: Error"],
    )
    cm_path = output_dir / "confusion_matrix.csv"
    cm_df.to_csv(cm_path)
    logger.info(f"Written: {cm_path}")

    # 4. run_summary.md
    summary_path = output_dir / "run_summary.md"
    summary = _build_summary(
        metrics=metrics,
        predictions=predictions,
        schema_report=schema_report,
        classification_report=generate_classification_report(aligned_predictions),
        confusion_matrix_df=cm_df,
        ground_truth_df=ground_truth_df,
    )
    summary_path.write_text(summary)
    logger.info(f"Written: {summary_path}")

    return output_dir


def _build_summary(
    metrics: EvaluationMetrics,
    predictions: list[PaperPrediction],
    schema_report: Optional[PaperContentSchemaReport],
    classification_report: str,
    confusion_matrix_df: pd.DataFrame,
    ground_truth_df: Optional[pd.DataFrame],
) -> str:
    """Build the run_summary.md markdown content."""
    lines: list[str] = []

    lines.append("# Paperena Verification Pipeline — Run Summary")
    lines.append(f"\n**Generated:** {datetime.now().isoformat()}")
    lines.append("")

    # Dataset statistics
    lines.append("## Dataset Statistics")
    lines.append("")
    if schema_report:
        lines.append(f"- **Total papers:** {schema_report.total_rows}")
        lines.append(f"- **Columns:** {', '.join(schema_report.column_names)}")
        lines.append(f"- **Text items:** {schema_report.text_item_count}")
        lines.append(f"- **Image items:** {schema_report.image_item_count}")
        lines.append(
            f"- **Papers with images:** {schema_report.rows_with_images}"
        )
        lines.append(
            f"- **Papers with local context:** {schema_report.rows_with_local_content}"
        )
        lines.append("")

        lines.append("### Paper Categories")
        lines.append("")
        for cat in schema_report.paper_categories:
            lines.append(f"- **{cat['category']}**: {cat['count']}")
        lines.append("")

        lines.append("### Error Categories")
        lines.append("")
        for cat in schema_report.error_categories:
            lines.append(f"- **{cat['category']}**: {cat['count']}")
        lines.append("")

        lines.append("### Error Severities")
        lines.append("")
        for sev in schema_report.error_severities:
            lines.append(f"- **{sev['severity']}**: {sev['count']}")
        lines.append("")

    # Pipeline statistics
    lines.append("## Pipeline Statistics")
    lines.append("")

    total_snippets = sum(p.total_snippets for p in predictions)
    total_verified = sum(p.snippets_verified for p in predictions)
    total_errors = sum(p.errors_detected for p in predictions)

    lines.append(f"- **Papers processed:** {len(predictions)}")
    lines.append(f"- **Total snippets:** {total_snippets}")
    lines.append(f"- **Snippets verified:** {total_verified}")
    lines.append(f"- **Errors detected:** {total_errors}")
    lines.append("")

    # Verifier usage
    lines.append("### Verifier Usage Breakdown")
    lines.append("")
    verifier_totals: dict[str, int] = {}
    for p in predictions:
        for vname, count in p.verifier_usage.items():
            verifier_totals[vname] = verifier_totals.get(vname, 0) + count
    for vname, count in sorted(verifier_totals.items()):
        lines.append(f"- **{vname}**: {count} snippets")
    lines.append("")

    # Overall metrics
    lines.append("## Overall Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| ------ | ----- |")
    lines.append(f"| Accuracy | {metrics.accuracy:.4f} |")
    lines.append(f"| Precision | {metrics.precision:.4f} |")
    lines.append(f"| Recall | {metrics.recall:.4f} |")
    lines.append(f"| F1 Score | {metrics.f1_score:.4f} |")
    lines.append(f"| True Positives | {metrics.true_positives} |")
    lines.append(f"| False Positives | {metrics.false_positives} |")
    lines.append(f"| False Negatives | {metrics.false_negatives} |")
    lines.append("")

    # Confusion matrix
    lines.append("### Confusion Matrix")
    lines.append("")
    lines.append(confusion_matrix_df.to_markdown())
    lines.append("")

    # Category-level performance
    lines.append("## Category-Level Performance")
    lines.append("")

    if metrics.by_error_category:
        lines.append("### By Error Category")
        lines.append("")
        lines.append(
            "| Category | Precision | Recall | F1 | TP | FP | FN | Support |"
        )
        lines.append(
            "| -------- | --------- | ------ | -- | -- | -- | -- | ------- |"
        )
        for cm in metrics.by_error_category:
            lines.append(
                f"| {cm.category_name} | {cm.precision:.3f} | "
                f"{cm.recall:.3f} | {cm.f1_score:.3f} | "
                f"{cm.true_positives} | {cm.false_positives} | "
                f"{cm.false_negatives} | {cm.support} |"
            )
        lines.append("")

    if metrics.by_error_severity:
        lines.append("### By Error Severity")
        lines.append("")
        lines.append(
            "| Severity | Precision | Recall | F1 | TP | FP | FN | Support |"
        )
        lines.append(
            "| -------- | --------- | ------ | -- | -- | -- | -- | ------- |"
        )
        for cm in metrics.by_error_severity:
            lines.append(
                f"| {cm.category_name} | {cm.precision:.3f} | "
                f"{cm.recall:.3f} | {cm.f1_score:.3f} | "
                f"{cm.true_positives} | {cm.false_positives} | "
                f"{cm.false_negatives} | {cm.support} |"
            )
        lines.append("")

    if metrics.by_paper_category:
        lines.append("### By Paper Category")
        lines.append("")
        lines.append(
            "| Category | Precision | Recall | F1 | TP | FP | FN | Support |"
        )
        lines.append(
            "| -------- | --------- | ------ | -- | -- | -- | -- | ------- |"
        )
        for cm in metrics.by_paper_category:
            lines.append(
                f"| {cm.category_name} | {cm.precision:.3f} | "
                f"{cm.recall:.3f} | {cm.f1_score:.3f} | "
                f"{cm.true_positives} | {cm.false_positives} | "
                f"{cm.false_negatives} | {cm.support} |"
            )
        lines.append("")

    # Classification report
    lines.append("## Classification Report (sklearn)")
    lines.append("")
    lines.append("```")
    lines.append(classification_report)
    lines.append("```")
    lines.append("")

    # Common failure modes (from low-confidence / high-FP categories)
    lines.append("## Common Failure Modes")
    lines.append("")
    # Identify categories with lowest F1
    all_cats = (
        metrics.by_error_category
        + metrics.by_error_severity
    )
    low_performing = sorted(all_cats, key=lambda c: c.f1_score)[:5]
    if low_performing:
        for cat in low_performing:
            lines.append(
                f"- **{cat.category_name}**: F1={cat.f1_score:.3f} "
                f"(Precision={cat.precision:.3f}, Recall={cat.recall:.3f}, "
                f"FP={cat.false_positives}, FN={cat.false_negatives})"
            )
    else:
        lines.append("_No failure modes identified (insufficient data)._")
    lines.append("")

    lines.append("---")
    lines.append("\n*Report generated by Paperena Verification Pipeline*")

    return "\n".join(lines)
