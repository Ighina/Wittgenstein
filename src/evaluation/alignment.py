"""Phase 11: Ground-truth alignment.

Matches predicted errors against the dataset's annotated ground truth.

By default this mirrors the original benchmark (see ``src/run_eval.py``) and
uses an LLM *judge* to decide, per paper, which predictions semantically
correspond to which annotated errors. The judge compares the substance of an
error rather than its surface form, so a correct finding referenced with a
different location string is still credited. A structural fuzzy location
matcher is retained as a fallback for offline/mock runs and whenever a judge
call fails.
"""

from __future__ import annotations

from typing import Optional

from loguru import logger
import pandas as pd

from src.config import PipelineConfig
from src.evaluation.judge import judge_matches
from src.models import AlignedPrediction, PaperPrediction, PredictedError
from src.parser.location_parser import fuzzy_match_locations


def match_predictions_to_ground_truth(
    predictions: list[PaperPrediction],
    ground_truth_df: pd.DataFrame,
    config: Optional[PipelineConfig] = None,
    match_threshold: float = 0.6,
) -> list[AlignedPrediction]:
    """Align all predictions against the ground-truth dataset.

    The matching strategy is chosen from ``config``:
      * If ``config.use_llm_judge`` is set and the LLM provider is not "mock",
        an LLM judge decides matches semantically (the benchmark's approach).
      * Otherwise (no config, mock provider, or judge disabled), predictions
        are matched with the structural fuzzy location matcher.

    Args:
        predictions: List of paper-level predictions from the pipeline.
        ground_truth_df: DataFrame with ground-truth annotations.
        config: Pipeline configuration controlling the judge backend. When
            ``None``, the fuzzy fallback is used (keeps unit tests offline).
        match_threshold: Minimum fuzzy match score (0.0-1.0) to consider a
            prediction matched — used only by the fuzzy fallback.

    Returns:
        List of AlignedPrediction objects (TP/FP plus synthetic FN entries).
    """
    gt_by_paper = _index_ground_truth(ground_truth_df)

    # -- Category filter: restrict to a single error category --
    category_filter = getattr(config, "eval_category_filter", None) if config else None
    if category_filter:
        logger.info(
            f"Filtering evaluation to category: {category_filter!r}"
        )
        # Filter predictions
        for p in predictions:
            p.predicted_errors = [
                pe for pe in p.predicted_errors
                if pe.error_category == category_filter
            ]
        # Filter ground truth
        gt_by_paper = {
            pid: [g for g in entries if g["error_category"] == category_filter]
            for pid, entries in gt_by_paper.items()
        }
        # Remove papers with no matching ground truth AND no predictions
        gt_by_paper = {pid: entries for pid, entries in gt_by_paper.items() if entries}

    use_judge = (
        config is not None
        and getattr(config, "use_llm_judge", False)
        and config.llm.provider.lower() != "mock"
    )

    logger.info(
        f"Aligning {sum(len(p.predicted_errors) for p in predictions)} predictions "
        f"against {len(ground_truth_df)} ground-truth annotations "
        f"using {'LLM judge' if use_judge else 'fuzzy location matching'}"
    )

    if use_judge:
        aligned = _align_with_judge(predictions, gt_by_paper, config, match_threshold)
    else:
        aligned = _align_fuzzy(predictions, gt_by_paper, match_threshold)

    tp = sum(1 for a in aligned if a.is_true_positive)
    fp = sum(1 for a in aligned if a.is_false_positive)
    fn = sum(1 for a in aligned if not a.is_true_positive and not a.is_false_positive)
    logger.info(f"Alignment complete: {tp} TP, {fp} FP, {fn} FN")

    return aligned


def _index_ground_truth(ground_truth_df: pd.DataFrame) -> dict[str, list[dict]]:
    """Group ground-truth annotation rows by paper id, preserving order."""
    gt_by_paper: dict[str, list[dict]] = {}
    for _, row in ground_truth_df.iterrows():
        paper_id = str(row["doi/arxiv_id"])
        gt_by_paper.setdefault(paper_id, []).append({
            "error_category": row["error_category"],
            "error_location": row["error_location"],
            "error_severity": row["error_severity"],
            "error_annotation": row.get("error_annotation", ""),
        })
    return gt_by_paper


# ---------------------------------------------------------------------------
# LLM-judge alignment (primary strategy — matches the original benchmark)
# ---------------------------------------------------------------------------


def _align_with_judge(
    predictions: list[PaperPrediction],
    gt_by_paper: dict[str, list[dict]],
    config: PipelineConfig,
    fuzzy_threshold: float,
) -> list[AlignedPrediction]:
    """Match predictions to ground truth one paper at a time via an LLM judge.

    If the judge call fails for a given paper, that paper falls back to the
    structural fuzzy matcher so a transient error never loses all its results.
    """
    judge_model = getattr(config, "judge_model", None) or config.llm.model
    aligned: list[AlignedPrediction] = []
    seen_paper_ids: set[str] = set()

    for prediction in predictions:
        paper_id = prediction.paper_id
        seen_paper_ids.add(paper_id)
        gt_entries = gt_by_paper.get(paper_id, [])
        preds = prediction.predicted_errors

        # No predictions for this paper → every annotation is a false negative.
        if not preds:
            aligned.extend(_false_negatives_for_paper(paper_id, gt_entries))
            continue

        # No ground truth for this paper → every prediction is a false positive.
        if not gt_entries:
            aligned.extend(
                _false_positive(paper_id, pe) for pe in preds
            )
            continue

        annotations = [
            {
                "location": str(gt.get("error_location", "")),
                "description": str(gt.get("error_annotation", "")),
            }
            for gt in gt_entries
        ]

        try:
            matches = judge_matches(
                annotations=annotations,
                predictions=preds,
                llm_config=config.llm,
                model=judge_model,
            )
        except Exception as exc:
            logger.warning(
                f"Judge failed for paper {paper_id} ({exc}); "
                f"falling back to fuzzy matching for this paper."
            )
            aligned.extend(
                _align_fuzzy_single_paper(paper_id, preds, gt_entries, fuzzy_threshold)
            )
            continue

        aligned.extend(
            _aligned_from_matches(paper_id, preds, gt_entries, matches)
        )

    # Papers that have ground truth but produced no prediction object at all.
    for paper_id, gt_entries in gt_by_paper.items():
        if paper_id not in seen_paper_ids:
            aligned.extend(_false_negatives_for_paper(paper_id, gt_entries))

    return aligned


def _aligned_from_matches(
    paper_id: str,
    preds: list[PredictedError],
    gt_entries: list[dict],
    matches: list[dict],
) -> list[AlignedPrediction]:
    """Build AlignedPrediction objects from validated judge matches."""
    pred_to_gt: dict[int, int] = {
        m["prediction_index"]: m["annotation_index"] for m in matches
    }
    matched_gt_indices: set[int] = set(pred_to_gt.values())

    aligned: list[AlignedPrediction] = []

    # True positives and false positives, in prediction order.
    for pred_idx, pred_error in enumerate(preds):
        if pred_idx in pred_to_gt:
            gt = gt_entries[pred_to_gt[pred_idx]]
            aligned.append(AlignedPrediction(
                paper_id=paper_id,
                predicted=pred_error,
                matched_ground_truth=True,
                ground_truth_category=gt["error_category"],
                ground_truth_location=gt["error_location"],
                ground_truth_severity=gt["error_severity"],
                ground_truth_annotation=gt.get("error_annotation", ""),
                match_quality=1.0,
                is_true_positive=True,
                is_false_positive=False,
            ))
        else:
            aligned.append(_false_positive(paper_id, pred_error))

    # Unmatched annotations → false negatives.
    for gt_idx, gt in enumerate(gt_entries):
        if gt_idx not in matched_gt_indices:
            aligned.append(_false_negative(paper_id, gt))

    return aligned


def _false_positive(paper_id: str, pred_error: PredictedError) -> AlignedPrediction:
    return AlignedPrediction(
        paper_id=paper_id,
        predicted=pred_error,
        matched_ground_truth=False,
        is_true_positive=False,
        is_false_positive=True,
    )


def _false_negative(paper_id: str, gt: dict) -> AlignedPrediction:
    return AlignedPrediction(
        paper_id=paper_id,
        predicted=PredictedError(
            error_category="",
            error_location="",
            confidence=0.0,
            supporting_evidence="",
        ),
        matched_ground_truth=True,
        ground_truth_category=gt["error_category"],
        ground_truth_location=gt["error_location"],
        ground_truth_severity=gt["error_severity"],
        ground_truth_annotation=gt.get("error_annotation", ""),
        match_quality=1.0,
        is_true_positive=False,
        is_false_positive=False,
    )


def _false_negatives_for_paper(
    paper_id: str, gt_entries: list[dict]
) -> list[AlignedPrediction]:
    return [_false_negative(paper_id, gt) for gt in gt_entries]


# ---------------------------------------------------------------------------
# Fuzzy location alignment (fallback strategy)
# ---------------------------------------------------------------------------


def _align_fuzzy(
    predictions: list[PaperPrediction],
    gt_by_paper: dict[str, list[dict]],
    threshold: float,
) -> list[AlignedPrediction]:
    """Structural fuzzy-location alignment (the pre-judge behavior)."""
    aligned: list[AlignedPrediction] = []
    matched_gt_indices: set[tuple[str, int]] = set()

    for prediction in predictions:
        paper_id = prediction.paper_id
        gt_entries = gt_by_paper.get(paper_id, [])

        for pred_error in prediction.predicted_errors:
            best_match = _find_best_match(
                pred_error=pred_error,
                gt_entries=gt_entries,
                paper_id=paper_id,
                matched_gt_indices=matched_gt_indices,
                threshold=threshold,
            )
            if best_match:
                aligned.append(best_match)
            else:
                aligned.append(_false_positive(paper_id, pred_error))

    # Ground-truth entries not matched → false negatives.
    for paper_id, gt_entries in gt_by_paper.items():
        for gt_idx, gt_entry in enumerate(gt_entries):
            if (paper_id, gt_idx) not in matched_gt_indices:
                aligned.append(_false_negative(paper_id, gt_entry))

    return aligned


def _align_fuzzy_single_paper(
    paper_id: str,
    preds: list[PredictedError],
    gt_entries: list[dict],
    threshold: float,
) -> list[AlignedPrediction]:
    """Fuzzy alignment restricted to a single paper (judge-failure fallback)."""
    aligned: list[AlignedPrediction] = []
    matched_gt_indices: set[tuple[str, int]] = set()

    for pred_error in preds:
        best_match = _find_best_match(
            pred_error=pred_error,
            gt_entries=gt_entries,
            paper_id=paper_id,
            matched_gt_indices=matched_gt_indices,
            threshold=threshold,
        )
        if best_match:
            aligned.append(best_match)
        else:
            aligned.append(_false_positive(paper_id, pred_error))

    for gt_idx, gt_entry in enumerate(gt_entries):
        if (paper_id, gt_idx) not in matched_gt_indices:
            aligned.append(_false_negative(paper_id, gt_entry))

    return aligned


def _find_best_match(
    pred_error: PredictedError,
    gt_entries: list[dict],
    paper_id: str,
    matched_gt_indices: set[tuple[str, int]],
    threshold: float,
) -> Optional[AlignedPrediction]:
    """Find the best ground-truth match for a single predicted error.

    Uses fuzzy location matching to compare predicted and actual locations.
    """
    best_score = 0.0
    best_gt_idx = -1
    best_gt_entry = None

    for gt_idx, gt_entry in enumerate(gt_entries):
        # Skip already matched ground-truth entries
        if (paper_id, gt_idx) in matched_gt_indices:
            continue

        # Compute match score between predicted and ground-truth locations
        score = fuzzy_match_locations(
            pred_error.error_location,
            gt_entry["error_location"],
        )

        # Bonus for matching error categories
        if pred_error.error_category == gt_entry["error_category"]:
            score = min(1.0, score + 0.15)

        if score > best_score:
            best_score = score
            best_gt_idx = gt_idx
            best_gt_entry = gt_entry

    if best_score >= threshold and best_gt_entry is not None:
        matched_gt_indices.add((paper_id, best_gt_idx))
        return AlignedPrediction(
            paper_id=paper_id,
            predicted=pred_error,
            matched_ground_truth=True,
            ground_truth_category=best_gt_entry["error_category"],
            ground_truth_location=best_gt_entry["error_location"],
            ground_truth_severity=best_gt_entry["error_severity"],
            ground_truth_annotation=best_gt_entry.get("error_annotation", ""),
            match_quality=best_score,
            is_true_positive=True,
            is_false_positive=False,
        )

    return None
