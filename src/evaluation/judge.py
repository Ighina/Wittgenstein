"""LLM-judge-based semantic matching of predictions to ground truth.

This reproduces the evaluation strategy of the original benchmark (see
``src/run_eval.py``), which used an LLM *judge* to decide whether a model's
predicted errors correspond to the human-annotated ground-truth errors.

The judge compares the *substance* of each error rather than its surface form,
so a correct finding that references a location differently ("Eq. (7)" vs.
"the displayed equation in Section 3") is still credited as a match. This is
far more forgiving — and more faithful to what the benchmark measures — than
the brittle location-string matching in ``alignment.py`` (kept only as a
fallback for offline/mock runs and when a judge call fails).
"""

from __future__ import annotations

import json
from typing import Any, Optional

from loguru import logger

from src.config import LLMConfig
from src.models import PredictedError
from src.utils.llm import llm_call, parse_json_response


JUDGE_SYSTEM_PROMPT = """You are an expert scientific reviewer acting as an impartial judge.

You are given, for a single paper:
- `annotations`: the ground-truth errors a human expert found. Each has a `location` and a `description`.
- `predictions`: errors an automated system claims to have found. Each has an `error_category`, an `error_location`, and `supporting_evidence`.

Your task is to decide which predictions correctly identify which ground-truth errors.

A prediction MATCHES an annotation when it refers to the SAME underlying mistake in the paper. Judge by substance, not surface wording:
- The location strings need NOT be identical. "Eq. (7)", "equation 7", and "the displayed equation in Section 3" can all denote the same place.
- The category labels need NOT be identical, as long as the described problem is the same.
- A prediction matches only if it captures the actual error the annotation describes — not merely the same region of the paper with an unrelated or vague complaint.

Matching rules:
- Each annotation matches AT MOST one prediction (choose the single best one).
- Each prediction matches AT MOST one annotation.
- Leaving annotations or predictions unmatched is expected and fine.

Return ONLY a JSON object of this exact form:
{
  "matches": [
    {"annotation_index": <int>, "prediction_index": <int>, "reasoning": "<why they describe the same error>"}
  ]
}
Indices are 0-based and refer to positions in the provided `annotations` and `predictions` arrays. If nothing matches, return {"matches": []}."""


def judge_matches(
    annotations: list[dict[str, str]],
    predictions: list[PredictedError],
    llm_config: Optional[LLMConfig] = None,
    model: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Ask an LLM judge which predictions match which ground-truth annotations.

    Args:
        annotations: Ground-truth errors as ``{"location", "description"}`` dicts.
        predictions: The model's predicted errors for the same paper.
        llm_config: LLM backend configuration (provider, API key, etc.).
        model: Optional judge-model override; falls back to ``llm_config.model``.

    Returns:
        A list of validated, de-duplicated match dicts, each with
        ``annotation_index``, ``prediction_index`` and ``reasoning``. Each
        annotation and each prediction appears in at most one match.

    Raises:
        Exception: Propagates any LLM/parse error so callers can fall back.
    """
    payload = {
        "annotations": annotations,
        "predictions": [
            {
                "error_category": p.error_category,
                "error_location": p.error_location,
                "supporting_evidence": p.supporting_evidence,
            }
            for p in predictions
        ],
    }
    user_content = json.dumps(payload, ensure_ascii=False, indent=2)

    response = llm_call(
        prompt=user_content,
        system_prompt=JUDGE_SYSTEM_PROMPT,
        model=model,
        config=llm_config,
    )
    parsed = parse_json_response(response)

    return _sanitize_matches(
        parsed.get("matches", []),
        n_annotations=len(annotations),
        n_predictions=len(predictions),
    )


def _sanitize_matches(
    raw_matches: Any,
    n_annotations: int,
    n_predictions: int,
) -> list[dict[str, Any]]:
    """Validate judge output: keep in-range indices, enforce 1:1 matching."""
    if not isinstance(raw_matches, list):
        logger.warning(f"Judge returned non-list 'matches': {type(raw_matches)}")
        return []

    valid: list[dict[str, Any]] = []
    used_annotations: set[int] = set()
    used_predictions: set[int] = set()

    for match in raw_matches:
        if not isinstance(match, dict):
            continue
        try:
            ann_idx = int(match["annotation_index"])
            pred_idx = int(match["prediction_index"])
        except (KeyError, TypeError, ValueError):
            continue

        if not (0 <= ann_idx < n_annotations):
            continue
        if not (0 <= pred_idx < n_predictions):
            continue
        # Enforce at-most-one match per side (judge is instructed to, but be safe).
        if ann_idx in used_annotations or pred_idx in used_predictions:
            continue

        used_annotations.add(ann_idx)
        used_predictions.add(pred_idx)
        valid.append({
            "annotation_index": ann_idx,
            "prediction_index": pred_idx,
            "reasoning": str(match.get("reasoning", "")),
        })

    return valid
