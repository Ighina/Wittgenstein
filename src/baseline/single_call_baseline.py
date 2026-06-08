"""Single-call baseline: read the whole paper, identify the error(s) in one shot.

This is the obvious thing to compare the orchestrated pipeline against — hand the
*entire* paper to one LLM call and ask for the errors directly, with no
segmentation, routing, triage, or specialized verifiers. It emits the SAME
`PaperPrediction` / `PredictedError` shape as the orchestrator, so the existing
alignment + metrics (`src/evaluation`) score it with no changes.

It is intentionally minimal: one prompt, one response, structured output. The
point is to measure how much the decomposition + specialist machinery actually
buys over "just ask the model".
"""

from __future__ import annotations

import time
from typing import Optional

from loguru import logger

from src.config import PipelineConfig, default_config
from src.models import NormalizedPaper, PaperPrediction, PredictedError
from src.utils.llm import llm_call, parse_json_response

# Marker string kept verbatim in the system prompt so the mock LLM backend (and
# anyone grepping) can recognize a single-call-baseline request.
_BASELINE_MARKER = "SINGLE-CALL WHOLE-PAPER REVIEW"

BASELINE_SYSTEM_PROMPT = f"""You are a scientific-integrity reviewer performing a {_BASELINE_MARKER}. You are given the FULL TEXT of one paper. Identify every CRITICAL error — a factual, logical, mathematical, statistical, or methodological mistake serious enough that, if confirmed, it would require a published correction (erratum) or a retraction.

## Guidance
- A typical paper has ZERO or ONE such error. Be precise; do NOT pad the list with minor or stylistic issues.
- For each error, give the most specific location you can (e.g. "Lemma 3", "Eq. (12)", "Section 4.2", "Figure 2", "Theorem 2.2"), the category, your confidence in [0,1], and a short justification quoting or paraphrasing the offending content.
- Do NOT flag typos, missing citations, unclear writing, or anything that would not change the paper's conclusions.

## Categories (choose the closest)
"Equation / proof", "Statistical reporting", "Data Inconsistency (text-text)",
"Data Inconsistency (figure-text)", "Data Inconsistency (figure-figure)",
"Figure duplication", "Experiment setup", "Reagent identity",
"Methodology inconsistency".

## Output Format
Return ONLY a JSON object:
```json
{{
  "errors": [
    {{
      "error_category": "Equation / proof",
      "error_location": "Lemma 3",
      "confidence": 0.0,
      "supporting_evidence": "..."
    }}
  ]
}}
```
Return {{"errors": []}} if you find no correction-worthy error.
"""


class SingleCallBaseline:
    """One-LLM-call whole-paper error detector with orchestrator-compatible output."""

    name: str = "single_call_baseline"

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        max_input_chars: int = 60000,
    ) -> None:
        self.config = config or default_config
        # Papers can be long; cap the input to a generous budget. Truncation is
        # logged so it is never silent.
        self.max_input_chars = max_input_chars

    def run(self, paper: NormalizedPaper, progress=None) -> PaperPrediction:
        """Verify a paper in a single call. Signature mirrors the orchestrator."""
        t0 = time.monotonic()
        full = self._full_text(paper)

        # Try the configured budget; on failure (a heavy whole-paper request can
        # drop the connection mid-generation) degrade to progressively smaller
        # inputs rather than failing to zero predictions. We add fallbacks down
        # to a floor REGARDLESS of whether the full text already fit, because the
        # failure correlates with request heaviness, not just raw length. This
        # keeps the baseline a fair comparator instead of a silent no-op.
        top = min(len(full), self.max_input_chars)
        budgets = [top]
        while budgets[-1] // 2 >= 12000:
            budgets.append(budgets[-1] // 2)

        predicted: list[PredictedError] = []
        last_exc: Optional[Exception] = None
        for budget in budgets:
            truncated = len(full) > budget
            text = full[:budget]
            prompt = (
                f"Paper title: {paper.title}\n"
                f"Category: {paper.paper_category}\n"
                f"{'[NOTE: paper text truncated to fit context]' if truncated else ''}\n\n"
                f"FULL PAPER TEXT:\n{text}\n\n"
                "Identify the critical error(s). Return ONLY the JSON object."
            )
            try:
                parsed = parse_json_response(
                    llm_call(
                        prompt=prompt,
                        system_prompt=BASELINE_SYSTEM_PROMPT,
                        config=self.config.llm,
                        temperature=self.config.llm.temperature,
                    )
                )
                for e in parsed.get("errors", []) or []:
                    predicted.append(PredictedError(
                        error_category=e.get("error_category") or "Unknown",
                        error_location=str(e.get("error_location", "")) or "Unknown",
                        confidence=float(e.get("confidence", 0.0) or 0.0),
                        supporting_evidence=str(e.get("supporting_evidence", ""))[:1000],
                        verifier_name=self.name,
                        snippet_id=f"{paper.paper_id}_baseline",
                    ))
                if truncated and budget < self.max_input_chars:
                    logger.warning(
                        f"[baseline] {paper.paper_id}: succeeded only after reducing "
                        f"input to {budget} chars."
                    )
                break  # success
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    f"[baseline] {paper.paper_id}: call failed at budget {budget} "
                    f"chars: {exc}"
                )
        else:
            logger.error(f"[baseline] {paper.paper_id} failed at all budgets: {last_exc}")

        predicted.sort(key=lambda p: p.confidence, reverse=True)
        elapsed = time.monotonic() - t0
        logger.info(
            f"[baseline] {paper.paper_id}: {len(predicted)} error(s) in {elapsed:.1f}s"
        )

        return PaperPrediction(
            paper_id=paper.paper_id,
            title=paper.title,
            paper_category=paper.paper_category,
            predicted_errors=predicted,
            total_snippets=1,
            snippets_verified=1,
            errors_detected=len(predicted),
            verifier_usage={self.name: 1},
            raw_results=[],
        )

    def _full_text(self, paper: NormalizedPaper) -> str:
        """Best available full-text rendering of the paper."""
        if paper.tagged_full_text and paper.tagged_full_text.strip():
            return paper.tagged_full_text
        # Fallback: reconstruct from structured parts.
        parts: list[str] = []
        for s in paper.sections:
            parts.append(f"## {s.section_title}\n{s.content}")
        for e in paper.equations:
            parts.append(f"[EQUATION {e.equation_label}] {e.latex}")
        for t in paper.theorems:
            parts.append(f"[{t.theorem_type} {t.label}] {t.statement}")
        return "\n\n".join(parts)
