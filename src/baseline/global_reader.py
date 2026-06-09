"""Global Reader — whole-paper baseline that produces a structured context map.

Extends the single-call baseline concept by also extracting:
- ``paper_claims`` — every verifiable claim the paper makes, with location and type
- ``suspicious_regions`` — regions the model is uncertain about and wants a
  specialist to examine more closely

This structured output feeds the ``GlobalReadOrchestrator``, which uses it as a
rich triage signal: specialists receive the full paper-claims map and cross-
references, so they no longer work in snippet-isolated vacuums.

The key insight: the single-call baseline outperforms the agentic pipeline because
it SEES THE WHOLE PAPER. This reader captures that whole-paper understanding in a
structured form, so every downstream specialist inherits it.
"""

from __future__ import annotations

import time
from typing import Optional

from loguru import logger
from pydantic import BaseModel, Field

from src.config import PipelineConfig, default_config
from src.models import NormalizedPaper
from src.utils.llm import llm_call, parse_json_response


# ---------------------------------------------------------------------------
# Structured output models
# ---------------------------------------------------------------------------


class PaperClaim(BaseModel):
    """A single verifiable claim extracted from the paper."""

    claim_text: str  # The claim as stated (paraphrased or quoted)
    location: str  # e.g., "Section 3.1", "Lemma 2", "Equation (12)"
    claim_type: str  # "equation", "theorem", "definition", "statistic", "assertion", etc.
    confidence: float = 1.0  # How central/consequential this claim is [0, 1]


class SuspiciousRegion(BaseModel):
    """A region the global reader flags for deeper specialist checking.

    These are NOT flagged as definite errors — they are places where the reader
    thinks a specialist should look more carefully. The uncertainty here is the
    reader's estimate that this region contains a correction-worthy error.
    """

    location: str  # e.g., "Proof of Theorem 2.2", "Equation (8)", "Figure 3"
    reason: str  # Why this region warrants a closer look
    suggested_verifier: str = "text"  # "math", "statistical", "vision", "text", "citation"
    uncertainty: float = 0.5  # [0, 1] — error likelihood estimate
    excerpt: str = ""  # The relevant text snippet, if identifiable


class GlobalReadResult(BaseModel):
    """Structured output from the global reader — a whole-paper analysis.

    This is the "context map" that feeds the downstream specialist pipeline.
    """

    paper_id: str
    title: str = ""
    paper_category: str = ""

    # Critical errors the reader is confident about (same as baseline output)
    errors: list[dict] = Field(default_factory=list)

    # Every verifiable claim the paper makes, with location and type
    paper_claims: list[PaperClaim] = Field(default_factory=list)

    # Regions flagged for specialist attention (not yet confirmed errors)
    suspicious_regions: list[SuspiciousRegion] = Field(default_factory=list)

    # The reader's overall assessment of the paper
    overall_assessment: str = ""

    # Execution metadata
    execution_time_ms: float = 0.0
    input_chars: int = 0


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_GLOBAL_READ_MARKER = "GLOBAL-READ WHOLE-PAPER ANALYSIS"

GLOBAL_READ_SYSTEM_PROMPT = f"""You are a scientific-integrity reviewer performing a {_GLOBAL_READ_MARKER}. You are given the FULL TEXT of one paper. Your job is to produce a STRUCTURED analysis with three parts:

## Part 1: Critical Errors
Identify every CRITICAL error — a factual, logical, mathematical, statistical, or methodological mistake serious enough that, if confirmed, it would require a published correction (erratum) or a retraction.

- A typical paper has ZERO or ONE such error. Be precise; do NOT pad the list with minor or stylistic issues.
- For each error, give the most specific location you can (e.g. "Lemma 3", "Eq. (12)", "Section 4.2", "Figure 2", "Theorem 2.2"), the category, your confidence in [0,1], and a short justification quoting or paraphrasing the offending content.
- Do NOT flag typos, missing citations, unclear writing, or anything that would not change the paper's conclusions.

## Part 2: Paper Claims Map
Extract EVERY non-trivial, verifiable claim the paper makes. This is a structured inventory for downstream specialists to use as context. Include:
- Equations and their claimed significance
- Theorem/lemma/proposition statements (just the claim, not the full proof)
- Numerical/statistical claims (reported values, percentages, p-values, etc.)
- Methodological claims ("we used X technique", "Y was measured using Z")
- Definitional claims (new terminology, notation conventions)
- Interpretive claims ("this shows that...", "therefore...")
- Citation/attribution claims ("as shown by [ref]...", "this is the first...")

For each claim, provide:
- ``claim_text``: the claim (paraphrased concisely or quoted briefly)
- ``location``: where in the paper it appears
- ``claim_type``: one of "equation", "theorem", "lemma", "definition", "statistic", "methodology", "interpretation", "citation", "assertion"
- ``confidence``: 0.0-1.0 — how central/consequential this claim is to the paper (1.0 = the paper's main result depends on it; 0.3 = minor background point)

A typical paper has 10-40 claims. Be thorough but don't list boilerplate.

## Part 3: Suspicious Regions
Identify specific regions that, while you are NOT confident enough to flag as definite errors, warrant a closer look by a specialist. These are places where:
- The reasoning seems rushed, incomplete, or has a potential gap
- An equation looks unusual or might have a subtle mistake
- A numerical claim seems surprising or inconsistent with another part of the paper
- A figure or table description suggests possible data issues
- A citation claim might overstate novelty or misrepresent prior work
- Statistical methods or reported values raise questions

For each suspicious region, provide:
- ``location``: where in the paper
- ``reason``: what specifically concerns you (one sentence)
- ``suggested_verifier``: which specialist should check — "math" (equations/derivations), "statistical" (numbers/statistics), "vision" (figures/tables), "text" (prose/logic), "citation" (attribution/novelty)
- ``uncertainty``: your estimate [0, 1] that this region contains a real error
- ``excerpt``: the relevant text (copy-pasted from the paper, up to 500 chars)

A typical paper has 2-10 suspicious regions. Be discriminating — flag only regions where specialist attention could realistically surface an error.

## Categories (for errors and suspicious regions)
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
  ],
  "paper_claims": [
    {{
      "claim_text": "...",
      "location": "Section 2",
      "claim_type": "theorem",
      "confidence": 0.9
    }}
  ],
  "suspicious_regions": [
    {{
      "location": "Proof of Theorem 2.2",
      "reason": "The induction step skips from n to n+2 without justifying the even-odd transition.",
      "suggested_verifier": "text",
      "uncertainty": 0.55,
      "excerpt": "..."
    }}
  ],
  "overall_assessment": "Brief assessment of the paper's overall correctness and the most important thing to verify."
}}
```
Return empty lists for errors/claims/regions if you find none.
"""


# ---------------------------------------------------------------------------
# Global Reader
# ---------------------------------------------------------------------------


class GlobalReader:
    """Whole-paper reader that produces a structured context map for downstream specialists.

    Like ``SingleCallBaseline``, this makes ONE LLM call over the full paper text.
    Unlike the baseline, it produces structured ``paper_claims`` and ``suspicious_regions``
    in addition to ``errors``.

    Usage:
        reader = GlobalReader(config=config)
        result = reader.run(paper)  # GlobalReadResult
        # result.paper_claims, result.suspicious_regions, result.errors
    """

    name: str = "global_reader"

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        max_input_chars: int = 60000,
    ) -> None:
        self.config = config or default_config
        self.max_input_chars = max_input_chars

    def run(self, paper: NormalizedPaper) -> GlobalReadResult:
        """Run the global read on a paper and return structured analysis."""
        t0 = time.monotonic()
        full = self._full_text(paper)
        input_chars = min(len(full), self.max_input_chars)
        truncated = len(full) > input_chars
        text = full[:input_chars]

        prompt = (
            f"Paper title: {paper.title}\n"
            f"Category: {paper.paper_category}\n"
            f"Paper ID: {paper.paper_id}\n"
            f"{'[NOTE: paper text truncated to fit context window]' if truncated else ''}\n\n"
            f"FULL PAPER TEXT:\n{text}\n\n"
            "Produce the structured analysis (errors, paper_claims, suspicious_regions, "
            "overall_assessment). Return ONLY the JSON object."
        )

        # Fallback budgets: same strategy as SingleCallBaseline
        top = input_chars
        budgets = [top]
        while budgets[-1] // 2 >= 12000:
            budgets.append(budgets[-1] // 2)

        parsed = None
        last_exc = None
        for budget in budgets:
            try:
                if budget < top:
                    text = full[:budget]
                    prompt = (
                        f"Paper title: {paper.title}\n"
                        f"Category: {paper.paper_category}\n"
                        f"Paper ID: {paper.paper_id}\n"
                        f"[NOTE: paper text truncated to {budget} chars to fit context]\n\n"
                        f"FULL PAPER TEXT:\n{text}\n\n"
                        "Produce the structured analysis. Return ONLY the JSON object."
                    )
                parsed = parse_json_response(
                    llm_call(
                        prompt=prompt,
                        system_prompt=GLOBAL_READ_SYSTEM_PROMPT,
                        config=self.config.llm,
                        temperature=self.config.llm.temperature,
                    )
                )
                if budget < self.max_input_chars:
                    logger.warning(
                        f"[global_read] {paper.paper_id}: succeeded only after "
                        f"reducing input to {budget} chars."
                    )
                break
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    f"[global_read] {paper.paper_id}: call failed at budget {budget} "
                    f"chars: {exc}"
                )
        else:
            logger.error(
                f"[global_read] {paper.paper_id} failed at all budgets: {last_exc}"
            )
            parsed = {}

        elapsed = (time.monotonic() - t0) * 1000

        # Parse the structured output
        errors = parsed.get("errors", []) or []
        paper_claims = [
            PaperClaim(
                claim_text=c.get("claim_text", ""),
                location=c.get("location", ""),
                claim_type=c.get("claim_type", "assertion"),
                confidence=float(c.get("confidence", 0.5)),
            )
            for c in (parsed.get("paper_claims", []) or [])
        ]
        suspicious_regions = [
            SuspiciousRegion(
                location=r.get("location", ""),
                reason=r.get("reason", ""),
                suggested_verifier=r.get("suggested_verifier", "text"),
                uncertainty=float(r.get("uncertainty", 0.5)),
                excerpt=r.get("excerpt", ""),
            )
            for r in (parsed.get("suspicious_regions", []) or [])
        ]

        logger.info(
            f"[global_read] {paper.paper_id}: {len(errors)} error(s), "
            f"{len(paper_claims)} claim(s), "
            f"{len(suspicious_regions)} suspicious region(s) "
            f"in {elapsed:.0f}ms"
        )

        return GlobalReadResult(
            paper_id=paper.paper_id,
            title=paper.title,
            paper_category=paper.paper_category,
            errors=errors,
            paper_claims=paper_claims,
            suspicious_regions=suspicious_regions,
            overall_assessment=parsed.get("overall_assessment", ""),
            execution_time_ms=elapsed,
            input_chars=input_chars,
        )

    def _full_text(self, paper: NormalizedPaper) -> str:
        """Best available full-text rendering of the paper."""
        if paper.tagged_full_text and paper.tagged_full_text.strip():
            return paper.tagged_full_text
        parts: list[str] = []
        for s in paper.sections:
            parts.append(f"## {s.section_title}\n{s.content}")
        for e in paper.equations:
            parts.append(f"[EQUATION {e.equation_label}] {e.latex}")
        for t in paper.theorems:
            parts.append(f"[{t.theorem_type} {t.label}] {t.statement}")
        return "\n\n".join(parts)
