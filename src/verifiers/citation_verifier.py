"""Citation / attribution / novelty verifier.

Targets a class the other verifiers miss: claims *about prior work*. Without
external lookup this verifier checks what is decidable from the text itself:

* Novelty over-claims — a technique/result claimed as new/original that the
  excerpt itself also attributes to (or cites from) prior work.
* Attribution mismatches — a citation invoked for a claim it plainly does not
  support, or a reference used inconsistently.
* Internal contradiction between a stated contribution and cited literature.

It is conservative: like the strict text reviewer, it flags only when the
excerpt *itself* contains the contradiction, never on the basis of outside
knowledge it cannot verify here. (The dataset's annotation #13 — "decomposition
technique claimed as original ... was previously established by other authors" —
is the canonical target.)
"""

from __future__ import annotations

import time

from src.models import (
    CitationVerificationResult,
    VerificationSnippet,
    VerificationStatus,
)
from src.verifiers.base import BaseVerifier


CITATION_SYSTEM_PROMPT = """You are a scientific-integrity reviewer examining ONE excerpt for CITATION, ATTRIBUTION, and NOVELTY problems that are decidable FROM THIS EXCERPT ALONE.

## Flag an error (error_detected = true) ONLY for:

1. **Novelty over-claim**: The excerpt claims a method/result is new, original, or the paper's contribution, while the SAME excerpt also says (or cites) that it was already established by others.
2. **Attribution mismatch**: A citation is invoked to support a specific claim that it clearly cannot support given what the excerpt states, or a reference is described inconsistently within the excerpt.
3. **Self-contradiction about prior work**: The excerpt simultaneously asserts incompatible things about what prior work did or did not do.

## NEVER flag (these are NOT errors):

- A missing citation, or a claim that "could use" a reference.
- Anything requiring outside knowledge of what a cited paper actually says — if you cannot decide it from THIS excerpt, do not flag.
- Style, formatting, citation format, or completeness of the bibliography.
- Ordinary background citations or standard "building on prior work" framing.

## Critical guidance

- Most excerpts have NO such error. Flagging must be RARE and supported by a direct quote from the excerpt.
- If not highly confident the excerpt is internally self-contradictory about attribution/novelty, return error_detected = false.

## Output Format

Return ONLY JSON:
```json
{
  "error_detected": false,
  "confidence": 0.0,
  "reasoning": "Quote the conflicting statements if flagging...",
  "predicted_error_category": null
}
```
When flagging, set confidence >= 0.8 and choose predicted_error_category from:
- "Equation / proof"  (novelty/priority of a technique or proof)
- "Data Inconsistency (text-text)"  (internal attribution contradiction)
- null
"""


class CitationVerifier(BaseVerifier):
    """Checks attribution/novelty claims for excerpt-internal contradictions."""

    name: str = "citation"

    _TEXT_TYPES = {
        "SECTION", "SUBSECTION", "PARAGRAPH", "THEOREM",
        "LEMMA", "PROPOSITION", "COROLLARY", "ALGORITHM", "APPENDIX",
    }

    def verify(self, snippet: VerificationSnippet) -> CitationVerificationResult:
        start_time = time.monotonic()

        if not self.can_verify(snippet):
            return CitationVerificationResult(
                snippet_id=snippet.snippet_id,
                verifier_name=self.name,
                status=VerificationStatus.SKIPPED,
                snippet_type=snippet.snippet_type.value,
                execution_time_ms=(time.monotonic() - start_time) * 1000,
            )

        def analyze_chunk(chunk: str) -> dict:
            resp = self._call_llm_json(
                prompt=(
                    f"Analyze this {snippet.snippet_type.value.lower()} for citation, "
                    f"attribution, or novelty contradictions decidable from the text alone.\n\n"
                    f"Location: {snippet.location}\n\nContent:\n{chunk[:4000]}\n\n"
                    "Return ONLY the JSON response."
                ),
                system_prompt=CITATION_SYSTEM_PROMPT,
            )
            return {
                "error_detected": bool(resp.get("error_detected", False)),
                "confidence": float(resp.get("confidence", 0.0)),
                "reasoning": resp.get("reasoning", ""),
                "predicted_error_category": resp.get("predicted_error_category"),
            }

        chosen, n_chunks, _ = self._analyze_in_chunks(snippet.content, analyze_chunk)

        if chosen is None:
            return CitationVerificationResult(
                snippet_id=snippet.snippet_id,
                verifier_name=self.name,
                status=VerificationStatus.UNVERIFIABLE,
                reasoning=f"All {n_chunks} chunk(s) failed to verify (e.g. empty LLM response).",
                snippet_type=snippet.snippet_type.value,
                execution_time_ms=(time.monotonic() - start_time) * 1000,
            )

        error_detected = chosen["error_detected"]
        return CitationVerificationResult(
            snippet_id=snippet.snippet_id,
            verifier_name=self.name,
            status=(
                VerificationStatus.ERROR_DETECTED if error_detected
                else VerificationStatus.NO_ERROR
            ),
            error_detected=error_detected,
            confidence=chosen["confidence"],
            reasoning=chosen["reasoning"],
            predicted_error_category=chosen["predicted_error_category"],
            snippet_type=snippet.snippet_type.value,
            execution_time_ms=(time.monotonic() - start_time) * 1000,
        )

    def can_verify(self, snippet: VerificationSnippet) -> bool:
        if not self.verifier_config.enabled:
            return False
        return (
            snippet.snippet_type.value in self._TEXT_TYPES
            and bool(snippet.content.strip())
        )
