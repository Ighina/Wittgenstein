"""Phase 9: General text consistency verification.

Analyzes text snippets for logical contradictions, unsupported claims,
internal inconsistencies, and methodology issues.
"""

from __future__ import annotations

import time
from typing import Optional

from loguru import logger

from src.config import PipelineConfig, default_config
from src.models import (
    TextVerificationResult,
    VerificationSnippet,
    VerificationStatus,
)
from src.verifiers.base import BaseVerifier


STRICT_TEXT_SYSTEM_PROMPT = """You are a scientific-integrity reviewer examining ONE excerpt from a scientific paper. Decide ONLY whether this excerpt contains a CRITICAL error — a factual, logical, mathematical, or methodological mistake serious enough that, if confirmed, it would require a published correction (erratum) or a retraction of the paper.

## Flag an error (error_detected = true) ONLY for:

1. **Provably wrong results**: A derivation, calculation, or stated result that is demonstrably incorrect.
2. **Conclusion-undermining contradictions**: A statement that directly contradicts data, a definition, or another explicit statement in a way that invalidates a finding.
3. **Invalidating methodology flaws**: A methodological mistake (e.g., the wrong statistical test applied to a central result) that makes a reported result unsound.
4. **Impossible/self-contradictory statistics**: Reported numbers or statistics that are internally impossible or incoherent AND material to a conclusion.

## NEVER flag (these are NOT errors):

- Typos, grammar, spelling, formatting, or stylistic issues.
- Missing citations, "could be clearer", incomplete explanations, or merely unusual notation.
- Claims you cannot verify from this excerpt alone, or that require external knowledge.
- Anything minor, cosmetic, or that would not change the paper's conclusions.
- Subjective disagreements, alternative approaches, or "I would have done it differently".

## Critical guidance

- A typical paper has ZERO or ONE such error. Flagging should be RARE.
- This is a single excerpt out of many; missing surrounding context is NOT an error — do not flag something merely because the excerpt seems incomplete.
- If you are not highly confident this excerpt contains a conclusion-invalidating mistake, return error_detected = false.
- When you do flag, report confidence >= 0.8 and cite the specific offending statement in `reasoning`.

## Output Format

Return a JSON object with exactly these fields:
```json
{
  "error_detected": false,
  "confidence": 0.0,
  "reasoning": "Detailed explanation; quote the specific offending text if flagging...",
  "predicted_error_category": null
}
```

Possible predicted_error_category values (use null when no error):
- "Equation / proof"
- "Data Inconsistency (text-text)"
- "Statistical reporting"
- "Experiment setup"
- "Reagent identity"
- "Methodology inconsistency"
- null (if no error detected)
"""


LEGACY_TEXT_SYSTEM_PROMPT = """You are a scientific text analysis expert. Your task is to examine text from a scientific paper and identify potential errors or issues.

## What to Check

1. **Logical contradictions**: Does any statement contradict another statement in the text?
2. **Unsupported claims**: Are there claims presented without evidence or citation?
3. **Internal inconsistencies**: Are definitions, notations, or terminology used consistently?
4. **Methodology issues**: Are there gaps or flaws in the described methodology?
5. **Theorem/proof mismatches**: Do theorem statements match their proofs? Are there missing steps?
6. **Data reporting issues**: Are numbers, statistics, or results reported consistently?

## Output Format

Return a JSON object with exactly these fields:
```json
{
  "error_detected": true,
  "confidence": 0.74,
  "reasoning": "Detailed explanation of findings...",
  "predicted_error_category": "Equation / proof"
}
```

Possible predicted_error_category values:
- "Equation / proof"
- "Data Inconsistency (text-text)"
- "Statistical reporting"
- "Experiment setup"
- "Reagent identity"
- "Methodology inconsistency"
- null (if no error detected)

Be thorough but precise. Only flag issues when you have reasonable confidence. Avoid false positives.
"""


class TextVerifier(BaseVerifier):
    """Verifies general text content for logical and factual consistency.

    Analyzes sections, paragraphs, theorems, and other text-based snippets
    for contradictions, unsupported claims, and internal inconsistencies.
    """

    name: str = "text"

    def verify(
        self,
        snippet: VerificationSnippet,
    ) -> TextVerificationResult:
        """Verify a text snippet for consistency issues.

        Args:
            snippet: Any text-based verification snippet (SECTION, THEOREM, etc.).

        Returns:
            TextVerificationResult with findings.
        """
        start_time = time.monotonic()

        logger.debug(
            f"Verifying text: {snippet.snippet_type.value} "
            f"'{snippet.location[:60]}' ({snippet.snippet_id})"
        )

        if not self.can_verify(snippet):
            return TextVerificationResult(
                snippet_id=snippet.snippet_id,
                verifier_name=self.name,
                status=VerificationStatus.SKIPPED,
                snippet_type=snippet.snippet_type.value,
                execution_time_ms=(time.monotonic() - start_time) * 1000,
            )

        system_prompt = (
            LEGACY_TEXT_SYSTEM_PROMPT
            if self.config.strictness == "lenient"
            else STRICT_TEXT_SYSTEM_PROMPT
        )

        def analyze_chunk(chunk: str) -> dict:
            response = self._call_llm_json(
                prompt=self._build_prompt(snippet, content=chunk),
                system_prompt=system_prompt,
            )
            return {
                "error_detected": bool(response.get("error_detected", False)),
                "confidence": float(response.get("confidence", 0.0)),
                "reasoning": response.get("reasoning", ""),
                "predicted_error_category": response.get("predicted_error_category"),
                "contradiction_locations": response.get("contradiction_locations", []),
            }

        # Long snippets are split into overlapping chunks and verified
        # independently; a single empty/garbled chunk no longer fails the snippet.
        chosen, n_chunks, n_failed = self._analyze_in_chunks(snippet.content, analyze_chunk)

        if chosen is None:
            return TextVerificationResult(
                snippet_id=snippet.snippet_id,
                verifier_name=self.name,
                status=VerificationStatus.UNVERIFIABLE,
                reasoning=f"All {n_chunks} chunk(s) failed to verify (e.g. empty LLM response).",
                snippet_type=snippet.snippet_type.value,
                execution_time_ms=(time.monotonic() - start_time) * 1000,
            )

        error_detected = chosen["error_detected"]
        return TextVerificationResult(
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
            contradiction_locations=chosen.get("contradiction_locations", []),
            execution_time_ms=(time.monotonic() - start_time) * 1000,
        )

    def can_verify(self, snippet: VerificationSnippet) -> bool:
        """Text verifier handles text-based snippet types."""
        if not self.verifier_config.enabled:
            return False
        text_types = {
            "SECTION", "SUBSECTION", "PARAGRAPH", "THEOREM",
            "LEMMA", "PROPOSITION", "COROLLARY", "ALGORITHM",
            "APPENDIX",
        }
        return snippet.snippet_type.value in text_types and bool(snippet.content.strip())

    def _build_prompt(self, snippet: VerificationSnippet, content: Optional[str] = None) -> str:
        """Build the analysis prompt.

        Args:
            snippet: The snippet being analyzed (for type/location).
            content: Text to analyze. Defaults to the full snippet content; the
                chunker passes one chunk at a time.
        """
        parts: list[str] = []

        snippet_type = snippet.snippet_type.value.lower()
        location = snippet.location
        body = snippet.content if content is None else content

        parts.append(
            f"Analyze the following {snippet_type} from a scientific paper for errors:"
        )
        parts.append(f"\nLocation: {location}")
        parts.append(f"\nContent:\n{body[:4000]}")

        if self.config.strictness == "lenient":
            parts.append(
                "\n\nIdentify any logical contradictions, unsupported claims, "
                "internal inconsistencies, or methodology issues. "
                "Return ONLY the JSON response."
            )
        else:
            parts.append(
                "\n\nFlag a CRITICAL error only if this excerpt contains a mistake "
                "serious enough to warrant an erratum or retraction (see the rules above). "
                "Otherwise set error_detected to false. Return ONLY the JSON response."
            )

        return "\n".join(parts)
