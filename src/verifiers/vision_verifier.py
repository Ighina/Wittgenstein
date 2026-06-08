"""Phase 8: Vision-based verification for figures and tables.

Uses a multimodal LLM to analyze figures and tables for inconsistencies,
misleading visualizations, and data integrity issues.
"""

from __future__ import annotations

import time
from typing import Optional

from loguru import logger

from src.config import PipelineConfig, default_config
from src.models import (
    VerificationSnippet,
    VerificationStatus,
    VisionVerificationResult,
)
from src.verifiers.base import BaseVerifier


STRICT_FIGURE_SYSTEM_PROMPT = """You are a scientific-integrity reviewer examining ONE figure from a scientific paper. Decide ONLY whether the figure contains a CRITICAL error — serious enough that, if confirmed, it would require a published correction (erratum) or a retraction of the paper.

## Flag an error (error_detected = true) ONLY for:

1. **Result-invalidating contradiction**: The figure demonstrably contradicts its caption, its data, or the text in a way that undermines a reported finding.
2. **Genuine duplication / fabrication**: Clear evidence the figure (or a panel) is duplicated or manipulated.
3. **Impossible data**: Data points or axis ranges that are impossible and material to a conclusion (not merely suboptimal presentation).

## NEVER flag (these are NOT errors):

- Aesthetic or presentation choices: axis scaling, missing/extra labels, color choices, "could be clearer", missing error bars where not strictly required.
- Anything you cannot verify from what is shown, or that needs external knowledge.
- Minor/cosmetic issues, or anything that would not change the paper's conclusions.

## Critical guidance

- A typical paper has ZERO such figure errors. Flagging should be RARE.
- If you are not highly confident the figure contains a conclusion-invalidating problem, return error_detected = false.
- When you flag, report confidence >= 0.75 and state the specific problem in `reasoning`.

## Output Format

Return a JSON object with exactly these fields:
```json
{
  "error_detected": false,
  "confidence": 0.0,
  "reasoning": "Detailed explanation; state the specific problem if flagging...",
  "predicted_error_category": null
}
```

Possible predicted_error_category values (use null when no error):
- "Figure duplication"
- "Data Inconsistency (figure-text)"
- "Data Inconsistency (figure-figure)"
- "Data inconsistency"
- "Statistical reporting"
- null (if no error detected)
"""


LEGACY_FIGURE_SYSTEM_PROMPT = """You are a scientific figure analysis expert. Your task is to examine a figure from a scientific paper and identify potential errors, inconsistencies, or misleading aspects.

## What to Check

1. **Inconsistencies**: Does the figure contradict its caption or the surrounding text?
2. **Misleading visualizations**: Are axes properly scaled and labeled? Are error bars present where expected? Are trends visually distorted?
3. **Impossible values**: Do any data points or axis ranges appear impossible given the context?
4. **Duplication**: Is there any indication this figure may be duplicated (from another paper, or internally)?
5. **Label accuracy**: Are all labels, legends, and annotations correct and consistent?

## Output Format

Return a JSON object with exactly these fields:
```json
{
  "error_detected": true,
  "confidence": 0.85,
  "reasoning": "Detailed explanation of findings...",
  "predicted_error_category": "Figure duplication"
}
```

Possible predicted_error_category values:
- "Figure duplication"
- "Data Inconsistency (figure-text)"
- "Data Inconsistency (figure-figure)"
- "Data inconsistency"
- "Statistical reporting"
- null (if no error detected)

Be thorough but precise. Only flag issues when you have reasonable confidence.
"""


STRICT_TABLE_SYSTEM_PROMPT = """You are a scientific-integrity reviewer examining ONE table from a scientific paper. Decide ONLY whether the table contains a CRITICAL error — serious enough that, if confirmed, it would require a published correction (erratum) or a retraction of the paper.

## Flag an error (error_detected = true) ONLY for:

1. **Provably wrong values**: Arithmetic that does not add up, or reported statistics that are internally impossible/incoherent, AND material to a conclusion.
2. **Result-invalidating contradiction**: Table values that directly contradict the caption or the text in a way that undermines a reported finding.
3. **Mislabeled data that changes interpretation**: Values clearly attached to the wrong row/column in a way that alters a conclusion.

## NEVER flag (these are NOT errors):

- Formatting, alignment, missing/placeholder cells, rounding, or presentation issues.
- Anything you cannot verify from the table content, or that needs external knowledge.
- Minor/cosmetic discrepancies that would not change the paper's conclusions.

## Critical guidance

- A typical paper has ZERO such table errors. Flagging should be RARE.
- If you are not highly confident the table contains a conclusion-invalidating problem, return error_detected = false.
- When you flag, report confidence >= 0.75 and state the specific problem in `reasoning`.

## Output Format

Return a JSON object with exactly these fields:
```json
{
  "error_detected": false,
  "confidence": 0.0,
  "reasoning": "Detailed explanation; state the specific problem if flagging...",
  "predicted_error_category": null
}
```

Possible predicted_error_category values (use null when no error):
- "Data Inconsistency (figure-text)"
- "Data Inconsistency (text-text)"
- "Data inconsistency"
- "Statistical reporting"
- null (if no error detected)
"""


LEGACY_TABLE_SYSTEM_PROMPT = """You are a scientific table analysis expert. Your task is to examine a table from a scientific paper and identify potential errors or inconsistencies.

## What to Check

1. **Arithmetic consistency**: Do row/column totals add up correctly?
2. **Statistical consistency**: Are reported statistics internally consistent? Are p-values and confidence intervals coherent?
3. **Row/column mismatches**: Do values align with their labels?
4. **Caption consistency**: Does the table content match its caption?
5. **Formatting issues**: Are there missing values, misaligned data, or placeholder artifacts?

## Output Format

Return a JSON object with exactly these fields:
```json
{
  "error_detected": true,
  "confidence": 0.85,
  "reasoning": "Detailed explanation of findings...",
  "predicted_error_category": "Data Inconsistency (figure-text)"
}
```

Possible predicted_error_category values:
- "Data Inconsistency (figure-text)"
- "Data Inconsistency (text-text)"
- "Data inconsistency"
- "Statistical reporting"
- null (if no error detected)

Be thorough but precise. Only flag issues when you have reasonable confidence.
"""


class VisionVerifier(BaseVerifier):
    """Verifies figures and tables using a multimodal LLM.

    Different prompts are used depending on whether the snippet is a figure
    or a table. Images are sent alongside the prompt text.
    """

    name: str = "vision"

    def verify(
        self,
        snippet: VerificationSnippet,
    ) -> VisionVerificationResult:
        """Verify a figure or table snippet.

        Args:
            snippet: A FIGURE or TABLE type verification snippet.

        Returns:
            VisionVerificationResult with findings.
        """
        start_time = time.monotonic()

        content_type = "table" if snippet.snippet_type.value == "TABLE" else "figure"
        logger.debug(
            f"Verifying {content_type}: {snippet.location} ({snippet.snippet_id})"
        )

        if not self.can_verify(snippet):
            return VisionVerificationResult(
                snippet_id=snippet.snippet_id,
                verifier_name=self.name,
                status=VerificationStatus.SKIPPED,
                content_type=content_type,
                execution_time_ms=(time.monotonic() - start_time) * 1000,
            )

        # Select prompt based on content type and strictness
        lenient = self.config.strictness == "lenient"
        if content_type == "table":
            system_prompt = LEGACY_TABLE_SYSTEM_PROMPT if lenient else STRICT_TABLE_SYSTEM_PROMPT
        else:
            system_prompt = LEGACY_FIGURE_SYSTEM_PROMPT if lenient else STRICT_FIGURE_SYSTEM_PROMPT

        prompt = self._build_prompt(snippet, content_type)
        image_path = snippet.image_path

        try:
            response = self._call_llm_json(
                prompt=prompt,
                system_prompt=system_prompt,
                image_path=image_path if image_path else None,
            )

            error_detected = response.get("error_detected", False)
            confidence = float(response.get("confidence", 0.0))
            reasoning = response.get("reasoning", "")
            predicted_category = response.get("predicted_error_category")

            status = (
                VerificationStatus.ERROR_DETECTED if error_detected
                else VerificationStatus.NO_ERROR
            )

            return VisionVerificationResult(
                snippet_id=snippet.snippet_id,
                verifier_name=self.name,
                status=status,
                error_detected=error_detected,
                confidence=confidence,
                reasoning=reasoning,
                predicted_error_category=predicted_category,
                content_type=content_type,
                image_path=image_path,
                caption_text=snippet.metadata.get("caption"),
                execution_time_ms=(time.monotonic() - start_time) * 1000,
            )

        except Exception as exc:
            logger.error(f"Vision verification failed for {snippet.snippet_id}: {exc}")
            return VisionVerificationResult(
                snippet_id=snippet.snippet_id,
                verifier_name=self.name,
                status=VerificationStatus.UNVERIFIABLE,
                reasoning=f"Verification error: {exc}",
                content_type=content_type,
                execution_time_ms=(time.monotonic() - start_time) * 1000,
            )

    def can_verify(self, snippet: VerificationSnippet) -> bool:
        """Vision verifier handles FIGURE and TABLE snippets."""
        if not self.verifier_config.enabled:
            return False
        return snippet.snippet_type.value in ("FIGURE", "TABLE")

    def _build_prompt(
        self,
        snippet: VerificationSnippet,
        content_type: str,
    ) -> str:
        """Build the analysis prompt for the vision LLM."""
        parts: list[str] = []

        if content_type == "table":
            parts.append("Analyze the following table from a scientific paper for errors:")
        else:
            parts.append("Analyze the following figure from a scientific paper for errors:")

        if snippet.metadata.get("caption"):
            parts.append(f"\nCaption: {snippet.metadata['caption']}")

        loc = snippet.location
        if loc:
            parts.append(f"\nLocation in paper: {loc}")

        if snippet.content:
            # For tables, include the text content (markdown table)
            if content_type == "table":
                parts.append(f"\nTable data:\n{snippet.content[:2000]}")
            else:
                # For figures, just include contextual text
                parts.append(f"\nContext:\n{snippet.content[:1500]}")

        parts.append("\n\nIdentify any errors, inconsistencies, or issues. Return ONLY the JSON response.")

        return "\n".join(parts)
