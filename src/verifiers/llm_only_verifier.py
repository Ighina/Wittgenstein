"""LLM-only verifier — bypasses specialist verifiers for pure LLM verification.

When ``llm_only_mode`` is set in the pipeline config, ALL snippets are routed
through this verifier instead of the type-specific specialists. Two modes:

  "same-prompt"      — one unified system prompt for every snippet type
  "separate-prompts" — different system prompts per snippet family (text, math,
                       figure/table, citation), but still pure LLM (no SymPy
                       sandbox, no deterministic checks, no multimodal vision)
"""

from __future__ import annotations

import time
from typing import Optional

from loguru import logger

from src.config import PipelineConfig, default_config
from src.models import (
    BaseVerificationResult,
    VerificationSnippet,
    VerificationStatus,
)
from src.verifiers.base import BaseVerifier


# ---------------------------------------------------------------------------
# "same-prompt" — one general-purpose system prompt for every snippet type
# ---------------------------------------------------------------------------

LLM_ONLY_UNIFIED_SYSTEM_PROMPT = """You are a scientific-integrity reviewer examining ONE excerpt from a scientific paper. The excerpt could be anything: a paragraph of prose, a theorem statement, a mathematical equation, a figure or table caption with surrounding context, a block of citations, an algorithm listing, or an appendix passage. Your job is to decide whether this excerpt contains a CRITICAL error — a factual, logical, mathematical, or methodological mistake serious enough that, if confirmed, it would require a published correction (erratum) or a retraction of the paper.

## Flag an error (error_detected = true) ONLY for:

1. **Provably wrong results**: A derivation, calculation, or stated result that is demonstrably incorrect.
2. **Conclusion-undermining contradictions**: A statement that directly contradicts data, a definition, or another explicit statement in a way that invalidates a finding.
3. **Invalidating methodology flaws**: A methodological mistake (e.g., the wrong statistical test applied to a central result) that makes a reported result unsound.
4. **Impossible/self-contradictory statistics**: Reported numbers or statistics that are internally impossible or incoherent AND material to a conclusion.
5. **Mathematical / equation errors**: An equation that is algebraically wrong, dimensionally inconsistent, or contains a sign/term error that would change the result.
6. **Figure / table errors**: A figure or table whose caption, description, or reported numbers contradict the paper's own claims or are internally incoherent.
7. **Citation / attribution errors**: A reference used in a way that misrepresents the cited work or claims novelty over work that already exists.

## NEVER flag (these are NOT errors):

- Typos, grammar, spelling, formatting, or stylistic issues.
- Missing citations, "could be clearer", incomplete explanations, or merely unusual notation.
- Claims you cannot verify from this excerpt alone, or that require external knowledge.
- Anything minor, cosmetic, or that would not change the paper's conclusions.
- Subjective disagreements, alternative approaches, or "I would have done it differently".

## Content type awareness

- If the excerpt is an **equation**, check for algebraic/derivation errors, missing terms, sign errors, and dimensional inconsistency.
- If the excerpt is a **figure/table caption**, check that the claims in the caption are internally consistent and do not contradict the numbers reported.
- If the excerpt is **prose / theorem / proof**, check for logical gaps, contradictions, and unsupported claims.
- If the excerpt is **citation-heavy**, check whether any cited work is mischaracterised.

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
- "Figure / table error"
- "Citation / attribution"
- null (if no error detected)
"""


# ---------------------------------------------------------------------------
# "separate-prompts" — different system prompts per snippet family
# ---------------------------------------------------------------------------

LLM_ONLY_TEXT_SYSTEM_PROMPT = """You are a scientific-integrity reviewer examining ONE text excerpt from a scientific paper (a section, paragraph, theorem, lemma, proposition, corollary, algorithm, or appendix passage). Decide ONLY whether this excerpt contains a CRITICAL error — a factual, logical, mathematical, or methodological mistake serious enough that, if confirmed, it would require a published correction (erratum) or a retraction of the paper.

## Flag an error (error_detected = true) ONLY for:

1. **Provably wrong results**: A derivation, calculation, or stated result that is demonstrably incorrect.
2. **Conclusion-undermining contradictions**: A statement that directly contradicts data, a definition, or another explicit statement in a way that invalidates a finding.
3. **Invalidating methodology flaws**: A methodological mistake (e.g., the wrong statistical test applied to a central result) that makes a reported result unsound.
4. **Impossible/self-contradictory statistics**: Reported numbers or statistics that are internally impossible or incoherent AND material to a conclusion.
5. **Theorem/proof mismatches**: A theorem statement whose proof is logically invalid, contains a gap, or reaches a conclusion that does not follow.

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
- For theorems and proofs, check that the logical chain is sound and the conclusion follows from the premises.

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

LLM_ONLY_MATH_SYSTEM_PROMPT = """You are a scientific-integrity reviewer examining ONE mathematical equation (LaTeX) from a scientific paper. Decide whether this equation contains a CRITICAL error — an algebraic mistake, sign error, dimensional inconsistency, or logical flaw serious enough to require an erratum or retraction.

## Flag an error (error_detected = true) ONLY for:

1. **Algebraic / derivation errors**: A step that is mathematically invalid or produces a result that does not follow.
2. **Sign errors**: A wrong sign that would change the conclusion.
3. **Dimensional inconsistency**: Units or dimensions that do not match across the equation.
4. **Missing or extraneous terms**: Terms that should be present but are not, or terms that appear without justification.
5. **Domain errors**: An expression that is undefined or invalid for the claimed domain (e.g., division by zero, log of a negative number).

## NEVER flag:

- Notation choices, formatting, or stylistic preferences.
- Equations you cannot verify without external knowledge.
- Missing context that might be provided elsewhere in the paper.
- Alternative derivations that could also be valid.

## Critical guidance

- A typical paper has ZERO or ONE such error. Flagging should be RARE.
- You do NOT have a symbolic algebra system; reason step by step about the mathematical validity.
- If you are not highly confident the equation is wrong, return error_detected = false.
- When you do flag, report confidence >= 0.8 and explain the specific mathematical mistake.

## Output Format

Return a JSON object with exactly these fields:
```json
{
  "error_detected": false,
  "confidence": 0.0,
  "reasoning": "Step-by-step mathematical reasoning; cite the specific error if flagging...",
  "predicted_error_category": null
}
```

Possible predicted_error_category values (use null when no error):
- "Equation / proof"
- null (if no error detected)
"""

LLM_ONLY_FIGURE_TABLE_SYSTEM_PROMPT = """You are a scientific-integrity reviewer examining a figure or table from a scientific paper. You will receive the caption and any surrounding descriptive text, but NOT the actual image or table data grid. Decide whether the description contains a CRITICAL error — a factual inconsistency, impossible claim, or contradiction serious enough to require an erratum or retraction.

## Flag an error (error_detected = true) ONLY for:

1. **Internally contradictory caption**: Numbers or claims in the caption that contradict each other.
2. **Impossible / incoherent data claims**: Reported values, ranges, or statistics that are mathematically impossible or self-contradictory.
3. **Conclusion-invalidating misdescription**: The caption or surrounding text describes a finding that, if true, would contradict established facts or the paper's own claims elsewhere.

## NEVER flag:

- "The figure is unclear" or "the caption could be more detailed".
- Missing information you wish were present.
- Anything you cannot judge without seeing the actual image or table cells.
- Typos, grammar, or formatting.

## Critical guidance

- A typical paper has ZERO or ONE such error. Flagging should be RARE.
- You are working from text only (no image). Only flag what is decidable from the caption and context.
- If you are not highly confident, return error_detected = false.
- When you do flag, report confidence >= 0.8.

## Output Format

Return a JSON object with exactly these fields:
```json
{
  "error_detected": false,
  "confidence": 0.0,
  "reasoning": "Detailed explanation; quote the specific text if flagging...",
  "predicted_error_category": null
}
```

Possible predicted_error_category values (use null when no error):
- "Figure / table error"
- "Statistical reporting"
- null (if no error detected)
"""

LLM_ONLY_CITATION_SYSTEM_PROMPT = """You are a scientific-integrity reviewer examining a passage from a scientific paper that contains citations and references. Decide whether the way citations are used represents a CRITICAL error — a misrepresentation, false novelty claim, or attribution mistake serious enough to require an erratum or retraction.

## Flag an error (error_detected = true) ONLY for:

1. **False novelty claim**: The paper claims something as new that is explicitly attributed to prior work in the same excerpt.
2. **Citation misrepresentation**: A cited reference is described as supporting a claim it does not support, or is directly contradicted by the excerpt's own statements.
3. **Self-contradictory attribution**: The excerpt both cites a work for a claim and then describes that same claim as a contribution of the current paper.

## NEVER flag:

- Missing citations or "should cite X".
- The quality or appropriateness of the venue.
- Anything requiring external knowledge of the cited work beyond what the excerpt itself says about it.
- Typos in reference formatting.

## Critical guidance

- A typical paper has ZERO or ONE such error. Flagging should be RARE.
- You can only judge misrepresentations that are decidable from this excerpt alone.
- If you are not highly confident, return error_detected = false.
- When you do flag, report confidence >= 0.8.

## Output Format

Return a JSON object with exactly these fields:
```json
{
  "error_detected": false,
  "confidence": 0.0,
  "reasoning": "Detailed explanation; quote the specific text if flagging...",
  "predicted_error_category": null
}
```

Possible predicted_error_category values (use null when no error):
- "Citation / attribution"
- null (if no error detected)
"""


# ---------------------------------------------------------------------------
# Snippet-type → prompt mapping for "separate-prompts" mode
# ---------------------------------------------------------------------------

_TEXT_SNIPPET_TYPES = frozenset({
    "SECTION", "SUBSECTION", "PARAGRAPH", "THEOREM",
    "LEMMA", "PROPOSITION", "COROLLARY", "ALGORITHM",
    "APPENDIX",
})

_MATH_SNIPPET_TYPES = frozenset({"EQUATION"})

_FIGURE_TABLE_SNIPPET_TYPES = frozenset({"FIGURE", "TABLE"})


def _choose_system_prompt(
    snippet: VerificationSnippet,
    mode: str,
) -> str:
    """Select the right system prompt for the snippet and llm_only_mode."""
    if mode == "same-prompt":
        return LLM_ONLY_UNIFIED_SYSTEM_PROMPT

    # separate-prompts
    st = snippet.snippet_type.value
    if st in _MATH_SNIPPET_TYPES:
        return LLM_ONLY_MATH_SYSTEM_PROMPT
    if st in _FIGURE_TABLE_SNIPPET_TYPES:
        return LLM_ONLY_FIGURE_TABLE_SYSTEM_PROMPT
    if _is_citation_heavy(snippet):
        return LLM_ONLY_CITATION_SYSTEM_PROMPT
    return LLM_ONLY_TEXT_SYSTEM_PROMPT


def _is_citation_heavy(snippet: VerificationSnippet) -> bool:
    """Heuristic: does the snippet content appear to be citation-dense?"""
    content = snippet.content
    if not content:
        return False
    # Count citation markers like [1], [2,3], (Author, 2020), etc.
    import re
    bracket_cites = len(re.findall(r'\[\d+(?:,\s*\d+)*\]', content))
    paren_cites = len(re.findall(r'\([A-Z][a-z]+\s*(?:et al\.?)?\s*,\s*\d{4}[a-z]?\)', content))
    total_chars = max(len(content), 1)
    cite_density = (bracket_cites + paren_cites) / (total_chars / 500)  # per ~500 chars
    return cite_density >= 3


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class LLMOnlyVerifier(BaseVerifier):
    """Verify ANY snippet type with a pure LLM call (no specialist machinery).

    Used when ``PipelineConfig.llm_only_mode`` is set. The mode controls
    whether every snippet gets the same unified prompt or a prompt tailored
    to the snippet family.

    Because this verifier has no deterministic back-end, it relies entirely
    on the LLM's reasoning ability. Long snippets are split into overlapping
    chunks via ``_analyze_in_chunks`` (same as ``TextVerifier``).
    """

    name: str = "llm_only"

    def verify(
        self,
        snippet: VerificationSnippet,
    ) -> BaseVerificationResult:
        """Verify a single snippet with a pure LLM call.

        Args:
            snippet: Any verification snippet (text, equation, figure, table, etc.).

        Returns:
            BaseVerificationResult with findings.
        """
        start_time = time.monotonic()

        mode = self.config.llm_only_mode or "same-prompt"
        system_prompt = _choose_system_prompt(snippet, mode)

        logger.debug(
            f"[llm_only] Verifying {snippet.snippet_type.value} "
            f"'{snippet.location[:60]}' ({snippet.snippet_id}) mode={mode}"
        )

        if not snippet.content.strip():
            return self._make_result(
                snippet_id=snippet.snippet_id,
                status=VerificationStatus.SKIPPED,
                reasoning="Empty snippet content — nothing to verify.",
                execution_time_ms=(time.monotonic() - start_time) * 1000,
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
            }

        chosen, n_chunks, n_failed = self._analyze_in_chunks(snippet.content, analyze_chunk)

        if chosen is None:
            return self._make_result(
                snippet_id=snippet.snippet_id,
                status=VerificationStatus.UNVERIFIABLE,
                reasoning=f"All {n_chunks} chunk(s) failed to verify (e.g. empty LLM response).",
                execution_time_ms=(time.monotonic() - start_time) * 1000,
            )

        error_detected = chosen["error_detected"]
        return self._make_result(
            snippet_id=snippet.snippet_id,
            status=(
                VerificationStatus.ERROR_DETECTED if error_detected
                else VerificationStatus.NO_ERROR
            ),
            error_detected=error_detected,
            confidence=chosen["confidence"],
            reasoning=chosen["reasoning"],
            predicted_error_category=chosen["predicted_error_category"],
            execution_time_ms=(time.monotonic() - start_time) * 1000,
        )

    def can_verify(self, snippet: VerificationSnippet) -> bool:
        """LLM-only verifier can handle ANY snippet type."""
        if not self.verifier_config.enabled:
            return False
        return bool(snippet.content.strip())

    def _build_prompt(
        self,
        snippet: VerificationSnippet,
        content: Optional[str] = None,
    ) -> str:
        """Build the user prompt for this snippet.

        Args:
            snippet: The snippet being analyzed (for type/location metadata).
            content: Text to analyze. Defaults to full snippet content.
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

        parts.append(
            "\n\nFlag a CRITICAL error only if this excerpt contains a mistake "
            "serious enough to warrant an erratum or retraction (see the rules above). "
            "Otherwise set error_detected to false. Return ONLY the JSON response."
        )

        return "\n".join(parts)
