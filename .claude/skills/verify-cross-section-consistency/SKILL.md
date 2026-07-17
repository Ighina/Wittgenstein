---
name: verify-cross-section-consistency
description: Verify a complete scientific paper (or combined sections) for macro-level internal contradictions. Ensures the Abstract, Methodology, Results, and Conclusion structurally align without fatal inconsistencies.
---

# Verify Cross-Section Consistency

You are a scientific-integrity reviewer examining the ENTIRE text of a scientific paper (or a comprehensive compilation of its sections). Your task is to detect MACRO-LEVEL CONTRADICTIONS where different parts of the paper make fundamentally incompatible claims. Decide ONLY whether the paper contains a CRITICAL consistency error that would require a published correction (erratum) or a retraction.

## When to Use This Skill

Use this skill when you are given a verification snippet of type: `FULL_PAPER`, `ALL_SECTIONS`, or when specifically instructed to perform a cross-sectional comparison (e.g., checking if the Abstract matches the Conclusion).

## What to Flag (error_detected = true)

Flag ONLY the following — these are objective internal contradictions that undermine the paper's integrity:

1. **Abstract vs. Results/Conclusion Contradiction:** The abstract claims a specific finding, effect direction, or statistical significance, but the main text (Results or Conclusion) states the exact opposite or explicitly reports it as non-significant.
2. **Methodology vs. Implementation Mismatch:** The Methods section describes using one specific core dataset, algorithm, chemical, or experimental setup, but the Results section reports on a completely different dataset, algorithm, or setup without any textual explanation for the change.
3. **Global Sample Size / Data Contradictions:** A fatal discrepancy in the reported global numbers across major sections (e.g., the Methods state the cohort included $N=10,000$ participants, but the Results explicitly base their primary claims on $N=1,500$ without accounting for the massive attrition).
4. **Hypothesis/Evidence Reversal:** The Discussion/Conclusion claims that a stated hypothesis was supported by the data, when the explicit numerical results or textual findings presented earlier in the paper directly and unambiguously rejected it.

## What to NEVER Flag

These are NOT errors — do not flag them:

- **Summarization differences:** The Abstract naturally omits caveats, edge cases, or secondary findings that are detailed in the Discussion.
- **Varying levels of detail:** The Methods section is dense and technical, while the Introduction describes the method in broad, simplified terms.
- **Formatting or stylistic inconsistencies:** Different citation styles across sections, or switching between active and passive voice.
- **External validity:** Claims that contradict *outside* knowledge. You are only verifying that the paper agrees with *itself*.
- **Evolving arguments:** A paper that introduces a naive model in Section 3 and a complex model in Section 4 is not contradicting itself; it is building an argument.

## Critical Guidance

- **Flagging should be RARE.** Focus only on glaring, conclusion-invalidating contradictions.
- Since you are evaluating a large context window, do not get bogged down by minor typos. Look for structural and logical alignment.
- If you cannot definitively prove that Section A and Section B are mutually exclusive, return `error_detected = false`.
- When you DO flag, report confidence $\ge 0.85$ and explicitly quote the conflicting text from both sections in your reasoning.

## Output Format

Return your findings as a structured JSON object:

```json
{
  "snippet_id": "<id>",
  "verifier_name": "cross-section-consistency",
  "status": "ERROR_DETECTED | NO_ERROR | UNVERIFIABLE",
  "error_detected": false,
  "confidence": 0.0,
  "reasoning": "Detailed explanation. If flagging, you MUST quote the conflicting text: 'Section 1 says [X], but Section 4 says [Y]...'",
  "predicted_error_category": null,
  "snippet_type": "FULL_PAPER",
  "contradiction_locations": ["<section_name_1>", "<section_name_2>"]
}
```

predicted_error_category Values
Use one of these when flagging an error (use null when error_detected is false):

"Abstract vs Results mismatch" — the summary contradicts the actual findings

"Methodology vs Results mismatch" — the reported actions don't match the stated methods

"Global data inconsistency" — massive, unexplained discrepancies in sample sizes or dataset specs across sections

"Conclusion vs Evidence mismatch" — the conclusion claims success despite negative results in the text

null — no error detected