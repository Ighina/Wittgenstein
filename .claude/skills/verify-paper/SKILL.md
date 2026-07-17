---
name: verify-paper
description: Verify an entire scientific paper for errors by parsing, segmenting, routing to specialist verifiers, and aggregating findings. This is the main orchestrator skill — it mirrors the main.py verify pipeline.
---

# Verify Paper

You are the main orchestrator for the Paperena scientific paper verification pipeline. Your job is to verify an entire paper end-to-end: parse it, segment it into verification snippets, route each snippet to the appropriate specialist verifier, and aggregate the findings into a comprehensive report.

## When to Use This Skill

Invoke this skill to verify a complete scientific paper. The paper can be:
1. **Markdown file**: A Markdown file input including a number of sections and any number of tables, equations, theorems, etc. 
2. **LaTex input**: A .tex file to be read including same as above.

## Orchestration Modes

You support two orchestration modes (mirroring the Python pipeline):

### Exhaustive Mode (default)
Every snippet is routed to a specialist verifier by type:
- `EQUATION` → `/verify-math`
- `FIGURE`, `TABLE` → `/verify-vision`
- `SECTION`, `SUBSECTION`, `PARAGRAPH`, `THEOREM`, `LEMMA`, `PROPOSITION`, `COROLLARY`, `ALGORITHM`, `APPENDIX` → `/verify-text`
- Text with significant numeric content → also `/verify-statistical`
- Citation-heavy text → also `/verify-citation`

### Uncertainty Mode
1. First, run `/verify-triage` on every snippet to produce an uncertainty map.
2. Select only snippets with uncertainty ≥ threshold (default 0.30).
3. Route those high-uncertainty snippets to specialist verifiers.
4. Low-uncertainty snippets are accepted as low-risk without specialist review.

## Pipeline Steps

### Step 1: Obtain the paper

**From Markdown input:**
Use the provided markdown file location and read it to pass it as it is to the next step.

**From LaTex file:**
Use the provided path provided in the prompt and read the relative .tex file
In this case convert the LaTex file in markdown first and then pass that to the next step.

### Step 2: Parse and segment the paper

Invoke the `/segment_paper` by passing to it the markdown paper from the step before.

This returns snippets grouped by type: sections, equations, figures, tables, theorems.

### Step 3: Route snippets to verifiers (Exhaustive mode)

For each snippet, invoke the appropriate skill based on its type using the Skill tool:

| Snippet Type | Skill to Invoke | Notes |
|---|---|---|
| `EQUATION` | `/verify-math` | Use Skill tool with skill="verify-text", make sure to communicate to the skill that input will be equations|
| `FIGURE` | `/verify-vision` | Pass image_path if available |
| `TABLE` | `/verify-text` | Pass table content |
| `SECTION`, `SUBSECTION`, `PARAGRAPH` | `/verify-text` | Primary text verifier |
| `THEOREM`, `LEMMA`, `PROPOSITION`, `COROLLARY` | `/verify-text` | |
| `ALGORITHM`, `APPENDIX` | `/verify-text` | |

Additionally, for any text snippet:
- If it contains citations/references, also invoke `/verify-citation`.

Finally, pass all the snippets pasted together to the general cross-section verifier `/verify-cross-section-consistency` which will verify that all sections are consistent with each other

### Step 3 (Alternative): Uncertainty mode

1. First, invoke `/verify-triage` for EVERY snippet (use the Triage skill's instructions to score each one).
2. Build the uncertainty map: for each snippet, record its uncertainty score and suggested route.
3. Filter snippets: keep only those with uncertainty ≥ the configured threshold (default 0.30).
4. For selected snippets, invoke the specialist verifier suggested by the triage route:
   - `math` → `/verify-math`
   - `proof` → `/verify-math`
   - `citation` → `/verify-citation`
   - `vision` → `/verify-vision`
   - `text` → `/verify-text`
   - `none` → skip specialist verification
5. For unselected (low-uncertainty) snippets, record them as NO_ERROR with confidence = 1.0 - uncertainty.

### Step 4: Aggregate findings

After all verifier invocations return:

1. Collect all verification results.
2. Filter to only findings where `error_detected = true` AND `confidence ≥ threshold` (use 0.7 for math/statistics, 0.8 for text/citation, 0.75 for vision).
3. Sort predicted errors by confidence (highest first).
4. Produce the final paper-level report.

### Step 5: Generate the report

Output a comprehensive verification report:

```json
{
  "paper_id": "<id>",
  "title": "<title>",
  "paper_category": "<category>",
  "predicted_errors": [
    {
      "error_category": "Equation / proof",
      "error_location": "Equation 7",
      "confidence": 0.90,
      "supporting_evidence": "The equation simplifies to a non-zero residual...",
      "verifier_name": "verify-text",
      "snippet_id": "<paper_id>_eq_6"
    }
  ],
  "statistics": {
    "total_snippets": 45,
    "snippets_verified": 45,
    "errors_detected": 1,
    "verifier_usage": {
      "math_equation": 12,
      "vision": 6,
      "text": 25,
      "statistical": 2
    }
  },
  "verdict": "PASS | NEEDS_REVIEW | NEEDS_CORRECTION",
  "summary": "Human-readable summary of findings...",
  "uncertainty_map": [...]
}
```

## Verdict Guidelines

- **PASS**: No errors detected, or only low-confidence (<0.7) findings.
- **NEEDS_REVIEW**: One finding with medium confidence (0.7-0.85).
- **NEEDS_CORRECTION**: One or more findings with high confidence (≥0.85).

## Configuration Reference

The pipeline's behavior is controlled by these parameters (defaults shown):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `strictness` | `strict` | Error sensitivity: `strict` (only erratum-worthy) or `lenient` (broader) |
| `mode` | `exhaustive` | Orchestration: `exhaustive` or `uncertainty` |
| `uncertainty_threshold` | `0.30` | In uncertainty mode, escalate snippets at/above this score |
| `workers` | `8` | Number of concurrent verifications (not applicable in skill mode) |
