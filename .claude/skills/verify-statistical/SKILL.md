---
name: verify-statistical
description: Verify statistical claims and numeric relationships in papers by extracting checkable expressions and deterministically recomputing them. Use for snippets with reported numbers, percentages, or statistics.
---

# Verify Statistical / Numeric Claims

You extract checkable NUMERIC relationships from one excerpt of a scientific paper, then deterministically re-check them using the safe arithmetic evaluator. You do NOT decide correctness by reasoning alone — you restate claims as closed arithmetic and let the calculator verify.

## When to Use This Skill

Use this skill when a verification snippet contains reported numbers, percentages, statistics, p-values, or quantitative results. The orchestrator will route snippets to you, or you may be called directly.

## Workflow

### Step 1: Extract numeric claims

Read the excerpt and identify numeric claims that are **fully self-contained** — both sides of the relationship must be determined by numbers written in the excerpt. Examples:

- Percentages/proportions that should sum to a total: "groups of 33%, 33%, 34%" → check if 33+33+34 = 100
- A derived value from stated inputs: "mean of 2, 4, 6 is 4.0" → check mean([2,4,6]) vs 4.0
- An arithmetic identity stated in prose: "12 of 50, i.e. 24%" → check 12/50*100 vs 24
- A unit conversion: "5 km = 5000 m" → check 5 km → 5000 m

**STRICT RULES for expressions:**
- May contain ONLY literal numbers taken from THIS excerpt and operators: +, -, *, /, **, %
- Functions allowed: sqrt, log, ln, log10, log2, exp, abs, round, min, max, sum, mean, floor, ceil, pow
- Constants: pi, e
- NO variable names, NO symbols, NO units inside the expression
- Only emit a check when BOTH sides are fully determined by numbers in the excerpt
- When in doubt, omit it

### Step 2: Check each claim deterministically

For each claim, use the `check_numeric_claim` MCP tool:

```
mcp__paperena__check_numeric_claim(expression="33 + 33 + 34", expected=100.0, tolerance=0.01)
```

The tool returns:
- `computed`: The recomputed value
- `expected`: The reported value
- `relative_error`: |computed - expected| / |expected|
- `passed`: true if relative_error ≤ tolerance

For unit conversions, use the `safe_arithmetic_eval` MCP tool.

### Step 3: Aggregate findings

- If ANY check deterministically fails (passed = false): flag as INVALID
- If ALL checks pass (passed = true): VALID
- If no checkable numeric claims were found: UNVERIFIABLE
- If extraction failed: UNVERIFIABLE

## What to NEVER Flag

- Claims that need external data not present in the excerpt
- Relationships with symbolic/unknown quantities
- Rounding differences within reasonable tolerance (adjust tolerance for rounded numbers)
- Anything requiring statistical inference beyond simple arithmetic

## Output Format

Return your findings as a structured JSON object:

```json
{
  "snippet_id": "<id>",
  "verifier_name": "statistical",
  "status": "VALID | INVALID | UNVERIFIABLE | SKIPPED",
  "error_detected": false,
  "confidence": 0.0,
  "reasoning": "Description of checks performed and their outcomes...",
  "predicted_error_category": "Statistical reporting",
  "checks": [
    {
      "description": "Percentages sum to 100%",
      "expression": "33 + 33 + 34",
      "expected": 100.0,
      "computed": 100.0,
      "tolerance": 0.01,
      "passed": true
    }
  ]
}
```

## Confidence Guidelines

- INVALID (deterministic contradiction): confidence = 0.90
- VALID (all checks pass): confidence = 0.85
- UNVERIFIABLE (no checkable claims): confidence = 0.0
