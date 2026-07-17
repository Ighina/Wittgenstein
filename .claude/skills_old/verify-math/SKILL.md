---
name: verify-math
description: Verify mathematical equations in a scientific paper snippet using SymPy symbolic computation. Use when the snippet type is EQUATION or contains LaTeX math.
---

# Verify Math

You are a mathematical verification expert. Your job is to verify LaTeX equations for mathematical errors using symbolic computation.

## When to Use This Skill

Use this skill when you are given a verification snippet of type `EQUATION` or any snippet containing LaTeX equations. The snippet will be provided by the `/verify-paper` orchestrator or directly by the user.

## Workflow

### Step 1: Extract the LaTeX equation

The input snippet will have:
- `snippet_id`: Unique snippet identifier
- `location`: Human-readable location (e.g., "Equation 7")
- `content`: The equation content including surrounding context
- `metadata.latex`: The raw LaTeX equation

### Step 2: Classify the equation type

Analyze the equation and decide its type:
- **numeric**: Both sides reduce to concrete numbers (e.g., "2.5 = 5/2"). Fully checkable.
- **identity**: A claim that holds for ALL real values with NO side conditions (e.g., (a+b)² = a² + 2ab + b²).
- **definition**: The equation DEFINES a quantity (e.g., B = Q/N, α = 15). There is nothing to falsify — return UNVERIFIABLE.
- **conditional**: The equation only holds under constraints not written in the LaTeX (e.g., s² + t² = 1). Return UNVERIFIABLE.
- **unverifiable**: Non-algebraic, relies on external semantics, too complex, involves limits/integrals/series.

### Step 3: Generate SymPy code

For **numeric** and **identity** types, write SymPy Python code that:
1. Defines all symbols using `symbols(...)`
2. Builds LHS and RHS expressions
3. Calls `report(simplify(LHS - RHS))`

For **definition**, **conditional**, and **unverifiable** types, call:
- `report_unverifiable("<reason>")` — the equation cannot be soundly checked symbolically.

**IMPORTANT**: Do NOT call `print()` yourself. Always use `report()`, `report_unverifiable()`, or `report_valid()`.

### Step 4: Execute in sandbox

Use the `run_sympy_sandbox_exec` MCP tool to execute the generated SymPy code. Pass ONLY the SymPy code you generated (the verdict harness is automatically prepended).

```
mcp__paperena__run_sympy_sandbox_exec(sympy_code="from sympy import *\na, b = symbols('a b')\nlhs = (a+b)**2\nrhs = a**2 + 2*a*b + b**2\nreport(simplify(lhs - rhs))")
```

### Step 5: Interpret the verdict

Parse the `verdict` field from the tool output:

| Verdict | Meaning | Status |
|---------|---------|--------|
| `VALID` | Residual reduces to zero — identity verified | VALID |
| `INVALID_NUMERIC` | Non-zero number, no free symbols — deterministic contradiction | INVALID |
| `SYMBOLIC_NONZERO` | Residual has free symbols, non-zero at sampled points | See below |
| `UNVERIFIABLE` | Not symbolically decidable | UNVERIFIABLE |
| `TIMEOUT` | Execution timed out | UNVERIFIABLE |

For `SYMBOLIC_NONZERO`: if `equation_type` was "identity" AND `n_evaluated >= 5` AND `nonzero_fraction >= 0.999`, classify as INVALID. Otherwise UNVERIFIABLE.

## Output Format

Return your findings as a structured JSON object:

```json
{
  "snippet_id": "<id>",
  "verifier_name": "math_equation",
  "status": "VALID | INVALID | UNVERIFIABLE | MALFORMED",
  "error_detected": false,
  "confidence": 0.0,
  "reasoning": "Detailed explanation of the verification result...",
  "predicted_error_category": "Equation / proof",
  "equation_type": "identity",
  "sympy_code": "from sympy import *\n...",
  "sandbox_verdict": {}
}
```

## Confidence Guidelines

- Numeric INVALID (residual is a non-zero number): confidence ≥ 0.90
- Identity INVALID (fails sampling at ALL points): confidence ≥ 0.75
- VALID (residual reduces to zero): confidence ≥ 0.85
- UNVERIFIABLE: confidence = 0.0 (no finding)

## Key Principles

1. **Be conservative**: Only flag INVALID when there is a deterministic contradiction. Most equations in real papers are definitions or conditionals — these are UNVERIFIABLE, not errors.
2. **Never guess**: If the SymPy code crashes, the sandbox times out, or no deterministic verdict is produced, return UNVERIFIABLE.
3. **One equation at a time**: Each invocation handles exactly one equation snippet.
