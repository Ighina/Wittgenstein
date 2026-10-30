---
name: verify-math-python
description: Verify ONE mathematical claim from a scientific paper computationally — recompute the quantity with sympy/numpy, reproduce the algebra symbolically, or search for a counterexample — and return the computational verdict plus the script and its output as evidence. Use for arithmetic, numerics, closed forms, bounds, units, and any claim a concrete witness settles.
---

# Verify Math with Python

You verify ONE mathematical claim by **computing it**, and you let the
computation decide. Unlike `/verify-math`, which reasons in prose, and
`/verify-math-lean`, which proves, this skill produces a witness: a number, a
simplified expression, or a counterexample, together with the script that
produced it.

This is the default verifier under `TOOL_ROUTING_POLICY.md`. Most published
mathematical errors are sign slips, dropped factors, wrong constants, misapplied
bounds and bad arithmetic — all of which a few lines of `sympy` settle in
seconds and settle decisively.

## Invocation

```
verify-math-python [--work-dir <dir>] [--timeout <seconds>]
```

| Flag | Default | Effect |
|---|---|---|
| `--work-dir <dir>` | a fresh temp dir | Where the script and its output are written. |
| `--timeout <seconds>` | `120` | Per-script wall clock. A script that exceeds it is killed and the attempt is `INCONCLUSIVE`. |

You are given a claim with its hypotheses and symbol definitions (from a
detector's `claim_content`), and usually a `probe_plan` from `/math-routing`
naming what to compute and over what instance space.

## When to Use This Skill

Use it when a **concrete witness settles the question**:

- a value, constant, unit or dimension recomputable from the paper's inputs;
- an algebraic identity, expansion, simplification, derivative or integral that
  can be reproduced symbolically and compared to the printed line;
- a closed form checked against the recurrence, sum or integral it claims to
  solve;
- an inequality or bound evaluated at the points that matter, including the
  endpoints and the degenerate cases;
- an asymptotic or complexity claim evaluated over a range of `n`;
- a counterexample search over small integers, matrices, graphs or polynomials;
- a probability, norm or measure asserted to lie in a range.

Return `UNVERIFIABLE` immediately when the claim is quantified over a space you
cannot enumerate or sample meaningfully, when its objects have no computational
representation, or when the defect is structural (a dependency between results,
a proof assuming its conclusion). Those belong to `/verify-math-lean` or
`/verify-math`.

## Step 1 — Write the claim down

In your own words, in one place, before writing any code:

- the claim exactly as the paper states it, with its hypotheses;
- every symbol's definition and type;
- **what result would mean the paper is wrong**, stated before you run anything.

That last line is the one that matters. Decide the falsification condition
first — "the paper prints $c = 1/2$ and the derivation gives $c = 1/4$", "there
is an $n \le 50$ with LHS > RHS" — and you cannot talk yourself into reading an
ambiguous output as confirmation afterwards.

### Faithfulness gate

The computation must test **the paper's claim**, not a convenient neighbour of
it.

- Do not narrow the domain to where the claim happens to hold. If the paper
  asserts something for all real $x$, probing only $x > 0$ proves nothing.
- Do not substitute values that trivialise the expression (all parameters 1, all
  matrices identity) and report agreement. Degenerate instances agree with
  almost anything.
- Do not swap an exact claim for an approximate one. If the paper writes `=`,
  test equality to tolerance and say what tolerance; do not silently accept
  "close enough".
- Carry the paper's standing assumptions in as constraints, and say which ones
  you imposed.

If a faithful computation is impossible — the claim's content is exactly the
thing you would have to assume — stop and return `UNVERIFIABLE`.

## Step 2 — Write one script

Write a single self-contained Python file to `$WORK_DIR/check.py`. Rules:

- **Print the verdict, do not infer it.** The script ends by printing an
  explicit line — `VERDICT: MISMATCH` / `VERDICT: MATCH` /
  `VERDICT: COUNTEREXAMPLE n=7` — computed from the falsification condition you
  wrote in Step 1. Reading a verdict out of a wall of numbers afterwards is how
  a wrong answer survives.
- **Print the quantities, not just the verdict.** Both sides of the comparison,
  the residual, the witness. These become the evidence.
- **Prefer exact arithmetic.** `sympy.Rational`, `sympy.simplify`,
  `sympy.nsimplify` over floats. When floats are unavoidable, state the
  tolerance and test against it explicitly.
- **Symbolic first, numeric second.** `sympy.simplify(lhs - rhs) == 0` is a
  stronger answer than agreement at twenty sample points. Fall back to sampling
  only when simplification does not close.
- **Search exhaustively where you can.** For a counterexample hunt, enumerate
  the whole small-instance space rather than sampling it, and print the bound of
  what was searched (`searched all n <= 60, all 3x3 0/1 matrices`). The size of
  a clean search is what makes a `NO_ERROR` meaningful.
- **No network, no writes outside `$WORK_DIR`.** The paper's sources are
  read-only to this skill.

Run it:

```bash
cd "$WORK_DIR" && timeout <timeout> python3 -I check.py
```

`-I` because the script is in its own directory and must not pick up modules
from anywhere else. Capture stdout and stderr verbatim — they are the evidence,
whatever the outcome.

Available: `sympy`, `numpy`, `scipy`, `mpmath` and the standard library. If an
import is missing, install it into the session (`pip install --quiet <pkg>`) or
work without it; do not report a verdict you did not compute.

## Step 3 — Iterate, with a budget

At most **four** script revisions. Legitimate reasons to revise:

- a syntax or API error;
- `simplify` did not close and a different normal form (`trigsimp`, `radsimp`,
  `together`, `nsimplify`) might;
- the instance space was too small to be informative and can be widened;
- the first script tested the wrong reading of an ambiguous claim — fix the
  reading and say so.

**Not** a legitimate reason: the result disagreed with what you expected. A
`MISMATCH` you then "fix" by adding a hypothesis until it matches is the
faithfulness gate failing on the second pass. If the claim as stated gives a
mismatch, that is the finding — record the mismatch and the script that shows
it.

Out of budget with nothing settled → `INCONCLUSIVE`.

## Step 4 — Read the verdict

| `python_status` | Condition |
|---|---|
| `REFUTED` | The script exhibits a concrete, faithful falsification: a counterexample, or a recomputed quantity that differs from the printed one beyond the stated tolerance. |
| `CONFIRMED` | A symbolic identity closed, or an exhaustive search over a space worth searching found nothing, and the computation was faithful. |
| `INCONCLUSIVE` | Simplification did not close, the search space was too small to mean anything, the script timed out, or the budget ran out. |
| `NOT_ATTEMPTED` | Ruled out at Step 1 — no computational representation, or no faithful computation exists. |

## Step 5 — Map to a verification result

| `python_status` | `status` | `error_detected` | Confidence |
|---|---|---|---|
| `REFUTED` | `ERROR_DETECTED` | `true` | `0.85`–`0.95`: `0.95` for an exact symbolic or exhaustive refutation, `0.85` for a floating-point one |
| `CONFIRMED` | `NO_ERROR` | `false` | — |
| `INCONCLUSIVE` | `UNVERIFIABLE` | `false` | — |
| `NOT_ATTEMPTED` | `UNVERIFIABLE` | `false` | — |

**`INCONCLUSIVE` is never a finding.** A `simplify` that did not close says
nothing about the claim; `sympy` fails to close true identities constantly. Only
a concrete witness justifies `error_detected = true`.

**A `CONFIRMED` is evidence about what was computed**, which is narrower than
the paper's claim whenever the claim is general. An exhaustive check over
`n <= 60` does not establish a statement about all `n` — say so in `reasoning`,
and let a `hybrid` route escalate it to Lean if a proof is what is needed.

## Output Format

Return a structured JSON object:

```json
{
  "snippet_id": "<id>",
  "verifier_name": "math-python",
  "status": "ERROR_DETECTED | NO_ERROR | UNVERIFIABLE",
  "error_detected": false,
  "confidence": 0.0,
  "reasoning": "What was computed, the falsification condition set in Step 1, what the script printed, and why that settles (or fails to settle) the claim. If REFUTED, give the witness explicitly.",
  "predicted_error_category": null,
  "snippet_type": "<EQUATION|DERIVATION|FORMULA_BLOCK|PROOF_STEP|MATHEMATICAL_MODEL>",
  "contradiction_locations": [],
  "python": {
    "python_status": "REFUTED | CONFIRMED | INCONCLUSIVE | NOT_ATTEMPTED",
    "approach": "symbolic | numeric | exhaustive_search | random_search",
    "script": "<the final check.py, verbatim>",
    "stdout": "<verbatim>",
    "stderr": "<verbatim, or ''>",
    "witness": "<the counterexample or the mismatching values, or null>",
    "search_space": "<what was searched, and its bound — or null for a symbolic check>",
    "tolerance": "<the numeric tolerance used, or null for exact arithmetic>",
    "revisions": 0
  }
}
```

`predicted_error_category` — use when `error_detected` is true, else `null`:

- `"Algebraic manipulation"` — invalid factorization, distribution, or inverse operations
- `"Fractional/Rational error"` — invalid simplification or combination of fractional terms
- `"Sign/Arithmetic error"` — dropped negatives, wrong constants, sign propagation failures
- `"Domain/Constraint violation"` — a quantity outside its mathematically possible range
- `"Calculus/Proof error"` — invalid limits, derivatives, or integral evaluations
- `"Numerically refuted"` — a counterexample was found and no narrower category fits

## Critical Guidance

- **The script is the authority, and it must be in the output.** A verdict
  without the script and its stdout is an opinion. Quote both verbatim, whatever
  the outcome — a reader has to be able to re-run it.
- **Decide the falsification condition before you run anything.** Written after
  the fact, it will match whatever came out.
- **Never report a refutation you cannot name.** "The simplification did not
  agree" is `INCONCLUSIVE`. `REFUTED` requires the witness: the values, the
  index, the instance.
- **Degenerate agreement is not agreement.** All-ones parameters and identity
  matrices satisfy nearly every wrong identity. Probe where the claim has
  content.
- **Say what you did not check.** An exhaustive search has a bound; a numeric
  check has a tolerance; a symbolic check has a branch it may have assumed.
  State them. The hybrid route and the orchestrator both need to know how far
  the evidence reaches.
