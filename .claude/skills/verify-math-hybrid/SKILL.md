---
name: verify-math-hybrid
description: Verify ONE general mathematical claim by the hybrid workflow of TOOL_ROUTING_POLICY.md — search for a counterexample in Python first, and escalate to a Lean 4 proof only if the search finds none and the surviving claim formalises unambiguously. Returns the empirical result and the formal result side by side.
---

# Verify Math — Hybrid (Python, then Lean)

You verify ONE **general** mathematical claim in two stages, in this order:

1. **Python** searches for a counterexample. If it finds one, the claim is
   refuted and there is nothing to prove.
2. **Lean 4** proves the claim that survived the search — but only if the search
   was clean and the claim formalises unambiguously.

This is the hybrid workflow of `TOOL_ROUTING_POLICY.md`, verbatim:

> 1. Use Python to explore examples and search for counterexamples.
> 2. If no counterexample is found, formulate the claim precisely.
> 3. Use Lean to prove the precise claim.
> 4. Report the Python result as empirical evidence and the Lean result as a
>    formal proof.
>
> Escalate to Lean only when the statement can be formalized unambiguously.

Read that file before running. The order is the point: the cheap stage runs
first, and a found counterexample means the expensive stage never runs. On a
false claim — which is what a verifier is looking for — the search usually wins,
and it wins in seconds.

## Invocation

```
verify-math-hybrid [--project=<Name>] [--keep-lean] [--work-dir <dir>] [--timeout <seconds>]
```

| Flag | Default | Effect |
|---|---|---|
| `--project=<Name>` | derived from the paper | Lean project under `MATH_VERIFICATION/`, passed straight through to `/verify-math-lean`. |
| `--keep-lean` | off | Keep the Lean project. Passed straight through. |
| `--work-dir <dir>` | a fresh temp dir | Where the Python stage writes its script and output. |
| `--timeout <seconds>` | `120` | Per-script wall clock for the Python stage. |

**Pass `--project` and `--keep-lean` through unchanged.** This skill owns
neither the Lean project's name nor its lifetime — an orchestrator walking a
paper's candidates does, and a project created or deleted behind its back costs
every later candidate a multi-gigabyte Mathlib rebuild.

## When to Use This Skill

Use it when `/math-routing` returned `hybrid`: the claim is **quantified or
structural** — "for all", an identity, an inequality, a bound, a limit, a
monotonicity or convergence statement — **and** its instance space is
enumerable or samplable: small integers, matrices of modest size, graphs on few
vertices, polynomials of low degree, a bounded parameter range.

Do not use it when:

- a single concrete computation settles the claim → `/verify-math-python`;
- the claim is general but has no probeable instances → `/verify-math-lean`;
- the defect is structural or prose-level → `/verify-math`.

## Step 1 — Formulate the claim once, for both stages

Write down, before running anything:

- the claim with every hypothesis and symbol definition;
- the **instance space** to search, and why it is the right one: the parameters
  that vary, their ranges, and the structural cases that matter (zero, empty,
  degenerate, boundary, the smallest non-trivial size);
- the **falsification condition** — what output means the paper is wrong;
- the **formal target** — the single proposition Lean would be asked to prove,
  which must be the same claim the search probes.

Both stages test one claim. If the Python stage probes a restriction and Lean
proves a generalisation, the two results are about different statements and
reporting them together is worse than reporting neither.

## Step 2 — Stage A: the counterexample search

Run the Python stage exactly as `/verify-math-python` specifies — its Step 2
script rules, its faithfulness gate, its four-revision budget. Two differences,
because here the search is a means and not the whole answer:

- **Search for the counterexample, do not check for agreement.** Enumerate the
  instance space exhaustively where it is finite, and prioritise the structural
  cases from Step 1 over bulk sampling. The boundary and degenerate instances
  are where published claims actually break.
- **Record the bound of what was searched**, always. `searched all n <= 200, all
  3x3 matrices over {-1,0,1}` is what makes the clean result mean something at
  Step 3. An unbounded "no counterexample found" means nothing.

Outcomes:

| Stage A | Next |
|---|---|
| counterexample found | **stop.** Skip Lean entirely, go to Step 4 with `REFUTED`. |
| clean search over a space worth searching | go to Step 3 |
| search inconclusive — space too small, timed out, no faithful probe | go to Step 3, but Lean is now the only evidence |

A found counterexample is the end of the run. Do not escalate to Lean to
"confirm" it: the witness already settles it, and a Lean `FAILED` afterwards
would only muddy a clean refutation.

## Step 3 — The escalation gate

Escalate to Lean **only** if all three hold. Check them in order and record
which one stopped you if you do not escalate.

**E1 — The search did not refute it.** Stage A came back clean or inconclusive.

**E2 — The statement formalises unambiguously.** Every symbol in the formal
target has a definition you could render with Mathlib, and a faithful statement
needs no invented hypotheses. This is the policy's own escalation condition and
it is the one that usually fails: a claim about objects the paper defines loosely
cannot be formalised faithfully, and a Lean run on it returns `FAILED` after
spending the most expensive verifier in the set.

**E3 — A proof is what the question needs.** The claim is genuinely general. If
Stage A's search was exhaustive over the *entire* instance space the paper's
claim ranges over, the claim is already settled — report `CONFIRMED` from Stage A
and do not escalate. Exhaustive beats formal when the domain is finite.

If E2 or E3 fails, skip Lean and go to Step 4 with Stage A's verdict, recording
`escalated: false` and the gate that stopped it.

## Step 4 — Stage B: Lean

Invoke the formal verifier on the Step 1 formal target:

```
/verify-math-lean --project=<Name> [--keep-lean]
```

Hand it the formal target as the claim, with the hypotheses from Step 1 — and
hand it Stage A's result as context: a clean exhaustive search over small
instances tells the formalisation which cases are safe and often suggests the
induction. Its `lean_status` (`PASSED` / `REFUTED` / `FAILED` /
`NOT_ATTEMPTED`) comes back unchanged; do not reinterpret it.

## Step 5 — Combine the two verdicts

The two stages are different kinds of evidence and the table keeps them
distinct. Read it as: a witness beats everything, a proof beats a clean search,
a clean search beats nothing.

| Stage A | Stage B | `hybrid_status` | `status` | `error_detected` | Confidence |
|---|---|---|---|---|---|
| counterexample | *not run* | `REFUTED_EMPIRICAL` | `ERROR_DETECTED` | `true` | `0.90`–`0.95` |
| clean / inconclusive | `REFUTED` | `REFUTED_FORMAL` | `ERROR_DETECTED` | `true` | `0.95` |
| clean | `PASSED` | `CONFIRMED_BOTH` | `NO_ERROR` | `false` | — |
| inconclusive | `PASSED` | `CONFIRMED_FORMAL` | `NO_ERROR` | `false` | — |
| clean | `FAILED` / `NOT_ATTEMPTED` | `CONFIRMED_EMPIRICAL` | `NO_ERROR` | `false` | — |
| clean | *not escalated (E2/E3)* | `CONFIRMED_EMPIRICAL` | `NO_ERROR` | `false` | — |
| inconclusive | `FAILED` / `NOT_ATTEMPTED` | `UNSETTLED` | `UNVERIFIABLE` | `false` | — |
| inconclusive | *not escalated* | `UNSETTLED` | `UNVERIFIABLE` | `false` | — |

The rows that need care:

- **A counterexample is the strongest outcome here**, not a weaker version of a
  Lean refutation. It is a concrete instance anyone can re-run. `0.95` when the
  witness is exact, `0.90` when it rests on floating-point arithmetic.
- **`CONFIRMED_EMPIRICAL` is a clean search and nothing more.** It is `NO_ERROR`
  because the verifier found no error, not because the claim is proved. Say
  exactly what was searched in `reasoning`; a reader must be able to see that
  the general claim is untouched.
- **A Lean `FAILED` never flips a clean search into a finding.** Lean fails on
  true statements constantly. When both stages fail to settle anything the
  answer is `UNSETTLED` / `UNVERIFIABLE` — not suspicion.
- **Disagreement is impossible by construction** (a counterexample stops the
  run), so if you ever find yourself with a witness *and* a `PASSED`, one of
  them is unfaithful to the claim. Re-read both against Step 1's formal target,
  report `UNSETTLED`, and say which stage drifted.

## Output Format

Return a structured JSON object:

```json
{
  "snippet_id": "<id>",
  "verifier_name": "math-hybrid",
  "status": "ERROR_DETECTED | NO_ERROR | UNVERIFIABLE",
  "error_detected": false,
  "confidence": 0.0,
  "reasoning": "The claim as formulated at Step 1; what the search covered and found; whether Lean was escalated to and why or why not; what the combination settles. Empirical and formal evidence kept clearly apart.",
  "predicted_error_category": null,
  "snippet_type": "<EQUATION|DERIVATION|FORMULA_BLOCK|PROOF_STEP|THEOREM>",
  "contradiction_locations": [],
  "hybrid": {
    "hybrid_status": "REFUTED_EMPIRICAL | REFUTED_FORMAL | CONFIRMED_BOTH | CONFIRMED_FORMAL | CONFIRMED_EMPIRICAL | UNSETTLED",
    "escalated": false,
    "escalation_gate": "E1 | E2 | E3 | null",
    "escalation_note": "Why Lean was or was not run, in one sentence.",
    "formal_target": "<the single proposition both stages were about>"
  },
  "python": { "...": "the /verify-math-python python{} block, verbatim" },
  "lean": { "...": "the /verify-math-lean lean{} block verbatim, or null if not escalated" }
}
```

Both sub-blocks go in **verbatim**. The point of this skill is that a reader can
see the empirical evidence and the formal evidence separately and judge each;
collapsing them into one summary destroys that.

`predicted_error_category` — same values as `/verify-math-python`, plus
`"Formally refuted"` when Lean proved the negation.

## Critical Guidance

- **Search first, always.** Running Lean before the search inverts the policy
  and wastes minutes on claims a counterexample kills in seconds.
- **Stop on a witness.** A found counterexample ends the run. Escalating anyway
  risks a `FAILED` that makes a clean refutation look uncertain.
- **The escalation gate is a real gate.** Most claims that reach it fail E2.
  Skipping Lean is the normal outcome, recorded as `CONFIRMED_EMPIRICAL` with
  the search bound — not a failure of the run.
- **Never merge the two verdicts into one number.** `CONFIRMED_EMPIRICAL` and
  `CONFIRMED_BOTH` are both `NO_ERROR` and they are not the same evidence. The
  orchestrator reads `hybrid_status` to tell them apart.
- **Pass `--project` and `--keep-lean` through untouched**, and never create or
  delete a Lean project yourself. That lifetime belongs to the caller.
