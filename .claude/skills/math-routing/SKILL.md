---
name: math-routing
description: Decide which verifier settles a candidate mathematical error — the Lean 4 formal verifier, the informal reviewer, the Python computational verifier, or the hybrid Python-then-Lean workflow. Implements TOOL_ROUTING_POLICY.md as a decision procedure over candidates produced by a detection pass. Returns one route per candidate with a rationale.
---

# Math Routing

You decide, for each **candidate mathematical error**, which verifier is most
likely to settle it. You do not verify anything yourself.

The policy you implement is `TOOL_ROUTING_POLICY.md` at the repo root. Read it
before routing. This skill turns that policy — written for user requests — into
a decision procedure over candidates, and nothing here may contradict it. If the
policy file and this skill disagree, the policy file wins and the disagreement is
a bug worth reporting in your output.

## Invocation

```
math-routing <path-to-candidates.json | inline candidate list>
```

Input is either a `math-detector-<tag>.json` written by
`scripts/run_math_detector.py`, or a list of candidates passed inline. Each
candidate carries `candidate_id`, `error_location`, `exact_quote`,
`claim_content`, `suspected_defect`, `check_kind`, `confidence`, `severity`.

`check_kind` is the detector's **hint**, not the decision. It was produced by a
model that had the whole paper in view and no obligation to know the verifiers'
strengths. Treat it as evidence and overrule it whenever `claim_content` says
otherwise — and say so in the rationale when you do.

## The four verifiers

| Route | Skill | Settles a claim by | Decisive when |
|---|---|---|---|
| `python` | `/verify-math-python` | computing: evaluating, simplifying symbolically, recomputing a bound, searching for a counterexample | a single concrete witness decides it |
| `lean` | `/verify-math-lean` | formalising in Lean 4 and compiling a proof or a refutation | the claim is general and formalises unambiguously |
| `hybrid` | `/verify-math-hybrid` | Python counterexample search first, then Lean on the surviving claim | general, formalisable, and cheap to probe numerically first |
| `informal` | `/verify-math` | close reading of the argument | neither computation nor formalisation can reach the defect |

## The decision procedure

Run the gates in order. The **first** gate that fires decides the route. Order
matters: it encodes the policy's "default to Python, escalate to Lean only when
the output must be a proof".

### Gate 0 — Unroutable

`claim_content` does not contain a checkable claim: it names a location without
stating anything ("Equation 12 looks wrong"), or every symbol in it is
undefined, or it reports something out of mathematical scope (a citation
problem, a protocol complaint, a typo).

→ route `none`, `status: UNROUTABLE`. Do not spend a verifier call on it. The
orchestrator records it as a dropped candidate.

This gate is narrow. A claim with *some* missing context still routes — the
verifier will say `UNVERIFIABLE` and that is useful information. Gate 0 is only
for candidates with no mathematical content at all.

### Gate 1 — A concrete witness decides it

The defect, if real, is exhibited by **one** computation on specific values:

- a numeric value, constant, unit or dimension that can be recomputed from the
  paper's own stated inputs;
- an algebraic identity, simplification, expansion, derivative or integral that
  can be reproduced symbolically and compared line to line;
- a closed form checked against the recurrence or sum it claims to solve, at
  several concrete `n`;
- a stated counterexample — the detector already wrote the witness down and it
  only needs evaluating;
- an asymptotic or complexity claim checkable by evaluating both sides over a
  range;
- a probability, norm or measure asserted to lie in a range it can be shown to
  leave.

→ route `python`.

This is the policy's default and it is the most common correct answer. The
policy is explicit: *"Do not use Lean merely because the task involves
mathematics."* A sign error in a derivation is settled by `sympy` in seconds and
costs a Lean run minutes to say the same thing less reliably.

### Gate 2 — The claim is general, and a counterexample is cheap to look for

The claim is quantified — "for all", "for every", "given any", or an identity,
inequality, bound, limit, monotonicity or convergence statement asserted in
general — **and** it has a finite, enumerable or samplable instance space: small
integers, matrices of modest size, graphs on few vertices, polynomials of low
degree, a parameter range.

→ route `hybrid`.

This is the policy's hybrid workflow, in its own order: Python searches for a
counterexample; if one is found the claim is refuted empirically and no proof is
needed; if none is found the claim is formulated precisely and handed to Lean.
Prefer `hybrid` over bare `lean` whenever the instance space is probeable — a
found counterexample is both faster and more legible evidence than a Lean
refutation, and a failed search is useful context for the formalisation.

### Gate 3 — The claim is general and formalises unambiguously

Quantified or structural, but with **no** practical way to probe instances: the
objects are infinite-dimensional, the claim is about an abstract structure with
no small models, or constructing one instance is itself the hard part. The
policy's Lean triggers apply — universal quantification, implications that must
hold in general, equality, equivalence, injectivity, surjectivity, monotonicity,
continuity, associativity, commutativity, distributivity, a proof obligation
about an algorithm, protocol, type or invariant, or a theorem, lemma or corollary
statement.

And the policy's escalation condition holds: **the statement can be formalised
unambiguously.** Every symbol has a definition you could render in Lean with
Mathlib, and a faithful statement needs no invented hypotheses.

→ route `lean`.

### Gate 4 — Everything else

The defect is real-but-informal: a dependency between results (an earlier
proposition is wrong and a later theorem leans on it), a proof that assumes what
it set out to show, a lemma invoked outside its stated conditions, a conclusion
stronger than the argument gives, a step whose gap is structural rather than
computational, or a claim resting on domain objects the paper never pins down.

→ route `informal`.

Gate 4 is not a dumping ground — it is the right route for a large class of real
errors. SPOT's equation/proof annotations include dependency failures and
over-strong conclusions that no amount of computation or formalisation reaches,
because the defect is in how the paper's results relate to each other.

## Escalation guard

Before returning `lean` or `hybrid`, check the policy's escalation condition once
more against `claim_content`:

- Would a faithful Lean statement need a hypothesis the paper does not state? If
  adding it makes the claim true, formalisation would verify a *different*
  claim. → downgrade to `informal`.
- Does the claim depend on an object with no Mathlib counterpart and no
  definition in `claim_content`? → downgrade to `informal`.
- Is the quantification actually over a finite set the paper enumerates? → that
  is Gate 1. → downgrade to `python`.

A Lean route that cannot produce a faithful statement wastes the most expensive
verifier in the set and returns `FAILED`, which is not evidence about anything.

## Output Format

Return a structured JSON object:

```json
{
  "policy": "TOOL_ROUTING_POLICY.md",
  "routes": [
    {
      "candidate_id": "<id>",
      "route": "python | lean | hybrid | informal | none",
      "status": "ROUTED | UNROUTABLE",
      "gate": "0 | 1 | 2 | 3 | 4",
      "detector_hint": "<the candidate's check_kind>",
      "overruled_hint": false,
      "rationale": "One or two sentences: which gate fired and on what in claim_content. Name the policy trigger when the route is lean or hybrid.",
      "probe_plan": "For python/hybrid only: what to compute or enumerate, and over what instance space. Null otherwise.",
      "formal_target": "For lean/hybrid only: the one claim to formalise, stated as a single proposition. Null otherwise."
    }
  ],
  "totals": {"python": 0, "lean": 0, "hybrid": 0, "informal": 0, "none": 0},
  "policy_conflicts": []
}
```

- **One entry per candidate, in input order.** A candidate with no route entry
  is a candidate silently dropped, and the orchestrator cannot tell that apart
  from one you decided was unroutable.
- `probe_plan` and `formal_target` are what the downstream verifier actually
  reads first. Vague ones ("check the equation", "prove the lemma") waste the
  call — name the quantity, the instance space, the proposition.
- `policy_conflicts` records any place where `TOOL_ROUTING_POLICY.md` and this
  skill's gates pointed in different directions, with the candidate id and both
  readings. Normally empty.

## Critical Guidance

- **Route on `claim_content`, not on `error_location` or `check_kind`.** The
  content is what the verifier will see. A candidate labelled `general_claim`
  whose content is "recompute 0.06 + 0.08 + 0.5" is a Python job.
- **Python is the default, and that is deliberate.** Most published
  mathematical errors are sign slips, dropped factors, wrong constants and bad
  arithmetic — all settled by computation. Reaching for Lean first is the single
  most expensive mistake available here.
- **Never route the same candidate twice.** `hybrid` already contains the
  Python-then-Lean sequence; do not also emit a separate `lean` route for it.
- **A route is a prediction, not a promise.** The verifier may return
  `UNVERIFIABLE`; that is a normal outcome and not a routing error. Routing
  errors are sending a prose dependency failure to Lean, or a
  for-all-reals identity to a numeric spot check and calling the result proof.
