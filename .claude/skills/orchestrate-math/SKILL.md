---
name: orchestrate-math
description: Orchestrate mathematical verification of a whole paper. Extracts every self-contained equation, theorem, and proof from a .md or .pdf manuscript, routes each to verify-math (informal review) or verify-math-lean (formal Lean 4 proof), retries Lean failures informally, and writes a <paper_name>_verified_math.json report next to the paper.
---

# Orchestrate Math Verification

You are the orchestrator for **mathematical** verification of one paper. Where
`/verify-paper` fans a whole manuscript out across every specialist, this skill
covers only the mathematics — but covers it in depth, with a formal Lean 4 proof
attempt on everything worth formalising.

## Invocation

```
orchestrate-math <path-to-paper.md | path-to-paper.pdf>
```

| Input | Description |
|---|---|
| paper path | **Required.** A `.md` or `.pdf` manuscript. |

Derived once, up front, and used throughout:

- `PAPER_PATH` — the absolute path you were given.
- `PAPER_DIR` — the directory containing it.
- `PAPER_NAME` — the filename **without** its extension (`spectral.pdf` →
  `spectral`).
- `REPORT_PATH` — `$PAPER_DIR/${PAPER_NAME}_verified_math.json`.
- `PROJECT_NAME` — a CamelCase Lean project name derived from `PAPER_NAME`
  (`spectral_gap_v2` → `SpectralGapV2`). Strip non-alphanumerics; if the result
  starts with a digit, prefix `Paper`.

Setup for the Lean side (toolchain, `lean-lsp` MCP, `lean4` plugin):
`docs/math-verification.md`.

## Step 1 — Read the manuscript

**`.md`** — read the file.

**`.pdf`** — read it with the Read tool's `pages` parameter (max 20 pages per
call; required at all for PDFs over 10 pages). Page through the whole document
and reconstruct it as markdown, preserving equations as LaTeX. Do not stop at
page 20 — a paper's mathematics is usually in the middle and the appendix.

If the file is neither `.md` nor `.pdf`, stop and say so.

## Step 2 — Extract mathematical objects

Walk the manuscript top-to-bottom in one pass and emit a snippet for every
**self-contained mathematically verifiable object**:

| Type | Trigger |
|---|---|
| `EQUATION` | display math — `$$…$$`, ```` ```math ````, `\begin{equation|align|gather}` |
| `THEOREM`, `LEMMA`, `PROPOSITION`, `COROLLARY` | a stated result, with its hypotheses |
| `PROOF` | a proof body, or an individual `PROOF_STEP` where a step carries the load |
| `DEFINITION` | only when it asserts a property that could be false (e.g. "this is well-defined because…") |
| `FORMULA_BLOCK` | a derivation chain presented as one unit |
| `MATHEMATICAL_MODEL` | a model whose stated properties are checkable |

Snippet IDs use `<PAPER_NAME>::<TYPE>::<index>`, index zero-based **within
type** — the same convention as `/segment_paper`.

**Self-contained** is the bar. Carry into `content` everything needed to judge
the claim: the statement, its hypotheses, and the definitions of the symbols it
uses, quoting surrounding context where the snippet alone would be ambiguous.
Record `location` (the containing section heading).

Skip: inline math used only as prose notation, numeric results reported from
experiments, and pure notation declarations that assert nothing.

## Step 3 — Route every snippet BEFORE running anything

Decide the route for **all** snippets first and write the routing table down.
You cannot do this lazily: Step 4 needs to know which Lean snippet is the *last*
one before it runs the *first* one.

| Route to | When |
|---|---|
| `verify-math` | Straightforward algebra/calculus a careful reader can check by inspection; anything **unformalisable** — claims about experimental data, modelling assumptions, statements resting on undefined domain objects, prose arguments with no crisp formal content. |
| `verify-math-lean` | Formalisable and non-trivial: theorems, lemmas, inequalities and bounds, limits, convergence and combinatorial claims, closed-form identities, proofs whose correctness turns on a real argument rather than on arithmetic. |

When genuinely torn, prefer `verify-math-lean` — a machine-checked verdict beats
an informal one, and a `FAILED` there falls back to `verify-math` at Step 5
anyway, so the cost of guessing wrong is one extra pass, not a lost check.

Produce the routing table before continuing:

```
| snippet_id | type | location | route | rationale |
```

Let `LEAN_QUEUE` be the routed-to-Lean snippets in document order, and
`N = len(LEAN_QUEUE)`.

## Step 4 — Run the verifiers

### 4a. `verify-math` snippets

Invoke `/verify-math` per snippet. These are independent — batch them freely.

### 4b. `verify-math-lean` snippets — strictly serial

They share one Lean project, so they must run **one at a time, in order**. For
the `i`-th snippet of `LEAN_QUEUE` (1-based):

```
/verify-math-lean --project=<PROJECT_NAME> --keep-lean      # while i < N
/verify-math-lean --project=<PROJECT_NAME>                  # i == N — deletes the project
```

Every call is pinned to the same `--project` so the Mathlib build (~7.5 GB, many
minutes to create) is built once and reused. Only the final call omits
`--keep-lean`, and that call is what removes
`MATH_VERIFICATION/<PROJECT_NAME>`.

Special cases:

- **`N == 0`** — no Lean call at all; no project is ever created. Skip to Step 5.
- **`N == 1`** — the single call omits `--keep-lean`. Never pass `--keep-lean` to
  a queue of one; it would strand the project.
- **The run ends early** — you abort, hit a budget, or a snippet errors out after
  at least one `--keep-lean` call succeeded. The project is still on disk and it
  is still yours. Delete it before you finish:
  `rm -rf MATH_VERIFICATION/<PROJECT_NAME>`. Report every unverified snippet as
  `UNVERIFIABLE`; never leave the project behind because the run went sideways.

## Step 5 — Fall back to informal review

Every `verify-math-lean` result whose `lean.lean_status` is `FAILED` (compile
errors survived, or `sorry`/`sorryAx` outlived the budget) goes to
`/verify-math` for a second, best-effort judgement. `NOT_ATTEMPTED` results go
the same way when the reason was unformalisability rather than missing setup.

Tell `/verify-math` it is reviewing a snippet Lean could not settle, and give it
the Lean `compiler_feedback` — where the proof got stuck is often a genuine hint
about where the mathematics is weak.

The fallback result **supplements, it does not replace**. Keep both in the
report: `lean_result` and `fallback_result`. If they disagree, say so in
`summary` and prefer the Lean verdict only when it is `PASSED` or `REFUTED` —
a `FAILED` Lean run is not evidence about the claim either way.

Run the fallbacks **after** Step 4 finishes, so the Lean project deletion is not
held up behind informal reviews.

## Step 6 — Write the report

Write `$REPORT_PATH` — `<PAPER_NAME>_verified_math.json`, in the same directory
as the paper. Overwrite it if it exists.

```json
{
  "paper_name": "<PAPER_NAME>",
  "paper_path": "<PAPER_PATH>",
  "lean_project": "MATH_VERIFICATION/<PROJECT_NAME>",
  "lean_project_kept": false,
  "totals": {
    "snippets_extracted": 0,
    "routed_to_math": 0,
    "routed_to_lean": 0,
    "lean_passed": 0,
    "lean_refuted": 0,
    "lean_failed": 0,
    "fallbacks_run": 0,
    "errors_detected": 0
  },
  "results": [
    {
      "snippet_id": "<PAPER_NAME>::THEOREM::0",
      "snippet_type": "THEOREM",
      "location": "4.1 Convergence",
      "content": "<the snippet, verbatim>",
      "route": "verify-math-lean",
      "route_rationale": "Formalisable convergence bound; correctness turns on the argument.",
      "lean_result": { "...": "the verify-math-lean JSON, verbatim" },
      "fallback_result": null,
      "status": "ERROR_DETECTED | NO_ERROR | UNVERIFIABLE",
      "error_detected": false,
      "confidence": 0.0
    }
  ],
  "findings": [
    {
      "snippet_id": "<PAPER_NAME>::EQUATION::3",
      "category": "Formally refuted",
      "severity": "critical",
      "confidence": 0.95,
      "evidence": "Lean refuted the identity at $a = b = 1$: $\\sqrt{2} \\neq 2$.",
      "verifier": "verify-math-lean",
      "exact_quote": "$$\\sqrt{a+b} = \\sqrt{a} + \\sqrt{b}$$"
    }
  ],
  "verdict": "CRITICAL_ERRORS | MINOR_ERRORS | INCONCLUSIVE | NO_ERRORS",
  "verdict_confidence": 0.0,
  "summary": "Human-readable summary: what was checked, what Lean settled, what it could not, and where the two verifiers disagreed."
}
```

Rules:

- `results` holds **one entry per extracted snippet** — including ones that
  came back clean. The report is the record of what was checked, not only of
  what was wrong.
- `findings` holds only `error_detected = true` entries at or above confidence
  `0.7`, sorted by confidence descending.
- `exact_quote` is copied character-for-character from the manuscript. Never
  paraphrase and never emit character offsets.
- `evidence` and `summary` render as markdown — write math as `$…$` / `$$…$$`,
  never ASCII pseudo-math.
- `severity`: `critical` when the error invalidates a main claim, `minor` when
  the argument survives it.
- `lean_project_kept` must be `true` only if the project is genuinely still on
  disk. Check before writing it.

## Verdict

- **CRITICAL_ERRORS** — at least one confident `critical` finding. A Lean
  `REFUTED` is the strongest form of this.
- **MINOR_ERRORS** — confident findings, all `minor`.
- **INCONCLUSIVE** — too little was settled to judge: most snippets
  `UNVERIFIABLE`, or the run was cut short before the main results were checked.
- **NO_ERRORS** — the paper's main mathematical claims were checked and nothing
  confident remains.

`verdict_confidence` is your confidence in the **verdict**, not a copy of any
finding's confidence. Lots of `PASSED` Lean proofs on the main theorems = high.
A verdict resting on one `FAILED`-then-informal snippet = low.

## Critical Guidance

- **A Lean `FAILED` is not a finding.** It means Lean did not settle the claim.
  Only a `REFUTED` — a compiling proof of the negation — is formal evidence of
  an error. This is the single easiest way to turn this skill into a
  false-positive machine.
- **Route before you run.** The `--keep-lean` scheme depends on knowing the
  queue length in advance.
- **The project is deleted exactly once, by the last Lean call** — or by you, by
  hand, if the run ends early. Never by an intermediate call.
- **Report everything checked**, not just the failures. A snippet Lean proved
  clean is a result worth recording.
