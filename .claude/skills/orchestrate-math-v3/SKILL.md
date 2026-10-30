---
name: orchestrate-math-v3
description: Detect-then-verify mathematical verification of a whole paper. A lenient high-recall LLM pass over the entire manuscript proposes candidate mathematical errors, /math-routing sends each to one of four verifiers (Lean 4, informal, Python, or hybrid Python-then-Lean) per TOOL_ROUTING_POLICY.md, and the verifiers decide. Writes <paper_name>_verified_math[-<tag>].json next to the paper.
---

# Orchestrate Math Verification v3 — Detect, Route, Verify

You verify the **mathematics** of one paper by proposing far too many candidate
errors and then letting four verifiers throw out the ones that are not real.

## Why this is not v1 or v2

`/orchestrate-math` (v1) extracts every self-contained equation, theorem and
proof and verifies each in isolation. Measured on SPOT's equation/proof errors
under oracle localisation
(`task_specific_output/math-oracle-spot/oracle-opus5_per_paper.csv`), the
annotated statement was extracted and self-contained in only **59%** of papers.
Seven papers' annotated results never entered the pipeline at all; eight more
arrived with the load-bearing proof truncated or the two halves of a
cross-statement defect separated. **The extractor, not the verifier, was the
binding constraint** — and no verifier can check a statement it never receives.

v1 also loses context by construction: a snippet cut out of a paper arrives
without the conventions, standing assumptions and symbol definitions that live
three pages away, so the verifier either guesses or returns `UNVERIFIABLE`.

So v3 inverts the order. **Detection comes first, over the whole paper, with the
whole paper in context.** One lenient pass proposes candidates and writes each
one out self-contained — statement, hypotheses, symbol definitions, all gathered
from wherever they sit. Then the verifiers, which are the precision filter, say
which ones are real.

The asymmetry is the design: a false positive costs one verifier call, a false
negative costs the error.

`/orchestrate-math-v2` runs a whole-paper baseline and a per-statement Lean arm
*independently* and reconciles them. v3 is a cascade, not two arms: the detector
has no verdict of its own, it has candidates, and a candidate's verdict is
whatever its verifier says.

## Invocation

```
orchestrate-math-v3 <path-to-paper.md | path-to-paper.pdf> [--llm <model>] [--tag <tag>] [--keep-lean] [--detector-tag <tag>]
```

| Flag | Default | Effect |
|---|---|---|
| `--llm <model>` | *the agent itself* | Run the detection pass by shelling out to `scripts/run_math_detector.py --model <model>` instead of doing it inline. Any model `scripts/llm_providers.py` resolves. |
| `--tag <tag>` | none | Write `<PAPER_NAME>_verified_math-<tag>.json` and stamp `"tag"` into the report. |
| `--keep-lean` | **on, always** | See **The Lean project is shared** below. v3 passes `--keep-lean` to every Lean call whether or not you were given the flag. |
| `--detector-tag <tag>` | `detector` | Which `math-detector-<tag>.json` to read or write. |

**Tag your v3 runs.** Untagged, this skill writes to the same path v1 uses and
the second run silently overwrites the first.

Derived once, up front:

- `PAPER_PATH`, `PAPER_DIR`, `PAPER_NAME` (filename without extension).
- `REPORT_PATH` — `$PAPER_DIR/${PAPER_NAME}_verified_math.json`, or
  `${PAPER_NAME}_verified_math-<tag>.json` with `--tag`.
- `PROJECT_NAME` — CamelCase Lean project name from `PAPER_NAME`
  (`spectral_gap_v2` → `SpectralGapV2`). Strip non-alphanumerics; if it starts
  with a digit, prefix `Paper`.

### The Lean project is shared — never delete it

In the `data/*-workspace` layouts every paper is `paper.md`, so `PROJECT_NAME`
is **`Paper`** for all of them: one shared Mathlib-backed project at
`MATH_VERIFICATION/Paper`, ~7.4 GB, already built.

Consequences, and they are not optional:

- **Pass `--keep-lean` to every `/verify-math-lean` and `/verify-math-hybrid`
  call, including the last one.** v1 and v2 let the final call delete the
  project; here that would strand every later paper in the batch with a
  multi-gigabyte rebuild.
- **Never `rm -rf MATH_VERIFICATION/Paper`**, not at the end of a run, not on
  an aborted run, not to "clean up".
- Set `lean_project_kept: true` in the report.

Lean-side setup (toolchain, `lean-lsp` MCP, `lean4` plugin):
`docs/math-verification.md`.

## Step 1 — Read the manuscript

**`.md`** — read the file.

**`.pdf`** — read it with the Read tool's `pages` parameter (max 20 pages per
call; required at all for PDFs over 10 pages). Page through the **whole**
document and reconstruct it as markdown, preserving equations as LaTeX. A
paper's mathematics is usually in the middle and the appendix — do not stop at
page 20.

Hold the full markdown as `PAPER_MD`. The detector consumes it, and in `--llm`
mode it needs to exist as a file.

If the file is neither `.md` nor `.pdf`, stop and say so.

## Step 2 — Detect candidates, leniently

One pass, the whole paper, no segmentation. Exactly one of the two modes runs.

### 2a. Default — the agent itself

No `--llm` was passed. Do the detection pass yourself, in one shot, over
`PAPER_MD`.

Use the **same contract** as `scripts/run_math_detector.py:build_detector_prompt`
— read that function and follow it, so the two modes stay comparable. What is
load-bearing in it:

- **Recall is the objective, not precision.** Report a step you cannot follow,
  an identity you cannot reproduce, a bound whose direction you are unsure of, a
  hypothesis that looks too weak, a constant you cannot derive. Report it at
  whatever confidence you actually hold — a candidate at 0.2 is useful, silence
  is not. Report two nearby candidates separately rather than merging them.
- **Mathematics only.** Equations, derivations, identities, inequalities and
  bounds, theorem and lemma statements, proofs and proof steps, asymptotic and
  complexity claims, falsifiable definitions, and the arithmetic or units of any
  computed quantity. Not typos, not notation, not citations, not experimental
  protocol, not writing quality — those cannot be verified mathematically, so
  they only waste a verifier call.
- **Every candidate must be self-contained.** This is the fix for v1's context
  loss and the single most important field you write. `claim_content` carries the
  statement in full, every hypothesis including standing assumptions stated
  elsewhere in the paper, the definition of every symbol, and for a proof step
  the preceding step it should follow from. Write it as if handing the claim to a
  mathematician who has never seen the paper — because that is exactly what the
  verifier is.
- Preserve LaTeX exactly as the paper writes it.

Emit the same JSON the script would, one entry per candidate:

```json
{"candidates": [{"error_location": "…", "exact_quote": "…", "claim_content": "…",
  "suspected_defect": "…", "check_kind": "numeric | symbolic | general_claim | proof_step | prose",
  "confidence": 0.0, "severity": "critical | minor"}]}
```

Assign `candidate_id` as `<PAPER_NAME>::CAND::<index>`, zero-based in emission
order, and write the whole thing to
`$PAPER_DIR/math-detector-<detector-tag>.json` in the script's schema so the two
modes leave the same artefact behind.

### 2b. `--llm <model>` — shell out to the detector script

```bash
python3 scripts/run_math_detector.py \
    --workspace-dir "$(dirname "$PAPER_DIR")" \
    --paper-id "$(basename "$PAPER_DIR")" \
    --model <model> \
    --tag <detector-tag>
```

The script walks a *workspace of prepared papers*. When the paper already sits
in that layout (`data/spot-workspace/<paper_id>/paper.md`) the call above works
as written. Otherwise stage it first:

```bash
STAGING="$(mktemp -d)/ws"
mkdir -p "$STAGING/$PAPER_NAME"
cp "$PAPER_PATH" "$STAGING/$PAPER_NAME/paper.md"   # or write PAPER_MD there, for a PDF input
```

and point `--workspace-dir` at `$STAGING`, then copy the resulting
`math-detector-<tag>.json` into `$PAPER_DIR` and `rm -rf` the staging directory.

Read `candidates[]` out of that file. Failure handling:

- **Missing API key / provider error** — the script exits non-zero and names the
  env var it wanted. Do **not** silently fall back to 2a: the user named a model
  and a run under a different model is a different experiment. Stop and report.
- **`FAILED: …`** — the script retried three times already. Record the detection
  stage as unavailable, write the report with `verdict: INCONCLUSIVE`, and say
  so. With no candidates there is nothing to verify, and that is the honest
  outcome.

### 2c. Deduplicate, do not prune

Merge two candidates **only** when they are the same claim at the same location
with the same suspected defect. Keep the higher confidence and note the merge.

Everything else survives Step 2, including the low-confidence ones. Pruning here
defeats the entire design: the detector's job was recall, and the verifiers are
what decide. The only candidates that leave without a verifier call are the ones
`/math-routing` rules `UNROUTABLE` at its Gate 0.

## Step 3 — Route every candidate

Invoke the routing skill on the candidate list:

```
/math-routing $PAPER_DIR/math-detector-<detector-tag>.json
```

It returns one route per candidate — `python`, `lean`, `hybrid`, `informal` or
`none` — with a `gate`, a `rationale`, and a `probe_plan` / `formal_target`
where they apply. The policy it implements is `TOOL_ROUTING_POLICY.md`.

Write the routing table down **before running anything**, because Step 4 needs
to know how many Lean-touching candidates there are and in what order:

```
| candidate_id | location | route | gate | rationale |
```

Let `LEAN_QUEUE` be the candidates routed `lean` or `hybrid`, in input order,
and `N = len(LEAN_QUEUE)`.

A candidate routed `none` gets a `results[]` entry with
`status: UNROUTABLE`, `error_detected: false`, and the routing rationale. It is
recorded, not hidden — the count of unroutable candidates is how you tell a
detector that is proposing junk from one that is proposing hard problems.

## Step 4 — Verify

### Order

Run the cheap verifiers first: `python`, then `informal`, then the `LEAN_QUEUE`.
A Python refutation is minutes cheaper than a Lean one and the evidence is
easier to read, so front-loading them means an aborted run still produced
findings.

### The Lean queue is strictly serial

The `LEAN_QUEUE` candidates share one Lean project, so they run **one at a time,
in order**, every call pinned to the same project and every call keeping it:

```
/verify-math-lean   --project=<PROJECT_NAME> --keep-lean      # route = lean
/verify-math-hybrid --project=<PROJECT_NAME> --keep-lean      # route = hybrid
```

- **`N == 0`** — no Lean call at all. Every Lean-routed count is zero and the
  report rests on the other verifiers.
- **`--keep-lean` is unconditional here**, including on the last call. See
  **The Lean project is shared**.
- **The run ends early** — you abort, hit a budget, or a candidate errors out.
  The shared project stays on disk; that is correct, do not clean it up. Mark
  every unreached candidate `NOT_ATTEMPTED` and continue to Step 5 — a partial
  verification still produces a report.

**One candidate's failure ends that candidate, not the queue.** A verifier that
returns `UNVERIFIABLE`, or a Lean `FAILED`, is a normal outcome: record it and
move to the next candidate. A queue where every Lean call failed is a valid
run — Step 5 reads it as "Lean settled nothing".

### What each verifier is handed

Hand the verifier the candidate's `claim_content` as the claim — **not** a
re-extracted snippet from the paper. That field is the whole point of Step 2:
it already carries the hypotheses and symbol definitions. Pass along
`exact_quote` as the location evidence, `suspected_defect` as what to look for,
and the `probe_plan` / `formal_target` from Step 3.

Tell the verifier what is suspected, and do not tell it what to conclude. "Check
whether the factor of 2 in the third line follows" is context; "confirm that the
factor of 2 is wrong" is a leading question, and a verifier that agrees with the
detector by construction adds nothing to the detector.

## Step 5 — Adjudicate

The detector proposes; the verifier disposes. For each candidate, with its
verifier's `status` and the verifier-specific sub-verdict:

| Verifier outcome | `status` | `error_detected` | Confidence | `source` |
|---|---|---|---|---|
| Lean `REFUTED` | ERROR_DETECTED | `true` | `0.95` | `lean` |
| Hybrid `REFUTED_FORMAL` | ERROR_DETECTED | `true` | `0.95` | `hybrid-formal` |
| Hybrid `REFUTED_EMPIRICAL` | ERROR_DETECTED | `true` | `0.95` exact witness, `0.90` floating-point | `hybrid-empirical` |
| Python `REFUTED` | ERROR_DETECTED | `true` | `0.95` exact or exhaustive, `0.85` floating-point | `python` |
| Informal `ERROR_DETECTED` | ERROR_DETECTED | `true` | `min(verifier_confidence, 0.85)` | `informal` |
| Lean `PASSED` | NO_ERROR | `false` | — | `lean` |
| Hybrid `CONFIRMED_BOTH` / `CONFIRMED_FORMAL` | NO_ERROR | `false` | — | `hybrid` |
| Python `CONFIRMED`, exact or exhaustive | NO_ERROR | `false` | — | `python` |
| Hybrid `CONFIRMED_EMPIRICAL` | NO_ERROR | `false` | — | `hybrid-empirical` |
| Python `CONFIRMED`, sampled only | **unsettled — see 5b** | | | |
| Informal `NO_ERROR` | **unsettled — see 5b** | | | |
| Anything `UNVERIFIABLE` / `FAILED` / `UNSETTLED` / `NOT_ATTEMPTED` | **unsettled — see 5b** | | | |
| `UNROUTABLE` (Gate 0) | UNROUTABLE | `false` | — | `routing` |

### 5a. A cleared candidate is dropped, and recorded

A candidate the verifier cleared decisively — a Lean `PASSED`, an exact symbolic
identity, an exhaustive search — leaves the findings. It keeps its `results[]`
entry with `status: NO_ERROR` and `cleared_by` naming the verifier.

This is where v3's precision comes from, so it has to be auditable: the count of
candidates cleared is the measure of how much work the verifier stage actually
did. A run where nothing was cleared means the verifiers are not filtering and
the report is just the detector's opinion with extra steps — say so in
`summary`.

### 5b. An unsettled candidate keeps the detector's suspicion, capped

Nobody settled it. Dropping all of these would recreate v1's failure from the
other end — a real error proposed by the detector and lost because Lean timed
out is exactly the error this design exists to catch. Keeping all of them would
make the report the detector's unfiltered output.

So an unsettled candidate becomes a finding **only if both** hold:

- **Specific.** `suspected_defect` names the step, quantity, index or hypothesis
  at issue. "The constant in Eq. (12) should be 1/4, not 1/2" is specific; "this
  step seems unjustified", "the assumptions look strong", "this may not hold in
  general" are vague doubts and are dropped.
- **The detector believed it.** `confidence >= 0.5`.

Then `error_detected: true` at `min(detector_confidence, 0.60)`, `source:
"detector-unsettled"`. The cap sits below every verifier-backed finding,
because nothing checked it.

An unsettled candidate that fails either test is dropped with
`status: UNVERIFIABLE` and a note. **A verifier failure is never itself
evidence of an error** — a Lean `FAILED` or a `simplify` that did not close says
nothing about the claim, and treating it as suspicion is the fastest way to turn
this skill into a false-positive machine.

### 5c. Merge duplicates found late

Two candidates that different verifiers confirmed as the same defect at the same
location collapse into one finding, keeping the highest confidence and listing
both `candidate_id`s. Count the merge in `totals.merged`.

## Step 6 — Write the report

**The report is unconditional.** Write it even when the detector failed, when
every verifier failed, or when the run was cut short — with the failures
recorded and the verdict `INCONCLUSIVE`. A run that produced no file produced
nothing, however much work went into it: the caller cannot tell it apart from a
crash, and the paper has to be redone from scratch.

Write `$REPORT_PATH`, overwriting if it exists. The `results[]` / `findings[]`
shape is what `evaluate_verifier.py:_math_predictions` reads — keep it exactly.

```json
{
  "skill": "orchestrate-math-v3",
  "tag": "<tag or omitted>",
  "paper_name": "<PAPER_NAME>",
  "paper_path": "<PAPER_PATH>",
  "lean_project": "MATH_VERIFICATION/<PROJECT_NAME>",
  "lean_project_kept": true,
  "detector": {
    "available": true,
    "mode": "self | script",
    "model": "<model, or 'agent' in self mode>",
    "report_path": "<math-detector-<tag>.json>",
    "n_candidates": 0,
    "n_merged": 0,
    "mean_confidence": 0.0
  },
  "routing": {
    "policy": "TOOL_ROUTING_POLICY.md",
    "python": 0, "lean": 0, "hybrid": 0, "informal": 0, "none": 0,
    "overruled_hints": 0
  },
  "totals": {
    "candidates_detected": 0,
    "candidates_routed": 0,
    "candidates_verified": 0,
    "confirmed_by_verifier": 0,
    "cleared_by_verifier": 0,
    "unsettled_kept": 0,
    "unsettled_dropped": 0,
    "unroutable": 0,
    "merged": 0,
    "errors_detected": 0,
    "lean_passed": 0, "lean_refuted": 0, "lean_failed": 0,
    "python_confirmed": 0, "python_refuted": 0, "python_inconclusive": 0
  },
  "results": [
    {
      "snippet_id": "<PAPER_NAME>::CAND::0",
      "snippet_type": "EQUATION | THEOREM | PROOF | …",
      "location": "4.1 Convergence",
      "content": "<the candidate's claim_content, verbatim>",
      "exact_quote": "<verbatim from the manuscript>",
      "suspected_defect": "<what the detector suspected>",
      "detector_confidence": 0.0,
      "route": "python | lean | hybrid | informal | none",
      "route_gate": "0 | 1 | 2 | 3 | 4",
      "route_rationale": "<from /math-routing>",
      "verifier_result": { "...": "the verifier's JSON, verbatim, or null" },
      "adjudication": {
        "rule": "verifier_confirmed | verifier_cleared | unsettled_kept | unsettled_dropped | unroutable",
        "cleared_by": "lean | python | hybrid | null",
        "note": "One sentence on why this outcome, in the verifier's own terms."
      },
      "status": "ERROR_DETECTED | NO_ERROR | UNVERIFIABLE | UNROUTABLE",
      "error_detected": false,
      "confidence": 0.0,
      "source": "lean | python | hybrid-formal | hybrid-empirical | informal | detector-unsettled | routing"
    }
  ],
  "findings": [
    {
      "snippet_id": "<PAPER_NAME>::CAND::3",
      "candidate_ids": ["<PAPER_NAME>::CAND::3"],
      "category": "Formally refuted",
      "severity": "critical",
      "confidence": 0.95,
      "evidence": "Lean refuted the identity at $a = b = 1$: $\\sqrt{2} \\neq 2$.",
      "verifier": "orchestrate-math-v3:lean",
      "source": "lean",
      "exact_quote": "$$\\sqrt{a+b} = \\sqrt{a} + \\sqrt{b}$$"
    }
  ],
  "verdict": "CRITICAL_ERRORS | MINOR_ERRORS | INCONCLUSIVE | NO_ERRORS",
  "verdict_confidence": 0.0,
  "summary": "How many candidates the detector proposed, how they routed, how many the verifiers confirmed and how many they cleared, and what remains unsettled. Name the cleared-to-confirmed ratio — it is the measure of whether the verifier stage did any work."
}
```

Rules:

- `results` holds **one entry per candidate**, including cleared, unsettled and
  unroutable ones. The report records what was checked, not only what was wrong.
- **`findings` holds every `error_detected = true` result at confidence
  `>= 0.5`**, sorted by confidence descending. This threshold is deliberately
  lower than v1's `0.7`: in v3 the precision filter is the verifier stage, not a
  confidence cut-off, and the `0.5`–`0.6` band is exactly the unsettled-but-
  specific candidates of 5b. Every result is in `results[]` with its honest
  confidence, so any other threshold can be re-derived offline with
  `scripts/threshold_sweep.py`.
- `verifier` stays a `math-*`/`orchestrate-math` style string so downstream
  filters recognise it; `source` carries the attribution.
- `exact_quote` is character-for-character from the manuscript. Never paraphrase,
  never emit character offsets.
- `evidence` and `summary` render as markdown — `$…$` / `$$…$$`, never ASCII
  pseudo-math.
- `severity`: `critical` when the error invalidates a main claim, `minor` when
  the argument survives it.
- `lean_project_kept` is `true`. Check the project is genuinely on disk before
  writing it.

## Verdict

- **CRITICAL_ERRORS** — at least one verifier-confirmed `critical` finding. A
  Lean `REFUTED` or a concrete counterexample is the strongest form.
- **MINOR_ERRORS** — confident findings, all `minor`.
- **INCONCLUSIVE** — the detector was unavailable, or nearly everything is
  unsettled: no verifier settled a main claim either way.
- **NO_ERRORS** — the detector's candidates over the main claims were checked
  and cleared, and nothing specific remains.

`verdict_confidence` is confidence in the **verdict**, not a copy of a finding's.
Many candidates, decisively cleared or confirmed = high. A verdict resting
mostly on `detector-unsettled` findings = low, and say why.

## Critical Guidance

- **Do not prune the detector's output.** Low confidence is not a reason to drop
  a candidate before routing — it is a number that follows the candidate
  through to Step 5. The one place candidates die before a verifier call is
  `/math-routing` Gate 0.
- **Hand over `claim_content`, never a re-extracted snippet.** Re-cutting the
  statement out of the paper throws away the context the detector gathered and
  reintroduces exactly the failure v3 was built to fix.
- **A verifier failure is not a finding.** `FAILED`, `INCONCLUSIVE` and
  `UNVERIFIABLE` mean the claim was not settled. Only 5b's specific,
  believed-by-the-detector candidates survive unsettled, and they are capped at
  `0.60`.
- **Never lead the verifier.** Give it the claim and what is suspected; let it
  reach its own verdict. A verifier that rubber-stamps the detector makes the
  whole cascade pointless.
- **The Lean project is shared and must survive.** `--keep-lean` on every call,
  no `rm -rf` ever, `lean_project_kept: true`.
- **Route before you run.** The Lean queue's serial order has to be known in
  advance.
- **Clean up any staging directory** from 2b even when the run fails — but never
  `MATH_VERIFICATION/`.
- **Tag v3 runs** so they do not overwrite v1's or v2's report.
