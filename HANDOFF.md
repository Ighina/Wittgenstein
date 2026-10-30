# Handoff — orchestrate-math-v3 on SPOT

Branch: `paperena-first-round`. Written 2026-10-30.

This picks up a half-finished experiment. Read **Where it stands** and **The
blocker** first; everything else is reference.

---

## Why v3 exists

Scoring `/orchestrate-math` (v1) on SPOT's 37 equation/proof errors under
*oracle localisation* showed the **extractor, not the verifier**, was the
binding constraint:

| v1, all 37 equation/proof papers | |
|---|---|
| Extractor ceiling — annotated statement extracted **and self-contained** | **0.595** |
| Achieved — oracle site actually flagged | **0.541** |

The verifier stage cost only **0.054**. The other ~0.40 was gone before any
verifier ran: 7 papers whose annotated result was never extracted, 8 more where
it arrived truncated or with the two halves of a cross-statement defect
separated. Improving Lean or the informal reviewer could not have moved v1 past
~0.60.

So v3 inverts the order: **detect first, over the whole paper**, with each
candidate written out self-contained; the verifiers become the precision filter
rather than the discovery mechanism.

Provisional confirmation, on the 18 papers scored before the credit ran out:

| ceiling on the same 18 papers | strict | incl. partial |
|---|---|---|
| v1 extractor | 0.556 | 0.833 |
| v3 detector | **0.778** | 0.833 |

The lenient ceiling is **identical**. The detector does not reach more
annotated statements than the extractor did — it delivers them **whole**. The
entire gain is `partial` → `covered`.

---

## Where it stands

| | status |
|---|---|
| v3 skills + detector script | **complete**, committed |
| Detector run (`detector-opus5`) | **25 / 37** papers |
| Detector scored by the localisation judge | **18 / 37** |
| v3 full pipeline (`orchestrate-math-v3-v3`) | **3 / 37** — batch was stopped deliberately |
| v1 + 4 baselines on SPOT | complete, from the earlier round |

**v3's end-to-end precision/recall does not exist yet.** Three papers is not a
measurement. That is the main outstanding deliverable.

---

## The blocker

```
Your credit balance is too low to access the Anthropic API.
```

The `ANTHROPIC_API_KEY` in `.env` is exhausted. That kills:

- `scripts/run_math_detector.py` (11 of its 12 failures were purely this)
- every judge call in `evaluate_math_oracle.py` and `evaluate_math_systems.py`

It does **not** affect `batch-verify-math.sh`, which drives `claude -p` through
the Claude Code subscription. The v3 batch can resume without a top-up; nothing
else can.

**Top up the API credit before doing anything in step 3 or later.**

---

## Resume, in order

From the repo root. Every step is resumable — completed papers are skipped and
judge verdicts are cached per paper, so a re-run after an interruption only
pays for what is missing.

### 1. Get the paper sources back

Git carries the *results* but not the 47 MB of `paper.md`:

```bash
python scripts/download_spot.py          # rewrites paper.md (+ ground_truth, metadata)
```

Deterministic from `amphora/SPOT`; it will not touch the result JSONs.

### 2. Resume the v3 pipeline  *(no API credit needed)*

The long pole: **~27 min/paper**, 34 papers left, so roughly **15 hours**.
Serial by necessity — all papers share one Lean project (see **Lean** below),
so do not parallelise it.

```bash
./scripts/batch-verify-math.sh --dataset spot --skill orchestrate-math-v3 \
    --tag v3 --model claude-opus-5 --keep-lean \
    --paper-list configs/spot-math-papers.txt
```

Skips the 3 papers already done. Run it detached (`nohup … &`) and watch
`data/spot-workspace/<paper_id>/verify-math.log`.

### 3. Finish the detector run  *(needs credit)*

```bash
python3 scripts/run_math_detector.py --workspace-dir data/spot-workspace \
    --paper-list configs/spot-math-papers.txt \
    --model claude-opus-5 --tag detector-opus5 --concurrency 4
```

12 papers left. One of them, `0003` (Iwasawa modules), previously failed by
spending 96k output tokens on thinking and emitting nothing; the brevity-on-retry
fix is in but **has never been exercised** — watch whether it recovers.

### 4. Score the detector's coverage  *(needs credit)*

```bash
python3 scripts/evaluate_math_oracle.py --from-detector detector-opus5 \
    --judge-model claude-sonnet-5 --concurrency 4
```

Read the **`cover`** column. In this mode `detect_*` equals coverage by
construction (every candidate is a flagged statement) and `clean FP` prints
`n/a`. This is the number that turns the provisional 0.778 into a result.

### 5. Score v3 end to end  *(needs credit; after step 2 finishes)*

Order matters — the comparison writes the per-paper CSV that the oracle ladder's
rung A reads.

```bash
# (a) precision/recall vs v1 and the baselines
python3 scripts/evaluate_math_systems.py --workspace-dir data/spot-workspace \
    --tag orchestrate-math-v3-v3 --judge-model claude-sonnet-5 \
    --output-dir task_specific_output/math-comparison-spot

# (b) the same at v1's operating point -- v3 promotes findings at 0.5, v1 at 0.7
python3 scripts/evaluate_math_systems.py --workspace-dir data/spot-workspace \
    --tag orchestrate-math-v3-v3 --judge-model claude-sonnet-5 \
    --min-confidence 0.7 \
    --output-dir task_specific_output/math-comparison-spot

# (c) oracle localisation + the protocol ladder
python3 scripts/evaluate_math_oracle.py --tag orchestrate-math-v3-v3 \
    --judge-model claude-sonnet-5 --concurrency 4
```

`--output-dir` is **not** optional: the default is `math-comparison/`
(PaperAudit), and the SPOT results live in `math-comparison-spot/`.

Do (b). Comparing v3's findings at 0.5 against v1's at 0.7 compares two
operating points, not two systems.

---

## New-laptop setup

### What git carries, and what it doesn't

| | |
|---|---|
| **Carried** | skills, scripts, configs, `ground_truth.json`, `metadata.json`, all result JSONs (detector candidates, routing tables, verification reports, **cached judge verdicts**), `task_specific_output/` |
| **Not carried** | `data/spot-workspace/**/*.md` — 47 MB, regenerate with step 1 |
| **Not carried** | `MATH_VERIFICATION/` — 7.4 GB Lean build, must be rebuilt |
| **Not carried** | `.env` — recreate it by hand |

The cached judge verdicts are the valuable part: they are ~55 files of real API
spend, and they travel. Do not delete them to "start clean".

### `.env`

At the repo root, not committed:

```
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...        # only for the gpt-* baselines
DEEPSEEK_API_KEY=...      # only for the deepseek baselines
```

### Lean

Full setup in `docs/math-verification.md`. Short version:

```bash
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y
export PATH="$HOME/.elan/bin:$PATH"
elan toolchain install leanprover/lean4:stable && elan default leanprover/lean4:stable
claude mcp add lean-lsp -s project uvx lean-lsp-mcp
claude plugin marketplace add cameronfreer/lean4-skills
claude plugin install lean4@lean4-skills
pip install sympy numpy scipy mpmath anthropic datasets pyarrow
```

**`MATH_VERIFICATION/Paper` is shared by every workspace paper** — all of them
are `paper.md`, so the project name derives to `Paper` for all 37. Consequences:

- v3 passes `--keep-lean` to **every** Lean call, including the last. v1 and v2
  let the final call delete the project; here that would charge the next paper a
  full Mathlib rebuild.
- **Never `rm -rf MATH_VERIFICATION/Paper`.**
- First run on a new machine rebuilds it: a multi-thousand-file Mathlib cache
  fetch plus several minutes of build. Verify before trusting any verdict:
  `cd MATH_VERIFICATION/Paper && lake build`.

---

## Architecture

```
scripts/run_math_detector.py          stage 1: lenient whole-paper detection
.claude/skills/math-routing/          stage 2: TOOL_ROUTING_POLICY.md, 5 gates
.claude/skills/verify-math/           stage 3: informal reviewer (pre-existing)
.claude/skills/verify-math-python/    stage 3: sympy/numpy + counterexamples
.claude/skills/verify-math-lean/      stage 3: Lean 4 (pre-existing)
.claude/skills/verify-math-hybrid/    stage 3: Python search, then Lean
.claude/skills/orchestrate-math-v3/   the cascade + adjudication
```

Routing gates, in order (first match wins): **0** unroutable · **1** a concrete
witness decides it → `python` · **2** general + probeable instance space →
`hybrid` · **3** general + formalises unambiguously → `lean` · **4** everything
else → `informal`. Gate order encodes the policy's "default to Python, escalate
to Lean only when the output must be a proof".

Adjudication: a verifier refutation becomes a finding at 0.85–0.95; a decisive
clear (Lean `PASSED`, exact symbolic identity, exhaustive search) **drops** the
candidate; anything unsettled survives only if `suspected_defect` is specific
**and** detector confidence ≥ 0.5, capped at 0.60. A verifier *failure* is never
itself evidence of an error.

### Artefacts per paper

```
math-detector-<tag>.json                        detector candidates
math-routing-<tag>.json                         routing table
paper_verified_math-<tag>.json                  verification report
judge-mathoracle-<tag>-<judge>.json             cached localisation verdicts
judge-math[-...]-<judge>.json                   cached alignment verdicts
```

### Naming

`batch-verify-math.sh` folds skill + `--llm-skill` + `--tag` into one segment
via `evaluate_verifier.compose_task_tag()` — the same function the scorers use
to *find* the report, so writer and reader cannot drift. `--skill
orchestrate-math-v3 --tag v3` ⇒ `paper_verified_math-orchestrate-math-v3-v3.json`
⇒ score with `--tag orchestrate-math-v3-v3`.

---

## Measurement vocabulary

`evaluate_math_oracle.py` asks a localisation judge — shown the annotated error
and the statements, **not** our verdicts — which statements contain the defect:

- **`covered`** — a statement contains it and carries enough to be caught on its own
- **`partial`** — the judge found the right statements but the defect is only
  partly visible (a dependency between two results with one side missing; a
  figure or table that was not extracted)
- **`absent`** — nothing contains it

`partial` is excluded from strict **denominators** but never from **numerators**:
4 of v1's 8 partial papers were still hits. `cover` is strict, `cover+` includes
partial, ladder rungs E/F use strict, rung G counts a partial as an extraction
miss unless already credited.

Strict is the right ceiling for a pipeline that verifies snippets in
**isolation** — a snippet not carrying enough of the defect to be caught alone
is not a usable input, even though it was extracted.

`partial` has two causes that behave oppositely, and the split is the whole
v1→v3 story:

1. **Truncation** — right statement, cut short. Fixable; all three papers that
   flipped `partial`→`covered` under v3 were this.
2. **Not a single-statement defect** — the annotation is about how two results
   *relate*. No per-statement pipeline reaches these. `0001` stays `partial`
   under both.

The **protocol ladder** (rungs A–G) exists because a recall number means nothing
without its protocol. It was built to place an external claim of ~89% on
"SPOT's Math/CS Equation and Proof subset": that figure sits exactly at v1's
rung-G **ceiling** (33/37 = 0.892; 26/29 = 0.897 on Math+CS), which is an upper
bound assuming every extraction miss would have been caught, not a measurement.
v1's defensible measured band was 0.73–0.82 (rungs E/F).

---

## Open issues

1. **v3 is unmeasured end to end.** 3/37. Step 2 then step 5.
2. **The verifier-stage cost is unknown for v3.** v1 lost 0.054 between ceiling
   and achieved. v3 has four verifiers and a looser detector feeding them, so
   its loss could be larger. This is the real risk to the v3 story and only
   step 5 answers it.
3. **The brevity-on-retry fix is untested** — written after the credit ran out.
   `0003` is the test case.
4. **Scope decision pending.** `0022`/`0023` (From Tripods to Bipods) regressed
   `partial`→`absent` under v3 because SPOT's annotation there is a
   **novelty/priority** complaint against prior work, and the detector prompt
   rules citation/attribution out of scope. Correct scope discipline, but it
   costs recall against SPOT's labels. Decide whether the detector should accept
   priority claims, or whether these two should be excluded as not being
   mathematical errors.
5. **v3 promotes findings at 0.5, v1 at 0.7.** Always report the matched
   operating point (step 5b) alongside. `scripts/threshold_sweep.py` can produce
   the curve from `results[]`.
6. **`MATH_VERIFICATION/Paper` is shared mutable state.** It is why the batch
   cannot be parallelised as written. Giving each worker its own project would
   cost ~7.4 GB each.
7. **The 18-paper detector coverage is a non-random slice** (first 18 in sort
   order). Do not quote 0.778 as a result until step 4 covers all 37.
