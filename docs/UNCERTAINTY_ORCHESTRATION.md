# Uncertainty-Driven Orchestration

This document describes the **uncertainty-driven** orchestration mode — an
alternative to the default exhaustive, type-routed pipeline — together with the
triage stage, the statistical and citation verifiers it can route to, and the
chunking layer that makes the text-based verifiers robust on long inputs.

> TL;DR — Instead of asking *“which verifier fits this section’s type?”* and
> running a verifier on **every** snippet, we first ask *“where is uncertainty
> concentrated?”* with one cheap triage pass, then run specialized verifiers
> **only** where the estimated error density is high. Routing emerges from
> expected error density, not document structure.

---

## 1. Motivation: the sparse-error problem

Errors in real papers are **sparse** — a typical annotated paper in the Paperena
benchmark has zero or one correction/retraction-worthy error. The exhaustive
orchestrator runs a verifier on all ~90–130 snippets per paper. That is wasteful
on cost and, worse, it gives every routine paragraph a chance to produce a
**false positive**. (See `docs/ARCHITECTURE.md` for the exhaustive pipeline and
`src/verifiers/math_verifier.py` for why over-flagging was a real problem.)

The reframing:

```
   exhaustive:  for every snippet → pick a verifier by TYPE → verify
 uncertainty:  triage every snippet → an UNCERTAINTY MAP
               → verify (with a specialist) only the HIGH-uncertainty nodes
```

This concentrates effort where errors are actually likely, and it shrinks the
surface area for false positives because most snippets never reach a specialist.

---

## 2. The pipeline

```
Paper
  │  segment (unchanged)
  ▼
Snippets ───────────────────────────────────────────────┐
  │  triage: one cheap LLM call per snippet               │
  ▼                                                       │
Uncertainty map        Introduction      0.03             │
                       Related Work      0.07             │
                       Methods           0.45             │
                       Equation Chain    0.78             │
                       Results           0.18             │
  │  select nodes with uncertainty ≥ threshold            │
  │  (optional budget cap keeps the top-K)                │
  ▼                                                       │
Specialized verifiers, routed by the triage’s suggestion: │
   math        → Equation Chain                           │
   statistical → Methods / Results                        │
   citation    → Related Work                             │
   text/proof  → prose, theorems                          │
   vision      → figures / tables                         │
  │                                                       │
  ▼                                                       │
Aggregate findings  ◄─────────────────────────────────────┘
(low-uncertainty snippets are accepted as low-risk, no specialist call)
```

Implemented in `src/orchestrator/uncertainty_orchestrator.py` as
`UncertaintyOrchestrator`, a subclass of `VerificationOrchestrator` that reuses
the parent’s verifier cache, confidence thresholding, and aggregation.

This is the single-paper realization of a broader **Adaptive Mixture of
Verifiers** idea: the triage stage is a router that predicts verifier *utility*
per node, and specialists operate on the high-utility nodes it surfaces. The
natural generalization (not implemented here) is a claim-dependency graph with
overlapping verifier neighborhoods and a meta-verifier aggregating findings.

---

## 3. Components

### 3.1 Triage verifier (`src/verifiers/triage_verifier.py`)

`TriageVerifier` makes **one cheap LLM call per snippet** and returns a
`TriageResult` (see `src/models.py`):

| field | meaning |
|---|---|
| `uncertainty` | `[0,1]` — estimated probability the snippet holds a correction-worthy error |
| `suggested_route` | which specialist should look closer: `math`, `proof`, `statistics`, `citation`, `vision`, `text`, `none` |
| `reason` | one-line justification (shown in the map) |
| `selected` / `routed_to` | filled in later: was it escalated, and to which registered verifier |

Key properties:

- It **does not decide errors** — that stays with the specialists, which are
  designed for it (SymPy for equations, the strict reviewer for prose, etc.).
- It **fails open**: if the triage call errors or returns unparseable output,
  the snippet is assigned a moderate uncertainty (≥ threshold) and a type-based
  route, so a triage hiccup never silently drops a snippet from review.

### 3.2 Route resolution (`src/orchestrator/router.py`)

`resolve_route_to_verifier(route, snippet, config)` maps a semantic route label
to a concrete, registered verifier using `config.triage_route_map`:

```
math/equation → math_equation
proof         → text
statistics/statistical/numeric → statistical
citation/reference            → citation
vision/figure/table           → vision
text/logic    → text
none          → (skip — accept as low-risk)
unknown route → fall back to type-based select_verifier_name()
```

The orchestrator’s `_resolve_specialist` adds a safety net: if the routed
verifier can’t actually handle the snippet (e.g. `vision` routed at a snippet
with no image), it falls back to the structural router, then to “no specialist”.

### 3.3 The uncertainty map

Every triage result is persisted on `PaperPrediction.uncertainty_map` and logged
as a region-level table (sections, the equation chain, theorems/proofs,
figures/tables), highest-max first, followed by the specific escalated nodes.
`main.py verify-one --mode uncertainty` prints the same table.

---

## 4. Configuration

All fields live on `PipelineConfig` (`src/config.py`):

| field | default | meaning |
|---|---|---|
| `orchestration_mode` | `"exhaustive"` | `"exhaustive"` or `"uncertainty"` |
| `uncertainty_threshold` | `0.30` | escalate snippets with `uncertainty ≥ threshold` |
| `uncertainty_budget` | `None` | optional hard cap on specialist calls per paper (keeps the top-K most uncertain) |
| `triage_route_map` | see above | semantic route → registered verifier name |
| `verify_chunk_chars` | `2000` | text/citation verifiers chunk content longer than this |
| `verify_chunk_overlap` | `200` | trailing-context overlap between chunks |

`triage` has a verifier-config entry with `confidence_threshold = 0.0` (it never
produces a final finding, so it is never thresholded out).

---

## 5. CLI usage

```bash
# Single paper, uncertainty mode, print the map:
python main.py verify-one 2405.01133v3 --mode uncertainty

# Tune the escalation threshold (higher = fewer specialist calls):
python main.py verify-one 2405.01133v3 --mode uncertainty --uncertainty-threshold 0.45

# Full dataset in uncertainty mode:
python main.py verify --mode uncertainty --uncertainty-threshold 0.30
```

`build_orchestrator(config)` in `main.py` selects the implementation from
`config.orchestration_mode`, so both `verify` and `verify-one` support `--mode`.

### Via Claude Code Skills

```bash
# Single paper with uncertainty mode
claude -p "/verify-paper Verify paper 2405.01133v3 from data/train-00000-of-00001.parquet --mode uncertainty"

# Batch verify with uncertainty mode
./scripts/batch-verify.sh --papers 10 --mode uncertainty --threshold 0.30
python scripts/batch_verify.py --papers 10 --mode uncertainty --threshold 0.30
```

---

## 6. Specialized verifiers added for this mode

### 6.1 Statistical verifier (`src/verifiers/statistical_verifier.py`)

Deterministic numeric verification for the `Statistical reporting` /
`Data inconsistency` error classes. Two stages, with the **decision kept
deterministic**:

1. An LLM **extracts** candidate numeric relationships as *closed arithmetic
   expressions* — numbers only, no symbols (e.g. “33%, 33%, 34% sum to 100” →
   `{"expr": "33 + 33 + 34", "expected": 100}`). It does **not** judge.
2. Python **recomputes** each expression with the injection-safe evaluator in
   `src/utils/safe_arithmetic.py` and flags `INVALID` only when a check fails
   beyond tolerance. No checkable claim → `UNVERIFIABLE`.

Optional unit-conversion checks use `pint` when installed (`pip install
'.[units]'`); absent, unit checks are skipped — never guessed. This mirrors the
math verifier’s philosophy: flag only a *provable* numeric contradiction.

### 6.2 Citation verifier (`src/verifiers/citation_verifier.py`)

Flags **novelty over-claims**, **attribution mismatches**, and self-contradiction
about prior work — but only what is decidable **from the excerpt itself** (it
never relies on outside knowledge it cannot verify here). The canonical target is
benchmark annotation #13: a technique “claimed as original … previously
established by other authors”.

Both are registered in `create_default_registry()` and reachable via the
`triage_route_map`. Adding a dedicated verifier for a new route is now a
one-line registration plus a route-map entry.

---

## 7. Chunking (text-based verifiers)

`src/utils/chunking.py :: chunk_text` and `BaseVerifier._analyze_in_chunks`.

Long, dense snippets (multi-step proofs, whole sections) are where the LLM most
often returns an **empty or unparseable** completion, and where a single
load-bearing wrong step is easiest to miss in a 2-page prompt. The text and
citation verifiers therefore:

1. Split content longer than `verify_chunk_chars` into overlapping chunks
   (`chunk_text` packs whole paragraphs → sentences → hard split as a last
   resort; `verify_chunk_overlap` carries trailing context across boundaries).
2. Verify each chunk independently and **aggregate**:
   - any chunk that detects an error → the highest-confidence such finding wins
     (its reasoning is prefixed with `[chunk i/n]` for provenance);
   - otherwise the highest-confidence “no error” finding;
   - a chunk whose LLM call fails (e.g. empty completion) is counted as failed
     and **skipped** — one bad chunk no longer sinks the whole snippet;
   - only if *every* chunk fails is the snippet `UNVERIFIABLE`.

Two complementary robustness fixes in `src/utils/llm.py`:

- **Honor `config.max_tokens` (root cause of empty responses).** `llm_call`
  previously ignored `config.max_tokens` and always used its own `4096` default.
  With a **reasoning-style model** (e.g. `deepseek-v4-pro`), chain-of-thought
  consumes that budget and `message.content` comes back **empty**. Confirmed
  directly: a trivial prompt returns empty at `max_tokens=20` but `"OK"` at
  `2000`/`8192`. `llm_call` now resolves `max_tokens` from the config (default
  raised to **8192**), which eliminated the empty responses (16+ → 0 on a paper
  run). This was the real driver behind the “empty on dense proofs” symptom that
  chunking was also mitigating.
- **Empty completions are retryable.** An empty/whitespace response now raises
  inside the retry loop instead of silently passing through to the parser.

> Operational note: heavy *whole-paper* single calls (≈15k input tokens) on a
> reasoning backend can still drop the connection mid-generation under provider
> load — independent of `max_tokens`. The `SingleCallBaseline` degrades its input
> budget on failure for exactly this reason (see baseline comparison below).

---

## 8. Threshold sweep tool

`scripts/threshold_sweep.py` characterizes the recall/cost tradeoff **cheaply**:
it triages each paper **once** and runs each selected specialist **once** (on the
union selected at the lowest threshold), caching every result by `snippet_id`,
then re-applies each threshold in the grid offline. Cost = one triage pass + one
specialist pass per paper, not one full run per threshold.

```bash
python scripts/threshold_sweep.py 2405.01133v3 2402.10307v2
# → table on stdout + full detail in outputs_new/threshold_sweep.json
```

---

## 9. Empirical results (two sample papers)

Specialist calls fall monotonically as the threshold rises — effort concentrates
as intended:

```
PAPER 2405.01133v3 (129 snippets)        PAPER 2402.10307v2 (88 snippets)
  thr  calls  routes                       thr  calls  routes
  0.10   32   math,text                     0.10   11   citation,math,statistical,text,vision
  0.30    7   math,text                     0.30    3   math,statistical,vision
  0.50    1   text                          0.50    1   statistical
```

The triage **localizes correctly**: on `2405.01133v3` the single highest node is
consistently `thm_3` (≈0.85), which *is* Lemma 3 — the annotated error site
(ground truth: “Equation / proof @ Lemma 3,4”). Compared to exhaustive mode, a
single paper drops from 129 specialist verifications to ~7–10 at the default
threshold.

---

## 10. Tradeoffs and caveats

- **Recall depends on the threshold.** A real error in a snippet the triage
  under-scores is never escalated, so it is missed. `--uncertainty-threshold`
  trades specialist budget for recall; `uncertainty_budget` caps cost directly.
- **Triage cost is per-snippet.** Uncertainty mode replaces *expensive*
  specialist calls (SymPy sandbox, vision, long prose) with *cheap* triage calls
  on the bulk of snippets; it does not reduce the raw number of LLM calls. The
  win is in expensive-call volume and in false-positive surface area.
- **The binding constraint observed on the two papers was the LLM, not the
  router**: dense proof snippets returned empty completions, which is exactly
  what the chunking layer (§7) targets.
- **Triage is non-deterministic** across runs (provider variance at
  `temperature=0`); region-level conclusions are stable, individual node scores
  vary somewhat.

---

## 11. Extending

- **Add a route → verifier:** register the verifier in
  `create_default_registry()` and point the relevant `triage_route_map` keys at
  its name. The triage prompt already emits `statistics`/`citation`/`proof`
  routes; map them to dedicated verifiers as you build them.
- **Add a route label:** add it to the triage system prompt’s route list
  (`TRIAGE_SYSTEM_PROMPT`) and to `triage_route_map`. Unknown routes fall back to
  type-based routing, so this is safe to do incrementally.
- **Change region bucketing** (for the printed map): edit
  `UncertaintyOrchestrator._region_of`.

See `docs/EXTENDING.md` for the general “add a verifier” walkthrough.

---

## 12. Baseline comparison

To measure what the decomposition + routing machinery actually buys, a
**single-call baseline** (`src/baseline/single_call_baseline.py`) hands the
*entire paper* to one LLM call and asks for the errors directly, emitting the
same `PaperPrediction` shape so the existing alignment + metrics score it
unchanged. `scripts/baseline_comparison.py` runs both systems on the same papers
and prints a side-by-side table (TP/FP/FN, precision/recall/F1) plus per-paper
hit/miss, using the same LLM judge (or `--no-judge` fuzzy matching).

```bash
python scripts/baseline_comparison.py 2405.01133v3 2402.10307v2 --mode uncertainty
python scripts/baseline_comparison.py --n 5 --mode exhaustive --no-judge   # first 5 papers
python scripts/baseline_comparison.py --provider mock                      # offline dry run
```

The single-call baseline is **operationally fragile** with a reasoning backend:
reasoning over a whole paper (~15k tokens) is slow and can drop the connection,
so the baseline degrades its input budget on failure rather than returning
nothing. This fragility is itself part of the comparison story — the decomposed
pipeline issues many *small* calls that complete reliably. (Note: meaningful
head-to-head numbers require a stable provider window; re-run when the API is
healthy.)

## 13. File map

| path | role |
|---|---|
| `src/orchestrator/uncertainty_orchestrator.py` | the orchestrator |
| `src/verifiers/triage_verifier.py` | general triage / uncertainty scoring |
| `src/verifiers/statistical_verifier.py` | deterministic numeric checks |
| `src/verifiers/citation_verifier.py` | attribution / novelty checks |
| `src/utils/safe_arithmetic.py` | injection-safe arithmetic evaluator |
| `src/utils/chunking.py` | boundary-aware chunking |
| `src/orchestrator/router.py` | `resolve_route_to_verifier`, registry |
| `src/baseline/single_call_baseline.py` | one-call whole-paper baseline |
| `scripts/threshold_sweep.py` | recall/cost sweep |
| `scripts/baseline_comparison.py` | baseline vs pipeline comparison |
| `scripts/batch-verify.sh` | shell script for Claude Code batch verification |
| `scripts/batch_verify.py` | Python script for Claude Code batch verification + evaluation |
| `.claude/skills/verify-triage/SKILL.md` | triage skill (Claude Code) |
| `.claude/skills/verify-paper/SKILL.md` | orchestrator skill (Claude Code) |
| `.claude/skills/verify-math/SKILL.md` | math verifier skill (Claude Code) |
| `.claude/skills/verify-statistical/SKILL.md` | statistical verifier skill (Claude Code) |
| `.claude/skills/verify-citation/SKILL.md` | citation verifier skill (Claude Code) |
| `tests/test_uncertainty.py`, `tests/test_statistical.py`, `tests/test_chunking.py`, `tests/test_baseline.py` | tests |
