# Paper-ready figures and tables

Three artifacts cover the whole math-verification comparison on **PaperAudit** and
**SPOT** — one figure and two tables. Everything is generated from the CSVs already
in `task_specific_output/` by a single script:

```bash
python task_specific_output/plots_and_tables/make_artifacts.py
```

| Artifact | Files | Answers |
|---|---|---|
| **Figure 1** | `fig1_results.pdf` / `.png` / `fig1_results_data.csv` | Where does each system sit on the precision/recall trade-off, and what does it spend its findings on? |
| **Table 1** | `table1_main_results.tex` / `.md` / `.csv` | The headline numbers, both benchmarks, three matching regimes. |
| **Table 2** | `table2_judge_sensitivity.tex` / `.md` / `.csv` | How much of the score is an artifact of the LLM judge? |

`.tex` files are drop-in `\input{}`s (needs `booktabs`); `.md` is the same table for
notes and issues; `.csv` is the machine-readable version, with micro-averages and
Holm-corrected p-values that the typeset table omits.

Both benchmarks are scored on their full paper sets — PaperAudit 46 papers / 94
seeded errors, SPOT 37 / 37 — with one backbone (Claude-Opus-5) and one judge
(Claude-Sonnet-5) throughout.

---

## Figure 1 — `fig1_results.pdf`

Four panels, two columns (PaperAudit | SPOT), **axes shared across columns** so the
two benchmarks can be read against each other.

* **(a), (b) — accuracy trade-off.** Macro precision vs. macro recall, with iso-F1
  contours. Our pipeline is the blue mark; single-call LLM baselines are the dark
  marks, each directly labelled, so identity never rests on color. Both panels show
  the same five systems on the same axes.
* **(c), (d) — where the findings go.** Findings reported *per paper*, split by the
  judge's verdict on each one. The dashed line is the number of seeded errors per
  paper, i.e. the reporting rate a perfectly calibrated system would have. This is
  the panel that explains the low absolute precision everywhere: most findings are
  *out of scope* — true observations about the paper that are not among the seeded
  errors — not *spurious*.

Suggested caption:

> **Figure 1: Mathematical-error detection on PaperAudit (left) and SPOT (right).**
> (a, b) Macro precision against macro recall with iso-F1 contours; our agentic
> pipeline in blue, single-call LLM baselines in grey. (c, d) Findings reported per
> paper, split by judge verdict; the dashed line marks the seeded-error rate of the
> benchmark. Our pipeline buys its recall by reporting far more than the baselines —
> 5.8 findings/paper on PaperAudit against a seeded rate of 2.0 — but the surplus is
> overwhelmingly *out of scope* (true observations that are not seeded errors) rather
> than wrong: 3.4% of its PaperAudit findings and 1.4% of its SPOT findings are
> spurious, the lowest rate of any system on PaperAudit.

## Table 1 — main results

Macro-averages over papers, every system aligned to ground truth by the *same* LLM
judge (Claude-Sonnet-5). Three F1 columns, increasingly favourable to the baselines:

* **F1** — every reported finding counts against precision.
* **F1⁺** — findings the judge ruled *out of scope* are removed from the precision
  denominator; this is the "don't punish a system for noticing a real but unseeded
  defect" view.
* **F1ˢ** — strict regime: a baseline finding counts only if the baseline itself
  labelled it *Equation / proof*.

All five systems are prompted with the same default taxonomy. The baselines that were
prompted with the PaperAudit taxonomy (`baseline:pa-*`, still present in the source
CSVs) are **excluded** — they are not like-for-like, and `make_artifacts.py` filters
them out by construction.

Significance (`*`) is a paired Wilcoxon test on per-paper F1, ours vs. each baseline,
Holm-corrected within benchmark.

## Table 2 — judge sensitivity

The *same* pipeline output re-aligned by four different LLM judges. Absolute F1 moves
by .09 (PaperAudit) and .12 (SPOT) depending on who judges — larger than several of
the system-to-system gaps in Table 1. That is why Table 1 fixes one judge
(Claude-Sonnet-5, the second row of Table 2) for all systems: the comparison is sound,
the absolute level is not. This table covers the same papers and the same pipeline
output as Table 1's "Ours" row, but scores them with the task-specific verifier scorer
rather than the system-comparison scorer, so the absolute values are not expected to
coincide; the four rows are mutually consistent, which is what the spread measures.

---

## Reading the result honestly

* **Ours leads both benchmarks on F1**, by .343 vs .335 (PaperAudit, Claude-Sonnet-5
  second) and .344 vs .294 (SPOT, DeepSeek-V4-pro second), and leads or ties every column
  except PaperAudit precision.
* **The win is a recall win, bought with volume.** On PaperAudit ours recalls .576
  against .330 for the best baseline, but reports 5.8 findings/paper against a seeded
  rate of 2.0 — the baselines all report *below* that rate (0.8–2.3). Precision is
  where it pays: .268, behind Claude-Sonnet-5 (.399), GPT-5.6-terra (.316) and even
  Claude-Opus-5 (.283). A reviewer will ask whether 5.8 findings/paper is usable
  output; the F1⁺ column (.631, far ahead of .388 for the next system) is the answer
  that favours us, because 192 of our 266 PaperAudit findings are judged *out of
  scope* — true but unseeded — not wrong.
* **Low spurious rate is the robust claim.** 3.4% of PaperAudit findings and 1.4% of
  SPOT findings are spurious, the lowest on PaperAudit and essentially tied with
  Claude-Opus-5 on SPOT (0.9%). The earlier "zero spurious findings" line no longer
  holds: that was the 34-paper Sonnet-4.5 run.
* **Significance is thin.** On PaperAudit only DeepSeek-V4-pro (p = .008) and
  Claude-Opus-5 (p = .020) lose to us after Holm correction; Claude-Sonnet-5 (p = .69)
  and GPT-5.6-terra (p = .19) are ties. On SPOT only Claude-Opus-5 (p = .017) survives,
  with Claude-Sonnet-5 borderline (p = .053). No baseline beats us significantly
  anywhere. With 37–46 papers, treat the rest as ties.
* **The backbone confound is gone.** Ours now runs the Claude-Opus-5 backbone on both
  benchmarks, so no system in Table 1 is advantaged by a stronger underlying model.

## Caveats baked into the artifacts

1. **Coverage gap.** Claude-Opus-5 on SPOT covers 36 of 37 papers and is averaged over
   the papers it covers; every other cell uses the full set. Stated in the caption.
2. **Table 2 is a different scorer.** Same papers, same pipeline output as Table 1's
   "Ours" row, but through the task-specific verifier scorer — it measures judge
   spread, not system quality, and its levels will not match Table 1.
3. **Benchmarks differ in density.** PaperAudit has 2.0 seeded errors per paper against
   SPOT's 1.0, which is why the per-paper reporting rates in (c) and (d) are on
   different scales.
4. **Out-of-scope is a judge call.** The F1⁺ column, and our case on PaperAudit, rest
   on the judge correctly separating "true but unseeded" from "wrong". Table 2 bounds
   how much that call moves with the judge, but does not eliminate it.

## Design notes

Colors follow the `dataviz` palette: categorical slot 1 (`#2a78d6`) for our system,
secondary ink (`#52514e`) for baselines (a highlight/context pair, not a two-category
palette — identity is carried by direct labels), and the status palette
good/warning/critical plus a neutral grey for the four judge verdicts, with the grey
reserved for the class that is *excluded* from the in-scope precision denominator.
Adjacent-pair CVD and normal-vision separation were checked with the palette
validator. Light surface only — these are print figures. `fig1_results_data.csv` is
the figure's table view: every number drawn in Figure 1.

## Provenance

The PaperAudit column of Table 1 comes from two runs made for these artifacts. First
the missing default-taxonomy `Claude-Sonnet-5` baseline (the two pre-existing
`verification-report-baseline-claude-sonnet-5.json` files had in fact been produced
with the **paperaudit** taxonomy despite the tag, hence `--overwrite`):

```bash
python scripts/run_baseline.py --workspace-dir data/paperaudit-workspace \
    --model claude-sonnet-5 --tag baseline-claude-sonnet-5 --overwrite
```

Then the whole PaperAudit comparison re-scored against the Claude-Opus-5 backbone
report (`paper_verified_math-opus5.json`), which also grows the paper set from 34 to
the full 46:

```bash
python scripts/evaluate_math_systems.py --workspace-dir data/paperaudit-workspace \
    --judge-model claude-sonnet-5 --tag opus5        # and again with --strict
```

`--judge-model claude-sonnet-5` is **not** the script default (`claude-sonnet-4-5`);
it is what the SPOT CSVs were produced with, and both benchmarks must share a judge
for Table 1 to mean anything. `--tag opus5` selects the Opus-5 backbone report; the
untagged `paper_verified_math.json` is the older Sonnet-4.5 run and covers only 34
papers.
