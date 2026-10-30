# Task-specific evaluation results

Results of **standalone, single-error-family evaluations** — a purpose-built
system runs over the whole dataset (PaperAudit or SPOT) and only the metrics for
its own error family are pulled out at the end.

One subfolder per task, created on first run:

| Task | Ad-hoc system | Report it reads | Ground-truth kept |
|---|---|---|---|
| `math` | `/orchestrate-math` skill | `<paper_name>_verified_math.json` | PaperAudit `method_logic_consistency` / SPOT `Equation / proof` |

Produced by:

```bash
# 1. run the ad-hoc system over the dataset
./scripts/batch-verify-math.sh --dataset paperaudit

# 2. score only the mathematical errors
python scripts/evaluate_verifier.py --llm claude-sonnet-4-5-20250929 \
    --evaluate --task math --workspace-dir data/paperaudit-workspace
```

which writes `task_specific_output/math/<llm>_metrics.csv` and
`task_specific_output/math/<llm>_summary.csv`.

## Comparing models with `--tag`

Tag each ad-hoc run to keep several models side by side in one workspace. The
tag is appended to the report filename and stamped into the report as a `tag`
field; on the evaluation side it selects that report, namespaces the judge cache
(`judge-math-<tag>-<judge>.json`) and names the CSVs `<llm>-<tag>_*.csv`.

```bash
./scripts/batch-verify-math.sh --model claude-opus-5              --tag opus5
./scripts/batch-verify-math.sh --model claude-sonnet-4-5-20250929 --tag sonnet45

python scripts/evaluate_verifier.py --llm <llm> --evaluate --task math --tag opus5
python scripts/evaluate_verifier.py --llm <llm> --evaluate --task math --tag sonnet45
```

```
data/paperaudit-workspace/<paper>/paper_verified_math-opus5.json
data/paperaudit-workspace/<paper>/paper_verified_math-sonnet45.json
task_specific_output/math/<llm>-opus5_summary.csv
task_specific_output/math/<llm>-sonnet45_summary.csv
```

A tagged evaluation reads only that tag's report — it never falls back to
another tag or to a full verify-paper report, so a missing run shows up as a
skipped paper rather than as another model's score.
