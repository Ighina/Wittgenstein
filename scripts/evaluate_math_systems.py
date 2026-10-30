#!/usr/bin/env python3
"""
evaluate_math_systems.py — Math-only comparison: orchestrate-math vs single-call baselines.

Scores the orchestrate-math skill (paper_verified_math.json) and every
single-call baseline (verification-report-baseline-<tag>.json) against the
MATHEMATICAL slice of ground truth only, so the systems are compared on the
error family the math skill actually targets:

  * Ground truth is filtered to corruption_type == method_logic_consistency
    (the PaperAudit family of equation/theorem/proof corruptions).
  * Baseline findings are filtered to mathematical categories
    ("Equation / proof", "method_logic_consistency", or any category matching
    the math regex from evaluate_verifier.TASK_REGISTRY["math"]).
  * Each (paper, system) pair is aligned with the same holistic LLM judge
    (many-to-many matches), cached per system so re-runs are free.

Outputs (task_specific_output/math-comparison/):
  math_systems_comparison.csv — one row per system (macro + micro P/R/F1)
  math_systems_per_paper.csv  — one row per system x paper

Usage:
  python scripts/evaluate_math_systems.py --workspace-dir data/paperaudit-workspace
  python scripts/evaluate_math_systems.py --workspace-dir data/paperaudit-workspace --limit 5
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_verifier import (  # noqa: E402
    DEFAULT_JUDGE_MODEL,
    TASK_REGISTRY,
    _warm_judge_cache,
    align_paper,
    compute_metrics,
    filter_ground_truth,
    judge_cache_path,
    load_anthropic_client,
    load_json,
    summarize,
    task_predictions,
)

PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "task_specific_output" / "math-comparison"

MATH_TASK = "math"
MATH_SPEC = TASK_REGISTRY[MATH_TASK]
MIN_PAPERS_DEFAULT = 10  # drop aborted 2-paper baseline runs by default

# ---------------------------------------------------------------------------
# Baseline math filtering
# ---------------------------------------------------------------------------


def is_math_category(category: Any) -> bool:
    """True when a baseline error_category belongs to the math family.

    Baselines written with the PaperAudit taxonomy label math corruptions
    "method_logic_consistency" directly; the default taxonomy uses
    "Equation / proof". Anything matching the math regex from the task
    registry also counts.
    """
    cat = str(category or "")
    if cat == "method_logic_consistency":
        return True
    return bool(MATH_SPEC["prediction_category_re"].search(cat))


def is_equation_proof_category(category: Any) -> bool:
    """Strict filter: only findings the baseline itself labeled Equation / proof.

    Used with --strict, which drops the PaperAudit-taxonomy systems entirely
    (they label math findings "method_logic_consistency" instead).
    """
    norm = re.sub(r"\s+", " ", str(category or "").strip().lower())
    return norm == "equation / proof"


def above_confidence(
    predictions: List[Dict[str, Any]], floor: Optional[float]
) -> List[Dict[str, Any]]:
    """Drop predictions below a confidence floor.

    Needed because the skills promote a result to findings[] at different
    thresholds: orchestrate-math and v2 at 0.7, v3 at 0.5 (v3's precision
    filter is its verifier stage, not a confidence cut-off). Scoring v3's
    findings[] as-is against v1's therefore compares two operating points, not
    two systems. `--min-confidence 0.7` puts them on the same one.

    A prediction with no confidence at all is KEPT: the single-call baselines
    sometimes omit the field, and dropping those would silently shrink a
    baseline's prediction set and flatter its precision.
    """
    if floor is None:
        return predictions
    kept: List[Dict[str, Any]] = []
    for p in predictions:
        c = p.get("confidence")
        if c is None:
            kept.append(p)
            continue
        try:
            if float(c) >= floor:
                kept.append(p)
        except (TypeError, ValueError):
            kept.append(p)
    return kept


def baseline_math_predictions(
    report: Dict[str, Any], strict: bool = False
) -> List[Dict[str, Any]]:
    """Keep only the mathematical findings of a single-call baseline report."""
    keep = is_equation_proof_category if strict else is_math_category
    return [
        p
        for p in report.get("predicted_errors") or []
        if isinstance(p, dict) and keep(p.get("error_category"))
    ]


def discover_baseline_tags(
    ws_base: Path, min_papers: int, strict: bool = False
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    """Find verification-report-baseline-<tag>.json files in the workspace.

    Returns ({tag: {model, papers}}, excluded) where excluded maps tags that
    were dropped (below min_papers, or — in strict mode — with zero
    "Equation / proof" findings) to the reason. The untagged
    verification-report-baseline.json is ignored.
    """
    counts: Dict[str, int] = defaultdict(int)
    models: Dict[str, str] = {}
    strict_preds: Dict[str, int] = defaultdict(int)
    excluded: Dict[str, str] = {}
    for d in sorted(x for x in ws_base.iterdir() if x.is_dir()):
        for f in d.glob("verification-report-baseline-*.json"):
            tag = f.stem[len("verification-report-baseline-") :]
            if not tag:
                continue  # untagged verification-report-baseline.json
            counts[tag] += 1
            if tag not in models:
                try:
                    baseline = load_json(f).get("baseline") or {}
                    models[tag] = baseline.get("model") or tag
                except Exception:
                    models[tag] = tag
            if strict:
                rep = load_json(f)
                strict_preds[tag] += len(baseline_math_predictions(rep, strict=True))

    included: Dict[str, Dict[str, Any]] = {}
    for tag in sorted(counts):
        if counts[tag] < min_papers:
            excluded[tag] = f"below --min-papers {min_papers} ({counts[tag]} papers)"
        elif strict and strict_preds[tag] == 0:
            excluded[tag] = (
                "zero 'Equation / proof' findings (uses the PaperAudit "
                "taxonomy: math findings are labeled method_logic_consistency)"
            )
        else:
            included[tag] = {"model": models[tag], "papers": counts[tag]}
    return included, excluded


def conf_suffix(tag: str, floor: Optional[float]) -> str:
    """Fold a confidence floor into a judge-cache tag.

    Without this, scoring the same system at 0.5 and then at 0.7 would reuse
    the first run's verdicts for a prediction list that has since shrunk, and
    the cache's own n_pred check would reject it — costing a full re-judge and
    reporting it as a cache miss rather than as two distinct conditions.
    """
    if floor is None:
        return tag
    return f"{tag}-c{floor:g}".replace(".", "p")


def paper_title(report: Dict[str, Any], paper_id: str) -> str:
    """A meaningful paper title for the judge prompt."""
    title = report.get("title") or ""
    paper_name = report.get("paper_name") or ""
    if title and title != "paper":
        return title
    if paper_name and paper_name != "paper":
        return paper_name
    return paper_id


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def math_report_name(tag: Optional[str] = None) -> str:
    """Filename of the orchestrate-math report, with or without a run tag.

    batch-verify-math.sh --tag <t> writes paper_verified_math-<t>.json, so a
    tagged run (e.g. a different backbone model) must be addressed by its tag
    or it is invisible here.
    """
    return f"paper_verified_math-{tag}.json" if tag else "paper_verified_math.json"


def build_tasks(
    ws_base: Path,
    paper_ids: List[str],
    baseline_tags: List[str],
    judge_model: str,
    strict: bool = False,
    math_tag: Optional[str] = None,
    min_confidence: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Assemble one judge task per (paper, system) pair.

    In strict mode the baseline cache tag carries the -ep- marker so its
    verdicts never collide with the broad math-filtered ones. `math_tag` selects
    a tagged orchestrate-math report and is folded into that system's name and
    judge-cache key, so two backbones scored in one workspace stay apart.
    """
    tasks: List[Dict[str, Any]] = []
    for paper_id in paper_ids:
        d = ws_base / paper_id
        gt_all = load_json(d / "ground_truth.json")
        gt_math = filter_ground_truth(gt_all, MATH_TASK)
        if not gt_math:
            print(f"  SKIP {paper_id}: no 'math' ground-truth errors")
            continue

        # System 1: the orchestrate-math skill report.
        rep = load_json(d / math_report_name(math_tag))
        tasks.append(
            {
                "paper_id": paper_id,
                "system": f"orchestrate-math:{math_tag}" if math_tag else "orchestrate-math",
                "gt": gt_math,
                "predicted": above_confidence(
                    task_predictions(rep, MATH_TASK), min_confidence
                ),
                "title": paper_title(rep, paper_id),
                "mode": MATH_TASK,
                "cache_path": judge_cache_path(
                    d,
                    conf_suffix(
                        f"{MATH_TASK}-{math_tag}" if math_tag else MATH_TASK,
                        min_confidence,
                    ),
                    judge_model,
                ),
            }
        )

        # Systems 2..N: each baseline tag, restricted to math findings.
        for tag in baseline_tags:
            rep_path = d / f"verification-report-baseline-{tag}.json"
            if not rep_path.exists():
                continue
            rep = load_json(rep_path)
            cache_tag = f"math-ep-baseline-{tag}" if strict else f"math-baseline-{tag}"
            cache_tag = conf_suffix(cache_tag, min_confidence)
            tasks.append(
                {
                    "paper_id": paper_id,
                    "system": f"baseline:{tag}",
                    "gt": gt_math,
                    "predicted": above_confidence(
                        baseline_math_predictions(rep, strict=strict),
                        min_confidence,
                    ),
                    "title": paper_title(rep, paper_id),
                    "mode": "baseline",
                    "cache_path": judge_cache_path(d, cache_tag, judge_model),
                }
            )
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Math-only comparison: orchestrate-math vs single-call baselines.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1],
    )
    parser.add_argument(
        "--workspace-dir",
        type=str,
        default="data/paperaudit-workspace",
        help="Paper workspace directory (default: data/paperaudit-workspace)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Score the tagged orchestrate-math report "
        "(paper_verified_math-<tag>.json, written by batch-verify-math.sh "
        "--tag <tag>) instead of the untagged one. The tag is folded into the "
        "system name and the judge-cache key, so runs with different backbones "
        "can be scored in the same workspace without colliding.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of papers (smoke testing)",
    )
    parser.add_argument(
        "--min-papers",
        type=int,
        default=MIN_PAPERS_DEFAULT,
        help=f"Minimum paper coverage for a baseline tag to be included "
        f"(default: {MIN_PAPERS_DEFAULT})",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help="Drop predictions below this confidence from EVERY system. Use it "
        "to compare skills that promote findings at different thresholds "
        "(orchestrate-math and v2 at 0.7, v3 at 0.5) on one operating point. "
        "Predictions with no confidence field are kept. The floor is folded "
        "into the judge-cache key, so two floors can be scored side by side.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Only Equation / proof: baselines contribute only findings "
        "labeled 'Equation / proof', and systems without any such findings "
        "(the PaperAudit-taxonomy pa-* baselines) are dropped.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=DEFAULT_JUDGE_MODEL,
        help=f"Model used for holistic LLM judging (default: {DEFAULT_JUDGE_MODEL})",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Anthropic API key for LLM-as-a-judge "
        "(default: ANTHROPIC_API_KEY env var, then project .env)",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Disable LLM judge; use the Jaccard-overlap heuristic instead. "
        "Smoke test only — NOT comparable to judged results.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Parallel judge calls (default: 8)",
    )
    parser.add_argument(
        "--overwrite-judge",
        action="store_true",
        help="Re-run judge calls even when a cached verdict exists",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for output CSVs",
    )
    args = parser.parse_args()

    ws_base = Path(args.workspace_dir)
    output_dir = Path(args.output_dir)

    # ── Paper set: papers the math skill actually ran on ───────────────────
    paper_ids = sorted(
        d.name
        for d in ws_base.iterdir()
        if d.is_dir()
        and (d / "ground_truth.json").exists()
        and (d / math_report_name(args.tag)).exists()
    )
    if not paper_ids:
        print(
            f"ERROR: no papers with {math_report_name(args.tag)} in {ws_base}.\n"
            "Run scripts/batch-verify-math.sh first"
            + (f" with --tag {args.tag}." if args.tag else ".")
        )
        sys.exit(1)
    if args.limit:
        paper_ids = paper_ids[: args.limit]
    print(f"Papers with math reports: {len(paper_ids)} (limit={args.limit})")

    # ── Systems: orchestrate-math + baseline tags above the coverage floor ─
    discovered, excluded = discover_baseline_tags(
        ws_base, args.min_papers, strict=args.strict
    )
    baseline_tags = sorted(discovered)
    math_system = f"orchestrate-math:{args.tag}" if args.tag else "orchestrate-math"
    systems = [math_system] + [f"baseline:{t}" for t in baseline_tags]
    print(f"Systems: {', '.join(systems)}")
    for tag in sorted(discovered):
        print(
            f"  baseline:{tag:<20} model={discovered[tag]['model']:<24} "
            f"papers={discovered[tag]['papers']}"
        )
    for tag in sorted(excluded):
        print(f"  (excluded {tag}: {excluded[tag]})")

    # ── Judge setup ────────────────────────────────────────────────────────
    client = None
    if not args.no_judge:
        client = load_anthropic_client(args.api_key)
        if client is not None:
            print(f"LLM-as-a-judge: ENABLED (holistic, model={args.judge_model})")
        else:
            print("LLM-as-a-judge: UNAVAILABLE (using heuristic smoke test)")
    else:
        print("LLM-as-a-judge: DISABLED — heuristic smoke test only")

    tasks = build_tasks(
        ws_base, paper_ids, baseline_tags, args.judge_model,
        strict=args.strict, math_tag=args.tag,
        min_confidence=args.min_confidence,
    )
    print(f"\nJudge tasks: {len(tasks)} (paper x system pairs)")

    if client is not None:
        _warm_judge_cache(client, args.judge_model, tasks, args.concurrency,
                          args.overwrite_judge)

    # ── Align & score ──────────────────────────────────────────────────────
    per_paper_rows: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for i, t in enumerate(tasks, 1):
        alignment = align_paper(
            t["predicted"],
            t["gt"],
            client,
            args.judge_model,
            t["title"],
            t["mode"],
            t["cache_path"],
        )
        if alignment is None:
            skipped.append(f"{t['paper_id']}/{t['system']}")
            print(f"  [{i}/{len(tasks)}] JUDGE FAILED {t['paper_id']} {t['system']}")
            continue
        metrics = compute_metrics(alignment)
        per_paper_rows.append(
            {
                "system": t["system"],
                "paper_id": t["paper_id"],
                "n_gt": alignment["n_gt"],
                "n_pred": alignment["n_pred"],
                "matched_gt": alignment["matched_gt_count"],
                "matched_pred": alignment["matched_pred_count"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "precision_in_scope": metrics["precision_in_scope"],
                "f1_in_scope": metrics["f1_in_scope"],
                "n_pred_in_scope": metrics["n_pred_in_scope"],
                "pred_true_positive": metrics["pred_true_positive"],
                "pred_duplicate": metrics["pred_duplicate"],
                "pred_out_of_scope": metrics["pred_out_of_scope"],
                "pred_spurious": metrics["pred_spurious"],
                "verdict_conflicts": metrics["verdict_conflicts"],
            }
        )
        print(
            f"  [{i}/{len(tasks)}] {t['paper_id']:<60} {t['system']:<28} "
            f"GT={alignment['n_gt']} Pred={alignment['n_pred']} "
            f"P={metrics['precision']:.2f} R={metrics['recall']:.2f} "
            f"F1={metrics['f1']:.2f}"
        )
    if skipped:
        print(f"\nWARNING: judge failed for: {', '.join(skipped)}")

    # ── Aggregate per system ───────────────────────────────────────────────
    rows_by_system: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in per_paper_rows:
        rows_by_system[r["system"]].append(r)

    comparison: List[Dict[str, Any]] = []
    for system in systems:
        rows = rows_by_system.get(system, [])
        if not rows:
            print(f"  NOTE: {system} has no scored papers — omitted from table")
            continue
        summary = summarize(rows)
        tag = system.split(":", 1)[1] if system.startswith("baseline:") else ""
        model = discovered.get(tag, {}).get("model", "")
        comparison.append(
            {
                "system": system,
                "model": model or system,
                "papers": len(rows),
                "total_gt": summary["total_gt"],
                "total_pred": summary["total_pred"],
                "matched_gt": summary["total_matched_gt"],
                "matched_pred": summary["total_matched_pred"],
                "macro_precision": summary["macro_precision"],
                "macro_recall": summary["macro_recall"],
                "macro_f1": summary["macro_f1"],
                "avg_f1_score": summary["avg_f1_score"],
                "micro_precision": summary["micro_precision"],
                "micro_recall": summary["micro_recall"],
                "micro_f1": summary["micro_f1"],
                "macro_precision_in_scope": summary["macro_precision_in_scope"],
                "micro_precision_in_scope": summary["micro_precision_in_scope"],
                "avg_f1_in_scope": summary["avg_f1_in_scope"],
                "total_pred_in_scope": summary["total_pred_in_scope"],
                "pred_true_positive": summary["pred_true_positive"],
                "pred_duplicate": summary["pred_duplicate"],
                "pred_out_of_scope": summary["pred_out_of_scope"],
                "pred_spurious": summary["pred_spurious"],
                "verdict_conflicts": summary["verdict_conflicts"],
            }
        )

    comparison.sort(key=lambda r: r["avg_f1_score"], reverse=True)

    # ── Comparison table ───────────────────────────────────────────────────
    print(f"\n{'=' * 130}")
    print(
        "MATH-ONLY COMPARISON (holistic judge; "
        "avg_F1 = F1 of macro P/R, comparable to PaperAudit headline; "
        "avg_F1* = same with in-scope precision)"
    )
    print(f"{'=' * 130}")
    header = (
        f"{'System':<28} {'Papers':>6} {'GT':>4} {'Pred':>5} "
        f"{'MacroP':>7} {'MacroR':>7} {'MacroF1':>7} {'avg_F1':>7} {'avg_F1*':>7} "
        f"{'MicroP':>7} {'MicroR':>7} {'MicroF1':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in comparison:
        print(
            f"{r['system']:<28} {r['papers']:>6} {r['total_gt']:>4} "
            f"{r['total_pred']:>5} {r['macro_precision']:>7.3f} "
            f"{r['macro_recall']:>7.3f} {r['macro_f1']:>7.3f} "
            f"{r['avg_f1_score']:>7.3f} {r['avg_f1_in_scope']:>7.3f} "
            f"{r['micro_precision']:>7.3f} "
            f"{r['micro_recall']:>7.3f} {r['micro_f1']:>8.3f}"
        )
    print(f"\nFinding verdicts (judge part 2) per system:")
    print(
        f"{'System':<28} {'TP':>4} {'dup':>4} {'out_of_scope':>12} {'spurious':>9} "
        f"{'conflicts':>9}"
    )
    for r in comparison:
        print(
            f"{r['system']:<28} {r['pred_true_positive']:>4} "
            f"{r['pred_duplicate']:>4} {r['pred_out_of_scope']:>12} "
            f"{r['pred_spurious']:>9} {r['verdict_conflicts']:>9}"
        )
    notes = [
        "ground truth restricted to method_logic_consistency (math) corruptions;"
    ]
    if args.min_confidence is not None:
        notes.append(
            f"every system's predictions filtered to confidence >= "
            f"{args.min_confidence:g} (predictions with no confidence kept)."
        )
    if args.strict:
        notes.append(
            "baselines restricted to findings labeled 'Equation / proof' only "
            "(PaperAudit-taxonomy systems dropped)."
        )
    else:
        notes.append(
            "baselines restricted to math-category findings (Equation / proof, "
            "method_logic_consistency)."
        )
    print("\nNotes: " + "\n       ".join(notes))

    # ── CSVs ───────────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    # The tag belongs in the FILENAME, not only in the rows. Without it a v3
    # run writes over the v1 run's per-paper CSV -- same path, different
    # systems -- and evaluate_math_oracle.py's rung A, which reads that file
    # for the headline recall, silently starts reporting whichever system was
    # scored last. Nothing errors; the number just quietly changes meaning.
    stem = "math_systems_strict" if args.strict else "math_systems"
    if args.tag:
        stem += f"_{args.tag}"
    if args.min_confidence is not None:
        stem += f"_c{args.min_confidence:g}".replace(".", "p")
    comp_path = output_dir / f"{stem}_comparison.csv"
    with open(comp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison[0].keys()))
        writer.writeheader()
        writer.writerows(comparison)
    print(f"\nComparison table saved to: {comp_path}")

    pp_path = output_dir / f"{stem}_per_paper.csv"
    with open(pp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_paper_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_paper_rows)
    print(f"Per-paper metrics saved to: {pp_path}")


if __name__ == "__main__":
    main()
