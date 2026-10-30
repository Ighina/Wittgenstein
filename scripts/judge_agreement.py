#!/usr/bin/env python3
"""
judge_agreement.py — Measure how much of a score is the detector and how much
is the judge.

Two independent checks:

1. AGREEMENT (free — reads cached verdicts, no API calls)
   Wherever two judges have scored the SAME predictions, compare them
   item by item: per-judge F1, raw agreement, Cohen's kappa, and which
   direction the disagreements run. Big F1 gaps with low kappa mean the
   number is a property of the judge, not the detector.

2. CALIBRATION (--calibrate, costs a few API calls per judge)
   Score each judge against cases whose answer is known in advance,
   so neither judge's opinion is the reference:
     - CEILING: ground truth fed back as the predictions. A judge that
       cannot recognise the errors as themselves is too strict; recall
       here is that judge's practical upper bound.
     - FLOOR:   predictions taken from a DIFFERENT paper. Anything
       matched is a false positive; precision lost here is leniency.
   A judge is usable only if it scores near 1.0 on the ceiling and near
   0.0 on the floor. The gap between judges on these two numbers bounds
   how much of any reported score is judge artifact.

Usage:
  python scripts/judge_agreement.py --workspace-dir data/paperaudit-workspace
  python scripts/judge_agreement.py --workspace-dir data/paperaudit-workspace \\
      --calibrate --judges claude-sonnet-4-5-20250929,deepseek-v4-pro --limit 5
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_verifier import (  # noqa: E402
    alignment_from_matches,
    compute_metrics,
    judge_paper_holistic,
    load_anthropic_client,
    load_json,
    summarize,
)


# ---------------------------------------------------------------------------
# 1. Agreement between judges on identical predictions (from cache)
# ---------------------------------------------------------------------------


def collect_cached_verdicts(ws_base: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Return {mode: {judge_model: {paper_id: cache_dict}}} from judge caches."""
    out: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
    for paper_dir in sorted(d for d in ws_base.iterdir() if d.is_dir()):
        for cache_path in sorted(paper_dir.glob("judge-*.json")):
            try:
                cache = load_json(cache_path)
            except Exception:
                continue
            mode = cache.get("mode")
            judge = cache.get("judge_model")
            if not mode or not judge or not isinstance(cache.get("matches"), list):
                continue
            out[mode][judge][paper_dir.name] = cache
    return {m: dict(j) for m, j in out.items()}


def cohens_kappa(a: List[bool], b: List[bool]) -> Optional[float]:
    """Cohen's kappa for two binary raters. None when it is undefined."""
    n = len(a)
    if n == 0:
        return None
    both = sum(1 for x, y in zip(a, b) if x and y)
    neither = sum(1 for x, y in zip(a, b) if not x and not y)
    po = (both + neither) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if abs(1 - pe) < 1e-12:
        # Both raters constant and identical: perfect but degenerate.
        return None
    return (po - pe) / (1 - pe)


def metrics_for(
    ws_base: Path, mode: str, judge_caches: Dict[str, Any], paper_ids: List[str]
) -> Dict[str, float]:
    """Recompute headline metrics for one judge over a fixed set of papers."""
    per_paper = []
    for pid in paper_ids:
        cache = judge_caches[pid]
        gt = load_json(ws_base / pid / "ground_truth.json")
        report = load_json(ws_base / pid / f"verification-report-{mode}.json")
        preds = report.get("predicted_errors", [])
        a = alignment_from_matches(cache["matches"], preds, gt)
        m = compute_metrics(a)
        per_paper.append(
            {
                "paper_id": pid,
                "n_gt": a["n_gt"],
                "n_pred": a["n_pred"],
                "matched_gt": a["matched_gt_count"],
                "matched_pred": a["matched_pred_count"],
                **m,
            }
        )
    return summarize(per_paper)


def compare_judges(ws_base: Path) -> List[Dict[str, Any]]:
    """Compare every pair of judges that scored the same predictions."""
    verdicts = collect_cached_verdicts(ws_base)
    rows: List[Dict[str, Any]] = []

    for mode in sorted(verdicts):
        judges = verdicts[mode]
        if len(judges) < 2:
            continue

        for i, ja in enumerate(sorted(judges)):
            for jb in sorted(judges)[i + 1 :]:
                shared = sorted(set(judges[ja]) & set(judges[jb]))
                # Only compare papers where both judges saw the same inputs.
                shared = [
                    p
                    for p in shared
                    if judges[ja][p].get("n_gt") == judges[jb][p].get("n_gt")
                    and judges[ja][p].get("n_pred") == judges[jb][p].get("n_pred")
                ]
                if not shared:
                    continue

                flags_a: List[bool] = []
                flags_b: List[bool] = []
                only_a = only_b = 0
                examples: List[Dict[str, Any]] = []
                for pid in shared:
                    ma = judges[ja][pid]["matches"]
                    mb = judges[jb][pid]["matches"]
                    for k, (x, y) in enumerate(zip(ma, mb)):
                        fa = bool(x.get("matched"))
                        fb = bool(y.get("matched"))
                        flags_a.append(fa)
                        flags_b.append(fb)
                        if fa != fb:
                            if fa:
                                only_a += 1
                            else:
                                only_b += 1
                            if len(examples) < 6:
                                examples.append(
                                    {
                                        "paper_id": pid,
                                        "gt_index": k + 1,
                                        "matched_by": ja if fa else jb,
                                        "rationale_a": str(x.get("rationale", ""))[:220],
                                        "rationale_b": str(y.get("rationale", ""))[:220],
                                    }
                                )

                n = len(flags_a)
                agree = sum(1 for x, y in zip(flags_a, flags_b) if x == y)
                rows.append(
                    {
                        "mode": mode,
                        "judge_a": ja,
                        "judge_b": jb,
                        "papers": len(shared),
                        "gt_items": n,
                        "raw_agreement": agree / n if n else 0.0,
                        "kappa": cohens_kappa(flags_a, flags_b),
                        "matched_rate_a": sum(flags_a) / n if n else 0.0,
                        "matched_rate_b": sum(flags_b) / n if n else 0.0,
                        "only_a_matched": only_a,
                        "only_b_matched": only_b,
                        "metrics_a": metrics_for(ws_base, mode, judges[ja], shared),
                        "metrics_b": metrics_for(ws_base, mode, judges[jb], shared),
                        "examples": examples,
                    }
                )
    return rows


def print_agreement(rows: List[Dict[str, Any]], show_examples: bool = True) -> None:
    if not rows:
        print(
            "No mode has verdicts from two different judges yet.\n"
            "Run --evaluate-llms (or --evaluate) twice with different "
            "--judge-model values to create a comparison."
        )
        return

    print(f"\n{'=' * 104}")
    print("JUDGE AGREEMENT — same predictions, different judges")
    print(f"{'=' * 104}")
    header = (
        f"{'Predictions (mode)':<28} {'Judge A':<20} {'Judge B':<20} "
        f"{'Papers':>6} {'Items':>6} {'Agree':>7} {'Kappa':>7} "
        f"{'F1(A)':>7} {'F1(B)':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        kappa = f"{r['kappa']:.3f}" if r["kappa"] is not None else "n/a"
        print(
            f"{r['mode'][:27]:<28} {r['judge_a'][:19]:<20} {r['judge_b'][:19]:<20} "
            f"{r['papers']:>6} {r['gt_items']:>6} {r['raw_agreement']:>7.3f} "
            f"{kappa:>7} {r['metrics_a']['avg_f1_score']:>7.3f} "
            f"{r['metrics_b']['avg_f1_score']:>7.3f}"
        )

    for r in rows:
        print(f"\n--- {r['mode']}: {r['judge_a']} vs {r['judge_b']} ---")
        ma, mb = r["metrics_a"], r["metrics_b"]
        print(
            f"  {r['judge_a']:<32} P={ma['macro_precision']:.3f} "
            f"R={ma['macro_recall']:.3f} avg_F1={ma['avg_f1_score']:.3f} "
            f"(matched {r['matched_rate_a']:.1%} of GT items)"
        )
        print(
            f"  {r['judge_b']:<32} P={mb['macro_precision']:.3f} "
            f"R={mb['macro_recall']:.3f} avg_F1={mb['avg_f1_score']:.3f} "
            f"(matched {r['matched_rate_b']:.1%} of GT items)"
        )
        print(
            f"  Disagreements: {r['only_a_matched']} matched only by "
            f"{r['judge_a']}, {r['only_b_matched']} only by {r['judge_b']} "
            f"({r['only_a_matched'] + r['only_b_matched']}/{r['gt_items']} items)"
        )
        if show_examples and r["examples"]:
            print("  Sample disagreements (same GT item, same predictions):")
            for ex in r["examples"][:3]:
                print(f"    [{ex['paper_id'][:40]} GT#{ex['gt_index']}] "
                      f"matched only by {ex['matched_by']}")
                print(f"       A: {ex['rationale_a']}")
                print(f"       B: {ex['rationale_b']}")


# ---------------------------------------------------------------------------
# 2. Calibration — ceiling and floor cases with a known answer
# ---------------------------------------------------------------------------


def gt_as_predictions(gt: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Turn ground-truth edits into predictions that restate them exactly.

    A judge that will not match these against their own source cannot
    recognise a correct detection at all, whatever the detector does.
    """
    preds = []
    for g in gt:
        preds.append(
            {
                "error_category": g.get("corruption_type", "unknown"),
                "error_location": g.get("location", "unknown"),
                "confidence": 0.95,
                "supporting_evidence": (
                    f"{g.get('error_explanation', '')} "
                    f"Offending text: {str(g.get('target_find', ''))[:400]}"
                ),
                "verifier_name": "calibration-ceiling",
                "snippet_id": "calibration::ceiling",
            }
        )
    return preds


def calibrate_judge(
    client: Any,
    judge_model: str,
    ws_base: Path,
    paper_ids: List[str],
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Run the ceiling and floor cases for one judge."""
    ceil_recalls: List[float] = []
    floor_precisions: List[float] = []

    for i, pid in enumerate(paper_ids):
        gt = load_json(ws_base / pid / "ground_truth.json")
        if not gt:
            continue

        # CEILING: the ground truth restated as predictions -> expect recall 1.0
        matches = judge_paper_holistic(
            client,
            judge_model,
            gt_as_predictions(gt),
            gt,
            f"calibration-ceiling::{pid}",
            "calibration",
            cache_path=ws_base / pid / f"calib-ceiling-{judge_model}.json",
            overwrite=overwrite,
        )
        if matches is not None:
            a = alignment_from_matches(matches, gt_as_predictions(gt), gt)
            ceil_recalls.append(compute_metrics(a)["recall"])

        # FLOOR: another paper's predictions -> expect recall 0.0
        other = paper_ids[(i + 1) % len(paper_ids)]
        if other != pid:
            other_gt = load_json(ws_base / other / "ground_truth.json")
            foreign = gt_as_predictions(other_gt)
            if foreign:
                matches = judge_paper_holistic(
                    client,
                    judge_model,
                    foreign,
                    gt,
                    f"calibration-floor::{pid}",
                    "calibration",
                    cache_path=ws_base / pid / f"calib-floor-{judge_model}.json",
                    overwrite=overwrite,
                )
                if matches is not None:
                    a = alignment_from_matches(matches, foreign, gt)
                    # Every match here is spurious.
                    floor_precisions.append(compute_metrics(a)["recall"])

    return {
        "judge_model": judge_model,
        "papers": len(paper_ids),
        "ceiling_recall": (
            sum(ceil_recalls) / len(ceil_recalls) if ceil_recalls else None
        ),
        "floor_false_match_rate": (
            sum(floor_precisions) / len(floor_precisions) if floor_precisions else None
        ),
    }


def print_calibration(results: List[Dict[str, Any]]) -> None:
    print(f"\n{'=' * 80}")
    print("JUDGE CALIBRATION — cases with a known answer")
    print(f"{'=' * 80}")
    print(f"{'Judge':<34} {'Ceiling recall':>15} {'Floor false-match':>19}")
    print(f"{'':<34} {'(want ~1.00)':>15} {'(want ~0.00)':>19}")
    print("-" * 72)
    for r in results:
        c = f"{r['ceiling_recall']:.3f}" if r["ceiling_recall"] is not None else "n/a"
        fl = (
            f"{r['floor_false_match_rate']:.3f}"
            if r["floor_false_match_rate"] is not None
            else "n/a"
        )
        print(f"{r['judge_model']:<34} {c:>15} {fl:>19}")
    print(
        "\nCeiling recall is that judge's practical maximum: a detector cannot "
        "score above it\neven with perfect predictions. Divide reported recall "
        "by it to get a judge-corrected figure."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure how much of a score is the judge rather than the detector.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--workspace-dir", type=str, required=True)
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Also run the ceiling/floor calibration (makes API calls)",
    )
    parser.add_argument(
        "--judges",
        type=str,
        default=None,
        help="--calibrate only: comma-separated judge models "
        "(default: every judge found in the cache)",
    )
    parser.add_argument(
        "--limit", type=int, default=5, help="Papers to calibrate on (default: 5)"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Re-run cached calibration verdicts"
    )
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "spot-eval"),
    )
    args = parser.parse_args()

    ws_base = Path(args.workspace_dir)
    if not ws_base.exists():
        print(f"ERROR: workspace not found: {ws_base}")
        sys.exit(1)

    rows = compare_judges(ws_base)
    print_agreement(rows)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        slim = [{k: v for k, v in r.items() if k != "examples"} for r in rows]
        with open(out_dir / "judge_agreement.json", "w", encoding="utf-8") as f:
            json.dump(slim, f, indent=2)
        print(f"\nAgreement detail saved to: {out_dir / 'judge_agreement.json'}")

    if args.calibrate:
        verdicts = collect_cached_verdicts(ws_base)
        known = sorted({j for judges in verdicts.values() for j in judges})
        judges = (
            [j.strip() for j in args.judges.split(",") if j.strip()]
            if args.judges
            else known
        )
        if not judges:
            print("\nNo judges to calibrate. Pass --judges.")
            return

        paper_ids = sorted(
            d.name
            for d in ws_base.iterdir()
            if d.is_dir() and (d / "ground_truth.json").exists()
        )[: args.limit]

        client = load_anthropic_client(args.api_key)
        results = [
            calibrate_judge(client, j, ws_base, paper_ids, args.overwrite)
            for j in judges
        ]
        print_calibration(results)
        with open(out_dir / "judge_calibration.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Calibration saved to: {out_dir / 'judge_calibration.json'}")


if __name__ == "__main__":
    main()
