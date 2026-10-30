#!/usr/bin/env python3
"""
judge_review.py — Build a blind human-labelling sheet for the items two judges
disagree on, then score the judges against those human labels.

This is what turns the judge from an unvalidated instrument into a calibrated
one. Until a human has labelled the disputed middle, "which judge is right"
is unanswerable — the judges are only checkable against each other.

Design choices that matter for the result being usable:

  * BLIND. The sheet shows the ground-truth error and the candidate
    prediction only. It does not show which judge said what, or that the
    judges disagreed at all — reading a judge's rationale first anchors the
    labeller onto that verdict. The rationales are kept in the key file for
    reading afterwards.
  * CONTROLS. A fraction of rows are items both judges agreed on (some
    matches, some non-matches), shuffled in and indistinguishable. These
    measure whether the labeller is systematically stricter or more lenient
    than both judges, and catch a sheet filled in carelessly.
  * SHUFFLED with a fixed seed, so a run is reproducible but the order
    carries no signal about item type or source paper.

Workflow:
  1. python scripts/judge_review.py --build --workspace-dir data/paperaudit-workspace
  2. Open outputs/spot-eval/judge_review_sheet.csv, fill the `human_label`
     column with: match | no_match | unsure   (leave blank to skip)
  3. python scripts/judge_review.py --score outputs/spot-eval/judge_review_sheet.csv

Do not open judge_review_key.json before labelling — it reveals the verdicts.
"""

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_verifier import load_json  # noqa: E402
from judge_agreement import cohens_kappa, collect_cached_verdicts  # noqa: E402

DEFAULT_OUT = PROJECT_ROOT / "outputs" / "spot-eval"
VALID_LABELS = {"match", "no_match", "unsure"}


def _truncate(text: Any, limit: int) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= limit else s[:limit] + " …"


def gather_items(
    ws_base: Path, judge_a: str, judge_b: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (disputed, agreed) item records comparable across both judges."""
    verdicts = collect_cached_verdicts(ws_base)
    disputed: List[Dict[str, Any]] = []
    agreed: List[Dict[str, Any]] = []

    for mode in sorted(verdicts):
        judges = verdicts[mode]
        if judge_a not in judges or judge_b not in judges:
            continue
        for paper_id in sorted(set(judges[judge_a]) & set(judges[judge_b])):
            ca, cb = judges[judge_a][paper_id], judges[judge_b][paper_id]
            if ca.get("n_gt") != cb.get("n_gt") or ca.get("n_pred") != cb.get("n_pred"):
                continue  # judged against different inputs; not comparable

            try:
                gt = load_json(ws_base / paper_id / "ground_truth.json")
                report = load_json(
                    ws_base / paper_id / f"verification-report-{mode}.json"
                )
            except Exception:
                continue
            preds = report.get("predicted_errors", [])

            for idx, (ma, mb) in enumerate(zip(ca["matches"], cb["matches"])):
                if idx >= len(gt):
                    continue
                a_matched = bool(ma.get("matched"))
                b_matched = bool(mb.get("matched"))
                pred_ids = sorted(
                    {
                        int(i)
                        for i in list(ma.get("matched_pred_ids", []))
                        + list(mb.get("matched_pred_ids", []))
                        if isinstance(i, int) and 1 <= i <= len(preds)
                    }
                )
                rec = {
                    "mode": mode,
                    "paper_id": paper_id,
                    "gt_index": idx,
                    "gt": gt[idx],
                    "pred_ids": pred_ids,
                    "preds": [preds[i - 1] for i in pred_ids],
                    "a_matched": a_matched,
                    "b_matched": b_matched,
                    "rationale_a": str(ma.get("rationale", "")),
                    "rationale_b": str(mb.get("rationale", "")),
                }
                if a_matched != b_matched:
                    disputed.append(rec)
                else:
                    agreed.append(rec)
    return disputed, agreed


def render_candidates(rec: Dict[str, Any], evidence_chars: int) -> str:
    """Format the prediction(s) a judge considered for this ground-truth item."""
    if not rec["preds"]:
        # Only reachable for agreed-no-match controls, where no judge named one.
        return "(the verifier reported nothing for this part of the paper)"
    parts = []
    for n, p in zip(rec["pred_ids"], rec["preds"]):
        parts.append(
            f"[prediction {n}] category: {p.get('error_category', '?')} | "
            f"location: {p.get('error_location', '?')}\n"
            f"{_truncate(p.get('supporting_evidence'), evidence_chars)}"
        )
    return "\n\n".join(parts)


def build_sheet(
    ws_base: Path,
    judge_a: str,
    judge_b: str,
    n_disputed: int,
    n_controls: int,
    out_dir: Path,
    seed: int,
    evidence_chars: int,
) -> None:
    disputed, agreed = gather_items(ws_base, judge_a, judge_b)
    if not disputed:
        print(
            f"No disputed items between '{judge_a}' and '{judge_b}'.\n"
            "Both judges must have scored the same predictions — check "
            "scripts/judge_agreement.py for available pairs."
        )
        sys.exit(1)

    rng = random.Random(seed)

    # Disputed items: simple random sample, so per-judge agreement measured on
    # the sheet is an unbiased estimate for the disputed population.
    sample_disputed = (
        rng.sample(disputed, n_disputed) if len(disputed) > n_disputed else list(disputed)
    )

    # Controls: half agreed-match, half agreed-no-match.
    agreed_yes = [r for r in agreed if r["a_matched"]]
    agreed_no = [r for r in agreed if not r["a_matched"]]
    half = n_controls // 2
    controls = (
        rng.sample(agreed_yes, min(half, len(agreed_yes)))
        + rng.sample(agreed_no, min(n_controls - half, len(agreed_no)))
    )

    rows: List[Dict[str, Any]] = []
    key: Dict[str, Any] = {}
    items = [(r, False) for r in sample_disputed] + [(r, True) for r in controls]
    rng.shuffle(items)

    for n, (rec, is_control) in enumerate(items, 1):
        item_id = f"item-{n:04d}"
        gt = rec["gt"]
        rows.append(
            {
                "item_id": item_id,
                "human_label": "",
                "human_notes": "",
                "paper": rec["paper_id"][:60],
                "gt_location": gt.get("location", ""),
                "gt_difficulty": gt.get("difficulty", ""),
                "gt_type": gt.get("corruption_type", ""),
                "ground_truth_error": _truncate(gt.get("error_explanation"), 900),
                "text_that_was_altered": _truncate(gt.get("target_find"), 400),
                "verifier_prediction": render_candidates(rec, evidence_chars),
            }
        )
        # Verdicts live only in the key, never in the sheet.
        key[item_id] = {
            "mode": rec["mode"],
            "paper_id": rec["paper_id"],
            "gt_index": rec["gt_index"],
            "pred_ids": rec["pred_ids"],
            "is_control": is_control,
            "control_type": (
                None if not is_control else ("agreed_match" if rec["a_matched"] else "agreed_no_match")
            ),
            "judge_a": judge_a,
            "judge_b": judge_b,
            "a_matched": rec["a_matched"],
            "b_matched": rec["b_matched"],
            "rationale_a": rec["rationale_a"],
            "rationale_b": rec["rationale_b"],
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = out_dir / "judge_review_sheet.csv"
    key_path = out_dir / "judge_review_key.json"

    with open(sheet_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(key_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "judge_a": judge_a,
                "judge_b": judge_b,
                "seed": seed,
                "disputed_population": len(disputed),
                "disputed_sampled": len(sample_disputed),
                "controls": len(controls),
                "items": key,
            },
            f,
            indent=2,
        )

    only_a = sum(1 for r in sample_disputed if r["a_matched"])
    print(f"Review sheet written: {sheet_path}")
    print(f"  {len(rows)} rows = {len(sample_disputed)} disputed "
          f"(sampled from {len(disputed)}) + {len(controls)} hidden controls")
    print(f"  Of the disputed rows, {only_a} were matched only by {judge_a}, "
          f"{len(sample_disputed) - only_a} only by {judge_b}")
    print(f"Answer key (do not read before labelling): {key_path}")
    print(
        "\nFill the `human_label` column with: match | no_match | unsure\n"
        "  match    = the prediction identifies the SAME underlying issue as the ground truth\n"
        "  no_match = it identifies a different issue, or is too vague to tell\n"
        "  unsure   = genuinely undecidable; these are excluded from scoring\n"
        "Leave a row blank to skip it. Then run:\n"
        f"  python scripts/judge_review.py --score {sheet_path}"
    )


def score_sheet(sheet_path: Path, key_path: Optional[Path]) -> None:
    key_path = key_path or sheet_path.parent / "judge_review_key.json"
    if not key_path.exists():
        print(f"ERROR: key file not found: {key_path}")
        sys.exit(1)
    key = load_json(key_path)
    items = key["items"]
    judge_a, judge_b = key["judge_a"], key["judge_b"]

    with open(sheet_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    labelled = []
    bad: List[str] = []
    for r in rows:
        raw = (r.get("human_label") or "").strip().lower().replace("-", "_")
        if not raw:
            continue
        if raw not in VALID_LABELS:
            bad.append(f"{r.get('item_id')}: {r.get('human_label')!r}")
            continue
        info = items.get(r.get("item_id", ""))
        if info:
            labelled.append((info, raw))

    if bad:
        print(f"WARNING: {len(bad)} unrecognised label(s), ignored. "
              f"Use match/no_match/unsure. e.g. {bad[:3]}")
    if not labelled:
        print(
            f"No labels found in {sheet_path}.\n"
            "Fill the `human_label` column with match | no_match | unsure first."
        )
        sys.exit(1)

    decided = [(i, l) for i, l in labelled if l != "unsure"]
    unsure = len(labelled) - len(decided)

    disputed = [(i, l) for i, l in decided if not i["is_control"]]
    controls = [(i, l) for i, l in decided if i["is_control"]]

    print(f"\n{'=' * 76}")
    print("HUMAN-LABELLED JUDGE VALIDATION")
    print(f"{'=' * 76}")
    print(f"Labelled {len(labelled)} rows: {len(decided)} decided, {unsure} unsure "
          f"({len(disputed)} disputed, {len(controls)} controls)")

    # --- Controls: is the labeller aligned with the judges where they agree? ---
    if controls:
        agree_ctrl = sum(
            1 for i, l in controls if (l == "match") == bool(i["a_matched"])
        )
        rate = agree_ctrl / len(controls)
        print(f"\nCONTROLS — items both judges agreed on: you agreed with them on "
              f"{agree_ctrl}/{len(controls)} ({rate:.1%})")
        if rate < 0.85:
            by_type = Counter(
                i["control_type"] for i, l in controls
                if (l == "match") != bool(i["a_matched"])
            )
            print("  ⚠ Below 85% — you and the judges differ even on the easy cases, "
                  "so the\n    disputed results below reflect a different matching "
                  f"standard, not just judge noise. Mismatches by type: {dict(by_type)}")

    if not disputed:
        print("\nNo disputed items labelled yet — those are what separate the judges.")
        return

    # --- Disputed: which judge tracks the human? ---
    human = [1 if l == "match" else 0 for _, l in disputed]
    a_flags = [1 if i["a_matched"] else 0 for i, _ in disputed]
    b_flags = [1 if i["b_matched"] else 0 for i, _ in disputed]
    a_correct = sum(1 for h, a in zip(human, a_flags) if h == a)
    b_correct = sum(1 for h, b in zip(human, b_flags) if h == b)
    n = len(disputed)

    print(f"\nDISPUTED ITEMS ({n} decided) — by construction exactly one judge is "
          f"right on each")
    print(f"{'Judge':<34} {'Agrees with human':>18} {'Kappa':>8}")
    print("-" * 62)
    for name, flags, correct in (
        (judge_a, a_flags, a_correct),
        (judge_b, b_flags, b_correct),
    ):
        k = cohens_kappa([bool(x) for x in human], [bool(x) for x in flags])
        ks = f"{k:.3f}" if k is not None else "n/a"
        print(f"{name:<34} {correct:>7}/{n} ({correct/n:>5.1%}) {ks:>8}")

    human_match_rate = sum(human) / n
    print(f"\nYou called {sum(human)}/{n} ({human_match_rate:.1%}) of the disputed "
          f"items a match.")
    if a_correct > b_correct:
        winner, loser = judge_a, judge_b
    elif b_correct > a_correct:
        winner, loser = judge_b, judge_a
    else:
        winner = loser = None

    if winner:
        margin = abs(a_correct - b_correct)
        # Binomial sign test against a coin flip on the disputed set.
        print(f"\n→ {winner} tracks your judgement more closely than {loser} "
              f"({margin} items apart).")
        if margin < max(3, int(0.1 * n)):
            print("  Margin is small relative to the sample — label more disputed "
                  "items before\n  treating this as settled.")
    else:
        print("\n→ The judges track your judgement equally well; neither is the "
              "better instrument\n  on this evidence.")

    # --- What this implies for reported scores ---
    print(f"\nIMPLICATION FOR REPORTED SCORES")
    lenient = judge_a if sum(a_flags) > sum(b_flags) else judge_b
    strict = judge_b if lenient == judge_a else judge_a
    if human_match_rate > 0.5:
        print(f"  You matched most disputed items, so the stricter judge ({strict}) "
              f"is\n  under-crediting the detector: its scores are a lower bound.")
    else:
        print(f"  You rejected most disputed items, so the more lenient judge "
              f"({lenient}) is\n  over-crediting the detector: its scores are an "
              f"upper bound.")
    print("  Either way, report the judge alongside the score, and prefer the judge "
          "that\n  tracked you above when quoting a single number.")

    out = sheet_path.parent / "judge_review_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "labelled": len(labelled),
                "decided": len(decided),
                "unsure": unsure,
                "disputed_decided": n,
                "human_match_rate_disputed": human_match_rate,
                "controls_decided": len(controls),
                "control_agreement": (
                    sum(1 for i, l in controls if (l == "match") == bool(i["a_matched"]))
                    / len(controls)
                    if controls
                    else None
                ),
                "judges": {
                    judge_a: {"agreement_disputed": a_correct / n, "correct": a_correct},
                    judge_b: {"agreement_disputed": b_correct / n, "correct": b_correct},
                },
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and score a blind human-labelling sheet for judge disagreements.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--build", action="store_true", help="Build the review sheet")
    parser.add_argument(
        "--score", type=str, default=None, help="Score a filled-in review sheet CSV"
    )
    parser.add_argument("--workspace-dir", type=str, default=None)
    parser.add_argument("--judge-a", type=str, default="claude-sonnet-4-5-20250929")
    parser.add_argument("--judge-b", type=str, default="deepseek-v4-pro")
    parser.add_argument(
        "--n-disputed",
        type=int,
        default=80,
        help="Disputed items to sample (default: 80)",
    )
    parser.add_argument(
        "--n-controls",
        type=int,
        default=20,
        help="Hidden agreed-item controls to mix in (default: 20)",
    )
    parser.add_argument(
        "--evidence-chars",
        type=int,
        default=700,
        help="Max characters of prediction evidence shown per row (default: 700)",
    )
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--key", type=str, default=None, help="--score: key file path")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUT))
    args = parser.parse_args()

    if args.score:
        score_sheet(Path(args.score), Path(args.key) if args.key else None)
        return

    if not args.build:
        parser.print_help()
        print("\nERROR: pass --build or --score <csv>.")
        sys.exit(1)

    if not args.workspace_dir:
        print("ERROR: --build requires --workspace-dir")
        sys.exit(1)
    ws = Path(args.workspace_dir)
    if not ws.exists():
        print(f"ERROR: workspace not found: {ws}")
        sys.exit(1)

    build_sheet(
        ws,
        args.judge_a,
        args.judge_b,
        args.n_disputed,
        args.n_controls,
        Path(args.output_dir),
        args.seed,
        args.evidence_chars,
    )


if __name__ == "__main__":
    main()
