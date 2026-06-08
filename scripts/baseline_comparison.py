"""Compare the single-call baseline against the orchestrated pipeline.

Runs BOTH systems on the same set of papers, scores each with the SAME alignment
+ metrics used everywhere else (`src/evaluation`), and prints a side-by-side
table plus per-paper outcomes. Writes full detail to
`outputs_new/baseline_comparison.json`.

Usage:
    # default: the two sample papers, our model in uncertainty mode, LLM judge
    python scripts/baseline_comparison.py

    # explicit papers / first-N / options
    python scripts/baseline_comparison.py 2405.01133v3 2402.10307v2
    python scripts/baseline_comparison.py --n 5 --mode exhaustive --no-judge -w 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from loguru import logger

from src.config import PipelineConfig
from src.baseline.single_call_baseline import SingleCallBaseline
from src.evaluation.alignment import match_predictions_to_ground_truth
from src.evaluation.metrics import evaluate_predictions
from src.orchestrator.orchestrator import VerificationOrchestrator
from src.orchestrator.uncertainty_orchestrator import UncertaintyOrchestrator
from src.parser.content_parser import parse_paper_content
from src.utils.logging import setup_logging

PARQUET = "data/train-00000-of-00001.parquet"


def _build_orchestrator(config: PipelineConfig):
    if config.orchestration_mode == "uncertainty":
        return UncertaintyOrchestrator(config=config)
    return VerificationOrchestrator(config=config)


def _score(predictions, subset_df, config) -> dict:
    aligned = match_predictions_to_ground_truth(predictions, subset_df, config=config)
    m = evaluate_predictions(aligned, subset_df)
    # Per-paper matched flag (was any ground-truth error for the paper hit?).
    matched_papers = {a.paper_id for a in aligned if a.is_true_positive}
    return {
        "precision": m.precision, "recall": m.recall, "f1": m.f1_score,
        "tp": m.true_positives, "fp": m.false_positives, "fn": m.false_negatives,
        "matched_papers": sorted(matched_papers),
        "predictions": {
            p.paper_id: [
                {"category": e.error_category, "location": e.error_location,
                 "confidence": round(e.confidence, 2)}
                for e in p.predicted_errors
            ] for p in predictions
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Baseline vs orchestrated pipeline.")
    ap.add_argument("paper_ids", nargs="*", help="Paper IDs to compare.")
    ap.add_argument("--n", type=int, default=None, help="Use the first N papers instead.")
    ap.add_argument("--mode", default="uncertainty", choices=["uncertainty", "exhaustive"])
    ap.add_argument("--workers", "-w", type=int, default=4)
    ap.add_argument("--no-judge", action="store_true", help="Use fuzzy matching, not the LLM judge.")
    ap.add_argument("--max-input-chars", type=int, default=60000)
    ap.add_argument("--uncertainty-threshold", type=float, default=0.30,
                    help="In uncertainty mode: escalate snippets at/above this score.")
    ap.add_argument("--llm-only", dest="llm_only", default=None,
                    choices=["same-prompt", "separate-prompts"],
                    help="Bypass specialist verifiers: use a single LLM for all snippets. "
                         "'same-prompt' uses one unified prompt; 'separate-prompts' uses "
                         "different prompts per type. Omit to use normal specialist verifiers.")
    ap.add_argument("--provider", default=None, help="Override LLM provider (e.g. 'mock' for a dry run).")
    ap.add_argument("--output", "-o", default="baseline_comparison", help="where to write the results")
    args = ap.parse_args()

    setup_logging("WARNING")
    df = pd.read_parquet(PARQUET)
    all_ids = df["doi/arxiv_id"].astype(str).tolist()

    if args.paper_ids:
        ids = args.paper_ids
    elif args.n:
        ids = all_ids[: args.n]
    else:
        ids = ["2405.01133v3", "2402.10307v2"]

    subset_df = df[df["doi/arxiv_id"].astype(str).isin(ids)].copy()

    config = PipelineConfig()
    config.llm.num_workers = args.workers
    config.orchestration_mode = args.mode
    config.uncertainty_threshold = args.uncertainty_threshold
    config.llm_only_mode = args.llm_only
    config.use_llm_judge = not args.no_judge
    if args.provider:
        config.llm.provider = args.provider

    baseline = SingleCallBaseline(config=config, max_input_chars=args.max_input_chars)
    orchestrator = _build_orchestrator(config)

    baseline_preds, model_preds = [], []
    for pid in ids:
        row = subset_df[subset_df["doi/arxiv_id"].astype(str) == pid].iloc[0]
        paper = parse_paper_content(
            paper_id=pid, title=row["title"], paper_category=row["paper_category"],
            paper_content=row["paper_content"], decode_images=False, image_output_dir=None,
        )
        logger.warning(f"[{pid}] baseline ...")
        baseline_preds.append(baseline.run(paper))
        logger.warning(f"[{pid}] orchestrated ({args.mode}) ...")
        model_preds.append(orchestrator.run(paper))

    baseline_score = _score(baseline_preds, subset_df, config)
    model_score = _score(model_preds, subset_df, config)

    # ---- report ----
    matcher = "fuzzy" if args.no_judge else "LLM judge"
    print("\n" + "=" * 72)
    print(f"BASELINE vs ORCHESTRATED ({args.mode})   papers={len(ids)}   match={matcher}")
    print("=" * 72)
    hdr = f"{'system':<26}{'TP':>4}{'FP':>4}{'FN':>4}{'prec':>7}{'rec':>7}{'F1':>7}"
    print(hdr)
    for label, s in (("single_call_baseline", baseline_score),
                     (f"orchestrated/{args.mode}", model_score)):
        print(f"{label:<26}{s['tp']:>4}{s['fp']:>4}{s['fn']:>4}"
              f"{s['precision']:>7.2f}{s['recall']:>7.2f}{s['f1']:>7.2f}")

    print("\nPer-paper (✓ = a ground-truth error was matched):")
    for pid in ids:
        gt = subset_df[subset_df['doi/arxiv_id'].astype(str) == pid].iloc[0]
        b_hit = "✓" if pid in baseline_score["matched_papers"] else "·"
        m_hit = "✓" if pid in model_score["matched_papers"] else "·"
        nb = len(baseline_score["predictions"].get(pid, []))
        nm = len(model_score["predictions"].get(pid, []))
        print(f"  {pid:<16} gt=[{gt['error_category']} @ {gt['error_location']}]")
        print(f"      baseline {b_hit}  ({nb} pred)   orchestrated {m_hit}  ({nm} pred)")

    out = {
        "paper_ids": ids, "mode": args.mode, "matcher": matcher,
        "baseline": baseline_score, "orchestrated": model_score,
    }
    Path(args.output).mkdir(exist_ok=True)
    dest = Path("outputs_new/baseline_comparison.json")
    dest.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nFull comparison written to {dest}")


if __name__ == "__main__":
    main()
