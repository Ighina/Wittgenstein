"""Characterize the recall/cost tradeoff of uncertainty-driven orchestration.

For each paper we run the triage pass ONCE and each selected specialist ONCE
(on the union of snippets selected at the lowest threshold), caching every
result by snippet_id. Each threshold in the grid is then a pure re-selection
over those cached results — so the whole sweep costs one triage pass plus one
specialist pass per paper, not one full pipeline run per threshold.

Usage:
    python scripts/threshold_sweep.py 2405.01133v3 2402.10307v2
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from loguru import logger

from src.config import PipelineConfig
from src.models import BaseVerificationResult, VerificationStatus
from src.orchestrator.uncertainty_orchestrator import UncertaintyOrchestrator
from src.parser.content_parser import parse_paper_content
from src.segmentation.segmenter import segment_paper
from src.utils.logging import setup_logging

GRID = [0.10, 0.20, 0.30, 0.40, 0.50]
PARQUET = "data/train-00000-of-00001.parquet"


def sweep_paper(paper_id: str, df: pd.DataFrame, workers: int = 8) -> dict:
    row = df[df["doi/arxiv_id"].astype(str) == paper_id].iloc[0]
    paper = parse_paper_content(
        paper_id=paper_id,
        title=row["title"],
        paper_category=row["paper_category"],
        paper_content=row["paper_content"],
        decode_images=False,
        image_output_dir=None,
    )

    cfg = PipelineConfig(orchestration_mode="uncertainty", uncertainty_threshold=min(GRID))
    cfg.llm.num_workers = workers
    orch = UncertaintyOrchestrator(config=cfg)

    snippets = segment_paper(paper, config=cfg.segmentation)

    # 1) Triage every snippet once.
    triage = orch._triage_all(snippets, progress=None, paper_id=paper_id)
    triage_by_id = {t.snippet_id: t for t in triage}

    # 2) Run each specialist once on the union (threshold = min(GRID)).
    union = orch._select_snippets(snippets, triage_by_id)
    plan = []
    for s in union:
        route = triage_by_id[s.snippet_id].suggested_route
        name = orch._resolve_specialist(route, s)
        if name:
            triage_by_id[s.snippet_id].selected = True
            triage_by_id[s.snippet_id].routed_to = name
            plan.append((s, name))

    cached: dict[str, BaseVerificationResult] = {}

    def _run(item):
        snip, name = item
        v = orch._get_verifier(name)
        try:
            return snip.snippet_id, v.verify(snip)
        except Exception as exc:  # noqa: BLE001
            return snip.snippet_id, BaseVerificationResult(
                snippet_id=snip.snippet_id, verifier_name=name,
                status=VerificationStatus.SKIPPED, reasoning=str(exc),
            )

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(_run, it) for it in plan]):
            sid, res = fut.result()
            cached[sid] = res

    # 3) Re-apply every threshold against the cached results (no new API calls).
    rows = []
    for thr in GRID:
        selected = [s for s in snippets if triage_by_id[s.snippet_id].uncertainty >= thr]
        sel_ids = {s.snippet_id for s in selected}
        results: list[BaseVerificationResult] = []
        for s in snippets:
            if s.snippet_id in sel_ids and s.snippet_id in cached:
                results.append(cached[s.snippet_id])
            else:
                t = triage_by_id[s.snippet_id]
                results.append(BaseVerificationResult(
                    snippet_id=s.snippet_id, verifier_name="triage",
                    status=VerificationStatus.NO_ERROR, error_detected=False,
                    confidence=max(0.0, 1.0 - t.uncertainty),
                ))
        predicted = orch._aggregate_findings(results, paper)
        rows.append({
            "threshold": thr,
            "specialist_calls": len(sel_ids),
            "errors_detected": len(predicted),
            "error_locations": [p.error_location for p in predicted],
            "error_categories": [p.error_category for p in predicted],
            "routes_used": sorted({triage_by_id[i].routed_to for i in sel_ids
                                   if triage_by_id[i].routed_to}),
        })

    return {
        "paper_id": paper_id,
        "total_snippets": len(snippets),
        "ground_truth": {
            "category": row["error_category"],
            "location": str(row["error_location"]),
        },
        "sweep": rows,
        "uncertainty_map": [t.model_dump() for t in triage],
    }


def main(paper_ids: list[str]) -> None:
    setup_logging("WARNING")  # quiet — we only want the table
    df = pd.read_parquet(PARQUET)
    out = {}
    for pid in paper_ids:
        logger.warning(f"Sweeping {pid} ...")
        out[pid] = sweep_paper(pid, df)

    # Pretty table to stdout.
    for pid, data in out.items():
        print("\n" + "=" * 78)
        print(f"PAPER {pid}  ({data['total_snippets']} snippets)")
        print(f"  ground truth: [{data['ground_truth']['category']}] "
              f"@ {data['ground_truth']['location']}")
        print(f"  {'thr':>5} {'calls':>6} {'errors':>7}  {'routes':<22} locations")
        for r in data["sweep"]:
            print(f"  {r['threshold']:>5.2f} {r['specialist_calls']:>6} "
                  f"{r['errors_detected']:>7}  {','.join(r['routes_used']):<22} "
                  f"{r['error_locations']}")

    Path("outputs_new").mkdir(exist_ok=True)
    dest = Path("outputs_new/threshold_sweep.json")
    dest.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nFull sweep written to {dest}")


if __name__ == "__main__":
    ids = sys.argv[1:] or ["2405.01133v3", "2402.10307v2"]
    main(ids)
