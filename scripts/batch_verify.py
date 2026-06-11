#!/usr/bin/env python3
"""Programmatic batch verification of papers using Claude Code.

Reads papers from the parquet dataset and calls Claude Code CLI with the
/verify-paper skill for each one. Saves structured results and optionally
evaluates against ground truth.

Usage:
    python scripts/batch_verify.py [OPTIONS]

Options:
    --papers N          Number of papers to verify (default: all)
    --offset N          Skip first N papers (default: 0)
    --output DIR        Output directory (default: outputs/claude-batch)
    --mode MODE         Orchestration mode: exhaustive|uncertainty (default: exhaustive)
    --threshold FLOAT   Uncertainty threshold (default: 0.30)
    --dry-run           List papers without verifying
    --paper-id ID       Verify a single paper by ID
    --list              List available papers in the dataset
    --evaluate          Run evaluation after batch verification
    --help              Show help message
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

app = typer.Typer(name="batch-verify", help="Batch paper verification via Claude Code")
console = Console()

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PARQUET = PROJECT_ROOT / "data" / "train-00000-of-00001.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "claude-batch"


def get_paper_list(
    parquet_path: Path,
    max_papers: Optional[int] = None,
    offset: int = 0,
) -> list[dict]:
    """Load paper metadata from the parquet file."""
    if not parquet_path.exists():
        console.print(f"[red]Parquet file not found: {parquet_path}[/red]")
        raise typer.Exit(code=1)

    df = pd.read_parquet(parquet_path)
    if max_papers:
        df = df.iloc[offset : offset + max_papers]
    else:
        df = df.iloc[offset:]

    papers = []
    for _, row in df.iterrows():
        papers.append({
            "paper_id": str(row["doi/arxiv_id"]),
            "title": str(row.get("title", ""))[:200],
            "paper_category": str(row.get("paper_category", "")),
            "error_category": str(row.get("error_category", "")),
            "error_location": str(row.get("error_location", "")),
            "error_severity": str(row.get("error_severity", "")),
        })
    return papers


def build_verify_prompt(
    paper_id: str,
    mode: str = "exhaustive",
    threshold: float = 0.30,
) -> str:
    """Build the Claude Code prompt for verifying a single paper."""
    parts = [
        "/verify-paper",
        "",
        f"Verify the paper with ID '{paper_id}' from the dataset.",
        f"Use orchestration mode: {mode}.",
    ]

    if mode == "uncertainty":
        parts.append(f"Use uncertainty threshold: {threshold}.")

    parts.extend([
        "",
        "Steps:",
        "1. Use the get_paper_from_dataset MCP tool to fetch the paper.",
        "2. Use the segment_paper MCP tool to parse and segment the paper.",
        "3. For each snippet, route to the appropriate specialist verifier skill using the Skill tool.",
        "4. Aggregate all findings into a verification report.",
        "5. Include comparison with the ground truth annotation from the dataset.",
        "6. Output the final report as structured JSON.",
    ])

    return "\n".join(parts)


def run_claude_verify(prompt: str, output_file: Path) -> tuple[bool, str]:
    """Run Claude Code CLI to verify a paper.

    Returns (success, output_text).
    """
    try:
        result = subprocess.run(
            ["claude", "-p", prompt, "--print"],
            capture_output=True,
            text=True,
            timeout=600,  # 10-minute timeout per paper
            cwd=str(PROJECT_ROOT),
        )

        output = result.stdout
        if result.stderr:
            output += "\n\n## stderr\n```\n" + result.stderr + "\n```"

        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(output)

        success = result.returncode == 0
        return success, output

    except subprocess.TimeoutExpired:
        msg = f"Timeout verifying paper (10 min limit)"
        output_file.write_text(f"# TIMEOUT\n{msg}\n")
        return False, msg
    except FileNotFoundError:
        msg = "Claude Code CLI not found. Install it from https://claude.ai/code"
        output_file.write_text(f"# ERROR\n{msg}\n")
        return False, msg
    except Exception as exc:
        msg = f"Error: {exc}"
        output_file.write_text(f"# ERROR\n{msg}\n")
        return False, msg


def parse_verification_result(output_text: str, paper_id: str) -> dict:
    """Try to extract structured verification result from Claude's output."""
    result = {
        "paper_id": paper_id,
        "verification_timestamp": datetime.now().isoformat(),
        "success": False,
        "predicted_errors": [],
        "verdict": "UNKNOWN",
        "raw_output_length": len(output_text),
    }

    # Try to find a JSON block with verification results
    # Look for ```json ... ``` blocks
    import re
    json_blocks = re.findall(r'```json\s*\n(.*?)\n```', output_text, re.DOTALL)
    for block in json_blocks:
        try:
            parsed = json.loads(block)
            if "paper_id" in parsed or "predicted_errors" in parsed:
                result.update(parsed)
                result["success"] = True
                break
        except json.JSONDecodeError:
            continue

    # Also look for bare JSON objects
    if not result["success"]:
        brace_blocks = re.findall(r'\{[^{}]*"paper_id"[^{}]*\}', output_text)
        for block in brace_blocks:
            try:
                parsed = json.loads(block)
                result.update(parsed)
                result["success"] = True
                break
            except json.JSONDecodeError:
                continue

    return result


@app.command()
def main(
    papers: Optional[int] = typer.Option(
        None, "--papers", "-n",
        help="Number of papers to verify (default: all).",
    ),
    offset: int = typer.Option(
        0, "--offset",
        help="Skip first N papers.",
    ),
    output_dir: str = typer.Option(
        "outputs/claude-batch",
        "--output", "-o",
        help="Output directory for results.",
    ),
    mode: str = typer.Option(
        "exhaustive",
        "--mode",
        help="Orchestration mode: exhaustive or uncertainty.",
    ),
    threshold: float = typer.Option(
        0.30,
        "--threshold",
        help="Uncertainty threshold for uncertainty mode.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="List papers without verifying.",
    ),
    paper_id: Optional[str] = typer.Option(
        None, "--paper-id",
        help="Verify a single paper by DOI/arXiv ID.",
    ),
    list_papers: bool = typer.Option(
        False, "--list",
        help="List available papers in the dataset.",
    ),
    evaluate: bool = typer.Option(
        False, "--evaluate",
        help="Run evaluation after batch verification.",
    ),
    parquet_path: str = typer.Option(
        "data/train-00000-of-00001.parquet",
        "--parquet", "-p",
        help="Path to the parquet dataset file.",
    ),
) -> None:
    """Batch-verify papers from the parquet dataset using Claude Code."""
    parquet_full = PROJECT_ROOT / parquet_path
    output_full = PROJECT_ROOT / output_dir

    # --list mode
    if list_papers:
        all_papers = get_paper_list(parquet_full)
        console.print(f"\n[bold]📚 Papers in dataset: {len(all_papers)}[/bold]\n")
        table = Table(title="Available Papers")
        table.add_column("#", style="dim")
        table.add_column("Paper ID", style="cyan")
        table.add_column("Title", style="green")
        table.add_column("Error Category", style="yellow")

        for i, p in enumerate(all_papers[:50], 1):
            table.add_row(
                str(i),
                p["paper_id"],
                p["title"][:80],
                p["error_category"],
            )

        console.print(table)
        if len(all_papers) > 50:
            console.print(f"\n[dim]... and {len(all_papers) - 50} more papers[/dim]")
        return

    # --dry-run mode
    if dry_run:
        paper_list = get_paper_list(parquet_full, max_papers=papers, offset=offset)
        console.print(f"\n[bold]🔍 DRY RUN — Would verify {len(paper_list)} papers[/bold]\n")
        for i, p in enumerate(paper_list, 1):
            console.print(f"  {i}. [{p['paper_category']}] {p['paper_id']}")
            console.print(f"     {p['title'][:100]}")
        console.print(f"\n  Mode: {mode}")
        console.print(f"  Output: {output_full}")
        return

    # --paper-id mode (single paper)
    if paper_id:
        console.print(f"[bold cyan]🔬 Verifying single paper: {paper_id}[/bold cyan]\n")
        paper_list = get_paper_list(parquet_full)
        match = next((p for p in paper_list if p["paper_id"] == paper_id), None)
        if match is None:
            console.print(f"[red]Paper not found: {paper_id}[/red]")
            raise typer.Exit(code=1)

        console.print(f"  Title: {match['title'][:120]}")
        console.print(f"  Category: {match['paper_category']}")
        console.print(f"  GT Error: {match['error_category']}")
        console.print("")

        prompt = build_verify_prompt(paper_id, mode, threshold)
        output_file = output_full / f"{paper_id.replace('/', '_')}.md"

        with console.status(f"[cyan]Verifying {paper_id}..."):
            success, output = run_claude_verify(prompt, output_file)

        if success:
            console.print(f"[green]✅ Verification complete → {output_file}[/green]")
            result = parse_verification_result(output, paper_id)
            summary_file = output_full / f"{paper_id.replace('/', '_')}_result.json"
            summary_file.write_text(json.dumps(result, indent=2))
        else:
            console.print(f"[red]❌ Verification failed[/red]")
        return

    # ---- Batch mode ----
    paper_list = get_paper_list(parquet_full, max_papers=papers, offset=offset)
    total = len(paper_list)

    console.print(f"\n[bold cyan]🚀 Batch Paper Verification[/bold cyan]")
    console.print(f"  Papers: {total}")
    console.print(f"  Mode: {mode}")
    console.print(f"  Output: {output_full}")
    console.print("")

    output_full.mkdir(parents=True, exist_ok=True)

    all_results = []
    verified = 0
    failed = 0
    start_time = time.monotonic()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"[cyan]Verifying {total} papers...", total=total
        )

        for i, paper in enumerate(paper_list, 1):
            pid = paper["paper_id"]
            progress.update(
                task,
                description=f"[cyan][{i}/{total}] Verifying {pid}..."
            )

            prompt = build_verify_prompt(pid, mode, threshold)
            output_file = output_full / f"{pid.replace('/', '_')}.md"

            success, output = run_claude_verify(prompt, output_file)

            if success:
                verified += 1
                result = parse_verification_result(output, pid)
                result["ground_truth"] = {
                    "error_category": paper["error_category"],
                    "error_location": paper["error_location"],
                    "error_severity": paper["error_severity"],
                }
            else:
                failed += 1
                result = {
                    "paper_id": pid,
                    "success": False,
                    "error": output[:200],
                }

            all_results.append(result)
            progress.advance(task)

            # Brief pause between papers
            time.sleep(1)

    elapsed = time.monotonic() - start_time

    # Save aggregate results
    results_file = output_full / "batch_results.json"
    results_file.write_text(json.dumps({
        "batch_timestamp": datetime.now().isoformat(),
        "total_papers": total,
        "verified": verified,
        "failed": failed,
        "mode": mode,
        "threshold": threshold,
        "duration_seconds": elapsed,
        "results": all_results,
    }, indent=2))

    # Summary
    console.print(f"\n[bold green]═══════════════════════════════════════════[/bold green]")
    console.print(f"[bold green]Batch verification complete![/bold green]")
    console.print(f"  Total:   {total}")
    console.print(f"  ✅ Done: {verified}")
    console.print(f"  ❌ Failed: {failed}")
    console.print(f"  ⏱️  Time:  {elapsed:.0f}s ({elapsed/60:.1f}m)")
    console.print(f"  📁 Saved: {results_file}")
    console.print(f"[bold green]═══════════════════════════════════════════[/bold green]")

    # Optionally evaluate
    if evaluate and verified > 0:
        console.print(f"\n[bold]Running evaluation against ground truth...[/bold]")
        _run_evaluation(results_file, parquet_full, output_full)


def _run_evaluation(results_file: Path, parquet_path: Path, output_dir: Path) -> None:
    """Run evaluation comparing batch results to ground truth."""
    try:
        # Load predictions from the batch results
        batch_data = json.loads(results_file.read_text())
        predictions = []

        for r in batch_data["results"]:
            if not r.get("success"):
                continue
            # Convert to the format expected by the evaluation module
            from src.models import PaperPrediction, PredictedError

            paper_pred = PaperPrediction(
                paper_id=r["paper_id"],
                predicted_errors=[
                    PredictedError(
                        error_category=e.get("error_category", "Unknown"),
                        error_location=e.get("error_location", ""),
                        confidence=e.get("confidence", 0.0),
                        supporting_evidence=e.get("supporting_evidence", ""),
                        verifier_name=e.get("verifier_name", ""),
                        snippet_id=e.get("snippet_id", ""),
                    )
                    for e in r.get("predicted_errors", [])
                ],
            )
            predictions.append(paper_pred)

        # Run the standard evaluation
        df = pd.read_parquet(parquet_path)
        from src.evaluation.alignment import match_predictions_to_ground_truth
        from src.evaluation.metrics import evaluate_predictions
        from src.reporting.reporter import generate_report
        from src.parser.schema_analyzer import analyze_dataset_schema

        schema_report = analyze_dataset_schema(parquet_path)
        aligned = match_predictions_to_ground_truth(predictions, df)
        metrics = evaluate_predictions(aligned, df)

        console.print(f"\n[bold]Evaluation Results:[/bold]")
        console.print(f"  Accuracy:  {metrics.accuracy:.4f}")
        console.print(f"  Precision: {metrics.precision:.4f}")
        console.print(f"  Recall:    {metrics.recall:.4f}")
        console.print(f"  F1 Score:  {metrics.f1_score:.4f}")

        report_dir = generate_report(
            metrics=metrics,
            aligned_predictions=aligned,
            predictions=predictions,
            schema_report=schema_report,
            output_dir=output_dir,
            ground_truth_df=df,
        )
        console.print(f"\n[green]Evaluation report: {report_dir}[/green]")

    except Exception as exc:
        console.print(f"[red]Evaluation failed: {exc}[/red]")


if __name__ == "__main__":
    main()
