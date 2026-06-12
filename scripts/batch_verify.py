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
        papers.append(
            {
                "paper_id": str(row["doi/arxiv_id"]),
                "title": str(row.get("title", ""))[:200],
                "paper_category": str(row.get("paper_category", "")),
                "error_category": str(row.get("error_category", "")),
                "error_location": str(row.get("error_location", "")),
                "error_severity": str(row.get("error_severity", "")),
            }
        )
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

    parts.extend(
        [
            "",
            "Steps:",
            "1. Use the get_paper_from_dataset MCP tool to fetch the paper.",
            "2. Use the segment_paper MCP tool to parse and segment the paper.",
            "3. For each snippet, route to the appropriate specialist verifier skill using the Skill tool.",
            "4. Aggregate all findings into a verification report.",
            "5. Include comparison with the ground truth annotation from the dataset.",
            "6. Output the final report as structured JSON.",
        ]
    )

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


def parse_verification_result(
    output_text: str, paper_id: str, output_md_file: Optional[Path] = None
) -> dict:
    """Try to extract structured verification result from Claude's output.

    First attempts to parse JSON embedded in the markdown output text.
    If that fails, looks for companion JSON report files saved alongside
    the markdown output by the /verify-paper skill
    (e.g., ``verification_report_<paper_id>.json`` or
    ``verification_<paper_id>.json``).
    """
    import re

    result = {
        "paper_id": paper_id,
        "verification_timestamp": datetime.now().isoformat(),
        "success": False,
        "predicted_errors": [],
        "verdict": "UNKNOWN",
        "raw_output_length": len(output_text),
    }

    # -- 1. Try to find a JSON block embedded in the markdown output --
    json_blocks = re.findall(r"```json\s*\n(.*?)\n```", output_text, re.DOTALL)
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

    # -- 2. Fall back to companion JSON report files saved by /verify-paper --
    if not result["success"] and output_md_file is not None:
        output_dir = output_md_file.parent

        # The skill produces JSON reports with naming variations:
        #   verification_report_<id>.json
        #   verification_<id>.json
        #   <id>_verification_report.json
        # Additionally, the ID may have / and . replaced with _ (e.g. DOIs).
        # Strategy: glob for all *report*.json / verification*.json files and
        # match by looking inside each file for the correct paper_id.
        json_candidates = sorted(
            set(output_dir.glob("*report*.json"))
            | set(output_dir.glob("verification*.json"))
        )

        # Exclude known non-per-paper files
        skip_names = {"batch_results.json", "metrics.json", "predictions.json",
                       "confusion_matrix.json"}
        json_candidates = [
            p for p in json_candidates if p.name not in skip_names
        ]

        for json_path in json_candidates:
            try:
                parsed = json.loads(json_path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            # Match by paper_id inside the JSON
            if parsed.get("paper_id") != paper_id:
                continue
            predicted_errors = parsed.get("predicted_errors", [])
            if predicted_errors:
                result["predicted_errors"] = predicted_errors
                result["verdict"] = parsed.get("verdict", result["verdict"])
                result["title"] = parsed.get("title", "")
                result["paper_category"] = parsed.get("paper_category", "")
                if "statistics" in parsed:
                    result["statistics"] = parsed["statistics"]
                result["success"] = True
                break

    return result


@app.command()
def main(
    papers: Optional[int] = typer.Option(
        None,
        "--papers",
        "-n",
        help="Number of papers to verify (default: all).",
    ),
    offset: int = typer.Option(
        0,
        "--offset",
        help="Skip first N papers.",
    ),
    output_dir: str = typer.Option(
        "outputs/claude-batch",
        "--output",
        "-o",
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
        False,
        "--dry-run",
        help="List papers without verifying.",
    ),
    paper_id: Optional[str] = typer.Option(
        None,
        "--paper-id",
        help="Verify a single paper by DOI/arXiv ID.",
    ),
    list_papers: bool = typer.Option(
        False,
        "--list",
        help="List available papers in the dataset.",
    ),
    evaluate: bool = typer.Option(
        False,
        "--evaluate",
        help="Run evaluation after batch verification.",
    ),
    parquet_path: str = typer.Option(
        "data/train-00000-of-00001.parquet",
        "--parquet",
        "-p",
        help="Path to the parquet dataset file.",
    ),
    eval_dir: Optional[str] = typer.Option(
        None,
        "--eval-dir",
        help="Run evaluation only on an existing output directory (skips verification).",
    ),
    llm_judge: bool = typer.Option(
        False,
        "--llm-judge",
        "-j",
        help="Use an LLM judge for semantic error matching instead of fuzzy location matching.",
    ),
    judge_model: Optional[str] = typer.Option(
        None,
        "--judge-model",
        help="Model to use for the LLM judge (default: auto-detected from provider).",
    ),
    judge_provider: Optional[str] = typer.Option(
        None,
        "--judge-provider",
        help="LLM provider for the judge: anthropic, openai, or deepseek (default: auto-detect).",
    ),
) -> None:
    """Batch-verify papers from the parquet dataset using Claude Code."""
    # Resolve paths here, after typer has parsed the actual string values
    parquet_full = (
        Path(parquet_path)
        if Path(parquet_path).is_absolute()
        else PROJECT_ROOT / parquet_path
    )
    output_full = (
        Path(output_dir)
        if Path(output_dir).is_absolute()
        else PROJECT_ROOT / output_dir
    )

    # --eval-dir mode: rebuild batch_results.json from existing outputs and evaluate
    if eval_dir:
        eval_full = (
            Path(eval_dir)
            if Path(eval_dir).is_absolute()
            else PROJECT_ROOT / eval_dir
        )
        _eval_existing_directory(
            eval_full, parquet_full, llm_judge, judge_model, judge_provider,
        )
        return

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
        console.print(
            f"\n[bold]🔍 DRY RUN — Would verify {len(paper_list)} papers[/bold]\n"
        )
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
            result = parse_verification_result(output, paper_id, output_file)
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
        task = progress.add_task(f"[cyan]Verifying {total} papers...", total=total)

        for i, paper in enumerate(paper_list, 1):
            pid = paper["paper_id"]
            progress.update(task, description=f"[cyan][{i}/{total}] Verifying {pid}...")

            prompt = build_verify_prompt(pid, mode, threshold)
            output_file = output_full / f"{pid.replace('/', '_')}.md"

            success, output = run_claude_verify(prompt, output_file)

            if success:
                verified += 1
                result = parse_verification_result(output, pid, output_file)
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
    results_file.write_text(
        json.dumps(
            {
                "batch_timestamp": datetime.now().isoformat(),
                "total_papers": total,
                "verified": verified,
                "failed": failed,
                "mode": mode,
                "threshold": threshold,
                "duration_seconds": elapsed,
                "results": all_results,
            },
            indent=2,
        )
    )

    # Summary
    console.print(
        f"\n[bold green]═══════════════════════════════════════════[/bold green]"
    )
    console.print(f"[bold green]Batch verification complete![/bold green]")
    console.print(f"  Total:   {total}")
    console.print(f"  ✅ Done: {verified}")
    console.print(f"  ❌ Failed: {failed}")
    console.print(f"  ⏱️  Time:  {elapsed:.0f}s ({elapsed/60:.1f}m)")
    console.print(f"  📁 Saved: {results_file}")
    console.print(
        f"[bold green]═══════════════════════════════════════════[/bold green]"
    )

    # Optionally evaluate
    if evaluate and verified > 0:
        console.print(f"\n[bold]Running evaluation against ground truth...[/bold]")
        _run_evaluation(
            results_file, parquet_full, output_full,
            llm_judge, judge_model, judge_provider,
        )


def _eval_existing_directory(
    eval_dir: Path, parquet_path: Path,
    llm_judge: bool = False,
    judge_model: Optional[str] = None,
    judge_provider: Optional[str] = None,
) -> None:
    """Rebuild batch_results.json from existing .md + companion JSON files, then evaluate.

    This is useful when you have already run verification and want to re-evaluate
    without re-running Claude Code (e.g. after a fix to the result parser).
    """
    # Index companion JSON files by paper_id
    json_by_paper: dict[str, Path] = {}
    for json_path in sorted(
        set(eval_dir.glob("*report*.json"))
        | set(eval_dir.glob("verification*.json"))
    ):
        skip_names = {
            "batch_results.json", "metrics.json", "predictions.json",
            "confusion_matrix.json",
        }
        if json_path.name in skip_names:
            continue
        try:
            parsed = json.loads(json_path.read_text())
            pid = parsed.get("paper_id", "")
            if pid:
                json_by_paper[pid] = json_path
        except (json.JSONDecodeError, OSError):
            continue

    if not json_by_paper:
        console.print(f"[red]No companion JSON reports found in {eval_dir}[/red]")
        raise typer.Exit(code=1)

    console.print(
        f"\n[bold cyan]🔍 Rebuilding results from {len(json_by_paper)} companion JSON(s) "
        f"in {eval_dir}[/bold cyan]\n"
    )

    paper_list = get_paper_list(parquet_path)
    gt_by_paper = {
        p["paper_id"]: {
            "error_category": p["error_category"],
            "error_location": p["error_location"],
            "error_severity": p["error_severity"],
        }
        for p in paper_list
    }

    all_results = []
    verified = 0
    failed = 0

    for paper_id, json_path in sorted(json_by_paper.items()):
        # Find the matching .md file using the sanitized paper_id
        sanitized = paper_id.replace("/", "_")
        md_file = eval_dir / f"{sanitized}.md"

        if not md_file.exists():
            # Try with dots also replaced (some DOI naming)
            sanitized2 = sanitized.replace(".", "_")
            md_file = eval_dir / f"{sanitized2}.md"

        if not md_file.exists():
            failed += 1
            console.print(f"  [red]✗[/red] {paper_id}: companion JSON exists but no .md file found")
            continue

        text = md_file.read_text()
        result = parse_verification_result(text, paper_id, md_file)

        if not result["success"]:
            failed += 1
            console.print(f"  [red]✗[/red] {paper_id}: failed to parse verification result")
            continue

        verified += 1
        num_errors = len(result.get("predicted_errors", []))
        console.print(
            f"  [green]✓[/green] {paper_id}: "
            f"{num_errors} predicted error(s), verdict={result.get('verdict', '?')}"
        )

        gt = gt_by_paper.get(paper_id)
        if gt:
            result["ground_truth"] = gt
        all_results.append(result)

    # Write the corrected batch_results.json
    results_file = eval_dir / "batch_results.json"
    results_file.write_text(
        json.dumps(
            {
                "batch_timestamp": datetime.now().isoformat(),
                "total_papers": len(json_by_paper),
                "verified": verified,
                "failed": failed,
                "mode": "eval-dir (reconstructed)",
                "threshold": 0.0,
                "duration_seconds": 0,
                "results": all_results,
            },
            indent=2,
        )
    )

    console.print(
        f"\n[bold green]Rebuilt {results_file}[/bold green] "
        f"({verified} ok, {failed} failed)"
    )

    if verified > 0:
        console.print(f"\n[bold]Running evaluation against ground truth...[/bold]")
        _run_evaluation(
            results_file, parquet_path, eval_dir,
            llm_judge, judge_model, judge_provider,
        )


def _run_evaluation(
    results_file: Path, parquet_path: Path, output_dir: Path,
    llm_judge: bool = False,
    judge_model: Optional[str] = None,
    judge_provider: Optional[str] = None,
) -> None:
    """Run evaluation comparing batch results to ground truth.

    When ``llm_judge`` is True, uses an LLM to semantically compare predicted
    errors against ground-truth annotations, which handles paraphrased error
    descriptions and differing location references.  Otherwise falls back to
    fuzzy location-string matching.
    """
    try:
        # Load predictions from the batch results
        batch_data = json.loads(results_file.read_text())
        predictions = []

        for r in batch_data["results"]:
            if not r.get("success"):
                continue
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

        # Build config for LLM judge if requested
        config = None
        if llm_judge:
            from src.config import PipelineConfig, LLMConfig
            import os as _os

            # Auto-detect provider and model from available API keys
            if not judge_provider:
                if _os.environ.get("ANTHROPIC_API_KEY"):
                    judge_provider = "anthropic"
                elif _os.environ.get("DEEPSEEK_API_KEY"):
                    judge_provider = "deepseek"
                elif _os.environ.get("OPENAI_API_KEY"):
                    judge_provider = "openai"
                else:
                    judge_provider = "deepseek"  # will fail with a clear message

            # Sensible model defaults per provider
            if not judge_model:
                defaults = {
                    "anthropic": "claude-sonnet-4-6",
                    "deepseek": "deepseek-v4-pro",
                    "openai": "gpt-4o",
                }
                judge_model = defaults.get(judge_provider, "deepseek-v4-pro")

            api_key_env = {
                "anthropic": "ANTHROPIC_API_KEY",
                "deepseek": "DEEPSEEK_API_KEY",
                "openai": "OPENAI_API_KEY",
            }.get(judge_provider, "DEEPSEEK_API_KEY")

            llm_cfg = LLMConfig(
                provider=judge_provider,
                model=judge_model,
                api_key_env=api_key_env,
            )
            config = PipelineConfig(llm=llm_cfg, use_llm_judge=True)
            config.judge_model = judge_model
            console.print(
                f"\n[bold cyan]🧠 Using LLM judge ({judge_provider}/{judge_model}) "
                f"for semantic error matching[/bold cyan]"
            )
        else:
            console.print(
                f"\n[dim]Using fuzzy location matching "
                f"(pass --llm-judge for semantic matching)[/dim]"
            )

        # Run the standard evaluation
        df = pd.read_parquet(parquet_path)
        from src.evaluation.alignment import match_predictions_to_ground_truth
        from src.evaluation.metrics import evaluate_predictions
        from src.reporting.reporter import generate_report
        from src.parser.schema_analyzer import analyze_dataset_schema

        schema_report = analyze_dataset_schema(parquet_path)
        aligned = match_predictions_to_ground_truth(predictions, df, config=config)
        metrics = evaluate_predictions(aligned, df)

        console.print(f"\n[bold]Evaluation Results:[/bold]")
        console.print(f"  Accuracy:  {metrics.accuracy:.4f}")
        console.print(f"  Precision: {metrics.precision:.4f}")
        console.print(f"  Recall:    {metrics.recall:.4f}")
        console.print(f"  F1 Score:  {metrics.f1_score:.4f}")

        # ── Per-paper breakdown of TP / FP / FN ──
        _print_per_paper_breakdown(aligned, predictions)

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
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")


def _print_per_paper_breakdown(
    aligned: list, predictions: list,
) -> None:
    """Print a per-paper breakdown of true positives, false positives, and false negatives."""
    from collections import defaultdict

    # Group aligned predictions by paper
    by_paper: dict[str, dict] = defaultdict(lambda: {"tp": [], "fp": [], "fn": []})
    for a in aligned:
        if a.is_true_positive:
            by_paper[a.paper_id]["tp"].append(a)
        elif a.is_false_positive:
            by_paper[a.paper_id]["fp"].append(a)
        else:
            by_paper[a.paper_id]["fn"].append(a)

    # Only show papers that have predictions or matched ground truth
    predicted_paper_ids = {p.paper_id for p in predictions}
    relevant = {pid for pid in by_paper if pid in predicted_paper_ids or by_paper[pid]["fn"]}

    if not relevant:
        return

    console.print(f"\n[bold]Per-Paper Breakdown:[/bold]")

    for paper_id in sorted(relevant):
        info = by_paper[paper_id]
        tp_count = len(info["tp"])
        fp_count = len(info["fp"])
        fn_count = len(info["fn"])

        parts = []
        if tp_count:
            parts.append(f"[green]{tp_count} TP[/green]")
        if fp_count:
            parts.append(f"[yellow]{fp_count} FP[/yellow]")
        if fn_count:
            parts.append(f"[red]{fn_count} FN[/red]")

        console.print(f"  [cyan]{paper_id}[/cyan]: {', '.join(parts)}")

        # Show FP details — predictions with no matching ground truth
        for fp in info["fp"]:
            console.print(
                f"    [yellow]FP:[/yellow] [{fp.predicted.error_category}] "
                f"{fp.predicted.error_location[:80]}"
            )
        # Show FN details — ground truth errors that were missed
        for fn in info["fn"]:
            console.print(
                f"    [red]FN:[/red] [{fn.ground_truth_category}] "
                f"{fn.ground_truth_location[:80]}"
            )

    total_fp = sum(len(info["fp"]) for info in by_paper.values() if info["fp"])
    if total_fp:
        console.print(
            f"\n[yellow]⚠ {total_fp} prediction(s) did not match any ground-truth "
            f"error — potential false alarms.[/yellow]"
        )


if __name__ == "__main__":
    app()
