#!/usr/bin/env python3
"""Paperena Verification Pipeline — CLI Entry Point.

Commands:
    analyze         Run dataset schema analysis
    verify          Run full verification on all papers
    verify-one ID   Verify a single paper
    evaluate        Compute evaluation metrics from saved predictions
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from src.config import PipelineConfig, default_config
from src.models import PaperPrediction
from src.parser.content_parser import parse_paper_content
from src.parser.location_parser import parse_error_location
from src.parser.schema_analyzer import analyze_dataset_schema
from src.orchestrator.orchestrator import VerificationOrchestrator
from src.orchestrator.uncertainty_orchestrator import UncertaintyOrchestrator
from src.orchestrator.router import create_default_registry


def build_orchestrator(config, registry=None):
    """Pick the orchestrator implementation for the configured mode."""
    if config.orchestration_mode == "uncertainty":
        return UncertaintyOrchestrator(config=config, registry=registry)
    return VerificationOrchestrator(config=config, registry=registry)
from src.evaluation.alignment import match_predictions_to_ground_truth
from src.evaluation.metrics import evaluate_predictions
from src.reporting.reporter import generate_report
from src.utils.logging import setup_logging

app = typer.Typer(
    name="paperena",
    help="Automated Scientific Paper Verification Pipeline",
)
console = Console()


@app.command()
def analyze(
    parquet_path: str = typer.Option(
        "data/train-00000-of-00001.parquet",
        "--parquet", "-p",
        help="Path to the parquet dataset file.",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output", "-o",
        help="Save schema report to JSON file.",
    ),
    sample_rows: int = typer.Option(
        5,
        "--sample-rows", "-n",
        help="Number of rows to sample for detailed inspection.",
    ),
) -> None:
    """Analyze dataset schema and produce a schema report."""
    setup_logging("INFO")
    console.print("[bold cyan]Paperena Dataset Schema Analyzer[/bold cyan]\n")

    parquet_path_obj = Path(parquet_path)
    if not parquet_path_obj.exists():
        console.print(f"[red]Error: Parquet file not found: {parquet_path}[/red]")
        raise typer.Exit(code=1)

    with console.status("[cyan]Analyzing dataset schema...[/cyan]"):
        report = analyze_dataset_schema(
            parquet_path=parquet_path_obj,
            sample_rows=sample_rows,
        )

    # Display report
    _display_schema_report(report)

    # Optionally save
    if output:
        output_path = Path(output)
        output_path.write_text(report.model_dump_json(indent=2))
        console.print(f"\n[green]Report saved to: {output_path}[/green]")


@app.command()
def verify(
    parquet_path: str = typer.Option(
        "data/train-00000-of-00001.parquet",
        "--parquet", "-p",
        help="Path to the parquet dataset file.",
    ),
    output_dir: str = typer.Option(
        "outputs",
        "--output", "-o",
        help="Directory for output files.",
    ),
    max_papers: Optional[int] = typer.Option(
        None,
        "--max-papers", "-n",
        help="Maximum number of papers to verify (for testing).",
    ),
    decode_images: bool = typer.Option(
        False,
        "--decode-images/--no-decode-images",
        help="Whether to decode base64 images to files.",
    ),
    skip_evaluation: bool = typer.Option(
        False,
        "--skip-evaluation",
        help="Skip evaluation and reporting (verify only).",
    ),
    workers: int = typer.Option(
        8,
        "--workers", "-w",
        help="Number of concurrent API workers per paper (1 = sequential).",
    ),
    strictness: str = typer.Option(
        "strict",
        "--strictness",
        help="Error sensitivity: 'strict' (only erratum/retraction-worthy errors) "
             "or 'lenient' (original broader prompts).",
    ),
    mode: str = typer.Option(
        "exhaustive",
        "--mode",
        help="Orchestration: 'exhaustive' (route every snippet by type) or "
             "'uncertainty' (triage first, route specialists by error density).",
    ),
    uncertainty_threshold: float = typer.Option(
        0.30,
        "--uncertainty-threshold",
        help="In --mode uncertainty: escalate snippets at/above this score.",
    ),
    llm_only: Optional[str] = typer.Option(
        None,
        "--llm-only",
        help="Bypass specialist verifiers: use a single LLM for all snippets. "
             "'same-prompt' (default) uses one unified prompt for every snippet type; "
             "'separate-prompts' uses different prompts per type (text, math, figure, "
             "citation). Omit to use normal specialist verifiers.",
    ),
    eval_category: Optional[str] = typer.Option(
        None,
        "--eval-category",
        help="Filter evaluation to a single error category (e.g., 'Equation / proof'). "
             "Only predictions and ground-truth entries in this category are counted. "
             "Omit to evaluate all categories.",
    ),
    parser_mode: str = typer.Option(
        "regex",
        "--parser-mode",
        help="Parser mode: 'regex' (default, regex-based) or 'llm' (LLM-based "
             "with context assembly and verifier attribution).",
    ),
) -> None:
    """Run full verification pipeline on all papers."""
    setup_logging("INFO")
    console.print("[bold cyan]Paperena Verification Pipeline[/bold cyan]\n")

    # Load dataset
    parquet_path_obj = Path(parquet_path)
    if not parquet_path_obj.exists():
        console.print(f"[red]Error: Parquet file not found: {parquet_path}[/red]")
        raise typer.Exit(code=1)

    df = pd.read_parquet(parquet_path_obj)
    if max_papers:
        df = df.head(max_papers)

    console.print(f"Loaded [bold]{len(df)}[/bold] papers from dataset\n")

    # Schema analysis
    with console.status("[cyan]Analyzing schema...[/cyan]"):
        schema_report = analyze_dataset_schema(parquet_path_obj)

    # Initialize orchestrator
    if strictness not in ("strict", "lenient"):
        console.print(f"[red]Invalid --strictness: {strictness!r} (use 'strict' or 'lenient')[/red]")
        raise typer.Exit(code=1)
    if mode not in ("exhaustive", "uncertainty"):
        console.print(f"[red]Invalid --mode: {mode!r} (use 'exhaustive' or 'uncertainty')[/red]")
        raise typer.Exit(code=1)
    config = PipelineConfig(strictness=strictness)
    config.llm.num_workers = workers
    config.orchestration_mode = mode
    config.uncertainty_threshold = uncertainty_threshold
    if llm_only is not None and llm_only not in ("same-prompt", "separate-prompts"):
        console.print(f"[red]Invalid --llm-only: {llm_only!r} (use 'same-prompt' or 'separate-prompts')[/red]")
        raise typer.Exit(code=1)
    if parser_mode not in ("regex", "llm"):
        console.print(f"[red]Invalid --parser-mode: {parser_mode!r} (use 'regex' or 'llm')[/red]")
        raise typer.Exit(code=1)
    config.llm_only_mode = llm_only
    config.eval_category_filter = eval_category
    config.parser_mode = parser_mode
    llm_tag = f" | LLM-only: [bold]{llm_only}[/bold]" if llm_only else ""
    parser_tag = f" | Parser: [bold]{parser_mode}[/bold]"
    cat_tag = f" | Category filter: [bold]{eval_category}[/bold]" if eval_category else ""
    console.print(
        f"Strictness: [bold]{strictness}[/bold] | Workers: [bold]{workers}[/bold] | "
        f"Mode: [bold]{mode}[/bold]{llm_tag}{parser_tag}{cat_tag}\n"
    )
    registry = create_default_registry()
    orchestrator = build_orchestrator(config, registry=registry)

    # Process each paper
    predictions: list[PaperPrediction] = []
    image_dir = Path(output_dir) / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            "[cyan]Verifying papers...", total=len(df)
        )

        for _, row in df.iterrows():
            paper_id = str(row["doi/arxiv_id"])
            title = row["title"]
            paper_category = row["paper_category"]
            paper_content = row["paper_content"]

            progress.update(task, description=f"[cyan]Verifying {paper_id}...")

            try:
                # Parse paper (parser mode selects the path)
                if config.parser_mode == "llm":
                    from src.parser.llm_content_parser import llm_parse_paper
                    paper = llm_parse_paper(
                        paper_id=paper_id,
                        title=title,
                        paper_category=paper_category,
                        paper_content=paper_content,
                        config=config,
                        decode_images=decode_images,
                        image_output_dir=str(image_dir) if decode_images else None,
                    )
                else:
                    paper = parse_paper_content(
                        paper_id=paper_id,
                        title=title,
                        paper_category=paper_category,
                        paper_content=paper_content,
                        decode_images=decode_images,
                        image_output_dir=image_dir if decode_images else None,
                    )

                # Verify
                prediction = orchestrator.run(paper)
                predictions.append(prediction)

            except Exception as exc:
                logger.error(f"Failed to verify paper {paper_id}: {exc}")
                predictions.append(PaperPrediction(
                    paper_id=paper_id,
                    title=title,
                    paper_category=paper_category,
                ))

            progress.advance(task)

    console.print(f"\n[green]Verified {len(predictions)} papers[/green]")

    # Save raw predictions
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    pred_file = output_path / "raw_predictions.json"
    pred_file.write_text(
        json.dumps([p.model_dump() for p in predictions], indent=2, default=str)
    )
    console.print(f"[green]Raw predictions saved to: {pred_file}[/green]")

    # Evaluation
    if not skip_evaluation:
        console.print("\n[bold]Running evaluation...[/bold]")
        aligned = match_predictions_to_ground_truth(predictions, df, config=config)
        metrics = evaluate_predictions(aligned, df)

        _display_metrics(metrics)

        # Generate full report
        report_dir = generate_report(
            metrics=metrics,
            aligned_predictions=aligned,
            predictions=predictions,
            schema_report=schema_report,
            output_dir=output_path,
            ground_truth_df=df,
        )
        console.print(f"\n[bold green]Reports generated in: {report_dir}[/bold green]")


@app.command()
def verify_one(
    paper_id: str = typer.Argument(..., help="Paper DOI/arXiv ID to verify."),
    parquet_path: str = typer.Option(
        "data/train-00000-of-00001.parquet",
        "--parquet", "-p",
    ),
    decode_images: bool = typer.Option(
        True,
        "--decode-images/--no-decode-images",
    ),
    workers: int = typer.Option(
        8,
        "--workers", "-w",
        help="Number of concurrent API workers (1 = sequential).",
    ),
    strictness: str = typer.Option(
        "strict",
        "--strictness",
        help="Error sensitivity: 'strict' or 'lenient'.",
    ),
    mode: str = typer.Option(
        "exhaustive",
        "--mode",
        help="Orchestration: 'exhaustive' or 'uncertainty'.",
    ),
    uncertainty_threshold: float = typer.Option(
        0.30,
        "--uncertainty-threshold",
        help="In --mode uncertainty: escalate snippets at/above this score.",
    ),
    llm_only: Optional[str] = typer.Option(
        None,
        "--llm-only",
        help="Bypass specialist verifiers: use a single LLM for all snippets. "
             "'same-prompt' (default) uses one unified prompt; "
             "'separate-prompts' uses different prompts per type. "
             "Omit to use normal specialist verifiers.",
    ),
    parser_mode: str = typer.Option(
        "regex",
        "--parser-mode",
        help="Parser mode: 'regex' (default) or 'llm' (LLM-based with context).",
    ),
) -> None:
    """Verify a single paper by ID."""
    setup_logging("DEBUG")
    console.print(f"[bold cyan]Verifying single paper: {paper_id}[/bold cyan]\n")

    parquet_path_obj = Path(parquet_path)
    df = pd.read_parquet(parquet_path_obj)

    # Find the paper
    row = df[df["doi/arxiv_id"].astype(str) == paper_id]
    if len(row) == 0:
        console.print(f"[red]Paper not found: {paper_id}[/red]")
        raise typer.Exit(code=1)

    row = row.iloc[0]
    paper_content = row["paper_content"]

    # Validate parser mode early
    if parser_mode not in ("regex", "llm"):
        console.print(f"[red]Invalid --parser-mode: {parser_mode!r} (use 'regex' or 'llm')[/red]")
        raise typer.Exit(code=1)

    with console.status("[cyan]Parsing paper...[/cyan]"):
        if parser_mode == "llm":
            from src.parser.llm_content_parser import llm_parse_paper
            paper = llm_parse_paper(
                paper_id=paper_id,
                title=row["title"],
                paper_category=row["paper_category"],
                paper_content=paper_content,
                decode_images=decode_images,
                image_output_dir=str(Path("outputs/images")) if decode_images else None,
            )
        else:
            paper = parse_paper_content(
                paper_id=paper_id,
                title=row["title"],
                paper_category=row["paper_category"],
                paper_content=paper_content,
                decode_images=decode_images,
                image_output_dir=Path("outputs/images") if decode_images else None,
            )

    if parser_mode == "llm":
        n_verifiable = sum(1 for u in paper.verifiable_units if u.is_verifiable)  # type: ignore[union-attr]
        n_unverifiable = sum(1 for u in paper.verifiable_units if not u.is_verifiable)  # type: ignore[union-attr]
        console.print(f"  LLM Verifiable Units: {n_verifiable} (skipped {n_unverifiable} unverifiable)")
        console.print(f"  LLM Symbol Registry: {len(paper.symbol_registry)} symbols")  # type: ignore[union-attr]
        console.print(f"  Images: {len(paper.images)}")
        # Show verifier route distribution
        routes: dict[str, int] = {}
        for u in paper.verifiable_units:  # type: ignore[union-attr]
            if u.is_verifiable and u.verifier_route:
                routes[u.verifier_route] = routes.get(u.verifier_route, 0) + 1
        if routes:
            console.print(f"  Verifier Routes: {dict(sorted(routes.items()))}")
    else:
        console.print(f"  Sections: {len(paper.sections)}")
        console.print(f"  Equations: {len(paper.equations)}")
        console.print(f"  Images: {len(paper.images)}")
        console.print(f"  Tables: {len(paper.tables)}")
        console.print(f"  Theorems: {len(paper.theorems)}")

    # Verify
    if strictness not in ("strict", "lenient"):
        console.print(f"[red]Invalid --strictness: {strictness!r} (use 'strict' or 'lenient')[/red]")
        raise typer.Exit(code=1)
    if mode not in ("exhaustive", "uncertainty"):
        console.print(f"[red]Invalid --mode: {mode!r} (use 'exhaustive' or 'uncertainty')[/red]")
        raise typer.Exit(code=1)
    config = PipelineConfig(strictness=strictness)
    config.llm.num_workers = workers
    config.orchestration_mode = mode
    config.uncertainty_threshold = uncertainty_threshold
    config.parser_mode = parser_mode
    if llm_only is not None and llm_only not in ("same-prompt", "separate-prompts"):
        console.print(f"[red]Invalid --llm-only: {llm_only!r} (use 'same-prompt' or 'separate-prompts')[/red]")
        raise typer.Exit(code=1)
    config.llm_only_mode = llm_only
    parser_tag = f" | Parser: {parser_mode}" if parser_mode != "regex" else ""
    llm_tag = f" | LLM-only: {llm_only}" if llm_only else ""
    console.print(f"[dim]Mode: {mode}{llm_tag}{parser_tag}[/dim]")
    orchestrator = build_orchestrator(config)
    prediction = orchestrator.run(paper)

    console.print(f"\n[bold]Results for {paper_id}:[/bold]")
    console.print(f"  Snippets: {prediction.total_snippets}")
    console.print(f"  Verified (specialist checks): {prediction.snippets_verified}")
    console.print(f"  Errors detected: {prediction.errors_detected}")

    # Uncertainty-mode: show the region-level map and where specialists fired.
    if prediction.uncertainty_map:
        from collections import defaultdict as _dd
        from src.models import TriageResult as _TR
        from src.orchestrator.uncertainty_orchestrator import (
            UncertaintyOrchestrator as _UO,
        )

        regions: dict = _dd(list)
        for entry in prediction.uncertainty_map:
            region = _UO._region_of(_TR(**entry))
            regions[region].append(entry.get("uncertainty", 0.0))

        console.print(
            f"\n[bold]Uncertainty map[/bold] (threshold={uncertainty_threshold:.2f}):"
        )
        for region, scores in sorted(
            regions.items(), key=lambda kv: max(kv[1]), reverse=True
        ):
            mean = sum(scores) / len(scores)
            console.print(
                f"  {region:<22} mean={mean:0.2f}  max={max(scores):0.2f}  n={len(scores)}"
            )
        escalated = [e for e in prediction.uncertainty_map if e.get("selected")]
        console.print(
            f"\n  Escalated to specialists: [bold]{len(escalated)}[/bold] / "
            f"{len(prediction.uncertainty_map)} snippets"
        )
        for e in sorted(escalated, key=lambda x: x.get("uncertainty", 0), reverse=True)[:15]:
            console.print(
                f"    {e['snippet_id']:<26} u={e.get('uncertainty',0):0.2f} "
                f"→ {e.get('routed_to') or e.get('suggested_route')}"
            )

    if prediction.predicted_errors:
        console.print(f"\n[bold]Predicted Errors:[/bold]")
        for pe in prediction.predicted_errors:
            console.print(
                f"  - [{pe.error_category}] {pe.error_location} "
                f"(confidence: {pe.confidence:.2f})"
            )
            console.print(f"    {pe.supporting_evidence[:200]}")

    # Compare to ground truth
    console.print(f"\n[bold]Ground Truth:[/bold]")
    console.print(f"  Error category: {row['error_category']}")
    console.print(f"  Error location: {row['error_location']}")
    console.print(f"  Error severity: {row['error_severity']}")
    console.print(f"  Annotation: {str(row['error_annotation'])[:300]}")

    # Parse ground truth location
    gt_loc = parse_error_location(row["error_location"])
    console.print(f"\n[bold]Parsed Location:[/bold] {gt_loc.normalized}")


@app.command()
def evaluate(
    predictions_path: str = typer.Option(
        "outputs/raw_predictions.json",
        "--predictions", "-p",
    ),
    parquet_path: str = typer.Option(
        "data/train-00000-of-00001.parquet",
        "--parquet", "-d",
    ),
    output_dir: str = typer.Option(
        "outputs",
        "--output", "-o",
    ),
    use_llm_judge: bool = typer.Option(
        True,
        "--llm-judge/--no-llm-judge",
        help="Match predictions to ground truth with an LLM judge (the original "
             "benchmark's strategy). Disable to use structural fuzzy matching.",
    ),
    judge_model: Optional[str] = typer.Option(
        None,
        "--judge-model",
        help="Model to use as the LLM judge (defaults to the configured reviewer model).",
    ),
    eval_category: Optional[str] = typer.Option(
        None,
        "--eval-category",
        help="Filter evaluation to a single error category (e.g., 'Equation / proof').",
    ),
) -> None:
    """Compute evaluation metrics from saved predictions."""
    setup_logging("INFO")
    console.print("[bold cyan]Paperena Evaluation[/bold cyan]\n")

    config = PipelineConfig()
    config.use_llm_judge = use_llm_judge
    config.judge_model = judge_model
    config.eval_category_filter = eval_category
    if eval_category:
        console.print(f"Category filter: [bold]{eval_category}[/bold]\n")

    # Load predictions
    pred_path = Path(predictions_path)
    if not pred_path.exists():
        console.print(f"[red]Predictions file not found: {predictions_path}[/red]")
        raise typer.Exit(code=1)

    raw_data = json.loads(pred_path.read_text())
    predictions = [PaperPrediction(**p) for p in raw_data]

    # Load ground truth
    parquet_path_obj = Path(parquet_path)
    if not parquet_path_obj.exists():
        console.print(f"[red]Parquet file not found: {parquet_path}[/red]")
        raise typer.Exit(code=1)

    df = pd.read_parquet(parquet_path_obj)

    # Schema analysis
    schema_report = analyze_dataset_schema(parquet_path_obj)

    # Align and evaluate
    console.print(f"Loaded {len(predictions)} predictions\n")

    judge_desc = "LLM judge" if config.use_llm_judge else "fuzzy location matching"
    with console.status(f"[cyan]Aligning predictions to ground truth ({judge_desc})...[/cyan]"):
        aligned = match_predictions_to_ground_truth(predictions, df, config=config)

    with console.status("[cyan]Computing metrics...[/cyan]"):
        metrics = evaluate_predictions(aligned, df)

    _display_metrics(metrics)

    # Generate full report
    report_dir = generate_report(
        metrics=metrics,
        aligned_predictions=aligned,
        predictions=predictions,
        schema_report=schema_report,
        output_dir=Path(output_dir),
        ground_truth_df=df,
    )
    console.print(f"\n[bold green]Reports generated in: {report_dir}[/bold green]")


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _display_schema_report(report) -> None:
    """Display schema report in a Rich table."""
    console.print(f"\n[bold]Dataset Overview:[/bold]")
    console.print(f"  Rows: {report.total_rows}")
    console.print(f"  Columns: {report.total_columns}")

    console.print(f"\n[bold]Content Types:[/bold]")
    table = Table(title="Paper Content Structure")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Content types", ", ".join(report.content_types))
    table.add_row("Keys found", ", ".join(report.keys_found))
    table.add_row("Text items", str(report.text_item_count))
    table.add_row("Image items", str(report.image_item_count))
    table.add_row("Papers with images", str(report.rows_with_images))
    table.add_row("Papers with local content", str(report.rows_with_local_content))

    console.print(table)

    console.print(f"\n[bold]Error Categories:[/bold]")
    for cat in report.error_categories:
        console.print(f"  {cat['category']}: {cat['count']}")

    console.print(f"\n[bold]Error Severities:[/bold]")
    for sev in report.error_severities:
        console.print(f"  {sev['severity']}: {sev['count']}")


def _display_metrics(metrics) -> None:
    """Display evaluation metrics in a Rich table."""
    table = Table(title="Overall Evaluation Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Accuracy", f"{metrics.accuracy:.4f}")
    table.add_row("Precision", f"{metrics.precision:.4f}")
    table.add_row("Recall", f"{metrics.recall:.4f}")
    table.add_row("F1 Score", f"{metrics.f1_score:.4f}")
    table.add_row("True Positives", str(metrics.true_positives))
    table.add_row("False Positives", str(metrics.false_positives))
    table.add_row("False Negatives", str(metrics.false_negatives))

    console.print(table)

    # Category-level
    if metrics.by_error_category:
        console.print(f"\n[bold]By Error Category:[/bold]")
        for cm in metrics.by_error_category:
            console.print(
                f"  {cm.category_name}: P={cm.precision:.3f} "
                f"R={cm.recall:.3f} F1={cm.f1_score:.3f}"
            )


@app.command()
def processbench(
    split: str = typer.Option(
        "gsm8k",
        "--split", "-s",
        help="ProcessBench split to evaluate: gsm8k, math, olympiadbench, omnimath.",
    ),
    max_cases: Optional[int] = typer.Option(
        None,
        "--max-cases", "-n",
        help="Maximum number of cases to process (for testing).",
    ),
    workers: int = typer.Option(
        1,
        "--workers", "-w",
        help="Number of concurrent workers (1 = sequential for progressive math).",
    ),
    strictness: str = typer.Option(
        "strict",
        "--strictness",
        help="Error sensitivity: 'strict' or 'lenient'.",
    ),
    output_dir: str = typer.Option(
        "outputs/processbench",
        "--output", "-o",
        help="Directory for output files.",
    ),
) -> None:
    """Evaluate on ProcessBench — identify errors in CoT math reasoning steps.

    ProcessBench (Zheng et al., ACL 2025) measures a model's ability to find
    the first erroneous step in a chain-of-thought mathematical solution.
    This command runs the progressive math verifier on each step and compares
    predictions against the human-annotated labels.
    """
    setup_logging("INFO")
    console.print(
        "[bold cyan]ProcessBench — CoT Math Error Identification[/bold cyan]\n"
    )

    try:
        from src.datasets.processbench import (
            ProcessBenchResult,
            case_to_snippets,
            compute_processbench_metrics,
            describe_split,
            load_processbench,
        )
    except ImportError as exc:
        console.print(f"[red]Failed to import ProcessBench loader: {exc}[/red]")
        console.print(
            "[yellow]Install with: pip install datasets[/yellow]"
        )
        raise typer.Exit(code=1)

    info = describe_split(split)
    if not info:
        console.print(
            f"[red]Unknown split: {split!r}. "
            f"Use: gsm8k, math, olympiadbench, omnimath[/red]"
        )
        raise typer.Exit(code=1)

    console.print(
        f"Split: [bold]{info['name']}[/bold] ({info['difficulty']}) | "
        f"Expected: {info['n_cases']} cases\n"
    )

    # Load dataset
    with console.status(f"[cyan]Loading ProcessBench/{split}...[/cyan]"):
        cases = load_processbench(split)
    if max_cases:
        cases = cases[:max_cases]
    console.print(f"Loaded [bold]{len(cases)}[/bold] test cases\n")

    # Initialize the progressive math verifier
    from src.orchestrator.router import create_default_registry
    from src.verifiers.progressive.progressive_verifier import ProgressiveMathVerifier

    config = PipelineConfig(strictness=strictness)
    config.llm.num_workers = workers
    config.use_progressive_math = True
    verifier = ProgressiveMathVerifier(config=config)

    results: list[ProcessBenchResult] = []
    n_correct = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"[cyan]Verifying {split} cases...", total=len(cases)
        )

        for case in cases:
            progress.update(
                task,
                description=f"[cyan]Case {case.id} ({case.n_steps} steps)...",
            )

            try:
                snippets = case_to_snippets(case)
                step_predictions: list[dict] = []
                predicted_error_idx = -1

                # Verify each step sequentially (context accumulates)
                for snip_dict in snippets:
                    from src.models import VerificationSnippet, SnippetType

                    snippet = VerificationSnippet(
                        snippet_id=snip_dict["snippet_id"],
                        snippet_type=SnippetType.PARAGRAPH,
                        paper_id=snip_dict["paper_id"],
                        location=snip_dict["location"],
                        content=snip_dict["content"],
                        metadata=snip_dict["metadata"],
                    )

                    if not verifier.can_verify(snippet):
                        step_predictions.append({
                            "step_index": snip_dict["metadata"]["step_index"],
                            "status": "skipped",
                            "confidence": 0.0,
                            "reasoning": "No verifiable math in step.",
                        })
                        continue

                    vr = verifier.verify(snippet)
                    is_error = vr.error_detected or vr.status.value == "INVALID"
                    step_predictions.append({
                        "step_index": snip_dict["metadata"]["step_index"],
                        "status": vr.status.value,
                        "confidence": vr.confidence,
                        "reasoning": vr.reasoning,
                        "statement_class": vr.statement_class,
                    })

                    if is_error and predicted_error_idx < 0:
                        predicted_error_idx = snip_dict["metadata"]["step_index"]

                # Clean up context for this case
                verifier.cleanup_paper(case.id)

                is_correct = (predicted_error_idx == case.label)
                if is_correct:
                    n_correct += 1

                results.append(ProcessBenchResult(
                    case_id=case.id,
                    problem=case.problem,
                    n_steps=case.n_steps,
                    true_label=case.label,
                    predicted_label=predicted_error_idx,
                    is_correct=is_correct,
                    step_predictions=step_predictions,
                ))

            except Exception as exc:
                logger.error(f"Failed on case {case.id}: {exc}")
                results.append(ProcessBenchResult(
                    case_id=case.id,
                    problem=case.problem,
                    n_steps=case.n_steps,
                    true_label=case.label,
                    predicted_label=-1,
                    is_correct=False,
                    step_predictions=[],
                ))

            progress.advance(task)

    # Compute and display metrics
    metrics = compute_processbench_metrics(results, split_name=split)

    console.print(f"\n[bold]Results for {info['name']}:[/bold]")
    console.print(f"  Accuracy: [bold]{metrics.accuracy:.3f}[/bold] "
                  f"({metrics.correct_predictions}/{metrics.total_cases})")
    console.print(f"  Correct on error cases: {metrics.correct_with_error}")
    console.print(f"  Correct on all-correct cases: {metrics.correct_all_correct}")
    console.print(f"  False positives: {metrics.false_positives}")
    console.print(f"  False negatives: {metrics.false_negatives}")

    if metrics.position_accuracy:
        console.print(f"\n[bold]Accuracy by error position:[/bold]")
        for pos, acc in sorted(metrics.position_accuracy.items()):
            label_name = "all-correct" if pos == -1 else f"step {pos}"
            console.print(f"  {label_name}: {acc:.3f}")

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    import json as _json
    results_file = output_path / f"processbench_{split}_results.json"
    results_file.write_text(_json.dumps(
        [
            {
                "case_id": r.case_id,
                "true_label": r.true_label,
                "predicted_label": r.predicted_label,
                "is_correct": r.is_correct,
                "step_predictions": r.step_predictions,
            }
            for r in results
        ],
        indent=2,
    ))
    console.print(f"\n[green]Results saved to: {results_file}[/green]")

    metrics_file = output_path / f"processbench_{split}_metrics.json"
    metrics_file.write_text(_json.dumps({
        "split": metrics.split_name,
        "total_cases": metrics.total_cases,
        "accuracy": metrics.accuracy,
        "correct_predictions": metrics.correct_predictions,
        "correct_with_error": metrics.correct_with_error,
        "correct_all_correct": metrics.correct_all_correct,
        "false_positives": metrics.false_positives,
        "false_negatives": metrics.false_negatives,
    }, indent=2))
    console.print(f"[green]Metrics saved to: {metrics_file}[/green]")


def main() -> None:
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
