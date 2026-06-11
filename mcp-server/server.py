#!/usr/bin/env python3
"""Paperena MCP Server — deterministic verification tools for Claude Code.

Exposes the non-LLM components of the paper verification pipeline as MCP tools:
parsing, segmentation, SymPy sandbox execution, safe arithmetic, and dataset
access. Claude Code skills invoke these tools for deterministic checks, while
keeping the LLM-based reasoning inside Claude itself.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Ensure the project root is on sys.path so we can import from src/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("paperena")


# ---------------------------------------------------------------------------
# Imports from the existing codebase (delay-loaded inside tools to keep
# startup fast — imports happen on first tool invocation).
# ---------------------------------------------------------------------------

_IMPORT_CACHE: dict[str, Any] = {}


def _lazy_import(module_path: str, name: str) -> Any:
    """Import a symbol lazily, caching it for subsequent calls."""
    key = f"{module_path}:{name}"
    if key not in _IMPORT_CACHE:
        mod = __import__(module_path, fromlist=[name])
        _IMPORT_CACHE[key] = getattr(mod, name)
    return _IMPORT_CACHE[key]


# ---------------------------------------------------------------------------
# Tool: parse_paper
# ---------------------------------------------------------------------------


@mcp.tool()
def parse_paper(
    paper_id: str,
    title: str,
    paper_category: str,
    paper_content: list[dict[str, Any]],
    decode_images: bool = False,
) -> dict[str, Any]:
    """Parse raw paper_content (list of dicts with type/text/image_url) into a
    structured paper with sections, equations, images, tables, and theorems.

    Args:
        paper_id: Unique paper identifier (DOI/arXiv ID).
        title: Paper title.
        paper_category: Scientific field/category.
        paper_content: Raw content list from the dataset.
        decode_images: If True, decode base64 images to files (slow).

    Returns:
        Dict with keys: paper_id, title, paper_category, sections (list),
        equations (list), images (list), tables (list), theorems (list),
        and summary counts.
    """
    parse_paper_content = _lazy_import("src.parser.content_parser", "parse_paper_content")

    paper = parse_paper_content(
        paper_id=paper_id,
        title=title,
        paper_category=paper_category,
        paper_content=paper_content,
        decode_images=decode_images,
        image_output_dir=None,
    )

    return {
        "paper_id": paper.paper_id,
        "title": paper.title,
        "paper_category": paper.paper_category,
        "sections": [
            {
                "id": s.id,
                "section_title": s.section_title,
                "section_level": s.section_level,
                "content_preview": s.content[:500],
                "content_length": len(s.content),
            }
            for s in paper.sections
        ],
        "equations": [
            {
                "id": e.id,
                "equation_label": e.equation_label,
                "latex": e.latex,
                "display_mode": e.display_mode,
            }
            for e in paper.equations
        ],
        "images": [
            {
                "id": i.id,
                "caption": i.caption,
                "has_image_file": i.image_path is not None,
            }
            for i in paper.images
        ],
        "tables": [
            {
                "id": t.id,
                "caption": t.caption,
                "row_count": len(t.rows) if t.rows else 0,
            }
            for t in paper.tables
        ],
        "theorems": [
            {
                "id": th.id,
                "theorem_type": th.theorem_type,
                "label": th.label,
                "has_proof": th.proof is not None,
                "statement_preview": th.statement[:300],
            }
            for th in paper.theorems
        ],
        "summary": {
            "n_sections": len(paper.sections),
            "n_equations": len(paper.equations),
            "n_images": len(paper.images),
            "n_tables": len(paper.tables),
            "n_theorems": len(paper.theorems),
        },
    }


# ---------------------------------------------------------------------------
# Tool: segment_paper
# ---------------------------------------------------------------------------


@mcp.tool()
def segment_paper(
    paper_id: str,
    title: str,
    paper_category: str,
    paper_content: list[dict[str, Any]],
) -> dict[str, Any]:
    """Segment a paper into verification snippets (SECTION, EQUATION, FIGURE,
    TABLE, THEOREM, LEMMA, etc.) suitable for per-snippet verification.

    First parses the paper, then segments it. Each snippet has a type, location,
    and content ready for a verifier to analyze.

    Args:
        paper_id: Unique paper identifier.
        title: Paper title.
        paper_category: Scientific field/category.
        paper_content: Raw content list from the dataset.

    Returns:
        Dict with paper_id, title, total_snippets, and a list of snippets each
        containing snippet_id, snippet_type, location, content (truncated to
        3000 chars), and metadata.
    """
    parse_paper_content = _lazy_import("src.parser.content_parser", "parse_paper_content")
    segment = _lazy_import("src.segmentation.segmenter", "segment_paper")

    paper = parse_paper_content(
        paper_id=paper_id,
        title=title,
        paper_category=paper_category,
        paper_content=paper_content,
        decode_images=False,
        image_output_dir=None,
    )

    snippets = segment(paper)

    return {
        "paper_id": paper_id,
        "title": title,
        "total_snippets": len(snippets),
        "snippets": [
            {
                "snippet_id": s.snippet_id,
                "snippet_type": s.snippet_type.value,
                "location": s.location,
                "content": s.content[:3000],
                "content_length": s.content_length,
                "metadata": s.metadata,
            }
            for s in snippets
        ],
    }


# ---------------------------------------------------------------------------
# Tool: run_sympy_check
# ---------------------------------------------------------------------------


@mcp.tool()
def run_sympy_check(
    latex: str,
    equation_context: str = "",
) -> dict[str, Any]:
    """Verify a LaTeX equation by generating SymPy code via LLM prompt
    instructions (the caller provides the LLM-generated code) and executing it
    in a sandboxed subprocess, then interpreting the deterministic verdict.

    This tool runs ONLY the sandbox execution — the LLM must first convert the
    LaTeX to SymPy code following the math-verifier conventions. Use together
    with the /verify-math skill which handles the LLM→code step.

    Args:
        latex: The LaTeX equation to check.
        equation_context: Optional surrounding text for context.

    Returns:
        Dict with verdict, residual, reasoning, and sandbox output.
    """
    # This is the determinisic half: given LLM-generated SymPy code, run it.
    # The actual LLM→code generation happens in the /verify-math skill.
    # Here we just expose the sandbox for direct use.
    run_sympy_sandbox = _lazy_import("src.utils.sandbox", "run_sympy_sandbox")
    SandboxTimeoutError = _lazy_import("src.utils.sandbox", "SandboxTimeoutError")
    SandboxError = _lazy_import("src.utils.sandbox", "SandboxError")

    return {
        "note": "This tool runs the sandbox. Generate SymPy code first using the math-verifier conventions, "
                "then call run_sympy_sandbox_exec with that code.",
        "latex": latex,
        "context": equation_context[:500] if equation_context else "",
    }


@mcp.tool()
def run_sympy_sandbox_exec(
    sympy_code: str,
    harness: str = "",
    timeout_seconds: int = 10,
) -> dict[str, Any]:
    """Execute SymPy code in a sandboxed subprocess and return the raw output.

    The code is prepended with the standard Paperena verdict harness (unless a
    custom harness is provided). The harness deterministically classifies the
    equation residual and emits a machine-readable VERDICT: line.

    Args:
        sympy_code: Python/SymPy code to execute (should call report() or
            report_unverifiable()).
        harness: Optional custom verdict harness. If empty, the default
            Paperena harness is used.
        timeout_seconds: Max execution time (default 10).

    Returns:
        Dict with stdout, stderr, returncode, and parsed verdict if found.
    """
    run_sympy_sandbox = _lazy_import("src.utils.sandbox", "run_sympy_sandbox")
    SandboxTimeoutError = _lazy_import("src.utils.sandbox", "SandboxTimeoutError")
    SandboxError = _lazy_import("src.utils.sandbox", "SandboxError")

    # Use the standard harness from the math verifier
    if not harness:
        from src.verifiers.math_verifier import _VERDICT_HARNESS

        harness = _VERDICT_HARNESS

    full_code = harness + "\n\n" + sympy_code

    try:
        stdout, stderr, returncode = run_sympy_sandbox(
            code=full_code,
            timeout_seconds=timeout_seconds,
        )
    except SandboxTimeoutError:
        return {
            "stdout": "",
            "stderr": "Execution timed out",
            "returncode": -1,
            "verdict": {"verdict": "TIMEOUT"},
            "error": "Sandbox execution timed out",
        }
    except SandboxError as exc:
        return {
            "stdout": "",
            "stderr": str(exc),
            "returncode": -1,
            "verdict": {"verdict": "SANDBOX_ERROR"},
            "error": str(exc),
        }
    except Exception as exc:
        return {
            "stdout": "",
            "stderr": str(exc),
            "returncode": -1,
            "verdict": {"verdict": "ERROR"},
            "error": str(exc),
        }

    # Parse the verdict from stdout
    verdict = None
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("VERDICT:"):
            try:
                verdict = json.loads(line[len("VERDICT:"):])
            except json.JSONDecodeError:
                continue

    return {
        "stdout": stdout[:2000],
        "stderr": stderr[:1000],
        "returncode": returncode,
        "verdict": verdict,
        "success": returncode == 0 and verdict is not None,
    }


# ---------------------------------------------------------------------------
# Tool: safe_arithmetic_eval
# ---------------------------------------------------------------------------


@mcp.tool()
def safe_arithmetic_eval(expression: str) -> dict[str, Any]:
    """Safely evaluate an arithmetic expression using the Paperena safe evaluator.

    Supports: +, -, *, /, **, %, sqrt, log, ln, log10, log2, exp, abs, round,
    min, max, sum, mean, floor, ceil, pow, pi, e.

    Args:
        expression: A closed arithmetic expression with only literal numbers
            and the supported operators/functions. No variable names allowed.

    Returns:
        Dict with computed value or error if evaluation fails.
    """
    safe_eval = _lazy_import("src.utils.safe_arithmetic", "safe_eval")
    ArithmeticEvalError = _lazy_import("src.utils.safe_arithmetic", "ArithmeticEvalError")

    try:
        result = safe_eval(expression)
        return {
            "expression": expression,
            "computed": result,
            "success": True,
        }
    except ArithmeticEvalError as exc:
        return {
            "expression": expression,
            "computed": None,
            "success": False,
            "error": str(exc),
        }
    except Exception as exc:
        return {
            "expression": expression,
            "computed": None,
            "success": False,
            "error": f"Unexpected error: {exc}",
        }


# ---------------------------------------------------------------------------
# Tool: check_numeric_claim
# ---------------------------------------------------------------------------


@mcp.tool()
def check_numeric_claim(expression: str, expected: float, tolerance: float = 0.01) -> dict[str, Any]:
    """Evaluate an arithmetic expression and compare it to an expected value.

    This is the deterministic step of statistical verification: given an
    expression extracted from paper text and the expected value, re-compute
    and check whether they match within tolerance.

    Args:
        expression: Arithmetic expression using literal numbers.
        expected: The expected/reported value from the paper.
        tolerance: Relative tolerance (default 0.01 = 1%).

    Returns:
        Dict with computed, expected, tolerance, passed (bool), and error if any.
    """
    safe_eval = _lazy_import("src.utils.safe_arithmetic", "safe_eval")
    ArithmeticEvalError = _lazy_import("src.utils.safe_arithmetic", "ArithmeticEvalError")

    try:
        computed = safe_eval(expression)
    except ArithmeticEvalError as exc:
        return {
            "expression": expression,
            "expected": expected,
            "tolerance": tolerance,
            "computed": None,
            "passed": None,
            "error": str(exc),
        }

    denom = max(abs(expected), 1e-12)
    relative_error = abs(computed - expected) / denom
    passed = relative_error <= tolerance

    return {
        "expression": expression,
        "expected": expected,
        "tolerance": tolerance,
        "computed": computed,
        "relative_error": relative_error,
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# Dataset tools
# ---------------------------------------------------------------------------


@mcp.tool()
def get_paper_from_dataset(
    paper_id: str,
    parquet_path: str = "data/train-00000-of-00001.parquet",
) -> dict[str, Any]:
    """Retrieve a single paper from the parquet dataset by its DOI/arXiv ID.

    Args:
        paper_id: The paper's DOI or arXiv ID.
        parquet_path: Path to the parquet file.

    Returns:
        Dict with the paper's metadata and raw content, or error if not found.
    """
    import pandas as pd

    parquet_full = Path(_PROJECT_ROOT) / parquet_path
    if not parquet_full.exists():
        return {"error": f"Parquet file not found: {parquet_full}"}

    df = pd.read_parquet(parquet_full)
    row = df[df["doi/arxiv_id"].astype(str) == paper_id]

    if len(row) == 0:
        return {"error": f"Paper not found: {paper_id}"}

    row = row.iloc[0]

    # Convert paper_content from ndarray (or list) to JSON-serializable list
    import numpy as np
    raw_content = row["paper_content"]
    if isinstance(raw_content, np.ndarray):
        raw_content = raw_content.tolist()

    # Truncate very large content items to keep MCP responses manageable
    content_summary = []
    for item in raw_content:
        item_copy = dict(item)
        if item_copy.get("text") and len(item_copy["text"]) > 3000:
            item_copy["text"] = item_copy["text"][:3000] + "... [truncated]"
        if item_copy.get("image_url") and isinstance(item_copy["image_url"], dict):
            url = item_copy["image_url"].get("url", "")
            if len(url) > 500:
                item_copy["image_url"]["url"] = url[:500] + "... [truncated]"
        content_summary.append(item_copy)

    return {
        "paper_id": str(row["doi/arxiv_id"]),
        "title": str(row.get("title", "")),
        "paper_category": str(row.get("paper_category", "")),
        "error_category": str(row.get("error_category", "")),
        "error_location": str(row.get("error_location", "")),
        "error_severity": str(row.get("error_severity", "")),
        "paper_content": content_summary,
        "full_content_available": True,
        "n_content_items": len(raw_content),
        "has_ground_truth": True,
    }


@mcp.tool()
def list_papers_in_dataset(
    parquet_path: str = "data/train-00000-of-00001.parquet",
    max_papers: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """List paper IDs and metadata from the parquet dataset.

    Args:
        parquet_path: Path to the parquet file.
        max_papers: Maximum number of papers to return.
        offset: Number of papers to skip.

    Returns:
        Dict with total_papers, papers list, and pagination info.
    """
    import pandas as pd

    parquet_full = Path(_PROJECT_ROOT) / parquet_path
    if not parquet_full.exists():
        return {"error": f"Parquet file not found: {parquet_full}"}

    df = pd.read_parquet(parquet_full)
    total = len(df)

    subset = df.iloc[offset : offset + max_papers]
    papers = []
    for _, row in subset.iterrows():
        papers.append({
            "paper_id": str(row["doi/arxiv_id"]),
            "title": str(row.get("title", ""))[:200],
            "paper_category": str(row.get("paper_category", "")),
            "error_category": str(row.get("error_category", "")),
            "error_severity": str(row.get("error_severity", "")),
        })

    return {
        "total_papers": total,
        "offset": offset,
        "returned": len(papers),
        "papers": papers,
    }


@mcp.tool()
def analyze_dataset_schema(
    parquet_path: str = "data/train-00000-of-00001.parquet",
    sample_rows: int = 5,
) -> dict[str, Any]:
    """Analyze the parquet dataset schema and produce a summary report.

    Args:
        parquet_path: Path to the parquet file.
        sample_rows: Number of sample rows to inspect.

    Returns:
        Dict with schema report including column info, content types,
        error categories, and paper categories.
    """
    analyze = _lazy_import("src.parser.schema_analyzer", "analyze_dataset_schema")

    parquet_full = Path(_PROJECT_ROOT) / parquet_path
    if not parquet_full.exists():
        return {"error": f"Parquet file not found: {parquet_full}"}

    report = analyze(parquet_path=parquet_full, sample_rows=sample_rows)
    return report.model_dump()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the Paperena MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
