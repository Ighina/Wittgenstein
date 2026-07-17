#!/usr/bin/env python3
"""Harmonized evaluation: zero-shot LLM baseline vs batch_verify.py pipeline.

Two systems are compared using the **same** LLM judge for error matching:

  A) **Zero-shot baseline** — the LLM reads the full paper and predicts errors
     directly via ``llm_reviewer_template`` (reproduces the original paper's
     evaluation methodology).

  B) **Eval-dir pipeline** — results produced by ``batch_verify.py`` (or any
     Claude Code /verify-paper run).  Structured predictions are extracted from
     the companion JSON reports (or, as a fallback, the ``.md`` files) and
     converted to the format the judge expects.

Both sets of predictions are then matched against the same ground-truth
annotations via ``llm_judge_template``, and the same metrics (micro/macro
precision, recall, PPR) are computed for each.

Usage::

    python scripts/original_eval.py \\
        --eval-dir outputs/claude-batch \\
        --model-name deepseek-v4-pro \\
        --judge-name deepseek-v4-pro

Outputs (written next to the script or beside ``--eval-dir``)::

    {eval_dir_name}_annotations.json   – intermediate structured predictions
    comparison_{eval_dir_name}.json    – metrics for both systems
    comparison_{eval_dir_name}.csv     – same metrics in tabular form
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

# -- project root -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import LLMConfig
from src.utils.llm import llm_call, parse_json_response

# ============================================================================
# Prompt templates (from the original paper evaluation)
# ============================================================================

LLM_REVIEWER_TEMPLATE = """You are a **scientific-rigor auditor**. You will receive the parsed contents of a research paper. Your job is to identify **only** those errors or flaws that directly undermine the **scientific validity** of the paper's methods, analyses, or conclusions.

Your sole focus is to identify flaws—such as errors in experimental design, data integrity, calculations, statistical inference, or reproducibility—that directly call into question the validity of a specific claim, paragraph, or the paper as a whole.

**Do not** report issues purely presentational, rhetorical, stylistic, or related to citation practices.

---

After you've done a **detailed walkthrough** of the paper, output exactly in this format—no extra keys or commentary:

```
<analysis>
{detailed walk-through of how you checked each section/figure and why you flagged (or did not flag) any flaw}
</analysis>

<response>
{
  "has_error": <true|false>,
  "errors": [
    {
      "location": "Section 2.1",
      "description": "Claim that 'all X are Y' is contradicted by citation [5], which shows examples of non-Y X."
    },
    {
      "location": "Figure 3",
      "description": "XAxis labeled 'Time (s)' but units appear to be milliseconds; scale bar mismatches caption."
    }
    // …more entries…
  ]
}
</response>
```

- **Do not** include any other keys or prose outside these two tagged blocks.
- **Do not** report stylistic or citation issues.
- Be as precise as possible about **where** (section ID or figure/table) and **what** the scientific flaw is.
- Each description for the errors must be rooted in a scientific rationale explaining why it is 'wrong' (not how it could be improved).

Begin your analysis now."""


LLM_JUDGE_TEMPLATE = """You are an expert LLM-as-a-Judge. You will receive a JSON object with two arrays:

1. "annotations": the ground-truth errors (each has "location" and "description").
2. "predictions": the model's reported errors (same format).

**Task**
1. Compare each prediction against each annotation.
2. A match occurs **only** when both "location" and "description" are identical.
3. Your output should be generated in the following format:

<analysis>
Analysis and comparison of each prediction and annotation.
</analysis>

<response>
{
  "matches": [
      {
      "location": the location of the matched object, this should be based on the annotated location,
      "description": your explanation on why you think it is a match.
      },
        {
      "location": ... ,
      "description": ...
      },
  ]
}
</response>

Be rigorous in considering matches; the location may be slightly differently named, but the description must match overall.
"""


# Template used to extract structured errors from free-form .md verification
# reports when no companion JSON report is available.
MD_EXTRACTION_TEMPLATE = """You are a precise data-extraction assistant. You will receive the **markdown output** of an automated scientific-paper verification system.

Extract every error or flaw that the system claims to have found. For each one, produce a JSON object with exactly two keys:

- **"location"**: the most specific location mentioned (e.g. "Fig 5", "Section 3.2", "Equation 7", "Theorem 1.1").  If no location is discernible, use "Unknown".
- **"description"**: a concise description of what the error is and why it matters, based on the evidence and reasoning in the report.  **Do not** include meta-commentary about the verifier or confidence scores — describe the *substance* of the finding.

Return **only** a JSON array of these objects, with no additional text or keys:

```json
[
  {"location": "Fig 5", "description": "..."},
  {"location": "Section 2.1", "description": "..."}
]
```

If the report found **zero** errors, return an empty array: `[]`

Begin your extraction now."""


# ============================================================================
# Response parsing (adapted from original — works with raw strings, not
# litellm response objects)
# ============================================================================

def parse_response_dict(s: str) -> Dict[str, Any]:
    """Extract and parse the JSON object between <response>...</response> tags.

    Raises:
        ValueError: if no <response> block is found or JSON is invalid.
    """
    pattern = r"<response>\s*(\{.*?\})\s*</response>"
    m = re.search(pattern, s, re.DOTALL)
    if not m:
        raise ValueError("No <response>...</response> block found")
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in <response> block: {e}") from e


def extract_response_dict(response_text: str) -> Dict[str, Any]:
    """Safely extract the JSON dict from an LLM response string.

    Returns a dict with at least a ``"parsed"`` key (bool).
    """
    if response_text is None or not response_text.strip():
        return {"parse_error": "empty response", "parsed": False}

    try:
        result = parse_response_dict(response_text)
        result.setdefault("parsed", True)
        return result
    except ValueError as e:
        return {"parse_error": str(e), "parsed": False, "raw": response_text[:2000]}


# ============================================================================
# Paper-content helpers
# ============================================================================

def paper_content_to_text(paper_content) -> str:
    """Convert a ``paper_content`` array/dict into a single plain-text string.

    Handles the numpy-array-of-dicts format stored in the parquet dataset.
    """
    if isinstance(paper_content, str):
        return paper_content

    parts: list[str] = []
    items = paper_content
    if hasattr(paper_content, "tolist"):
        items = paper_content.tolist()
    if not isinstance(items, (list, tuple)):
        items = [items]

    for item in items:
        if isinstance(item, dict):
            text = item.get("text", "")
            if text and isinstance(text, str):
                parts.append(text.strip())
        elif isinstance(item, str):
            parts.append(item.strip())

    return "\n\n".join(parts)


# ============================================================================
# PaperAudit dataset helpers
# ============================================================================

def extract_text_from_instruction(instruction: str) -> str:
    """Extract the paper text from a PaperAudit instruction string.

    The instruction contains a task prompt followed by the paper text
    between ``### Text to Review ###`` and ``---`` markers.

    Args:
        instruction: Raw instruction string from the PaperAudit dataset.

    Returns:
        The extracted paper text, stripped of surrounding whitespace.
        Falls back to the full instruction if the markers are not found.
    """
    pattern = r"###\s*Text\s+to\s+Review\s*###\s*\n(.*?)\n\s*---"
    match = re.search(pattern, instruction, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: return the instruction as-is (markers might vary)
    return instruction


def parse_paperaudit_output(output_str: str) -> list[dict]:
    """Parse the PaperAudit ``output`` column into ground-truth annotations.

    The output is a JSON array of error objects with keys:
    ``error_location``, ``error_type``, ``error_explanation``.

    These are mapped to the standard ground-truth format used throughout
    this script: ``error_location``, ``error_category``, ``error_severity``,
    ``error_annotation``.

    Args:
        output_str: Raw output string (JSON array) from the dataset.

    Returns:
        A list of annotation dicts, or an empty list on parse failure.
    """
    if not output_str or not isinstance(output_str, str):
        return []

    try:
        errors = json.loads(output_str)
    except (json.JSONDecodeError, TypeError):
        return []

    if not isinstance(errors, list):
        return []

    annotations: list[dict] = []
    for err in errors:
        if not isinstance(err, dict):
            continue
        annotations.append({
            "error_category": str(err.get("error_type", "")),
            "error_location": str(err.get("error_location", "")),
            "error_severity": "",
            "error_annotation": str(err.get("error_explanation", "")),
        })
    return annotations


def load_paperaudit_dataset() -> tuple[dict[str, Any], dict[str, list[dict]]]:
    """Load the PaperAudit dataset from HuggingFace hub.

    Downloads ``mayiwen/PaperAudit_Dataset`` (split ``sft_data``), extracts
    the paper text from each instruction, and parses the output column as
    ground-truth annotations.

    Papers are assigned synthetic IDs of the form ``"paperaudit_N"``.

    Returns:
        (paper_contents, ground_truth)
          - *paper_contents* maps paper_id → plain-text paper content.
          - *ground_truth* maps paper_id → list of annotation dicts
            (``error_location``, ``error_category``, ``error_severity``,
            ``error_annotation``).

    Raises:
        ImportError: if ``datasets`` is not installed.
        SystemExit: if the dataset fails to load.
    """
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError:
        print("ERROR: The 'datasets' library is required for --paperaudit. "
              "Install with: pip install datasets")
        sys.exit(1)

    try:
        ds = hf_load_dataset("mayiwen/PaperAudit_Dataset", split="sft_data")
    except Exception as exc:
        print(f"ERROR: Failed to load PaperAudit dataset from HuggingFace hub: {exc}")
        sys.exit(1)

    paper_contents: dict[str, Any] = {}
    ground_truth: dict[str, list[dict]] = {}

    for idx, row in enumerate(ds):
        pid = f"paperaudit_{idx}"
        instruction = row["instruction"]
        output = row["output"]

        # Extract paper text from the instruction
        text = extract_text_from_instruction(instruction)
        paper_contents[pid] = text

        # Parse ground-truth annotations from the output column
        annotations = parse_paperaudit_output(output)
        ground_truth[pid] = annotations

    total_gt = sum(len(v) for v in ground_truth.values())
    print(f"  {len(paper_contents)} papers, {total_gt} GT errors\n")

    return paper_contents, ground_truth


# ============================================================================
# Metrics (identical to the original paper)
# ============================================================================

def compute_all_metrics(resp_df: pd.DataFrame) -> dict:
    """Compute micro/macro precision, recall, and PPR from a response DataFrame.

    Expected columns:
        doi/arxiv_id, annotation (list), errors (list), matches (list)
    """
    df = resp_df.copy()

    # Ensure list columns
    df["matches"] = df["matches"].apply(lambda x: x if isinstance(x, list) else [])
    df["errors"] = df["errors"].apply(lambda x: x if isinstance(x, list) else [])

    # Per-paper counts
    df = df.assign(
        k_i=df["annotation"].apply(len),
        TP_i=df["matches"].apply(len),
    )
    df["FP_i"] = df["errors"].apply(len) - df["TP_i"]
    df["FN_i"] = df["k_i"] - df["TP_i"]

    N = len(df)

    # Micro
    TP_total = df["TP_i"].sum()
    FP_total = df["FP_i"].sum()
    FN_total = df["FN_i"].sum()
    precision_micro = TP_total / (TP_total + FP_total) if (TP_total + FP_total) > 0 else 0.0
    recall_micro = TP_total / (TP_total + FN_total) if (TP_total + FN_total) > 0 else 0.0

    # Macro
    per_prec = df.apply(
        lambda row: row["TP_i"] / (row["TP_i"] + row["FP_i"])
        if (row["TP_i"] + row["FP_i"]) > 0 else 0.0,
        axis=1,
    )
    per_rec = df.apply(
        lambda row: row["TP_i"] / (row["TP_i"] + row["FN_i"])
        if (row["TP_i"] + row["FN_i"]) > 0 else 0.0,
        axis=1,
    )
    precision_macro = float(per_prec.mean())
    recall_macro = float(per_rec.mean())

    # Perfect-Paper Rate
    perfect_mask = (df["TP_i"] == df["k_i"]) & (df["FP_i"] == 0)
    PPR = float(perfect_mask.sum() / N) if N > 0 else 0.0

    # Per-paper stats
    per_paper = df[["doi/arxiv_id", "k_i", "TP_i", "FP_i", "FN_i"]].to_dict(orient="records")

    return {
        "N": N,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "PPR": PPR,
        "per_paper": per_paper,
    }


# ============================================================================
# Eval-dir processing
# ============================================================================

def _extract_paper_id_from_md(text: str, filename: str) -> Optional[str]:
    """Extract paper_id from markdown content or filename."""
    # PaperAudit synthetic IDs (paperaudit_N)
    m = re.search(r"[Pp]aper\s*`(paperaudit_\d+)`", text[:500])
    if m:
        return m.group(1)

    # PaperAudit ID from filename
    m = re.match(r"(paperaudit_\d+)\.md", filename)
    if m:
        return m.group(1)

    # Try backtick-quoted paper ID
    m = re.search(r"[Pp]aper\s*`([^`]+)`", text[:500])
    if m:
        candidate = m.group(1)
        if re.match(r"^(?:\d{4}\.\d{4,5}|10\.\d{4,5}/)", candidate):
            return candidate

    # Try explicit arxiv ID
    m = re.search(r"paper[_ ]+(?:ID[_ ]+)?(\d{4}\.\d{4,5}(?:v\d+)?)", text[:500], re.IGNORECASE)
    if m:
        return m.group(1)

    # DOI pattern
    m = re.search(r"\b(10\.\d{4,5}/[\w.-]+)", text[:500])
    if m:
        candidate = m.group(1)
        if not candidate.endswith(".json"):
            return candidate

    # Derive from filename
    basename = filename.replace(".md", "")
    if basename.startswith("10.") and "_" in basename:
        parts = basename.split("_", 1)
        return f"{parts[0]}/{parts[1]}"
    if re.match(r"^\d{4}\.\d{4,5}", basename):
        return basename

    return None


def _collect_eval_dir_papers(eval_dir: Path) -> dict[str, dict]:
    """Discover papers in *eval_dir* from .json reports and .md files.

    Returns a dict mapping ``paper_id`` → ``{"json_path": ..., "md_path": ...}``.
    """
    # Index companion JSON files by paper_id — search eval_dir AND
    # the project root outputs/ (where /verify-paper writes JSONs by default).
    root_outputs = PROJECT_ROOT / "outputs"
    search_dirs = [eval_dir]
    if root_outputs.is_dir() and root_outputs.resolve() != eval_dir.resolve():
        search_dirs.append(root_outputs)

    json_by_paper: dict[str, Path] = {}
    for search_dir in search_dirs:
        for json_path in sorted(
            set(search_dir.glob("*report*.json"))
            | set(search_dir.glob("verification*.json"))
            | set(search_dir.glob("*_verification.json"))
        ):
            skip_names = {
                "batch_results.json", "metrics.json", "predictions.json",
                "confusion_matrix.json", "_ground_truth.json", "_snippets.json",
                "_umap.json",
            }
            if json_path.name in skip_names:
                continue
            try:
                parsed = json.loads(json_path.read_text())
                pid = parsed.get("paper_id", "")
                if pid and pid not in json_by_paper:
                    json_by_paper[pid] = json_path
            except (json.JSONDecodeError, OSError):
                continue

    # Discover .md files
    md_files = sorted(
        f for f in eval_dir.glob("*.md")
        if f.name not in {"run_summary.md", "README.md", "SUMMARY.md"}
    )
    md_by_paper: dict[str, Path] = {}
    for md_file in md_files:
        text = md_file.read_text()
        pid = _extract_paper_id_from_md(text, md_file.name)
        if pid:
            md_by_paper[pid] = md_file

    # Merge
    all_ids = set(json_by_paper.keys()) | set(md_by_paper.keys())
    papers: dict[str, dict] = {}
    for pid in sorted(all_ids):
        entry: dict = {}
        if pid in json_by_paper:
            entry["json_path"] = json_by_paper[pid]
        if pid in md_by_paper:
            entry["md_path"] = md_by_paper[pid]
        papers[pid] = entry

    return papers


def _predictions_from_json_report(json_path: Path) -> list[dict]:
    """Extract ``[{"location", "description"}]`` from a JSON verification report."""
    try:
        parsed = json.loads(json_path.read_text())
    except (json.JSONDecodeError, OSError):
        return []

    predicted_errors = parsed.get("predicted_errors", [])
    if not predicted_errors:
        return []

    results: list[dict] = []
    for pe in predicted_errors:
        if not isinstance(pe, dict):
            continue
        location = str(pe.get("error_location", "") or "")
        # Build a description from the available fields
        category = str(pe.get("error_category", "") or "")
        evidence = str(pe.get("supporting_evidence", "") or "")
        description_parts = []
        if category:
            description_parts.append(f"[{category}]")
        if evidence:
            # Truncate very long evidence strings
            evidence = evidence[:500]
            description_parts.append(evidence)
        if not description_parts:
            description_parts.append("Error detected by automated pipeline")
        description = " ".join(description_parts)

        if not location:
            location = "Unknown"

        results.append({"location": location, "description": description})

    return results


def _predictions_from_md_with_llm(
    md_path: Path,
    judge_config: LLMConfig,
    judge_model: str,
) -> list[dict]:
    """Use an LLM to extract structured errors from a .md verification report."""
    text = md_path.read_text()

    # Quick check: if the .md is clearly a timeout / error / short output,
    # don't bother calling the LLM.
    if text.strip().startswith("# TIMEOUT") or "Timeout verifying" in text:
        return []
    if len(text) < 200:
        return []

    prompt = (
        "Below is the markdown output of a paper verification system. "
        "Extract all detected errors as a JSON array.\n\n"
        f"```markdown\n{text[:8000]}\n```"
    )

    try:
        response = llm_call(
            prompt=prompt,
            system_prompt=MD_EXTRACTION_TEMPLATE,
            model=judge_model,
            config=judge_config,
            temperature=0.0,
        )
        parsed = parse_json_response(response)
        if isinstance(parsed, list):
            # Validate each entry has the expected keys
            result = []
            for item in parsed:
                if isinstance(item, dict) and "location" in item:
                    result.append({
                        "location": str(item.get("location", "Unknown")),
                        "description": str(item.get("description", "")),
                    })
            return result
        # If response is a dict with an "errors" key, try that
        if isinstance(parsed, dict):
            errors = parsed.get("errors", [])
            if isinstance(errors, list):
                return [
                    {"location": str(e.get("location", "Unknown")),
                     "description": str(e.get("description", ""))}
                    for e in errors if isinstance(e, dict)
                ]
        return []
    except Exception:
        return []


def _predictions_from_md_fallback(md_path: Path) -> list[dict]:
    """Structural fallback: parse error tables / sections from .md without LLM."""
    text = md_path.read_text()
    results: list[dict] = []

    # Pattern 1: Table rows like "| 1 | **Figure duplication** | Fig 5 | 0.75 | vision |"
    table_pattern = re.compile(
        r"^\|\s*\d+\s*\|\s*\*?\*?(.+?)\*?\*?\s*\|\s*(.+?)\s*\|\s*[0-9.]+\s*\|",
        re.MULTILINE,
    )
    for m in table_pattern.finditer(text):
        error_type = m.group(1).strip()
        location = m.group(2).strip()
        # Skip header rows
        if error_type.lower() in ("error", "finding", "type"):
            continue
        results.append({
            "location": location,
            "description": f"[{error_type}] Error detected at {location}",
        })

    if results:
        return results

    # Pattern 2: Error-detected sections.
    # Split only on explicit error-report section headers.  The word "Finding"
    # alone is too common (appears in "Key findings", "Aggregated Findings
    # Summary", etc.) and was causing phantom predictions with "Unknown" for
    # every field.  Require a colon and a preceding header marker or digit.
    error_sections = re.split(
        r"(?:###?\s*)?(?:Error\s*Detected|Predicted\s*Error)\s*:?",
        text, flags=re.IGNORECASE,
    )
    for section in error_sections[1:]:
        cat_m = re.search(
            r"\|\s*\*?\*?Error\s*(?:category|type)\*?\*?\s*\|\s*(.+?)\s*\|",
            section, re.IGNORECASE,
        )
        loc_m = re.search(
            r"\|\s*\*?\*?(?:Error\s*)?Location\*?\*?\s*\|\s*(.+?)\s*\|",
            section, re.IGNORECASE,
        )
        location = loc_m.group(1).strip("` ") if loc_m else None
        category = cat_m.group(1).strip("` ") if cat_m else None

        # Skip entries where neither location nor category could be
        # determined — these are phantom matches, not real findings.
        if not location and not category:
            continue

        results.append({
            "location": location or "Unknown",
            "description": f"[{category or 'Unknown'}] Error detected at {location or 'Unknown'}",
        })

    return results


def process_eval_dir(
    eval_dir: Path,
    judge_config: LLMConfig,
    judge_model: str,
) -> Tuple[dict[str, list[dict]], dict[str, dict]]:
    """Extract structured error predictions from an eval directory.

    For each paper:
      1. Prefer companion JSON reports (most reliable).
      2. Fall back to LLM extraction from .md files.
      3. Last resort: structural regex parsing of .md files.

    Returns:
        (predictions_by_paper, paper_meta)
          predictions_by_paper: paper_id → [{"location", "description"}, ...]
          paper_meta: paper_id → {"json_path", "md_path", "source"}
    """
    papers = _collect_eval_dir_papers(eval_dir)
    predictions: dict[str, list[dict]] = {}
    meta: dict[str, dict] = {}

    for pid, entry in papers.items():
        paper_meta = {"source": "unknown"}

        # 1) Prefer JSON report
        if "json_path" in entry:
            preds = _predictions_from_json_report(entry["json_path"])
            if preds or entry["json_path"].exists():  # even empty list is valid
                predictions[pid] = preds
                paper_meta["source"] = "json_report"
                paper_meta["json_path"] = str(entry["json_path"])
                if "md_path" in entry:
                    paper_meta["md_path"] = str(entry["md_path"])
                meta[pid] = paper_meta
                continue

        # 2) .md with LLM extraction
        if "md_path" in entry:
            preds = _predictions_from_md_with_llm(
                entry["md_path"], judge_config, judge_model,
            )
            if preds:
                predictions[pid] = preds
                paper_meta["source"] = "md_llm_extraction"
                paper_meta["md_path"] = str(entry["md_path"])
                meta[pid] = paper_meta
                continue

            # 3) .md structural fallback
            preds = _predictions_from_md_fallback(entry["md_path"])
            predictions[pid] = preds
            paper_meta["source"] = "md_structural_fallback"
            paper_meta["md_path"] = str(entry["md_path"])
            meta[pid] = paper_meta
            continue

        # Shouldn't get here (every entry has at least one path)
        predictions[pid] = []
        meta[pid] = paper_meta

    return predictions, meta


# ============================================================================
# Zero-shot baseline
# ============================================================================

def run_zero_shot_baseline(
    paper_ids: list[str],
    paper_contents: dict[str, Any],
    ground_truth: dict[str, list[dict]],
    reviewer_config: LLMConfig,
    reviewer_model: str,
    max_input_chars: int = 60000,
) -> pd.DataFrame:
    """Run the zero-shot LLM reviewer on each paper.

    Returns a DataFrame with columns:
        doi/arxiv_id, errors, annotation
    """
    rows: list[dict] = []

    for pid in paper_ids:
        content_raw = paper_contents.get(pid)
        if content_raw is None:
            print(f"  [skip] {pid}: no paper_content in dataset")
            continue

        paper_text = paper_content_to_text(content_raw)
        if len(paper_text) > max_input_chars:
            paper_text = paper_text[:max_input_chars]
            print(f"  [truncated] {pid}: {len(paper_text)} → {max_input_chars} chars")

        prompt = (
            f"Paper ID: {pid}\n\n"
            f"FULL PAPER TEXT:\n{paper_text}\n\n"
            "Identify all scientifically meaningful errors. Return ONLY the tagged JSON response."
        )

        print(f"  [reviewer] {pid}: calling {reviewer_model} ({len(paper_text)} chars) ...")

        errors: list[dict] = []
        parsed_flag = False
        for attempt in range(1, 6):
            try:
                response = llm_call(
                    prompt=prompt,
                    system_prompt=LLM_REVIEWER_TEMPLATE,
                    model=reviewer_model,
                    config=reviewer_config,
                    temperature=0.0,
                )
                parsed = extract_response_dict(response)
                if parsed.get("parsed"):
                    errors = parsed.get("errors", []) or []
                    parsed_flag = True
                    break
                else:
                    print(f"    attempt {attempt}: parse error — {parsed.get('parse_error', '?')}")
            except Exception as exc:
                print(f"    attempt {attempt}: {exc}")
                if attempt < 5:
                    time.sleep(2 ** attempt)

        gt = ground_truth.get(pid, [])
        rows.append({
            "doi/arxiv_id": pid,
            "parsed": parsed_flag,
            "is_error": bool(errors),
            "errors": errors,
            "annotation": [
                {"location": g["error_location"], "description": g["error_annotation"]}
                for g in gt
            ],
            "matches": [],  # filled in by judge
        })

        n_errs = len(errors)
        print(f"    → {n_errs} error(s) predicted, parsed={parsed_flag}")

    return pd.DataFrame(rows)


# ============================================================================
# LLM Judge
# ============================================================================

def run_judge_on_dataframe(
    df: pd.DataFrame,
    judge_config: LLMConfig,
    judge_model: str,
    label: str = "",
) -> pd.DataFrame:
    """Run the LLM judge on every row of *df* that has errors.

    The DataFrame must have columns:
        doi/arxiv_id, errors, annotation

    Returns *df* with the ``matches`` column populated.
    """
    df = df.copy()
    df["matches"] = df["matches"].apply(lambda x: x if isinstance(x, list) else [])

    for idx, row in df.iterrows():
        pid = row["doi/arxiv_id"]
        predictions = row.get("errors", []) or []
        annotations = row.get("annotation", []) or []

        if not predictions or not annotations:
            # No predictions or no ground truth → no matches possible
            continue

        payload = {
            "annotations": annotations,
            "predictions": predictions,
        }
        user_content = json.dumps(payload, ensure_ascii=False, indent=2)

        prefix = f"[{label}]" if label else ""
        print(f"  {prefix} [judge] {pid}: {len(predictions)} preds vs {len(annotations)} GTs ...")

        for attempt in range(1, 6):
            try:
                response = llm_call(
                    prompt=user_content,
                    system_prompt=LLM_JUDGE_TEMPLATE,
                    model=judge_model,
                    config=judge_config,
                    temperature=0.0,
                )
                parsed = extract_response_dict(response)
                if parsed.get("parsed"):
                    matches = parsed.get("matches", []) or []
                    df.at[idx, "matches"] = matches
                    print(f"    → {len(matches)} match(es)")
                    break
                else:
                    print(f"    attempt {attempt}: parse error — {parsed.get('parse_error', '?')}")
            except Exception as exc:
                print(f"    attempt {attempt}: {exc}")
                if attempt < 5:
                    time.sleep(2 ** attempt)
        else:
            print(f"    → judge failed after all attempts for {pid}")

    return df


# ============================================================================
# Helpers: build eval-dir dataframe and convert to judge format
# ============================================================================

def build_eval_dir_dataframe(
    eval_predictions: dict[str, list[dict]],
    ground_truth: dict[str, list[dict]],
) -> pd.DataFrame:
    """Build a DataFrame for eval-dir results, ready for judge + metrics.

    The eval-dir predictions are already in ``[{"location", "description"}]``
    format, matching what the judge expects.
    """
    rows = []
    for pid, preds in eval_predictions.items():
        gt = ground_truth.get(pid, [])
        rows.append({
            "doi/arxiv_id": pid,
            "parsed": True,
            "is_error": bool(preds),
            "errors": preds,
            "annotation": [
                {"location": g["error_location"], "description": g["error_annotation"]}
                for g in gt
            ],
            "matches": [],
        })
    return pd.DataFrame(rows)


# ============================================================================
# Main
# ============================================================================

def _auto_detect_provider() -> str:
    """Return the first available LLM provider based on env vars."""
    if os.environ.get("DEEPSEEK_API_KEY"):
        return "deepseek"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return "deepseek"


def _default_model_for_provider(provider: str) -> str:
    defaults = {
        "deepseek": "deepseek-v4-pro",
        "anthropic": "claude-sonnet-4-6",
        "openai": "gpt-4o",
    }
    return defaults.get(provider, "deepseek-v4-pro")


def _api_key_env_for_provider(provider: str) -> str:
    return {
        "deepseek": "DEEPSEEK_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
    }.get(provider, "DEEPSEEK_API_KEY")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Harmonized evaluation: zero-shot LLM vs batch_verify.py pipeline"
    )
    parser.add_argument(
        "--eval-dir", type=str, required=True,
        help="Directory containing batch_verify.py outputs (.md and .json files).",
    )
    parser.add_argument(
        "--model-name", type=str, default=None,
        help="LLM model for the zero-shot baseline reviewer "
             "(default: auto-detected from provider).",
    )
    parser.add_argument(
        "--judge-name", type=str, default=None,
        help="LLM model for the judge (default: auto-detected from provider).",
    )
    parser.add_argument(
        "--provider", type=str, default=None,
        help="LLM provider: anthropic, openai, or deepseek (default: auto-detect).",
    )
    parser.add_argument(
        "--judge-provider", type=str, default=None,
        help="Provider for the judge model (default: same as --provider).",
    )
    parser.add_argument(
        "--parquet", type=str, default="data/train-00000-of-00001.parquet",
        help="Path to the parquet dataset (default: data/train-00000-of-00001.parquet).",
    )
    parser.add_argument(
        "--paperaudit", action="store_true",
        help="Use PaperAudit_Dataset from HuggingFace hub (mayiwen/PaperAudit_Dataset) "
             "instead of the SPOT parquet dataset.",
    )
    parser.add_argument(
        "--max-input-chars", type=int, default=60000,
        help="Maximum characters of paper content sent to the zero-shot reviewer.",
    )
    parser.add_argument(
        "--skip-baseline", action="store_true",
        help="Skip the zero-shot baseline (only evaluate eval-dir results).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory for output files (default: same as --eval-dir).",
    )
    args = parser.parse_args()

    # Resolve paths
    eval_dir = Path(args.eval_dir)
    if not eval_dir.is_absolute():
        eval_dir = PROJECT_ROOT / eval_dir
    if not eval_dir.is_dir():
        print(f"ERROR: --eval-dir does not exist or is not a directory: {eval_dir}")
        sys.exit(1)

    parquet_path = Path(args.parquet)
    if not parquet_path.is_absolute():
        parquet_path = PROJECT_ROOT / parquet_path

    output_dir = Path(args.output_dir) if args.output_dir else eval_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Derive a clean name for output files
    eval_dir_name = eval_dir.name if eval_dir.name else "eval"

    # -- Provider & model resolution -----------------------------------------
    provider = args.provider or _auto_detect_provider()
    judge_provider = args.judge_provider or provider

    reviewer_model = args.model_name or _default_model_for_provider(provider)
    judge_model = args.judge_name or _default_model_for_provider(judge_provider)

    reviewer_config = LLMConfig(
        provider=provider,
        model=reviewer_model,
        api_key_env=_api_key_env_for_provider(provider),
    )
    judge_config = LLMConfig(
        provider=judge_provider,
        model=judge_model,
        api_key_env=_api_key_env_for_provider(judge_provider),
    )

    print("=" * 60)
    print("Harmonized Evaluation")
    print("=" * 60)
    print(f"  Eval dir:       {eval_dir}")
    if args.paperaudit:
        print(f"  Dataset:        PaperAudit (HuggingFace hub)")
    else:
        print(f"  Parquet:        {parquet_path}")
    print(f"  Output dir:     {output_dir}")
    print(f"  Reviewer:       {provider}/{reviewer_model}")
    print(f"  Judge:          {judge_provider}/{judge_model}")
    if args.skip_baseline:
        print(f"  Zero-shot:      SKIPPED")
    print()

    # ------------------------------------------------------------------
    # 1. Load ground truth (from PaperAudit or SPOT parquet)
    # ------------------------------------------------------------------
    print("Loading dataset ...")
    if args.paperaudit:
        paper_contents, ground_truth = load_paperaudit_dataset()
    else:
        df = pd.read_parquet(parquet_path)

        # Build ground-truth index
        ground_truth: dict[str, list[dict]] = {}
        paper_contents: dict[str, Any] = {}
        for _, row in df.iterrows():
            pid = str(row["doi/arxiv_id"])
            paper_contents[pid] = row["paper_content"]
            ground_truth.setdefault(pid, []).append({
                "error_category": str(row.get("error_category", "")),
                "error_location": str(row.get("error_location", "")),
                "error_severity": str(row.get("error_severity", "")),
                "error_annotation": str(row.get("error_annotation", "")),
            })
        print(f"  {len(ground_truth)} papers, {sum(len(v) for v in ground_truth.values())} GT errors\n")

    # ------------------------------------------------------------------
    # 2. Process eval-dir → structured predictions
    # ------------------------------------------------------------------
    print("Processing eval-dir ...")
    eval_predictions, eval_meta = process_eval_dir(eval_dir, judge_config, judge_model)

    # Filter eval-dir predictions to only include papers present in the loaded
    # dataset.  Otherwise process_eval_dir discovers papers from the filesystem
    # (every .json/.md in eval_dir), which can silently mix papers from a
    # different dataset — e.g. SPOT results matched against PaperAudit GT.
    valid_ids = set(paper_contents.keys())
    skipped_ids = [pid for pid in eval_predictions if pid not in valid_ids]
    if skipped_ids:
        print(f"  ⚠ Dropping {len(skipped_ids)} paper(s) not in the loaded dataset:")
        for pid in sorted(skipped_ids)[:10]:
            print(f"      - {pid}")
        if len(skipped_ids) > 10:
            print(f"      ... and {len(skipped_ids) - 10} more")
    eval_predictions = {pid: v for pid, v in eval_predictions.items() if pid in valid_ids}
    eval_meta = {pid: v for pid, v in eval_meta.items() if pid in valid_ids}

    n_with_errors = sum(1 for p in eval_predictions.values() if p)
    n_total_errors = sum(len(p) for p in eval_predictions.values())
    print(f"  {len(eval_predictions)} papers, {n_with_errors} with errors "
          f"({n_total_errors} total predicted errors)")

    # Save intermediate annotations
    annotations_output = {
        "eval_dir": str(eval_dir),
        "generated_at": pd.Timestamp.now().isoformat(),
        "num_papers": len(eval_predictions),
        "num_papers_with_errors": n_with_errors,
        "total_predicted_errors": n_total_errors,
        "paper_meta": {pid: meta for pid, meta in eval_meta.items()},
        "predictions": eval_predictions,
    }
    annotations_path = output_dir / f"{eval_dir_name}_annotations.json"
    annotations_path.write_text(json.dumps(annotations_output, indent=2, default=str))
    print(f"  Saved annotations → {annotations_path}\n")

    # ------------------------------------------------------------------
    # 3. Build eval-dir DataFrame and run judge (only if matching papers exist)
    # ------------------------------------------------------------------
    eval_metrics = None
    eval_df = None
    if eval_predictions:
        eval_df = build_eval_dir_dataframe(eval_predictions, ground_truth)
        n_eval_papers = len(eval_df)

        print(f"Running LLM judge on eval-dir predictions ({n_eval_papers} papers) ...")
        eval_df = run_judge_on_dataframe(eval_df, judge_config, judge_model, label="eval-dir")

        eval_metrics = compute_all_metrics(eval_df)
        print(f"  Eval-dir — Micro P: {eval_metrics['precision_micro']:.3f}  "
              f"R: {eval_metrics['recall_micro']:.3f}  "
              f"Macro P: {eval_metrics['precision_macro']:.3f}  "
              f"R: {eval_metrics['recall_macro']:.3f}  "
              f"PPR: {eval_metrics['PPR']:.3f}\n")
    else:
        print("No eval-dir papers matched the loaded dataset — skipping eval-dir judge.\n")

    # ------------------------------------------------------------------
    # 4. Zero-shot baseline (optional)
    # ------------------------------------------------------------------
    baseline_metrics = None
    baseline_df = None
    if not args.skip_baseline:
        if args.paperaudit:
            # Run baseline on ALL PaperAudit papers, not just those with eval-dir coverage
            paper_ids = sorted(paper_contents.keys())
        else:
            paper_ids = sorted(eval_predictions.keys())
        if not paper_ids:
            print("WARNING: No papers found in eval-dir — skipping zero-shot baseline.")
        else:
            print(f"Running zero-shot baseline on {len(paper_ids)} papers ...")
            baseline_df = run_zero_shot_baseline(
                paper_ids=paper_ids,
                paper_contents=paper_contents,
                ground_truth=ground_truth,
                reviewer_config=reviewer_config,
                reviewer_model=reviewer_model,
                max_input_chars=args.max_input_chars,
            )

            n_baseline = len(baseline_df)
            n_parsed = baseline_df["parsed"].sum()
            n_with_errs = baseline_df["is_error"].sum()
            print(f"  {n_baseline} papers, {n_parsed} parsed, {n_with_errs} with errors")

            print(f"\nRunning LLM judge on zero-shot baseline predictions ...")
            baseline_df = run_judge_on_dataframe(
                baseline_df, judge_config, judge_model, label="baseline",
            )

            baseline_metrics = compute_all_metrics(baseline_df)
            print(f"  Baseline — Micro P: {baseline_metrics['precision_micro']:.3f}  "
                  f"R: {baseline_metrics['recall_micro']:.3f}  "
                  f"Macro P: {baseline_metrics['precision_macro']:.3f}  "
                  f"R: {baseline_metrics['recall_macro']:.3f}  "
                  f"PPR: {baseline_metrics['PPR']:.3f}\n")

    # ------------------------------------------------------------------
    # 5. Assemble comparison output
    # ------------------------------------------------------------------
    comparison: dict[str, Any] = {
        "config": {
            "eval_dir": str(eval_dir),
            "dataset": ("paperaudit" if args.paperaudit else "spot"),
            "parquet": str(parquet_path) if not args.paperaudit else None,
            "reviewer_model": f"{provider}/{reviewer_model}",
            "judge_model": f"{judge_provider}/{judge_model}",
            "max_input_chars": args.max_input_chars,
            "generated_at": pd.Timestamp.now().isoformat(),
        },
    }

    if eval_metrics is not None:
        comparison["eval_dir_metrics"] = {
            "system": "batch_verify pipeline",
            "N": eval_metrics["N"],
            "precision_micro": eval_metrics["precision_micro"],
            "recall_micro": eval_metrics["recall_micro"],
            "precision_macro": eval_metrics["precision_macro"],
            "recall_macro": eval_metrics["recall_macro"],
            "PPR": eval_metrics["PPR"],
        }

    if baseline_metrics is not None:
        comparison["zero_shot_baseline_metrics"] = {
            "system": f"zero-shot ({reviewer_model})",
            "N": baseline_metrics["N"],
            "precision_micro": baseline_metrics["precision_micro"],
            "recall_micro": baseline_metrics["recall_micro"],
            "precision_macro": baseline_metrics["precision_macro"],
            "recall_macro": baseline_metrics["recall_macro"],
            "PPR": baseline_metrics["PPR"],
        }
        comparison["per_paper"] = []
        baseline_per_paper = {
            p["doi/arxiv_id"]: p for p in (baseline_metrics.get("per_paper") or [])
        }
        eval_per_paper = {
            p["doi/arxiv_id"]: p for p in ((eval_metrics or {}).get("per_paper") or [])
        }
        all_ids = sorted(set(baseline_per_paper.keys()) | set(eval_per_paper.keys()))
        for pid in all_ids:
            entry: dict = {"paper_id": pid}
            bp = baseline_per_paper.get(pid, {})
            ep = eval_per_paper.get(pid, {})
            entry["baseline"] = {
                "k_i": bp.get("k_i", 0), "TP": bp.get("TP_i", 0),
                "FP": bp.get("FP_i", 0), "FN": bp.get("FN_i", 0),
            }
            entry["eval_dir"] = {
                "k_i": ep.get("k_i", 0), "TP": ep.get("TP_i", 0),
                "FP": ep.get("FP_i", 0), "FN": ep.get("FN_i", 0),
            }
            comparison["per_paper"].append(entry)

    # -- Write comparison JSON --
    comp_json_path = output_dir / f"comparison_{eval_dir_name}.json"
    comp_json_path.write_text(json.dumps(comparison, indent=2, default=str))
    print(f"Comparison JSON → {comp_json_path}")

    # -- Write comparison CSV --
    csv_rows = []
    metric_pairs: list[tuple[str, dict]] = []
    if eval_metrics is not None:
        metric_pairs.append(("eval_dir", eval_metrics))
    if baseline_metrics is not None:
        metric_pairs.append(("zero_shot_baseline", baseline_metrics))

    for system_name, metrics in metric_pairs:
        csv_rows.append({
            "system": system_name,
            "N": metrics["N"],
            "precision_micro": round(metrics["precision_micro"], 4),
            "recall_micro": round(metrics["recall_micro"], 4),
            "precision_macro": round(metrics["precision_macro"], 4),
            "recall_macro": round(metrics["recall_macro"], 4),
            "PPR": round(metrics["PPR"], 4),
        })

    csv_df = pd.DataFrame(csv_rows)
    comp_csv_path = output_dir / f"comparison_{eval_dir_name}.csv"
    csv_df.to_csv(comp_csv_path, index=False)
    print(f"Comparison CSV  → {comp_csv_path}")

    # -- Print summary table --
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    header = f"{'System':<35}{'N':>4}{'P(micro)':>10}{'R(micro)':>10}{'P(macro)':>10}{'R(macro)':>10}{'PPR':>8}"
    print(header)
    print("-" * 72)
    for row in csv_rows:
        print(
            f"{row['system']:<35}"
            f"{row['N']:>4}"
            f"{row['precision_micro']:>10.3f}"
            f"{row['recall_micro']:>10.3f}"
            f"{row['precision_macro']:>10.3f}"
            f"{row['recall_macro']:>10.3f}"
            f"{row['PPR']:>8.3f}"
        )
    print("=" * 72)


if __name__ == "__main__":
    main()
