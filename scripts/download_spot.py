#!/usr/bin/env python3
"""
download_spot.py — Download the amphora/SPOT dataset from HuggingFace and convert it
to the paperaudit-workspace format.

Usage:
    python scripts/download_spot.py [--output-dir DATA_DIR] [--split SPLIT]

    DATA_DIR defaults to data/spot-workspace/
    SPLIT   defaults to "train" (the only split in SPOT)

Output structure (mirrors paperaudit-workspace):
    DATA_DIR/
        0001-Sanitized_Title__Subtitle/
            paper.md               # full paper as markdown (text + embedded images)
            ground_truth.json      # list of annotated errors (paperaudit schema)
        0002-Another_Paper/
            paper.md
            ground_truth.json
        ...

Mapping notes:
    - SPOT's paper_content (list of {type, text, image_url}) → concatenated paper.md
    - SPOT's error_annotation → ground_truth error_explanation
    - SPOT's error_category → ground_truth corruption_type (best-effort mapping)
    - SPOT's error_severity → ground_truth difficulty (retract→hard, errata→medium)
    - SPOT's error_local_content → ground_truth target_find (when available)
    - SPOT has NO equivalent of 'replacement' or 'rationale' — those are filled with
      sensible defaults.
    - No verification-report.json is written (that's the output of running verification).

Dependencies:
    pip install datasets pyarrow
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Mappings: SPOT → paperaudit
# ---------------------------------------------------------------------------

ERROR_CATEGORY_MAP = {
    "Equation / proof":                     "method_logic_consistency",
    "Figure duplication":                   "evidence_data_integrity",
    "Data inconsistency":                   "evidence_data_integrity",
    "Data Inconsistency (figure-figure)":   "evidence_data_integrity",
    "Data Inconsistency (figure-text)":     "evidence_data_integrity",
    "Data Inconsistency (text-text)":       "evidence_data_integrity",
    "Experiment setup":                     "experimental_design_protocol",
    "Reagent identity":                     "reference_background_fabrication",
    "Statistical reporting":                "evidence_data_integrity",
}

SEVERITY_DIFFICULTY_MAP = {
    "retract": "hard",
    "errata":  "medium",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sanitize_filename(title: str, max_len: int = 120) -> str:
    """Turn a paper title into a filesystem-safe directory slug."""
    # Replace problematic chars with underscores
    slug = title.strip()
    slug = re.sub(r'[\\/:*?"<>|]', '_', slug)
    slug = re.sub(r'\s+', '_', slug)
    slug = re.sub(r'_+', '_', slug)
    slug = slug.strip('_')
    # Truncate
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip('_')
    return slug


def build_paper_markdown(paper_content: list[dict]) -> str:
    """
    Concatenate the paper_content list into a single markdown string.
    - type='text'   → append the text
    - type='image_url' → embed as data URI markdown image
    """
    parts: list[str] = []
    img_counter = 0

    for item in paper_content:
        if item.get("type") == "text":
            txt = item.get("text", "")
            if txt:
                parts.append(txt)
        elif item.get("type") == "image_url":
            img_url = item.get("image_url")
            if img_url and isinstance(img_url, dict):
                url = img_url.get("url", "")
                if url:
                    parts.append(f"\n![Figure {img_counter}]({url})\n")
                    img_counter += 1

    return "\n\n".join(parts)


def build_ground_truth(row: dict, index: int) -> list[dict]:
    """
    Convert a single SPOT row into a paperaudit ground_truth entry.
    SPOT has one error per row; paperaudit expects a list of errors per paper.

    Fields we CAN populate accurately:
        id, corruption_type, difficulty, location, error_explanation

    Fields we approximate / derive:
        rationale       → derived from error_annotation
        target_find     → from error_local_content if available, else empty
        replacement     → NOT available in SPOT  (empty string)

    Fields defaulted:
        needs_cross_section → False
    """
    category = row.get("error_category", "")
    severity = row.get("error_severity", "errata").lower()
    annotation = row.get("error_annotation", "")
    location = row.get("error_location", "Unknown")
    local_content = row.get("error_local_content")

    # Best-effort target_find from error_local_content
    target_find = ""
    if local_content and len(local_content) > 0:
        texts = [item.get("text", "") for item in local_content if item.get("text")]
        target_find = "\n".join(texts)

    corruption_type = ERROR_CATEGORY_MAP.get(category, "evidence_data_integrity")
    difficulty = SEVERITY_DIFFICULTY_MAP.get(severity, "medium")

    # Build a short rationale from the annotation first sentence
    rationale = annotation.split(".")[0].strip() + "." if annotation else ""

    return [{
        "id": index + 1,
        "corruption_type": corruption_type,
        "difficulty": difficulty,
        "location": location,
        "rationale": rationale,
        "error_explanation": annotation,
        "needs_cross_section": False,
        "target_find": target_find,
        "replacement": "",
        "_source": "SPOT",                     # provenance marker
        "_spot_category": category,            # original category for reference
        "_spot_severity": severity,            # original severity
    }]


def process_dataset(ds, output_dir: Path) -> None:
    """Iterate over the SPOT dataset and write paperaudit-format directories."""
    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(ds)

    for idx in range(total):
        row = ds[idx]
        title = row.get("title", f"untitled_{idx}")
        slug = sanitize_filename(title)
        dir_name = f"{idx + 1:04d}-{slug}"

        paper_dir = output_dir / dir_name
        paper_dir.mkdir(parents=True, exist_ok=True)

        # ---- paper.md ----
        paper_content = row.get("paper_content", [])
        md = build_paper_markdown(paper_content)
        (paper_dir / "paper.md").write_text(md, encoding="utf-8")

        # ---- ground_truth.json ----
        gt = build_ground_truth(row, idx)
        with open(paper_dir / "ground_truth.json", "w", encoding="utf-8") as f:
            json.dump(gt, f, indent=2, ensure_ascii=False)

        # ---- metadata.json (extra context, not in original paperaudit) ----
        meta = {
            "doi_or_arxiv": row.get("doi/arxiv_id", ""),
            "title": title,
            "paper_category": row.get("paper_category", ""),
            "spot_index": idx,
            "_note": "Converted from amphora/SPOT dataset",
        }
        with open(paper_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        print(f"  [{idx + 1:04d}/{total:04d}]  {dir_name}", flush=True)

    # Summary
    print(f"\nDone.  Wrote {total} paper directories to {output_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download amphora/SPOT and convert to paperaudit-workspace format"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: data/spot-workspace/)",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to use (default: train)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be done, don't write files",
    )
    args = parser.parse_args()

    # Resolve output dir relative to repo root
    repo_root = Path(__file__).resolve().parent.parent
    output_dir = repo_root / (args.output_dir or "data/spot-workspace")

    print(f"Loading amphora/SPOT  (split={args.split}) ...", flush=True)

    try:
        from datasets import load_dataset
    except ImportError:
        print(
            "ERROR: 'datasets' library not installed.\n"
            "Install it with:  pip install datasets pyarrow",
            file=sys.stderr,
        )
        sys.exit(1)

    ds = load_dataset("amphora/SPOT", split=args.split)
    print(f"Loaded {len(ds)} rows.\n", flush=True)

    if args.dry_run:
        for idx in range(len(ds)):
            title = sanitize_filename(ds[idx].get("title", f"untitled_{idx}"))
            print(f"  [{idx + 1:04d}]  {idx + 1:04d}-{title}")
        print(f"\nDry run — no files written.  Would output to: {output_dir}")
        return

    process_dataset(ds, output_dir)


if __name__ == "__main__":
    main()
