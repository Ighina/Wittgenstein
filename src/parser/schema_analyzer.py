"""Phase 1: Dataset exploration and schema analysis.

Inspects the parquet dataset, analyzes the structure of paper_content,
and produces a PaperContentSchemaReport.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from src.models import ContentItemSchema, PaperContentSchemaReport


def analyze_dataset_schema(
    parquet_path: str | Path,
    sample_rows: int = 5,
) -> PaperContentSchemaReport:
    """Analyze the schema and structure of the paper dataset.

    Loads the parquet file, inspects every field, analyzes the structure
    of paper_content, and produces a comprehensive schema report.

    Args:
        parquet_path: Path to the parquet file.
        sample_rows: Number of rows to sample for detailed inspection.

    Returns:
        A PaperContentSchemaReport describing the dataset structure.
    """
    parquet_path = Path(parquet_path)
    logger.info(f"Analyzing dataset schema: {parquet_path}")

    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    # Basic dataset info
    total_rows = len(df)
    total_columns = len(df.columns)
    column_names = df.columns.tolist()
    column_dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

    logger.info(f"Loaded {total_rows} rows, {total_columns} columns")

    # Analyze paper_content structure
    content_types: set[str] = set()
    keys_found: set[str] = set()
    text_item_count = 0
    image_item_count = 0
    rows_with_images = 0
    rows_with_local_content = 0
    sample_content_items: list[dict[str, Any]] = []

    for idx, row in df.iterrows():
        pc = row.get("paper_content")
        if pc is None:
            continue

        row_has_images = False
        for item in pc:
            if isinstance(item, dict):
                keys_found.update(item.keys())

                item_type = item.get("type", "unknown")
                content_types.add(item_type)

                if item_type == "text":
                    text_item_count += 1
                elif item_type == "image_url":
                    image_item_count += 1
                    row_has_images = True

                # Collect samples from first few rows
                if idx < sample_rows and len(sample_content_items) < 20:
                    sample = {
                        "type": item_type,
                        "has_text": item.get("text") is not None,
                        "has_image_url": item.get("image_url") is not None,
                        "text_preview": (
                            str(item.get("text", ""))[:200]
                            if item.get("text")
                            else None
                        ),
                        "image_url_type": (
                            type(item.get("image_url")).__name__
                            if item.get("image_url")
                            else None
                        ),
                    }
                    sample_content_items.append(sample)

        if row_has_images:
            rows_with_images += 1

        # Check error_local_content
        elc = row.get("error_local_content")
        if elc is not None and (not isinstance(elc, list) or len(elc) > 0):
            rows_with_local_content += 1

    # Analyze categorical columns
    error_categories = (
        df["error_category"].value_counts().reset_index().to_dict("records")
    )
    # Convert to list of {value, count}
    error_categories = [
        {"category": r["error_category"], "count": r["count"]}
        for r in error_categories
    ]

    error_locations_sample = df["error_location"].unique().tolist()

    error_severities = (
        df["error_severity"].value_counts().reset_index().to_dict("records")
    )
    error_severities = [
        {"severity": r["error_severity"], "count": r["count"]}
        for r in error_severities
    ]

    paper_categories = (
        df["paper_category"].value_counts().reset_index().to_dict("records")
    )
    paper_categories = [
        {"category": r["paper_category"], "count": r["count"]}
        for r in paper_categories
    ]

    report = PaperContentSchemaReport(
        total_rows=total_rows,
        total_columns=total_columns,
        column_names=column_names,
        column_dtypes=column_dtypes,
        content_types=sorted(content_types),
        keys_found=sorted(keys_found),
        text_item_count=text_item_count,
        image_item_count=image_item_count,
        rows_with_images=rows_with_images,
        rows_with_local_content=rows_with_local_content,
        sample_content_items=sample_content_items,
        error_categories=error_categories,
        error_locations_sample=error_locations_sample,
        error_severities=error_severities,
        paper_categories=paper_categories,
        generated_at=datetime.now().isoformat(),
    )

    logger.info(
        f"Schema analysis complete: {text_item_count} text items, "
        f"{image_item_count} image items, "
        f"{len(content_types)} content types, "
        f"{len(keys_found)} keys found"
    )

    return report
