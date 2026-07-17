#!/usr/bin/env python3
"""Rebuild batch_results.json from .md files and companion JSON reports.

This script handles the case where /verify-paper wrote JSON reports to
outputs/ (root) but batch_verify.py expects them in the eval directory
(e.g., outputs/claude_all/). It also parses markdown files directly to
extract structured verification data when no companion JSON exists.

Usage:
    python scripts/rebuild_batch_results.py outputs/claude_all

This reads all .md files in the directory, looks for companion JSONs both
in the eval directory and in outputs/, parses the markdown for papers
without JSONs, and writes a corrected batch_results.json.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path


def extract_from_markdown(text: str) -> dict | None:
    """Extract structured verification data from a markdown output file.

    Handles three formats:
    1. Paper-level JSON blocks embedded in markdown
    2. Structured markdown with tables (Verification Report format)
    3. Per-snippet JSON blocks (aggregates them into paper-level data)
    """
    result = {
        "verdict": "UNKNOWN",
        "predicted_errors": [],
    }

    # Quick check: timeout / MCP approval pending / empty
    if text.strip().startswith("# TIMEOUT") or "Timeout verifying" in text:
        return None
    if "requires approval" in text[:300] or "MCP tool call requires" in text[:300]:
        return None

    # --- Strategy 0: Short summary format (no tables, just prose) ---
    # Pattern: "zero errors" / "0 errors detected" / "Verdict: PASS"
    if "0 errors detected" in text or "zero errors" in text.lower() or "PASS" in text[:500]:
        if not re.search(r"\|\s*\*?\*?Error", text):  # no structured error table
            result["verdict"] = "NO_ERROR"
            result["success"] = True
            return result

    # --- Strategy 1: Paper-level JSON block ---
    json_blocks = re.findall(r"```json\s*\n(.*?)\n```", text, re.DOTALL)
    for block in json_blocks:
        try:
            parsed = json.loads(block)
            if "paper_id" in parsed and "predicted_errors" in parsed:
                result.update(parsed)
                result["success"] = True
                return result
        except json.JSONDecodeError:
            continue

    # --- Strategy 2: Aggregated per-snippet JSON blocks ---
    per_snippet_errors = []
    for block in json_blocks:
        try:
            parsed = json.loads(block)
            # Handle both single object and list of objects
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and item.get("error_detected") and item.get("status") == "ERROR_DETECTED":
                        per_snippet_errors.append(item)
            elif isinstance(parsed, dict):
                if parsed.get("error_detected") and parsed.get("status") == "ERROR_DETECTED":
                    per_snippet_errors.append(parsed)
        except json.JSONDecodeError:
            continue

    # Also look for bare JSON objects with error_detected=true
    brace_blocks = re.findall(
        r'\{[^{}]*"error_detected"\s*:\s*true[^{}]*\}', text, re.DOTALL
    )
    for block in brace_blocks:
        try:
            cleaned = block.strip()
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                if parsed.get("error_detected") and parsed.get("status") == "ERROR_DETECTED":
                    # Avoid duplicating blocks we already found via ```json```
                    if not any(e.get("snippet_id") == parsed.get("snippet_id")
                              for e in per_snippet_errors):
                        per_snippet_errors.append(parsed)
        except json.JSONDecodeError:
            continue

    if per_snippet_errors:
        result["predicted_errors"] = [
            {
                "error_category": e.get("predicted_error_category", "Unknown"),
                "error_location": (
                    e.get("contradiction_locations", [None])[0]
                    if e.get("contradiction_locations")
                    else e.get("snippet_id", "")
                ),
                "confidence": e.get("confidence", 0.0),
                "supporting_evidence": e.get("reasoning", ""),
                "verifier_name": e.get("verifier_name", ""),
                "snippet_id": e.get("snippet_id", ""),
            }
            for e in per_snippet_errors
        ]
        result["verdict"] = "NEEDS_CORRECTION" if per_snippet_errors else "UNKNOWN"
        result["success"] = True
        return result

    # --- Strategy 3: Structured markdown parsing ---
    # Look for "Verdict:" or "**Verdict:**"
    verdict_match = re.search(
        r"(?:\*\*)?Verdict(?:\*\*)?\s*:\s*\*?\*?\s*(NEEDS_CORRECTION|NEEDS_REVIEW|NO_ERROR|VALID|UNVERIFIABLE)",
        text, re.IGNORECASE,
    )
    if verdict_match:
        result["verdict"] = verdict_match.group(1).upper()

    # Look for error blocks in markdown tables or explicit sections
    # Pattern: "Error Detected" or "Predicted Error" sections
    error_sections = re.split(
        r"(?:###?\s*)?(?:Error\s*Detected|Predicted\s*Error|Finding):?",
        text, flags=re.IGNORECASE,
    )

    # Also check for "### Finding:" pattern
    finding_blocks = re.findall(
        r"(?:###?\s*Finding:.*?)(?=###\s|##\s|---\s|$)",
        text, re.DOTALL | re.IGNORECASE,
    )

    predicted_errors = []

    # Parse error detail tables
    # Look for markdown tables with error details
    table_rows = re.findall(
        r"\|\s*\*?\*?(Error\s*(?:category|type|location|confidence|severity)|Category|Location|Confidence|Severity)"
        r"\*?\*?\s*\|\s*(.+?)\s*\|",
        text, re.IGNORECASE,
    )

    # More robust: find error table blocks
    error_table_patterns = [
        # Standard table: | Field | Value/Detail |
        r"\|\s*\*?\*?Error\s*(?:category|type)\*?\*?\s*\|\s*(.+?)\s*\|",
        r"\|\s*\*?\*?Location\*?\*?\s*\|\s*(.+?)\s*\|",
        r"\|\s*\*?\*?Confidence\*?\*?\s*\|\s*\*?\*?([0-9.]+)\*?\*?\s*\|",
        r"\|\s*\*?\*?Severity\*?\*?\s*\|\s*\*?\*?(.+?)\*?\*?\s*\|",
    ]

    # Try to find error entries from structured tables
    error_category = None
    error_location = None
    error_confidence = 0.0
    error_evidence = ""
    error_severity = None

    for line in text.split("\n"):
        line = line.strip()

        # Check for error category
        m = re.search(r"\|\s*\*?\*?Error\s*(?:category|type)\*?\*?\s*\|\s*(.+?)\s*\|", line, re.IGNORECASE)
        if m:
            error_category = m.group(1).strip("` ")

        # Check for location
        m = re.search(r"\|\s*\*?\*?(?:Error\s*)?Location\*?\*?\s*\|\s*(.+?)\s*\|", line, re.IGNORECASE)
        if m:
            error_location = m.group(1).strip("` ")

        # Check for confidence
        m = re.search(r"\|\s*\*?\*?Confidence\*?\*?\s*\|\s*\*?\*?([0-9.]+)\*?\*?\s*\|", line, re.IGNORECASE)
        if m:
            try:
                error_confidence = float(m.group(1))
            except ValueError:
                pass

        # Check for severity
        m = re.search(r"\|\s*\*?\*?Severity\*?\*?\s*\|\s*\*?\*?(.+?)\*?\*?\s*\|", line, re.IGNORECASE)
        if m:
            error_severity = m.group(1).strip("` **")

        # Reset after a section break or when we have a complete error
        if line.startswith("---") or line.startswith("## "):
            if error_category or error_location:
                # Check if this was a predicted error (not just ground truth)
                # by checking if we're in an "Error Detected" / "Predicted Error" section
                pass  # We'll build at the end

    # Build from sections
    for section_idx, section in enumerate(error_sections):
        if section_idx == 0:
            continue  # First split is before any error section

        # Extract error details from the section
        cat_m = re.search(
            r"\|\s*\*?\*?Error\s*(?:category|type)\*?\*?\s*\|\s*(.+?)\s*\|",
            section, re.IGNORECASE,
        )
        loc_m = re.search(
            r"\|\s*\*?\*?(?:Error\s*)?Location\*?\*?\s*\|\s*(.+?)\s*\|",
            section, re.IGNORECASE,
        )
        conf_m = re.search(
            r"\|\s*\*?\*?Confidence\*?\*?\s*\|\s*\*?\*?([0-9.]+)\*?\*?\s*\|",
            section, re.IGNORECASE,
        )
        sev_m = re.search(
            r"\|\s*\*?\*?Severity\*?\*?\s*\|\s*\*?\*?(.+?)\*?\*?\s*\|",
            section, re.IGNORECASE,
        )
        # Supporting evidence: the paragraph(s) after the table or in the section
        evid_m = re.search(
            r"(?:Supporting\s*Evidence|Root\s*Cause|What\s*it\s*does).*?\n(.*?)(?:\n\n|---|\Z)",
            section, re.DOTALL,
        )

        # Also try to find error description from "Predicted error:" line
        predicted_line = re.search(
            r"\*\*Predicted\s*error:?\*\*\s*(.+?)(?:\n|$)",
            text, re.IGNORECASE,
        )

        predicted_errors.append({
            "error_category": cat_m.group(1).strip("` ") if cat_m else "Equation / proof",
            "error_location": loc_m.group(1).strip("` ") if loc_m else (
                predicted_line.group(1)[:200] if predicted_line else ""
            ),
            "confidence": float(conf_m.group(1)) if conf_m else 0.85,
            "supporting_evidence": (
                evid_m.group(1).strip()[:500] if evid_m else (
                    predicted_line.group(1) if predicted_line else ""
                )
            ),
            "verifier_name": "text",
            "snippet_id": "",
        })

    # --- Strategy 4: Look for "Predicted error:" inline pattern ---
    if not predicted_errors:
        # Look for explicit predicted error statements
        predicted_error_match = re.search(
            r"\*\*Predicted\s*error:?\*\*\s*(.+?)(?:\n\n|###|---|\Z)",
            text, re.DOTALL | re.IGNORECASE,
        )
        if predicted_error_match:
            desc = predicted_error_match.group(1).strip()[:500]
            # Try to extract category and location from surrounding context
            cat_from_context = None
            loc_from_context = None

            # Check for category in nearby lines
            cat_m = re.search(r"\|\s*\*?\*?Error\s*(?:category|type)\*?\*?\s*\|\s*(.+?)\s*\|",
                             text, re.IGNORECASE)
            loc_m = re.search(r"\|\s*\*?\*?(?:Error\s*)?Location\*?\*?\s*\|\s*(.+?)\s*\|",
                             text, re.IGNORECASE)
            conf_m = re.search(r"\|\s*\*?\*?Confidence\*?\*?\s*\|\s*\*?\*?([0-9.]+)\*?\*?\s*\|",
                              text, re.IGNORECASE)

            predicted_errors.append({
                "error_category": cat_m.group(1).strip("` ") if cat_m else "Equation / proof",
                "error_location": loc_m.group(1).strip("` ") if loc_m else desc[:200],
                "confidence": float(conf_m.group(1)) if conf_m else 0.85,
                "supporting_evidence": desc,
                "verifier_name": "text",
                "snippet_id": "",
            })

    if predicted_errors:
        result["predicted_errors"] = predicted_errors
        result["verdict"] = result.get("verdict") or "NEEDS_CORRECTION"
        result["success"] = True
        return result

    # --- Strategy 5: Look for error indicators in prose (no structured tables) ---
    # Check for error-related keywords in short/medium prose summaries
    error_indicators = [
        r"(?:naming|contribution)\s+(?:mismatch|discrepancy)",
        r"(?:error|gap|flaw|incorrect|false|wrong|invalid|unsupported)",
        r"(?:does\s+not\s+(?:match|hold|follow|prove|support))",
    ]
    has_error_indicator = any(re.search(pat, text, re.IGNORECASE) for pat in error_indicators)

    if re.search(r"NEEDS_CORRECTION", text, re.IGNORECASE) or (
        has_error_indicator and len(text) < 2000
    ):
        # Try to find an error description from context
        error_desc = ""
        # Often the error is described after key indicators
        for pat in [
            r"(?:cross-snippet analysis|the paper)[^.]*?(?:error|gap|mismatch|flaw|incorrect|false|wrong)[^.]*\.",
            r"(?:detected|found|identified)[^.]*?(?:error|gap|mismatch|issue)[^.]*\.",
        ]:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                error_desc = m.group(0)
                break

        if not error_desc:
            # Just capture the relevant paragraph
            for keyword in ["naming mismatch", "error", "gap", "flaw"]:
                idx = text.lower().find(keyword)
                if idx > 0:
                    error_desc = text[max(0, idx-100):idx+300]
                    break

        result["predicted_errors"] = [{
            "error_category": "Unknown" if not has_error_indicator else "Equation / proof",
            "error_location": "",
            "confidence": 0.6 if not has_error_indicator else 0.75,
            "supporting_evidence": error_desc,
            "verifier_name": "text",
            "snippet_id": "",
        }]
        result["verdict"] = "NEEDS_CORRECTION"
        result["success"] = True
        return result

    # Could not find errors but verdict suggests no issues
    if re.search(r"\b(NO_ERROR|VALID|PASS)\b", text, re.IGNORECASE):
        result["verdict"] = "NO_ERROR"
        result["success"] = True
        return result

    # Could not extract meaningful data
    return None


def extract_paper_id_from_markdown(text: str, filename: str) -> str:
    """Extract paper_id from markdown content or filename."""
    # Try backtick-quoted paper ID: paper `2401.06455` or Paper `2401.06455`
    m = re.search(r"[Pp]aper\s*`([^`]+)`", text[:500])
    if m:
        candidate = m.group(1)
        # Validate it looks like a paper ID (arxiv or DOI)
        if re.match(r"^(?:\d{4}\.\d{4,5}|10\.\d{4,5}/)", candidate):
            return candidate

    # Try explicit paper ID patterns: paper_2401.06455 or paper 2401.06455
    m = re.search(r"paper[_ ]+(?:ID[_ ]+)?([240]\d{3,4}(?:\.\d{4,5})?(?:v\d+)?)",
                  text[:500], re.IGNORECASE)
    if m:
        candidate = m.group(1)
        # Don't capture .json extensions or verification_report suffixes
        if not candidate.endswith(".json") and "verification_report" not in candidate:
            return candidate

    # DOI patterns (but not file paths)
    m = re.search(r"\b(10\.\d{4,5}/[\w.-]+)", text[:500])
    if m:
        candidate = m.group(1)
        if not candidate.endswith(".json"):
            return candidate

    # Try to derive from filename
    basename = filename.replace(".md", "")
    # Convert sanitized filename back to paper ID
    # Common patterns: 10.1007_s10571 -> 10.1007/s10571
    # arxiv: 2401.06455 stays the same
    if basename.startswith("10.") and "_" in basename:
        # Try to convert back: replace first _ with /
        parts = basename.split("_", 1)
        candidate = f"{parts[0]}/{parts[1]}"
        return candidate

    return basename


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <eval_directory>")
        sys.exit(1)

    eval_dir = Path(sys.argv[1])
    if not eval_dir.is_dir():
        print(f"Error: {eval_dir} is not a directory")
        sys.exit(1)

    # Also look for companion JSONs in outputs/ (root)
    root_outputs = Path("outputs")  # project root outputs directory

    # Index report JSONs from both locations
    json_by_paper: dict[str, Path] = {}
    search_dirs = [eval_dir, root_outputs] if root_outputs != eval_dir else [eval_dir]

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

    print(f"Found {len(json_by_paper)} companion JSON report(s) across:")
    for sd in search_dirs:
        count = sum(1 for p in json_by_paper.values() if str(sd) in str(p))
        if count:
            print(f"  {sd}/ ({count} files)")

    # Find all .md files (exclude non-paper files)
    skip_names = {"run_summary.md", "README.md", "SUMMARY.md"}
    md_files = sorted(
        f for f in eval_dir.glob("*.md") if f.name not in skip_names
    )
    print(f"\nFound {len(md_files)} .md file(s) in {eval_dir}")

    # Process each .md file
    all_results = []
    verified = 0
    failed = 0

    for md_file in md_files:
        text = md_file.read_text()
        pid = extract_paper_id_from_markdown(text, md_file.name)

        result = None

        # Option 1: Use companion JSON if available
        if pid in json_by_paper:
            try:
                parsed = json.loads(json_by_paper[pid].read_text())
                result = {
                    "paper_id": pid,
                    "success": True,
                    "predicted_errors": parsed.get("predicted_errors", []),
                    "verdict": parsed.get("verdict", "UNKNOWN"),
                    "title": parsed.get("title", ""),
                    "paper_category": parsed.get("paper_category", ""),
                }
                if "statistics" in parsed:
                    result["statistics"] = parsed["statistics"]
                verified += 1
                print(f"  [json] {pid}: {len(result['predicted_errors'])} error(s), "
                      f"verdict={result['verdict']}")
            except Exception as e:
                print(f"  [WARN] Could not read JSON for {pid}: {e}")

        # Option 2: Parse markdown
        if result is None:
            extracted = extract_from_markdown(text)
            if extracted and extracted.get("success"):
                result = {
                    "paper_id": pid,
                    "success": True,
                    "predicted_errors": extracted.get("predicted_errors", []),
                    "verdict": extracted.get("verdict", "UNKNOWN"),
                }
                verified += 1
                print(f"  [md]   {pid}: {len(result['predicted_errors'])} error(s), "
                      f"verdict={result['verdict']}")
            else:
                # Check if this is a partial/no-verdict output
                if "TIMEOUT" in text or "timeout" in text.lower():
                    result = {
                        "paper_id": pid,
                        "success": False,
                        "error": "Timeout",
                    }
                elif len(text) < 500:
                    result = {
                        "paper_id": pid,
                        "success": False,
                        "error": f"Output too short ({len(text)} chars)",
                    }
                else:
                    result = {
                        "paper_id": pid,
                        "success": False,
                        "predicted_errors": [],
                        "verdict": "UNKNOWN",
                        "raw_output_length": len(text),
                    }
                failed += 1
                print(f"  [fail] {pid}: could not parse, verdict=UNKNOWN ({len(text)} chars)")

        # Add raw_output_length for reference
        if "raw_output_length" not in result:
            result["raw_output_length"] = len(text)

        # Add source tracking
        result["verification_timestamp"] = datetime.now().isoformat()
        all_results.append(result)

    # Build batch_results.json
    batch_data = {
        "batch_timestamp": datetime.now().isoformat(),
        "total_papers": len(md_files),
        "verified": verified,
        "failed": failed,
        "mode": "eval-dir (rebuilt from .md + companion JSONs)",
        "threshold": 0.0,
        "duration_seconds": 0,
        "results": all_results,
    }

    output_file = eval_dir / "batch_results.json"
    output_file.write_text(json.dumps(batch_data, indent=2))
    print(f"\n{'='*60}")
    print(f"Rebuilt {output_file}")
    print(f"  Total papers: {len(md_files)}")
    print(f"  Successfully parsed: {verified}")
    print(f"  Failed to parse: {failed}")
    print(f"\nYou can now run:")
    print(f"  python scripts/batch_verify.py --eval-dir {eval_dir} --llm-judge")


if __name__ == "__main__":
    main()
