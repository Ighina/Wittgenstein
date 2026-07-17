#!/usr/bin/env python3
"""
evaluate_verifier.py — Evaluate the verify-paper skill against the PaperAudit-Bench dataset.

Two-phase workflow:
  1. --prepare : Extract papers from PaperAudit, rebuild markdown, save ground truth.
  2. --evaluate: Compare verification reports against ground truth using LLM-as-a-judge.

Usage:
  python scripts/evaluate_verifier.py --llm claude-sonnet-4-5-20250929 --prepare --limit 5
  python scripts/evaluate_verifier.py --llm claude-sonnet-4-5-20250929 --evaluate --limit 5
  python scripts/evaluate_verifier.py --llm claude-sonnet-4-5-20250929 --prepare --evaluate --limit 3
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths (relative to this script's location)
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Default PaperAudit origin data path — matches the layout relative to
# paperena-agent/paperena_claude. Override with --paperaudit-base.
PAPERAUDIT_BASE_DEFAULT = (
    PROJECT_ROOT.parent
    / "paperena-agent"
    / "PaperAudit_Origin_Data"
    / "PaperAudit_Origin_Data"
    / "NeurIPS_25"
)

# WORKSPACE_BASE = PROJECT_ROOT / "outputs" / "paperaudit-workspace"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "paperaudit-eval"
WORKSPACE_BASE = PROJECT_ROOT / "data" / "paperaudit-workspace"

# LLM name mapping: the --llm argument uses a short name, mapped to paper_synth_<full>.json
LLM_NAME_MAP = {
    "claude-sonnet-4-5-20250929": "claude-sonnet-4-5-20250929",
    "doubao-seed-1-6-251015": "doubao-seed-1-6-251015",
    "gemini-2.5-pro": "gemini-2.5-pro",
    "glm-4.6": "glm-4.6",
    "gpt-5-2025-08-07": "gpt-5-2025-08-07",
    "grok-4": "grok-4",
    "o4-mini": "o4-mini",
    "qwen_qwen3-vl-235b-a22b-instruct": "qwen_qwen3-vl-235b-a22b-instruct",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sanitize_paper_id(dirname: str) -> str:
    """Derive a clean paper_id from the directory name.

    Directory names look like: '25671-RAG4GFM__Bridging_Knowledge_Gaps_...'
    We keep the numeric prefix + shortname part.
    """
    # Take up to 80 chars, replace problematic characters
    clean = dirname[:80].replace(" ", "_").replace("/", "_")
    return clean


def get_paper_dirs(base_path: Path) -> List[Path]:
    """Return sorted list of paper directories."""
    if not base_path.exists():
        print(f"ERROR: PaperAudit base path not found: {base_path}")
        sys.exit(1)
    dirs = sorted(
        [d for d in base_path.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )
    return dirs


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file with clear error messaging."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    """Save data as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Phase 1: Prepare — Extract papers from PaperAudit
# ---------------------------------------------------------------------------


def rebuild_markdown(content_list: List[Dict[str, Any]]) -> str:
    """Rebuild a markdown string from PaperAudit's paper.content list.

    Each item: {"type": "text", "text": "...", "index": N, "section": "..."}
    or:        {"type": "image_url", "image_url": {"url": "data:..."}, ...}
    """
    sections: List[str] = []
    current_section = None

    for item in content_list:
        item_type = item.get("type", "text")
        section_label = item.get("section", "")

        # Emit section header when section changes
        if section_label and section_label != current_section:
            current_section = section_label
            # Only emit a header if it looks like a real section name
            if section_label not in ("", "header"):
                sections.append(f"\n## {section_label}\n")

        if item_type == "text":
            text = item.get("text", "").strip()
            if text:
                sections.append(text)

        elif item_type == "image_url":
            img_data = item.get("image_url", {})
            img_url = img_data.get("url", "") if isinstance(img_data, dict) else ""
            idx = item.get("index", "?")
            if img_url.startswith("data:image"):
                # Base64 embedded image — include a placeholder with index
                sections.append(
                    f"\n![Figure {idx}](image_{idx}.jpg)\n"
                    f"<!-- Caption: {section_label}, index {idx} -->\n"
                )
            else:
                sections.append(f"\n![Figure {idx}]({img_url})\n")

    return "\n\n".join(sections)


def prepare_papers(
    llm_name: str,
    limit: Optional[int] = None,
    paperaudit_base: Optional[Path] = None,
    workspace_base: Optional[Path] = None,
) -> List[str]:
    """Extract papers from PaperAudit dataset.

    For each paper:
      - Load paper_synth_{llm_name}.json
      - Save ground truth from audit_log.edits
      - Rebuild markdown and save as paper.md

    Returns list of prepared paper IDs.
    """
    base_path = paperaudit_base or PAPERAUDIT_BASE_DEFAULT
    ws_base = workspace_base or WORKSPACE_BASE
    paper_dirs = get_paper_dirs(base_path)
    synth_filename = f"paper_synth_{llm_name}.json"
    prepared: List[str] = []

    total = len(paper_dirs)
    if limit:
        paper_dirs = paper_dirs[:limit]
        print(f"Processing {len(paper_dirs)} of {total} papers (limit={limit})\n")
    else:
        print(f"Processing all {total} papers\n")

    for i, paper_dir in enumerate(paper_dirs, 1):
        paper_id = sanitize_paper_id(paper_dir.name)
        synth_path = paper_dir / synth_filename

        if not synth_path.exists():
            print(
                f"  [{i}/{len(paper_dirs)}] SKIP {paper_id}: {synth_filename} not found"
            )
            continue

        try:
            synth_data = load_json(synth_path)
        except Exception as e:
            print(f"  [{i}/{len(paper_dirs)}] ERROR {paper_id}: {e}")
            continue

        # Extract ground truth
        audit_log = synth_data.get("audit_log", {})
        edits = audit_log.get("edits", [])

        # Extract paper content and rebuild markdown
        paper = synth_data.get("paper", {})
        content_list = paper.get("content", [])
        if not content_list:
            print(f"  [{i}/{len(paper_dirs)}] SKIP {paper_id}: empty paper.content")
            continue

        md_text = rebuild_markdown(content_list)

        # Determine output directory
        workspace_dir = ws_base / paper_id
        workspace_dir.mkdir(parents=True, exist_ok=True)

        # Save ground truth
        gt_path = workspace_dir / "ground_truth.json"
        save_json(gt_path, edits)

        # Save markdown paper
        md_path = workspace_dir / "paper.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_text)

        num_edits = len(edits)
        num_sections = len(content_list)
        print(
            f"  [{i}/{len(paper_dirs)}] OK  {paper_id}: "
            f"{num_edits} GT edits, {num_sections} sections → {md_path}"
        )
        prepared.append(paper_id)

    print(f"\nPrepared {len(prepared)} papers.")
    return prepared


# ---------------------------------------------------------------------------
# Phase 2: Evaluate — Compare verification reports against ground truth
# ---------------------------------------------------------------------------


def load_anthropic_client(api_key: Optional[str] = None):
    """Load the Anthropic client, trying multiple auth sources.

    Returns the client or None if anthropic is unavailable.
    """
    try:
        import anthropic  # type: ignore
    except ImportError:
        print(
            "WARNING: anthropic package not installed. "
            "Run: pip install anthropic\n"
            "  Falling back to text-overlap heuristic. "
            "Use --no-judge to suppress this warning."
        )
        return None

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        # Try reading from ~/.claude/credentials.json or similar
        cred_path = Path.home() / ".claude" / "credentials.json"
        if cred_path.exists():
            try:
                creds = json.loads(cred_path.read_text())
                key = creds.get("oauth_token") or creds.get("api_key")
                if key:
                    key = key if key.startswith("sk-") else None
            except Exception:
                pass

    if not key:
        print(
            "ERROR: No Anthropic API key found. Set ANTHROPIC_API_KEY env var "
            "or pass --api-key."
        )
        sys.exit(1)

    return anthropic.Anthropic(api_key=key)


def text_overlap_score(a: str, b: str) -> float:
    """Compute a simple token-overlap score between two strings.

    Returns a value in [0, 1] indicating how much of the shorter string's
    tokens appear in the longer string.
    """
    if not a or not b:
        return 0.0

    # Tokenize on word boundaries, lowercase
    def tokens(s: str) -> set:
        return set(re.findall(r"[a-zA-Z0-9]+", s.lower()))

    ta = tokens(a)
    tb = tokens(b)

    if not ta or not tb:
        return 0.0

    smaller = ta if len(ta) <= len(tb) else tb
    larger = tb if len(ta) <= len(tb) else ta

    overlap = smaller & larger
    return len(overlap) / len(smaller) if smaller else 0.0


def prefilter_candidate_pairs(
    predicted_errors: List[Dict[str, Any]],
    ground_truth_edits: List[Dict[str, Any]],
    min_overlap: float = 0.05,
) -> List[Tuple[int, int]]:
    """Return candidate (pred_idx, gt_idx) pairs worth sending to the LLM judge.

    Uses token overlap between predicted supporting_evidence/error_location
    and ground truth error_explanation/target_find to filter out obviously
    unrelated pairs.
    """
    candidates: List[Tuple[int, int]] = []

    for p_idx, pred in enumerate(predicted_errors):
        pred_text = " ".join(
            [
                str(pred.get("supporting_evidence", "")),
                str(pred.get("error_location", "")),
                str(pred.get("snippet_id", "")),
            ]
        )

        for g_idx, gt in enumerate(ground_truth_edits):
            gt_text = " ".join(
                [
                    str(gt.get("error_explanation", "")),
                    str(gt.get("target_find", "")),
                    str(gt.get("location", "")),
                    str(gt.get("corruption_type", "")),
                ]
            )

            score = text_overlap_score(pred_text, gt_text)
            if score >= min_overlap:
                candidates.append((p_idx, g_idx))

    return candidates


def _extract_text_from_response(response: Any) -> str:
    """Extract text content from an Anthropic SDK response, handling ThinkingBlock.

    The Anthropic SDK (≥0.50) may return ThinkingBlock objects alongside
    TextBlock objects in response.content.  We iterate over all blocks and
    concatenate only the TextBlock.text payloads.
    """
    parts: List[str] = []
    block_types: List[str] = []
    for block in response.content:
        block_types.append(type(block).__name__)
        # TextBlock has a .text attribute
        if hasattr(block, "text"):
            parts.append(block.text)
    if not parts and block_types:
        print(
            f"  [JUDGE WARN] No TextBlock found in response. "
            f"Blocks present: {block_types}. "
            f"This usually means max_tokens is too low and thinking consumed "
            f"the entire budget — try increasing max_tokens.",
            file=sys.stderr,
        )
    return " ".join(parts)


def _parse_judge_json(raw_text: str) -> Optional[Dict[str, Any]]:
    """Robustly extract a JSON object from an LLM judge response.

    Tries, in order:
      1. Parse the whole string as JSON (stripping markdown fences first).
      2. Find a JSON object with a balanced-brace search.
      3. Fall back to the simple regex (which fails on nested braces).
    """
    if not raw_text:
        return None

    # Strip ```json / ``` fences
    cleaned = re.sub(r"```(?:json)?\s*", "", raw_text).strip()
    cleaned = cleaned.strip("`").strip()

    # 1) Try parsing the whole string
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 2) Balanced-brace extraction: find the first '{' then track depth
    start = cleaned.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(cleaned)):
        if cleaned[i] == "{":
            depth += 1
        elif cleaned[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(cleaned[start : i + 1])
                except json.JSONDecodeError:
                    break
    # 3) Fallback to original simple regex (no nested braces)
    json_match = re.search(r"\{[^{}]+\}", cleaned, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def judge_pair(
    client: Any,
    predicted_error: Dict[str, Any],
    ground_truth_edit: Dict[str, Any],
    paper_title: str = "",
    mode: str = "exhaustive",
    strictness: str = "strict",
) -> Dict[str, Any]:
    """Use an LLM to judge whether a predicted error matches a ground truth edit.

    Returns: {"verdict": "match|partial_match|no_match", "confidence": float, "reasoning": str}
    """
    # Build a note about verification settings to inform the judge
    mode_note = {
        "exhaustive": "The verifier checked every snippet in the paper (exhaustive mode).",
        "uncertainty": "The verifier only checked high-uncertainty snippets (uncertainty mode, threshold ≥ 0.30). This means some low-risk regions were not verified.",
    }.get(mode, f"Verification mode: {mode}")

    strictness_note = {
        "strict": "The verifier was configured with STRICT error sensitivity — it should only flag erratum-worthy, high-confidence errors.",
        "lenient": "The verifier was configured with LENIENT error sensitivity — it may flag a broader range of potential issues including style and clarity concerns.",
    }.get(strictness, f"Error sensitivity: {strictness}")

    prompt = f"""You are evaluating an error-detection system for scientific papers. Your task is to determine whether a PREDICTED error (found by an automated verifier) correctly identifies the same underlying issue as a GROUND TRUTH error (a known, intentionally-inserted mistake).

Paper: {paper_title}

Verification context:
- {mode_note}
- {strictness_note}

--- GROUND TRUTH ERROR ---
Category: {ground_truth_edit.get('corruption_type', 'unknown')}
Difficulty: {ground_truth_edit.get('difficulty', 'unknown')}
Location: {ground_truth_edit.get('location', 'unknown')}
Error explanation: {ground_truth_edit.get('error_explanation', 'N/A')}
What text was changed (target_find): {str(ground_truth_edit.get('target_find', ''))[:500]}

--- PREDICTED ERROR ---
Category: {predicted_error.get('error_category', 'unknown')}
Location: {predicted_error.get('error_location', 'unknown')}
Confidence: {predicted_error.get('confidence', 'N/A')}
Evidence: {predicted_error.get('supporting_evidence', 'N/A')}
Verifier: {predicted_error.get('verifier_name', 'unknown')}

Judge whether the predicted error identifies the SAME UNDERLYING ISSUE as the ground truth error. Consider:
- Do they point to the same location / content in the paper?
- Does the verifier's reasoning align with what was actually wrong?
- Even if wording differs, is the core problem the same?
- Take the verification context into account: in STRICT mode, be more demanding about what counts as a match (the verifier should be precise); in LENIENT mode, accept broader matches.

Respond with a JSON object ONLY:
{{"verdict": "match|partial_match|no_match", "confidence": 0.0-1.0, "reasoning": "one sentence explaining your decision"}}

- "match": The predicted error correctly identifies the ground truth issue.
- "partial_match": The predicted error partially overlaps with the ground truth (e.g., right location but wrong explanation, or vice versa).
- "no_match": The predicted error identifies a different issue or misses the ground truth entirely."""

    try:
        response = client.messages.create(
            model="deepseek-v4-pro",
            max_tokens=8192,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )

        # Extract text robustly — handles ThinkingBlock alongside TextBlock
        raw_text = _extract_text_from_response(response)

        if not raw_text:
            print(
                f"  [JUDGE WARN] Empty response content. "
                f"Blocks: {[type(b).__name__ for b in response.content]}",
                file=sys.stderr,
            )
            return {
                "verdict": "error",
                "confidence": 0.0,
                "reasoning": "Empty response content from judge",
            }

        parsed = _parse_judge_json(raw_text)
        if parsed is not None:
            return parsed

        # If we get here, parsing failed
        print(
            f"  [JUDGE WARN] Failed to parse judge response:\n"
            f"    Raw text (first 300 chars): {raw_text[:300]}",
            file=sys.stderr,
        )
        return {
            "verdict": "no_match",
            "confidence": 0.0,
            "reasoning": f"Failed to parse judge response: {raw_text[:100]}",
        }

    except Exception as e:
        print(
            f"  [JUDGE ERROR] API call failed: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        return {
            "verdict": "error",
            "confidence": 0.0,
            "reasoning": f"Judge error: {type(e).__name__}: {e}",
        }


def align_errors(
    predicted_errors: List[Dict[str, Any]],
    ground_truth_edits: List[Dict[str, Any]],
    client: Optional[Any] = None,
    paper_title: str = "",
    mode: str = "exhaustive",
    strictness: str = "strict",
) -> Dict[str, Any]:
    """Align predicted errors with ground truth edits.

    Without an LLM client, uses text-overlap heuristic.
    With an LLM client, uses LLM-as-a-judge for candidate pairs.

    Returns alignment results including TP, FP, FN counts and matched pairs.
    """
    n_pred = len(predicted_errors)
    n_gt = len(ground_truth_edits)

    # Track which GT edits are matched and which predicted errors are matched
    matched_gt: set = set()
    matched_pred: set = set()
    partial_matches: List[Dict[str, Any]] = []

    # Track judge verdict statistics for diagnostics
    judge_stats: Dict[str, int] = Counter()

    if client is not None:
        # LLM-as-a-judge path
        candidates = prefilter_candidate_pairs(predicted_errors, ground_truth_edits)
        print(
            f"      LLM judge: {len(candidates)} candidate pairs to evaluate "
            f"(out of {n_pred * n_gt} possible)"
        )

        # Sort candidates — prioritize those with higher text overlap
        # Compute overlap for sorting
        def _overlap(pair: Tuple[int, int]) -> float:
            p_idx, g_idx = pair
            p = predicted_errors[p_idx]
            g = ground_truth_edits[g_idx]
            pred_text = " ".join(
                [
                    str(p.get("supporting_evidence", "")),
                    str(p.get("error_location", "")),
                ]
            )
            gt_text = " ".join(
                [str(g.get("error_explanation", "")), str(g.get("target_find", ""))]
            )
            return text_overlap_score(pred_text, gt_text)

        candidates.sort(key=_overlap, reverse=True)

        for p_idx, g_idx in candidates:
            # Skip if either side is already definitively matched
            if p_idx in matched_pred and g_idx in matched_gt:
                continue

            result = judge_pair(
                client,
                predicted_errors[p_idx],
                ground_truth_edits[g_idx],
                paper_title,
                mode=mode,
                strictness=strictness,
            )

            verdict = result.get("verdict", "unknown")
            judge_stats[verdict] += 1

            if verdict == "match":
                matched_pred.add(p_idx)
                matched_gt.add(g_idx)
                partial_matches.append(
                    {
                        "pred_idx": p_idx,
                        "gt_idx": g_idx,
                        "verdict": "match",
                        "judge_confidence": result.get("confidence", 1.0),
                    }
                )
            elif verdict == "partial_match":
                partial_matches.append(
                    {
                        "pred_idx": p_idx,
                        "gt_idx": g_idx,
                        "verdict": "partial_match",
                        "judge_confidence": result.get("confidence", 0.5),
                    }
                )
            # "error" verdict means the API call or parsing failed —
            # already logged to stderr by judge_pair(); counted in stats

        # Print judge verdict summary for this paper
        if judge_stats:
            stats_str = ", ".join(f"{k}={v}" for k, v in sorted(judge_stats.items()))
            print(f"      Judge verdicts: {stats_str}")
            if judge_stats.get("error", 0) > 0:
                print(
                    f"      ⚠ {judge_stats['error']} judge call(s) failed — "
                    f"check stderr for [JUDGE ERROR] messages"
                )
    else:
        # Heuristic path: high-overlap pairs → match
        candidates = prefilter_candidate_pairs(
            predicted_errors, ground_truth_edits, min_overlap=0.15
        )

        # Greedy matching: best overlap first
        def _overlap(pair: Tuple[int, int]) -> float:
            p_idx, g_idx = pair
            p = predicted_errors[p_idx]
            g = ground_truth_edits[g_idx]
            pred_text = " ".join(
                [
                    str(p.get("supporting_evidence", "")),
                    str(p.get("error_location", "")),
                ]
            )
            gt_text = " ".join(
                [str(g.get("error_explanation", "")), str(g.get("target_find", ""))]
            )
            return text_overlap_score(pred_text, gt_text)

        candidates.sort(key=_overlap, reverse=True)

        for p_idx, g_idx in candidates:
            if p_idx in matched_pred or g_idx in matched_gt:
                continue
            score = _overlap((p_idx, g_idx))
            if score >= 0.15:
                matched_pred.add(p_idx)
                matched_gt.add(g_idx)
                partial_matches.append(
                    {
                        "pred_idx": p_idx,
                        "gt_idx": g_idx,
                        "verdict": "match",
                        "overlap_score": score,
                    }
                )

    # Classify
    tp = len(matched_gt)
    fp = n_pred - len(matched_pred)
    fn = n_gt - len(matched_gt)

    # Per-difficulty breakdown
    difficulty_gt_found: Dict[str, int] = Counter()
    difficulty_gt_total: Dict[str, int] = Counter()
    for g_idx, gt in enumerate(ground_truth_edits):
        diff = gt.get("difficulty", "unknown")
        difficulty_gt_total[diff] += 1
        if g_idx in matched_gt:
            difficulty_gt_found[diff] += 1

    # Per-corruption-type breakdown
    type_gt_found: Dict[str, int] = Counter()
    type_gt_total: Dict[str, int] = Counter()
    for g_idx, gt in enumerate(ground_truth_edits):
        ctype = gt.get("corruption_type", "unknown")
        type_gt_total[ctype] += 1
        if g_idx in matched_gt:
            type_gt_found[ctype] += 1

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_pred": n_pred,
        "n_gt": n_gt,
        "matched_pred_indices": sorted(matched_pred),
        "matched_gt_indices": sorted(matched_gt),
        "partial_matches": partial_matches,
        "difficulty_gt_found": dict(difficulty_gt_found),
        "difficulty_gt_total": dict(difficulty_gt_total),
        "type_gt_found": dict(type_gt_found),
        "type_gt_total": dict(type_gt_total),
        "judge_stats": dict(judge_stats),
    }


def compute_metrics(alignment: Dict[str, Any]) -> Dict[str, float]:
    """Compute precision, recall, F1 from alignment results."""
    tp = alignment["tp"]
    fp = alignment["fp"]
    fn = alignment["fn"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    # Per-difficulty recall
    difficulty_recall: Dict[str, float] = {}
    for diff, total in alignment["difficulty_gt_total"].items():
        found = alignment["difficulty_gt_found"].get(diff, 0)
        difficulty_recall[diff] = found / total if total > 0 else 0.0

    # Per-type recall
    type_recall: Dict[str, float] = {}
    for ctype, total in alignment["type_gt_total"].items():
        found = alignment["type_gt_found"].get(ctype, 0)
        type_recall[ctype] = found / total if total > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        **{f"recall_{diff}": v for diff, v in difficulty_recall.items()},
        **{f"recall_type_{ctype}": v for ctype, v in type_recall.items()},
    }


def evaluate_papers(
    llm_name: str,
    limit: Optional[int] = None,
    api_key: Optional[str] = None,
    use_llm_judge: bool = True,
    mode: str = "exhaustive",
    strictness: str = "strict",
    workspace_base: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Evaluate verification reports against ground truth for all papers.

    Returns (per_paper_results, summary).
    """
    ws_base = workspace_base or WORKSPACE_BASE

    client = None
    if use_llm_judge:
        client = load_anthropic_client(api_key)
        if client is not None:
            print("LLM-as-a-judge: ENABLED\n")
        else:
            print("LLM-as-a-judge: UNAVAILABLE (using text-overlap heuristic)\n")
    else:
        print("LLM-as-a-judge: DISABLED (using text-overlap heuristic)\n")

    # Discover prepared papers (those with ground_truth.json)
    if not ws_base.exists():
        print(f"ERROR: Workspace directory not found: {ws_base}")
        print("Run with --prepare first.")
        sys.exit(1)

    paper_ids = sorted(
        [
            d.name
            for d in ws_base.iterdir()
            if d.is_dir() and (d / "ground_truth.json").exists()
        ]
    )

    if not paper_ids:
        print("ERROR: No prepared papers found. Run with --prepare first.")
        sys.exit(1)

    if limit:
        paper_ids = paper_ids[:limit]
        print(f"Evaluating {len(paper_ids)} papers (limit={limit})\n")
    else:
        print(f"Evaluating all {len(paper_ids)} papers\n")

    per_paper: List[Dict[str, Any]] = []

    for i, paper_id in enumerate(paper_ids, 1):
        workspace_dir = ws_base / paper_id
        gt_path = workspace_dir / "ground_truth.json"
        report_path = workspace_dir / "verification-report.json"

        if not report_path.exists():
            print(
                f"  [{i}/{len(paper_ids)}] SKIP {paper_id}: no verification-report.json"
            )
            continue

        try:
            ground_truth = load_json(gt_path)
            report = load_json(report_path)
        except Exception as e:
            print(f"  [{i}/{len(paper_ids)}] ERROR {paper_id}: {e}")
            continue

        predicted_errors = report.get("predicted_errors", [])
        paper_title = report.get("title", paper_id)

        # For each GT edit, determine which section/snippet it falls in
        # (ground truth edits have target_find text we can search)

        alignment = align_errors(
            predicted_errors,
            ground_truth,
            client,
            paper_title,
            mode=mode,
            strictness=strictness,
        )
        metrics = compute_metrics(alignment)

        result = {
            "paper_id": paper_id,
            "total_gt": alignment["n_gt"],
            "total_pred": alignment["n_pred"],
            "tp": alignment["tp"],
            "fp": alignment["fp"],
            "fn": alignment["fn"],
            "mode": mode,
            "strictness": strictness,
            **metrics,
        }
        per_paper.append(result)

        print(
            f"  [{i}/{len(paper_ids)}] {paper_id}: "
            f"GT={alignment['n_gt']} Pred={alignment['n_pred']} "
            f"TP={alignment['tp']} FP={alignment['fp']} FN={alignment['fn']} "
            f"P={metrics['precision']:.2f} R={metrics['recall']:.2f} F1={metrics['f1']:.2f}"
        )

    # Aggregate summary
    if not per_paper:
        print("\nNo papers evaluated.")
        return [], {}

    total_gt = sum(r["total_gt"] for r in per_paper)
    total_pred = sum(r["total_pred"] for r in per_paper)
    total_tp = sum(r["tp"] for r in per_paper)
    total_fp = sum(r["fp"] for r in per_paper)
    total_fn = sum(r["fn"] for r in per_paper)

    micro_precision = (
        total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    )
    micro_recall = (
        total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    )
    micro_f1 = (
        (2 * micro_precision * micro_recall / (micro_precision + micro_recall))
        if (micro_precision + micro_recall) > 0
        else 0.0
    )

    macro_precision = sum(r["precision"] for r in per_paper) / len(per_paper)
    macro_recall = sum(r["recall"] for r in per_paper) / len(per_paper)
    macro_f1 = sum(r["f1"] for r in per_paper) / len(per_paper)

    # Per-difficulty aggregation
    difficulty_totals: Dict[str, int] = Counter()
    difficulty_founds: Dict[str, int] = Counter()
    type_totals: Dict[str, int] = Counter()
    type_founds: Dict[str, int] = Counter()

    for paper_id in paper_ids:
        gt_path = ws_base / paper_id / "ground_truth.json"
        report_path = ws_base / paper_id / "verification-report.json"
        if not gt_path.exists() or not report_path.exists():
            continue
        try:
            ground_truth = load_json(gt_path)
            report = load_json(report_path)
        except Exception:
            continue

        predicted_errors = report.get("predicted_errors", [])
        alignment = align_errors(
            predicted_errors, ground_truth, None, "", mode=mode, strictness=strictness
        )
        for diff, count in alignment["difficulty_gt_total"].items():
            difficulty_totals[diff] += count
        for diff, count in alignment["difficulty_gt_found"].items():
            difficulty_founds[diff] += count
        for ctype, count in alignment["type_gt_total"].items():
            type_totals[ctype] += count
        for ctype, count in alignment["type_gt_found"].items():
            type_founds[ctype] += count

    difficulty_recall_agg = {
        f"recall_{diff}": (difficulty_founds.get(diff, 0) / total if total > 0 else 0.0)
        for diff, total in difficulty_totals.items()
    }
    type_recall_agg = {
        f"recall_type_{ctype}": (
            type_founds.get(ctype, 0) / total if total > 0 else 0.0
        )
        for ctype, total in type_totals.items()
    }

    summary = {
        "total_papers": len(per_paper),
        "total_gt_errors": total_gt,
        "total_pred_errors": total_pred,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "mode": mode,
        "strictness": strictness,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        **difficulty_recall_agg,
        **type_recall_agg,
    }

    print(f"\n{'='*60}")
    print(
        f"SUMMARY: {len(per_paper)} papers, {total_gt} GT errors, {total_pred} predicted errors"
    )
    print(f"  Micro: P={micro_precision:.3f} R={micro_recall:.3f} F1={micro_f1:.3f}")
    print(f"  Macro: P={macro_precision:.3f} R={macro_recall:.3f} F1={macro_f1:.3f}")
    if difficulty_recall_agg:
        print(
            f"  Difficulty recall: { {k: f'{v:.3f}' for k, v in difficulty_recall_agg.items()} }"
        )

    return per_paper, summary


def write_results(
    llm_name: str,
    per_paper: List[Dict[str, Any]],
    summary: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Write per-paper metrics and summary to CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-paper CSV
    per_paper_path = output_dir / f"{llm_name}_metrics.csv"
    if per_paper:
        # Collect all possible keys across all results
        all_keys: List[str] = []
        seen_keys = set()
        for r in per_paper:
            for k in r:
                if k not in seen_keys:
                    all_keys.append(k)
                    seen_keys.add(k)

        with open(per_paper_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(per_paper)

        print(f"\nPer-paper metrics saved to: {per_paper_path}")

    # Summary CSV
    summary_path = output_dir / f"{llm_name}_summary.csv"
    if summary:
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
            writer.writeheader()
            writer.writerow(summary)

        print(f"Summary saved to: {summary_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the verify-paper skill against PaperAudit-Bench.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare papers from the dataset (extract .md + ground truth)
  python scripts/evaluate_verifier.py --llm claude-sonnet-4-5-20250929 --prepare --limit 5

  # After running verify-paper on each paper, evaluate the results
  python scripts/evaluate_verifier.py --llm claude-sonnet-4-5-20250929 --evaluate --limit 5

  # Combine both: prepare + evaluate (useful when verification reports exist)
  python scripts/evaluate_verifier.py --llm claude-sonnet-4-5-20250929 --prepare --evaluate --limit 3

  # Specify a custom PaperAudit base path
  python scripts/evaluate_verifier.py --llm claude-sonnet-4-5-20250929 --prepare \\
      --paperaudit-base /path/to/PaperAudit_Origin_Data/PaperAudit_Origin_Data/NeurIPS_25

LLM names (paper_synth_{name}.json):
  claude-sonnet-4-5-20250929, gpt-5-2025-08-07, gemini-2.5-pro,
  grok-4, o4-mini, glm-4.6, doubao-seed-1-6-251015,
  qwen_qwen3-vl-235b-a22b-instruct
        """,
    )

    parser.add_argument(
        "--llm",
        required=True,
        help="Name of the LLM used to corrupt papers (e.g., claude-sonnet-4-5-20250929)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of papers to process",
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Extract papers from PaperAudit dataset and rebuild markdown",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate verification reports against ground truth",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(RESULTS_DIR),
        help=f"Directory for output CSV files (default: {RESULTS_DIR})",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Anthropic API key for LLM-as-a-judge (default: ANTHROPIC_API_KEY env var)",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Disable LLM judge; use text-overlap heuristic instead",
    )
    parser.add_argument(
        "--mode",
        choices=["exhaustive", "uncertainty"],
        default="exhaustive",
        help="Verification mode used by verify-paper skill (default: exhaustive). "
        "In exhaustive mode every snippet is verified; "
        "in uncertainty mode only high-uncertainty snippets are verified.",
    )
    parser.add_argument(
        "--strictness",
        choices=["strict", "lenient"],
        default="strict",
        help="Error sensitivity used by verify-paper skill (default: strict). "
        "'strict' only flags erratum-worthy errors; "
        "'lenient' applies broader error detection.",
    )
    parser.add_argument(
        "--paperaudit-base",
        type=str,
        default=str(PAPERAUDIT_BASE_DEFAULT),
        help=f"Path to the PaperAudit dataset directory "
        f"(default: {PAPERAUDIT_BASE_DEFAULT}). "
        f"The directory should contain subdirectories for each paper, "
        f"each with paper_synth_{{llm}}.json files.",
    )
    parser.add_argument(
        "--workspace-dir",
        type=str,
        default=str(WORKSPACE_BASE),
        help=f"Directory for prepared paper workspace files "
        f"(default: {WORKSPACE_BASE})",
    )

    args = parser.parse_args()

    # Validate: need at least one mode
    if not args.prepare and not args.evaluate:
        parser.print_help()
        print("\nERROR: Specify at least one of --prepare or --evaluate.")
        sys.exit(1)

    # Validate LLM name
    llm_name = args.llm
    # Try to match a known LLM or use as-is
    if llm_name in LLM_NAME_MAP:
        synth_name = LLM_NAME_MAP[llm_name]
    else:
        synth_name = llm_name

    output_dir = Path(args.output_dir)
    paperaudit_base = Path(args.paperaudit_base)

    # Allow overriding workspace via argument
    workspace_base = Path(args.workspace_dir)

    # --- Phase 1: Prepare ---
    if args.prepare:
        print(f"{'='*60}")
        print(f"PHASE 1: PREPARE (LLM: {synth_name})")
        print(f"  PaperAudit base: {paperaudit_base}")
        print(f"  Workspace:       {workspace_base}")
        print(f"{'='*60}\n")
        prepare_papers(
            synth_name,
            args.limit,
            paperaudit_base=paperaudit_base,
            workspace_base=workspace_base,
        )

    # --- Phase 2: Evaluate ---
    if args.evaluate:
        print(f"\n{'='*60}")
        print(f"PHASE 2: EVALUATE (LLM: {synth_name})")
        print(f"  Workspace: {workspace_base}")
        print(f"  Mode:      {args.mode}")
        print(f"  Strictness:{args.strictness}")
        print(f"{'='*60}\n")

        per_paper, summary = evaluate_papers(
            synth_name,
            args.limit,
            args.api_key,
            use_llm_judge=not args.no_judge,
            mode=args.mode,
            strictness=args.strictness,
            workspace_base=workspace_base,
        )

        write_results(synth_name, per_paper, summary, output_dir)


if __name__ == "__main__":
    main()
