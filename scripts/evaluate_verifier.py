#!/usr/bin/env python3
"""
evaluate_verifier.py — Evaluate the verify-paper skill against the PaperAudit-Bench dataset.

Three evaluation modes:
  1. --prepare            : Extract papers from PaperAudit, rebuild markdown, save ground truth.
  2. --evaluate           : Compare verification reports against ground truth using LLM-as-a-judge.
  3. --evaluate-comparison: Compare exhaustive vs uncertainty verification modes.

Judging protocol (matches the original PaperAudit evaluation):
  ONE holistic LLM call per paper — the judge receives the FULL ground-truth
  list and the FULL predicted-error list and aligns them in a single pass.
  Many-to-many matches are allowed:
    recall    = (# GT items matched by any prediction) / n_gt
    precision = (# distinct predictions matched to any GT) / n_pred
  The judge additionally returns one verdict per prediction
  (true_positive / duplicate / out_of_scope / spurious). That pass does not
  move the headline numbers, but it makes the judge attend to every finding
  explicitly instead of treating the credited-id list as an afterthought, and
  it yields precision_in_scope — the same numerator over a denominator that
  drops findings the judge deemed correct observations of defects this
  benchmark never injected.

Headline metric comparable to the original PaperAudit figures:
  avg_f1_score = F1(macro-avg precision, macro-avg recall)

Judge caching:
  Verdicts are cached per paper/mode/judge-model and keyed on a fingerprint of
  the exact rendered prompt. Re-running the verifier, editing the ground truth
  or bumping JUDGE_PROMPT_VERSION all invalidate the cache automatically; only
  a byte-identical adjudication request is ever reused.

Usage:
  python scripts/evaluate_verifier.py --llm claude-sonnet-4-5-20250929 --prepare --limit 5
  python scripts/evaluate_verifier.py --llm claude-sonnet-4-5-20250929 --evaluate --limit 5
  python scripts/evaluate_verifier.py --llm claude-sonnet-4-5-20250929 --evaluate-comparison --limit 5

Task-specific evaluation (--task):
  Scores a single error family produced by an ad-hoc system instead of the full
  verify-paper pipeline. --task defaults to None (every error is evaluated).
  With --task math, predictions are read from the <paper>_verified_math.json
  reports written by the orchestrate-math skill, the ground truth is filtered to
  mathematical errors, and results land in task_specific_output/math/.

  ./scripts/batch-verify-math.sh --dataset paperaudit
  python scripts/evaluate_verifier.py --llm claude-sonnet-4-5-20250929 \
      --evaluate --task math --workspace-dir data/paperaudit-workspace

  Add --tag to keep several ad-hoc runs side by side and compare them:
  ./scripts/batch-verify-math.sh --model claude-opus-5 --tag opus5
  python scripts/evaluate_verifier.py --llm claude-sonnet-4-5-20250929 \
      --evaluate --task math --tag opus5     # -> <llm>-opus5_summary.csv

  --skill and --llm-skill fold the producing skill (and the model its own --llm
  was pointed at) into that tag, so two skills over the same paper can never
  land on one filename. Pass batch-verify-math.sh the trio and pass the scorer
  the same trio; compose_task_tag() computes the name for both sides:
  ./scripts/batch-verify-math.sh --skill orchestrate-math-v2 --llm-skill gpt-5-6-terra
  python scripts/evaluate_verifier.py --llm claude-sonnet-4-5-20250929 \
      --evaluate --task math --skill orchestrate-math-v2 --llm-skill gpt-5-6-terra
"""

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths (relative to this script's location)
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from llm_providers import complete, resolve_provider  # noqa: E402
from model_costs import (  # noqa: E402
    estimate_cost,
    load_pricing,
    price_for,
    sampling_kwargs,
)

# Default PaperAudit origin data path — matches the layout relative to
# paperena-agent/paperena_claude. Override with --paperaudit-base.
PAPERAUDIT_BASE_DEFAULT = (
    PROJECT_ROOT.parent
    / "paperena-agent"
    / "PaperAudit_Origin_Data"
    / "PaperAudit_Origin_Data"
    / "NeurIPS_25"
)

RESULTS_DIR = PROJECT_ROOT / "outputs" / "spot-eval"
WORKSPACE_BASE = PROJECT_ROOT / "data" / "spot-workspace"

# Task-specific evaluations (--task) write their CSVs here, under a
# subfolder named after the task (created on first run).
TASK_SPECIFIC_OUTPUT_DIR = PROJECT_ROOT / "task_specific_output"

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
# Task-specific evaluation (--task)
#
# A "task" is a single error family checked by an ad-hoc system rather than by
# the full verify-paper pipeline — e.g. the orchestrate-math skill, which walks
# a paper's equations, theorems and proofs and attempts a Lean 4 proof of each
# formalisable claim. With --task set, the evaluation:
#   * reads that task's standalone report instead of verification-report-*.json,
#   * keeps only the ground-truth errors belonging to the task's categories,
#   * writes its CSVs to task_specific_output/<task>/.
# Add a task by adding an entry below.
# ---------------------------------------------------------------------------


def _math_predictions(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalise an orchestrate-math report into predicted_errors shape.

    The skill writes findings[] (confident errors) plus results[] (one entry per
    checked snippet, carrying the section location). A report that already has
    predicted_errors — e.g. a full verify-paper report reused for a math-only
    scoring — is passed straight through.
    """
    existing = report.get("predicted_errors")
    if isinstance(existing, list):
        return existing

    locations = {
        r.get("snippet_id"): r.get("location")
        for r in report.get("results") or []
        if isinstance(r, dict)
    }

    preds: List[Dict[str, Any]] = []
    for f in report.get("findings") or []:
        if not isinstance(f, dict):
            continue
        evidence = " ".join(
            str(x) for x in (f.get("evidence"), f.get("exact_quote")) if x
        )
        preds.append(
            {
                "error_category": f.get("category") or "Equation / proof",
                "error_location": (
                    f.get("location")
                    or locations.get(f.get("snippet_id"))
                    or f.get("snippet_id")
                    or ""
                ),
                "confidence": f.get("confidence"),
                "supporting_evidence": evidence,
                "verifier_name": f.get("verifier") or "orchestrate-math",
                "snippet_id": f.get("snippet_id"),
                "severity": f.get("severity"),
            }
        )
    return preds


TASK_REGISTRY: Dict[str, Dict[str, Any]] = {
    "math": {
        "label": "mathematics (equations, theorems, proofs)",
        # PaperAudit corruption_type values that are mathematical in nature
        "corruption_types": {"method_logic_consistency"},
        # SPOT _spot_category values — see ERROR_CATEGORY_MAP in download_spot.py
        "spot_categories": {"Equation / proof"},
        # Standalone report written next to the paper by /orchestrate-math.
        # The skill names it <paper_name>_verified_math.json; the second stem is
        # accepted as an alias. A --tag run appends -<tag> to the stem, so
        # several models' reports sit side by side in one workspace.
        "report_stems": ["_verified_math", "_math_verification"],
        "predictions": _math_predictions,
        # Used only when a full verify-paper report stands in for a missing
        # orchestrate-math report: keep the mathematical findings so precision
        # is not charged for the other verifiers' findings.
        "prediction_category_re": re.compile(
            r"equation|proof|theorem|lemma|algebra|derivation|arithmetic"
            r"|calculus|formula|sign/|domain/constraint|type mismatch|math",
            re.IGNORECASE,
        ),
        "judge_note": (
            "The predictions come from a mathematics-only verification pass "
            "(the orchestrate-math skill): every self-contained equation, "
            "theorem and proof was extracted and reviewed informally and, where "
            "formalisable, with a Lean 4 proof attempt. Only MATHEMATICAL "
            "ground-truth errors are listed here — corruptions of other kinds "
            "were filtered out and are out of scope for this comparison."
        ),
    },
}

TASK_NAMES = sorted(TASK_REGISTRY)


def gt_matches_task(edit: Dict[str, Any], task: str) -> bool:
    """True when a ground-truth edit belongs to the task's error family.

    SPOT rows carry the fine-grained _spot_category and are judged on it;
    PaperAudit rows only have corruption_type.
    """
    spec = TASK_REGISTRY[task]
    spot_category = edit.get("_spot_category")
    if spot_category:
        return spot_category in spec["spot_categories"]
    return edit.get("corruption_type") in spec["corruption_types"]


def filter_ground_truth(
    edits: List[Dict[str, Any]], task: str
) -> List[Dict[str, Any]]:
    """Keep only the ground-truth edits belonging to the task."""
    return [e for e in edits if isinstance(e, dict) and gt_matches_task(e, task)]


def task_report_globs(task: str, tag: Optional[str] = None) -> List[str]:
    """Filename patterns for a task's standalone report, optionally tagged.

    Untagged: <paper_name>_verified_math.json
    Tagged:   <paper_name>_verified_math-<tag>.json
    """
    suffix = f"-{tag}" if tag else ""
    return [f"*{stem}{suffix}.json" for stem in TASK_REGISTRY[task]["report_stems"]]


def resolve_task_report_path(
    workspace_dir: Path, task: str, tag: Optional[str] = None
) -> Optional[Path]:
    """Find the standalone report written by the task's ad-hoc system.

    With a tag, only that tag's report counts — falling back to an untagged one
    would silently mix runs from different models.
    """
    for pattern in task_report_globs(task, tag):
        matches = sorted(workspace_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def task_predictions(report: Dict[str, Any], task: str) -> List[Dict[str, Any]]:
    """Normalise a task report into the predicted_errors shape the judge reads."""
    return TASK_REGISTRY[task]["predictions"](report)


def filter_predictions(
    predictions: List[Dict[str, Any]], task: str
) -> List[Dict[str, Any]]:
    """Keep only the predictions belonging to the task's error family.

    Applied to a borrowed full verify-paper report, whose findings span every
    verifier. The task's own standalone report needs no filtering — it is
    single-family by construction.
    """
    category_re = TASK_REGISTRY[task].get("prediction_category_re")
    kept: List[Dict[str, Any]] = []
    for pred in predictions:
        if not isinstance(pred, dict):
            continue
        verifier = str(pred.get("verifier_name") or "").lower()
        if task in verifier:
            kept.append(pred)
            continue
        if category_re and category_re.search(str(pred.get("error_category") or "")):
            kept.append(pred)
    return kept


# Characters allowed in a report tag. Tags become filename segments
# (<paper>_verified_math-<tag>.json), so they must stay filename-safe.
TAG_SAFE_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def compose_task_tag(
    tag: Optional[str] = None,
    skill: Optional[str] = None,
    llm_skill: Optional[str] = None,
) -> Optional[str]:
    """Fold the producing skill and its --llm into the report tag.

    ONE source of truth for both sides of the contract: batch-verify-math.sh
    calls this to decide what to NAME a report, and --evaluate --task calls it
    to decide what to LOOK FOR. If the two ever disagreed the symptom would be
    a silent "no report found" for a run that completed fine, so the shell
    script shells out to this function rather than reimplementing it.

    Segments are joined most-general-first, so reports sort by producer:

        skill=orchestrate-math-v2, llm_skill=gpt-5-6-terra, tag=run3
          -> "orchestrate-math-v2-gpt-5-6-terra-run3"

    Each segment is optional; the plain `--tag run3` case is unchanged, and
    `--skill X` with no tag yields just "X". Returns None when nothing is set,
    which means the untagged report.

    Folding the skill in is what makes overwriting impossible: two skills
    writing the same paper's math report land on different filenames whether or
    not the user remembered to pass a distinguishing --tag. `llm_skill` folds in
    for the same reason — orchestrate-math-v2 run against gpt-5-6-terra and the
    same skill run against its default agent are different experiments and must
    not share a filename.
    """
    parts = [s for s in (skill, llm_skill, tag) if s]
    if not parts:
        return None
    for s in parts:
        if not TAG_SAFE_RE.match(s):
            raise ValueError(
                f"tag segment {s!r} is not filename-safe; allowed: [A-Za-z0-9._-]"
            )
    return "-".join(parts)


def task_output_dir(task: str, base: Optional[Path] = None) -> Path:
    """task_specific_output/<task>/, created on first run."""
    out = (base or TASK_SPECIFIC_OUTPUT_DIR) / task
    out.mkdir(parents=True, exist_ok=True)
    return out


DEFAULT_JUDGE_MODEL = "claude-sonnet-4-5-20250929"
JUDGE_MAX_TOKENS = 20_000
JUDGE_MAX_RETRIES = 3
TRUNCATE_FIELD_CHARS = 800  # Max chars per text field shown to the judge

# Bumped whenever the judge prompt or its output contract changes. Every cached
# verdict stores the fingerprint of the exact prompt that produced it, so a bump
# invalidates stale judgments instead of silently reusing them.
JUDGE_PROMPT_VERSION = "2"

# Verdicts the judge assigns to each prediction in part 2 of the adjudication.
PRED_VERDICTS = ("true_positive", "duplicate", "out_of_scope", "spurious")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sanitize_paper_id(dirname: str) -> str:
    """Derive a clean paper_id from the directory name.

    Directory names look like: '25671-RAG4GFM__Bridging_Knowledge_Gaps_...'
    We keep the numeric prefix + shortname part.
    """
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


def load_json(path: Path) -> Any:
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


def _model_tag(name: str) -> str:
    return name.replace("/", "_")


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


def _load_dotenv_key(name: str) -> Optional[str]:
    """Read a KEY=value line from the project-root .env file, if present."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return None
    try:
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    except Exception:
        pass
    return None


def load_anthropic_client(api_key: Optional[str] = None):
    """Load the Anthropic client, trying multiple auth sources.

    Order: explicit --api-key, ANTHROPIC_API_KEY env var, project .env file.
    Returns the client or None if the anthropic package is unavailable.
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

    key = (
        api_key
        or os.environ.get("ANTHROPIC_API_KEY")
        or _load_dotenv_key("ANTHROPIC_API_KEY")
    )

    if not key:
        print(
            "ERROR: No Anthropic API key found. Set ANTHROPIC_API_KEY env var, "
            "add it to .env, or pass --api-key."
        )
        sys.exit(1)

    return anthropic.Anthropic(api_key=key)


# ---------------------------------------------------------------------------
# Holistic LLM judge (one call per paper, matching the original PaperAudit
# evaluation protocol: full GT list + full findings list, many-to-many)
# ---------------------------------------------------------------------------


def _truncate(text: Any, limit: int = TRUNCATE_FIELD_CHARS) -> str:
    s = str(text) if text is not None else ""
    return s if len(s) <= limit else s[:limit] + " …[truncated]"


def gt_paper_texts(edit: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    """Return (text the verifier actually read, original wording or None).

    The two benchmarks store this differently and the distinction decides what
    the judge may compare a finding's quoted evidence against:

    * PaperAudit injects corruptions. `target_find` is the ORIGINAL wording,
      which was deleted from the manuscript; `replacement` is what ends up in
      paper.md. Measured over the 712 prepared PaperAudit edits, `replacement`
      is the string present in the paper 84% of the time and `target_find`
      only 5% — so showing the judge `target_find` as "the changed text" made
      it compare findings against text that is not in the paper at all.
    * SPOT catalogues errors that were already published. It has no
      `replacement`, and `target_find` IS the erroneous text in the paper.

    A non-empty `replacement` is therefore the marker of an injected edit.
    """
    original = str(edit.get("target_find") or "")
    replacement = str(edit.get("replacement") or "")
    if replacement.strip():
        return replacement, (original or None)
    return original, None


def build_holistic_prompt(
    ground_truth_edits: List[Dict[str, Any]],
    predicted_errors: List[Dict[str, Any]],
    paper_title: str,
    mode: str,
) -> str:
    """Build a single adjudication prompt with ALL GT items and ALL findings."""
    gt_min = []
    any_original = False
    for i, g in enumerate(ground_truth_edits, 1):
        in_paper, original = gt_paper_texts(g)
        item = {
            "gt_id": i,
            "difficulty": g.get("difficulty"),
            "corruption_type": g.get("corruption_type"),
            "location": g.get("location"),
            "corruption_intent": _truncate(g.get("rationale")),
            "error_explanation": _truncate(g.get("error_explanation")),
            "text_in_paper": _truncate(in_paper),
            "needs_cross_section": bool(g.get("needs_cross_section", False)),
        }
        # Only injected edits have a pre-corruption original to show, and it is
        # flagged as absent-from-the-paper so the judge never expects a finding
        # to quote it.
        if original is not None:
            item["original_text_removed_from_paper"] = _truncate(original)
            any_original = True
        gt_min.append(item)

    pred_min = []
    for i, p in enumerate(predicted_errors, 1):
        pred_min.append(
            {
                "pred_id": i,
                "error_category": p.get("error_category"),
                "error_location": p.get("error_location"),
                "supporting_evidence": _truncate(p.get("supporting_evidence")),
                "confidence": p.get("confidence"),
                "verifier_name": p.get("verifier_name"),
            }
        )

    gt_json = json.dumps(gt_min, ensure_ascii=False, indent=2)
    pred_json = json.dumps(pred_min, ensure_ascii=False, indent=2)

    mode_note = {
        "exhaustive": "The verifier checked every snippet in the paper (exhaustive mode).",
        "uncertainty": (
            "The verifier only checked high-uncertainty snippets (uncertainty "
            "mode), so some low-risk regions were not verified."
        ),
        "baseline": (
            "The predictions come from a single-call LLM baseline that read "
            "the whole paper in one shot."
        ),
    }.get(mode)
    if mode_note is None and mode in TASK_REGISTRY:
        mode_note = TASK_REGISTRY[mode]["judge_note"]
    if mode_note is None:
        mode_note = f"Verification mode: {mode}"

    original_note = (
        "\n- `original_text_removed_from_paper`, where present, is the clean "
        "wording that the corruption DELETED. It does NOT appear in the paper "
        "the verifier read. Never require a finding to quote it, and never "
        "treat a finding as wrong for contradicting it."
        if any_original
        else ""
    )

    return f"""You are an expert adjudicator scoring an automated paper-verification system on the paper "{paper_title}".

{mode_note}

GROUND_TRUTH lists known defects in that paper — the MECHANISM of each one. FINDINGS are what the verifier reported after reading ONLY the defective paper — the SYMPTOMS it observed. Your job is to decide which ground-truth defects the verifier actually caught.

How to read a ground_truth item:
- `text_in_paper` is the passage the verifier actually saw. Judge quote-level agreement against THIS field.{original_note}
- `error_explanation` and `corruption_intent` say why the passage is defective.
- `location` is where the defect was introduced, which is not always where it is detectable.

TASK — TWO PARTS, BOTH REQUIRED.

PART 1 — For EACH ground_truth item, decide whether ANY finding detects it.

A finding DETECTS a ground truth when it points at the same defect in the same material. Apply these rules:

R1 LOCUS. The finding must implicate the same specific material: the same number, cell, claim, citation or equation — or, for a cross-section defect, the same pair of conflicting passages. Verifiers routinely label sections, tables and figures differently from the ground truth, so judge by the content pointed at, not by the label. Flagging a DIFFERENT number in the same table, or a different claim in the same section, is NOT a detection.

R2 POLARITY IS IRRELEVANT. These defects invert, inflate or deflate a claim, so the verifier — which only ever saw the defective side — will often describe the problem from the opposite direction: calling an "understatement" what the ground truth records as an "exaggeration", or naming whichever of two inconsistent passages it believes to be the wrong one. If the finding identifies the same inconsistency or the same misstated quantity, it DETECTS the ground truth, regardless of which side it calls incorrect or which way round it says the error runs.

R3 MECHANISM vs SYMPTOM. The edit sits at `location`, but its detectable consequence often surfaces elsewhere — an altered abstract that now contradicts Section 3.1. When the defect is a conflict between two passages (`needs_cross_section` true, or an explanation naming two places), a finding DETECTS it by identifying that conflict, and it does not matter which of the two passages it reports as the error's location.

R4 SPECIFICITY. The finding must name a concrete defect. Generic reservations — "the statistics are underspecified", "results may not generalise", "this section is unclear" — detect nothing, even when aimed at the right paragraph. A finding whose stated reason for concern is a genuinely DIFFERENT scientific issue that happens to sit in the same sentence is NOT a detection.

R5 MANY-TO-MANY. One gt_id may be detected by several pred_ids — list all of them. One pred_id may detect several gt_ids, but only when it genuinely describes each of them.

PART 2 — For EACH pred_id, assign exactly one verdict:
- "true_positive": you credited it in Part 1 (it appears in some matched_pred_ids).
- "duplicate": it reports the same defect as another finding you already credited. Give that finding's id in `duplicate_of`.
- "out_of_scope": a fair and correct observation about a defect that is NOT in GROUND_TRUTH — a pre-existing weakness of the underlying paper (a typo, a missing ablation, an honest limitation). Use this only when the observation looks CORRECT about the paper.
- "spurious": it asserts a defect that the paper does not support, or it misreads the paper.

CONSISTENCY: the pred_ids you listed in Part 1's matched_pred_ids are exactly the ones with verdict "true_positive". Every other pred_id gets one of the other three.

OUTPUT (JSON object only, no other text):
{{
  "matches": [
    {{
      "gt_id": <int, 1-based, in order>,
      "matched": <true|false>,
      "matched_pred_ids": [<int, ...>],
      "rationale": "1-3 sentences explaining the judgment"
    }},
    ...
  ],
  "predictions": [
    {{
      "pred_id": <int, 1-based, in order>,
      "verdict": "true_positive" | "duplicate" | "out_of_scope" | "spurious",
      "duplicate_of": <int or null>,
      "rationale": "one short sentence"
    }},
    ...
  ]
}}

RULES:
- Exactly {len(gt_min)} items in "matches", one per ground_truth item, in order.
- Exactly {len(pred_min)} items in "predictions", one per finding, in order.
- matched_pred_ids must only contain pred_id values that appear in FINDINGS; empty list when matched is false.
- Judge on substance, not vocabulary: never require a finding to reuse the ground truth's wording, terminology or framing.
- Rules R1 and R4 are the strict ones — a different locus or a merely generic remark is never a detection. R2 and R3 are the permissive ones — direction of the error and choice of endpoint never break a match.
- If, after applying R1-R4, the call is still genuinely balanced, decide AGAINST the match.

GROUND_TRUTH:
{gt_json}

FINDINGS:
{pred_json}""".strip()


def _parse_judge_json(raw_text: str) -> Optional[Any]:
    """Robustly extract a JSON object from an LLM judge response."""
    if not raw_text:
        return None

    # Strip ```json / ``` fences
    cleaned = re.sub(r"```(?:json)?\s*", "", raw_text).strip()
    cleaned = cleaned.strip("`").strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Balanced-brace extraction: find the first '{' then track depth
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
                    return None
    return None


def _extract_text_from_response(response: Any) -> str:
    """Extract text content from an Anthropic SDK response, skipping ThinkingBlocks."""
    parts: List[str] = []
    for block in response.content:
        if hasattr(block, "text"):
            parts.append(block.text)
    return " ".join(parts)


def validate_matches(obj: Any, n_gt: int, n_pred: int) -> Tuple[bool, str]:
    """Validate the judge's output structure."""
    if not isinstance(obj, dict):
        return False, "not a dict"
    matches = obj.get("matches")
    if not isinstance(matches, list):
        return False, "missing 'matches' list"
    if len(matches) != n_gt:
        return False, f"'matches' length {len(matches)} != ground_truth length {n_gt}"
    for i, m in enumerate(matches):
        if not isinstance(m, dict):
            return False, f"match[{i}] not dict"
        if not isinstance(m.get("matched"), bool):
            return False, f"match[{i}].matched must be bool"
        ids = m.get("matched_pred_ids")
        if not isinstance(ids, list) or not all(isinstance(x, int) for x in ids):
            return False, f"match[{i}].matched_pred_ids must be list[int]"
        if any(x < 1 or x > n_pred for x in ids):
            return False, f"match[{i}].matched_pred_ids out of range 1..{n_pred}"
        gid = m.get("gt_id")
        if not isinstance(gid, int) or gid not in (i, i + 1):
            return False, f"match[{i}].gt_id must equal {i + 1} (1-based)"
    return True, "ok"


def validate_pred_verdicts(obj: Any, n_pred: int) -> Tuple[bool, str]:
    """Validate the judge's part-2 output: one verdict per prediction, in order."""
    if not isinstance(obj, dict):
        return False, "not a dict"
    preds = obj.get("predictions")
    if not isinstance(preds, list):
        return False, "missing 'predictions' list"
    if len(preds) != n_pred:
        return False, f"'predictions' length {len(preds)} != findings length {n_pred}"
    for i, v in enumerate(preds):
        if not isinstance(v, dict):
            return False, f"predictions[{i}] not dict"
        if v.get("verdict") not in PRED_VERDICTS:
            return False, (
                f"predictions[{i}].verdict must be one of {PRED_VERDICTS}, "
                f"got {v.get('verdict')!r}"
            )
        pid = v.get("pred_id")
        if not isinstance(pid, int) or pid not in (i, i + 1):
            return False, f"predictions[{i}].pred_id must equal {i + 1} (1-based)"
        dup = v.get("duplicate_of")
        if dup is not None and (
            not isinstance(dup, int) or dup < 1 or dup > n_pred or dup == i + 1
        ):
            return False, f"predictions[{i}].duplicate_of out of range 1..{n_pred}"
    return True, "ok"


def prompt_fingerprint(prompt: str) -> str:
    """Stable id for the exact adjudication request behind a cached verdict.

    Cache validity used to be checked with `n_gt == n_gt and n_pred == n_pred`,
    which silently reused a verdict whenever a re-run happened to produce the
    same NUMBER of findings, and never invalidated anything when the rubric
    itself changed. Hashing the rendered prompt covers both: any change to the
    ground truth, the predictions, the paper title, the mode note or
    JUDGE_PROMPT_VERSION produces a new fingerprint and forces a re-judge.
    """
    h = hashlib.sha256()
    h.update(JUDGE_PROMPT_VERSION.encode("utf-8"))
    h.update(b"\x00")
    h.update(prompt.encode("utf-8"))
    return h.hexdigest()[:16]


def read_judge_cache(
    cache_path: Optional[Path], n_gt: int, n_pred: int, fingerprint: str
) -> Optional[Dict[str, Any]]:
    """Return a cached judgment only when it was produced by THIS exact prompt.

    Returns {"matches": [...], "pred_verdicts": [...]} or None on any miss.
    """
    if cache_path is None or not cache_path.exists():
        return None
    try:
        cached = load_json(cache_path)
    except Exception:
        return None
    if not isinstance(cached, dict):
        return None
    if cached.get("prompt_fingerprint") != fingerprint:
        return None
    if cached.get("n_gt") != n_gt or cached.get("n_pred") != n_pred:
        return None
    matches = cached.get("matches")
    if not validate_matches({"matches": matches}, n_gt, n_pred)[0]:
        return None
    verdicts = cached.get("pred_verdicts") or []
    if verdicts and not validate_pred_verdicts({"predictions": verdicts}, n_pred)[0]:
        verdicts = []
    return {"matches": matches, "pred_verdicts": verdicts}


def write_judge_cache(
    cache_path: Optional[Path],
    judge_model: str,
    mode: str,
    n_gt: int,
    n_pred: int,
    fingerprint: str,
    matches: List[Dict[str, Any]],
    pred_verdicts: List[Dict[str, Any]],
) -> None:
    if cache_path is None:
        return
    save_json(
        cache_path,
        {
            "judge_model": judge_model,
            "mode": mode,
            "prompt_version": JUDGE_PROMPT_VERSION,
            "prompt_fingerprint": fingerprint,
            "n_gt": n_gt,
            "n_pred": n_pred,
            "matches": matches,
            "pred_verdicts": pred_verdicts,
        },
    )


def _client_matches(client: Any, judge_model: str) -> bool:
    """Whether a pre-built client can serve this judge model.

    The callers build an Anthropic client up front; it is reusable only when
    the judge model actually routes to Anthropic. Anything else gets a fresh
    provider client so the request goes to the right API.
    """
    if client is None:
        return False
    try:
        return resolve_provider(judge_model)[0] == "anthropic"
    except ValueError:
        # Unrecognised name: assume the caller's client is intended (e.g. an
        # Anthropic-compatible proxy serving a third-party model).
        return True


def judge_paper_holistic(
    client: Any,
    judge_model: str,
    predicted_errors: List[Dict[str, Any]],
    ground_truth_edits: List[Dict[str, Any]],
    paper_title: str,
    mode: str,
    cache_path: Optional[Path] = None,
    overwrite: bool = False,
) -> Optional[Dict[str, Any]]:
    """Adjudicate one paper with a single LLM call, using a disk cache.

    Returns {"matches": [...one per GT, in order...], "pred_verdicts": [...one
    per finding, in order...]}, or None if all attempts failed. `pred_verdicts`
    is [] when the judge could not produce a valid part-2 list; the matches are
    still usable, so that degrades rather than failing the paper.
    """
    n_gt = len(ground_truth_edits)
    n_pred = len(predicted_errors)

    if n_gt == 0:
        return {"matches": [], "pred_verdicts": []}

    # Built before the cache lookup: its fingerprint IS the cache key, so a
    # changed rubric, changed ground truth or changed findings all miss.
    prompt = build_holistic_prompt(
        ground_truth_edits, predicted_errors, paper_title, mode
    )
    fingerprint = prompt_fingerprint(prompt)

    if not overwrite:
        cached = read_judge_cache(cache_path, n_gt, n_pred, fingerprint)
        if cached is not None:
            return cached

    if n_pred == 0:
        matches = [
            {
                "gt_id": i + 1,
                "matched": False,
                "matched_pred_ids": [],
                "rationale": "No predictions to match.",
            }
            for i in range(n_gt)
        ]
        write_judge_cache(
            cache_path, judge_model, mode, n_gt, n_pred, fingerprint, matches, []
        )
        return {"matches": matches, "pred_verdicts": []}

    last_err = ""
    for attempt in range(1, JUDGE_MAX_RETRIES + 1):
        try:
            # Routed through the provider layer so any provider can judge.
            # A bare name picks its native API; prefix it (e.g.
            # "anthropic:deepseek-v4-pro") to force an Anthropic-compatible
            # proxy instead. `client` is only reused when it matches.
            response = complete(
                model=judge_model,
                prompt=prompt,
                max_tokens=JUDGE_MAX_TOKENS,
                # temp 0 first for determinism, nudged on retries to escape a
                # deterministic malformed output; dropped for models that
                # reject sampling params.
                temperature=0.0 if attempt == 1 else 0.3,
                client=client if _client_matches(client, judge_model) else None,
            )
        except Exception as e:
            last_err = f"API error: {type(e).__name__}: {e}"
            print(f"  [JUDGE] attempt {attempt} {last_err}", file=sys.stderr)
            time.sleep(min(2**attempt, 30))
            continue

        raw_text = response.text
        parsed = _parse_judge_json(raw_text)
        ok, msg = validate_matches(parsed, n_gt, n_pred)
        if ok:
            matches = parsed["matches"]
            v_ok, v_msg = validate_pred_verdicts(parsed, n_pred)
            if not v_ok and attempt < JUDGE_MAX_RETRIES:
                # Part 1 is fine but part 2 is not — retry for the full answer.
                last_err = f"prediction-verdict validation failed: {v_msg}"
                print(f"  [JUDGE] attempt {attempt} {last_err}", file=sys.stderr)
                continue
            pred_verdicts = parsed["predictions"] if v_ok else []
            if not v_ok:
                print(
                    f"  [JUDGE] accepting matches without prediction verdicts "
                    f"({v_msg}) — in-scope precision unavailable for this paper",
                    file=sys.stderr,
                )
            write_judge_cache(
                cache_path, judge_model, mode, n_gt, n_pred, fingerprint,
                matches, pred_verdicts,
            )
            return {"matches": matches, "pred_verdicts": pred_verdicts}

        last_err = f"validation failed: {msg}"
        print(
            f"  [JUDGE] attempt {attempt} {last_err} "
            f"(raw first 200 chars: {raw_text[:200]!r})",
            file=sys.stderr,
        )

    print(f"  [JUDGE FAILURE] giving up after {JUDGE_MAX_RETRIES} attempts: {last_err}",
          file=sys.stderr)
    return None


# ---------------------------------------------------------------------------
# Heuristic fallback (no-judge) — SMOKE TEST ONLY
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset(
    """a an and are as at be but by for from has have if in into is it its of on
    or that the this to was we were which with not no than then there these
    those they their our your you i he she his her them us also can may might
    would should could will shall do does did done being been more most such
    only other some any each between both after before during through over
    under about against""".split()
)


def _content_tokens(s: str) -> set:
    return {
        t
        for t in re.findall(r"[a-zA-Z0-9]+", s.lower())
        if t not in _STOPWORDS and len(t) > 1
    }


def jaccard_score(a: str, b: str) -> float:
    """Jaccard similarity over content tokens (stopwords removed)."""
    ta, tb = _content_tokens(a), _content_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


HEURISTIC_THRESHOLD = 0.20


def heuristic_align(
    predicted_errors: List[Dict[str, Any]],
    ground_truth_edits: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Greedy 1-1 alignment by Jaccard similarity. Rough smoke test only —
    NOT comparable to LLM-judge results or to the original PaperAudit figures.
    """

    def pred_text(p: Dict[str, Any]) -> str:
        return " ".join(
            [str(p.get("supporting_evidence", "")), str(p.get("error_location", ""))]
        )

    def gt_text(g: Dict[str, Any]) -> str:
        return " ".join(
            [str(g.get("error_explanation", "")), str(g.get("target_find", ""))]
        )

    scored = []
    for pi, p in enumerate(predicted_errors):
        pt = pred_text(p)
        for gi, g in enumerate(ground_truth_edits):
            s = jaccard_score(pt, gt_text(g))
            if s >= HEURISTIC_THRESHOLD:
                scored.append((s, pi, gi))
    scored.sort(reverse=True)

    used_p: set = set()
    used_g: set = set()
    matched_by_gt: Dict[int, List[int]] = {}
    for s, pi, gi in scored:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        matched_by_gt[gi] = [pi + 1]

    return [
        {
            "gt_id": gi + 1,
            "matched": gi in matched_by_gt,
            "matched_pred_ids": matched_by_gt.get(gi, []),
            "rationale": "heuristic jaccard match",
        }
        for gi in range(len(ground_truth_edits))
    ]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def alignment_from_matches(
    matches: List[Dict[str, Any]],
    predicted_errors: List[Dict[str, Any]],
    ground_truth_edits: List[Dict[str, Any]],
    pred_verdicts: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Derive counts and breakdowns from a validated judgment.

    Many-to-many semantics (same as the original PaperAudit evaluation):
      matched_gt_count   → recall numerator
      matched_pred_count → precision numerator (distinct pred ids)

    `pred_verdicts` is the judge's per-prediction pass. It does not change the
    headline numbers — those stay on the original protocol — but it supplies
    the diagnostic breakdown of what the uncredited findings actually were, and
    an in-scope precision that drops findings the judge deemed correct
    observations about defects the benchmark simply never injected.
    """
    n_gt = len(ground_truth_edits)
    n_pred = len(predicted_errors)

    matched_gt_indices = [i for i, m in enumerate(matches) if m.get("matched") is True]
    matched_pred_ids: set = set()
    for m in matches:
        for x in m.get("matched_pred_ids", []):
            if isinstance(x, int) and 1 <= x <= n_pred:
                matched_pred_ids.add(x)

    matched_gt_count = len(matched_gt_indices)
    matched_pred_count = len(matched_pred_ids)

    # Per-prediction verdicts (empty when the judge failed to produce them).
    verdict_counts: Dict[str, int] = Counter()
    out_of_scope_ids: set = set()
    for v in pred_verdicts or []:
        verdict = v.get("verdict")
        if verdict not in PRED_VERDICTS:
            continue
        verdict_counts[verdict] += 1
        pid = v.get("pred_id")
        if verdict == "out_of_scope" and isinstance(pid, int):
            out_of_scope_ids.add(pid)

    # A prediction the judge called out-of-scope while also crediting it to a
    # ground truth is a self-contradiction; the credit wins, and the conflict
    # is counted so a drifting judge shows up in the results.
    verdict_conflicts = len(out_of_scope_ids & matched_pred_ids)
    out_of_scope_ids -= matched_pred_ids

    n_pred_in_scope = n_pred - len(out_of_scope_ids)

    # Per-difficulty / per-corruption-type breakdown
    difficulty_gt_found: Dict[str, int] = Counter()
    difficulty_gt_total: Dict[str, int] = Counter()
    type_gt_found: Dict[str, int] = Counter()
    type_gt_total: Dict[str, int] = Counter()
    matched_set = set(matched_gt_indices)
    for g_idx, gt in enumerate(ground_truth_edits):
        diff = gt.get("difficulty", "unknown")
        ctype = gt.get("corruption_type", "unknown")
        difficulty_gt_total[diff] += 1
        type_gt_total[ctype] += 1
        if g_idx in matched_set:
            difficulty_gt_found[diff] += 1
            type_gt_found[ctype] += 1

    return {
        "n_gt": n_gt,
        "n_pred": n_pred,
        "n_pred_in_scope": n_pred_in_scope,
        "matched_gt_count": matched_gt_count,
        "matched_pred_count": matched_pred_count,
        "matched_gt_indices": matched_gt_indices,
        "matched_pred_ids": sorted(matched_pred_ids),
        "has_pred_verdicts": bool(pred_verdicts),
        "verdict_counts": dict(verdict_counts),
        "verdict_conflicts": verdict_conflicts,
        "difficulty_gt_found": dict(difficulty_gt_found),
        "difficulty_gt_total": dict(difficulty_gt_total),
        "type_gt_found": dict(type_gt_found),
        "type_gt_total": dict(type_gt_total),
    }


def compute_metrics(alignment: Dict[str, Any]) -> Dict[str, float]:
    """Compute precision, recall, F1 (many-to-many, original protocol)."""
    n_gt = alignment["n_gt"]
    n_pred = alignment["n_pred"]
    recall = alignment["matched_gt_count"] / n_gt if n_gt > 0 else 0.0
    precision = alignment["matched_pred_count"] / n_pred if n_pred > 0 else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    # Same numerator, denominator excluding findings the judge judged to be
    # correct observations of defects the benchmark never injected. Falls back
    # to plain precision when the judge produced no per-prediction verdicts.
    n_in_scope = alignment.get("n_pred_in_scope", n_pred)
    precision_in_scope = (
        alignment["matched_pred_count"] / n_in_scope if n_in_scope > 0 else 0.0
    )
    f1_in_scope = (
        (2 * precision_in_scope * recall / (precision_in_scope + recall))
        if (precision_in_scope + recall) > 0
        else 0.0
    )
    verdicts = alignment.get("verdict_counts", {})

    difficulty_recall: Dict[str, float] = {}
    for diff, total in alignment["difficulty_gt_total"].items():
        found = alignment["difficulty_gt_found"].get(diff, 0)
        difficulty_recall[diff] = found / total if total > 0 else 0.0

    type_recall: Dict[str, float] = {}
    for ctype, total in alignment["type_gt_total"].items():
        found = alignment["type_gt_found"].get(ctype, 0)
        type_recall[ctype] = found / total if total > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "precision_in_scope": precision_in_scope,
        "f1_in_scope": f1_in_scope,
        "n_pred_in_scope": n_in_scope,
        "pred_true_positive": verdicts.get("true_positive", 0),
        "pred_duplicate": verdicts.get("duplicate", 0),
        "pred_out_of_scope": verdicts.get("out_of_scope", 0),
        "pred_spurious": verdicts.get("spurious", 0),
        "verdict_conflicts": alignment.get("verdict_conflicts", 0),
        **{f"recall_{diff}": v for diff, v in difficulty_recall.items()},
        **{f"recall_type_{ctype}": v for ctype, v in type_recall.items()},
    }


def align_paper(
    predicted_errors: List[Dict[str, Any]],
    ground_truth_edits: List[Dict[str, Any]],
    client: Optional[Any],
    judge_model: str,
    paper_title: str,
    mode: str,
    cache_path: Optional[Path],
    overwrite_judge: bool = False,
) -> Optional[Dict[str, Any]]:
    """Align one paper's predictions with its GT (judge or heuristic).

    Returns the alignment dict, or None if the LLM judge failed.
    """
    if client is not None:
        judgment = judge_paper_holistic(
            client,
            judge_model,
            predicted_errors,
            ground_truth_edits,
            paper_title,
            mode,
            cache_path=cache_path,
            overwrite=overwrite_judge,
        )
        if judgment is None:
            return None
        matches = judgment["matches"]
        pred_verdicts = judgment["pred_verdicts"]
    else:
        matches = heuristic_align(predicted_errors, ground_truth_edits)
        pred_verdicts = []

    return alignment_from_matches(
        matches, predicted_errors, ground_truth_edits, pred_verdicts
    )


def resolve_report_path(workspace_dir: Path, mode: str) -> Optional[Path]:
    """Pick the verification report for a mode, preferring the mode-specific file."""
    mode_path = workspace_dir / f"verification-report-{mode}.json"
    if mode_path.exists():
        return mode_path
    generic = workspace_dir / "verification-report.json"
    if generic.exists():
        return generic
    return None


def judge_cache_path(workspace_dir: Path, mode: str, judge_model: str) -> Path:
    return workspace_dir / f"judge-{mode}-{_model_tag(judge_model)}.json"


def _warm_judge_cache(
    client: Any,
    judge_model: str,
    tasks: List[Dict[str, Any]],
    concurrency: int,
    overwrite_judge: bool,
) -> None:
    """Run all pending judge calls in parallel so the main loop hits cache.

    Each task: {paper_id, predicted, gt, title, mode, cache_path}
    """
    # Mirrors judge_paper_holistic's own cache test — a file that merely EXISTS
    # is not a hit, so a stale-fingerprint paper is re-judged here in parallel
    # instead of falling through to a serial call in the main loop.
    def _is_cached(t: Dict[str, Any]) -> bool:
        if not t["gt"]:
            return True
        prompt = build_holistic_prompt(
            t["gt"], t["predicted"], t["title"], t["mode"]
        )
        return (
            read_judge_cache(
                t["cache_path"],
                len(t["gt"]),
                len(t["predicted"]),
                prompt_fingerprint(prompt),
            )
            is not None
        )

    pending = [t for t in tasks if overwrite_judge or not _is_cached(t)]
    if not pending:
        print("Judge cache warm: all papers already cached.\n")
        return

    print(
        f"Judging {len(pending)} paper/mode combinations "
        f"with {judge_model} (concurrency={concurrency})..."
    )
    done = 0
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {
            ex.submit(
                judge_paper_holistic,
                client,
                judge_model,
                t["predicted"],
                t["gt"],
                t["title"],
                t["mode"],
                t["cache_path"],
                overwrite_judge,
            ): t
            for t in pending
        }
        for fut in as_completed(futures):
            t = futures[fut]
            done += 1
            try:
                res = fut.result()
                status = "ok" if res is not None else "FAILED"
            except Exception as e:
                status = f"EXC {type(e).__name__}"
            print(f"  [{done}/{len(pending)}] {t['paper_id']} ({t['mode']}): {status}")
    print()


def summarize(per_paper: List[Dict[str, Any]], prefix: str = "") -> Dict[str, Any]:
    """Aggregate per-paper metrics.

    Headline (comparable to the original PaperAudit evaluation):
      avg_f1_score = F1(macro-avg precision, macro-avg recall)
    Also reports macro_f1 (mean per-paper F1) and micro metrics.
    """
    n = len(per_paper)
    p = prefix

    total_gt = sum(r[f"{p}n_gt"] for r in per_paper)
    total_pred = sum(r[f"{p}n_pred"] for r in per_paper)
    total_matched_gt = sum(r[f"{p}matched_gt"] for r in per_paper)
    total_matched_pred = sum(r[f"{p}matched_pred"] for r in per_paper)

    macro_precision = sum(r[f"{p}precision"] for r in per_paper) / n
    macro_recall = sum(r[f"{p}recall"] for r in per_paper) / n
    macro_f1 = sum(r[f"{p}f1"] for r in per_paper) / n
    macro_precision_in_scope = (
        sum(r.get(f"{p}precision_in_scope", r[f"{p}precision"]) for r in per_paper) / n
    )

    avg_f1_score = (
        2 * macro_precision * macro_recall / (macro_precision + macro_recall)
        if (macro_precision + macro_recall) > 0
        else 0.0
    )

    micro_precision = total_matched_pred / total_pred if total_pred > 0 else 0.0
    micro_recall = total_matched_gt / total_gt if total_gt > 0 else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )

    total_in_scope = sum(
        r.get(f"{p}n_pred_in_scope", r[f"{p}n_pred"]) for r in per_paper
    )
    micro_precision_in_scope = (
        total_matched_pred / total_in_scope if total_in_scope > 0 else 0.0
    )
    avg_f1_in_scope = (
        2 * macro_precision_in_scope * macro_recall
        / (macro_precision_in_scope + macro_recall)
        if (macro_precision_in_scope + macro_recall) > 0
        else 0.0
    )
    verdict_totals = {
        key: sum(r.get(f"{p}{key}", 0) for r in per_paper)
        for key in (
            "pred_true_positive",
            "pred_duplicate",
            "pred_out_of_scope",
            "pred_spurious",
            "verdict_conflicts",
        )
    }

    return {
        f"{p}total_gt": total_gt,
        f"{p}total_pred": total_pred,
        f"{p}total_matched_gt": total_matched_gt,
        f"{p}total_matched_pred": total_matched_pred,
        f"{p}macro_precision": macro_precision,
        f"{p}macro_recall": macro_recall,
        f"{p}macro_f1": macro_f1,
        f"{p}avg_f1_score": avg_f1_score,
        f"{p}micro_precision": micro_precision,
        f"{p}micro_recall": micro_recall,
        f"{p}micro_f1": micro_f1,
        f"{p}total_pred_in_scope": total_in_scope,
        f"{p}macro_precision_in_scope": macro_precision_in_scope,
        f"{p}micro_precision_in_scope": micro_precision_in_scope,
        f"{p}avg_f1_in_scope": avg_f1_in_scope,
        **{f"{p}{k}": v for k, v in verdict_totals.items()},
    }


def evaluate_papers(
    llm_name: str,
    limit: Optional[int] = None,
    api_key: Optional[str] = None,
    use_llm_judge: bool = True,
    mode: str = "exhaustive",
    workspace_base: Optional[Path] = None,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    concurrency: int = 8,
    overwrite_judge: bool = False,
    task: Optional[str] = None,
    tag: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Evaluate verification reports against ground truth for all papers.

    With `task` set, only that error family is scored: predictions come from the
    task's standalone report (falling back to the mode report) and the ground
    truth is filtered to the task's categories.

    With `tag` set, only that tag's task report is read, and its judge verdicts
    are cached under the tag — so several models can be scored side by side in
    one workspace.

    Returns (per_paper_results, summary).
    """
    ws_base = workspace_base or WORKSPACE_BASE

    if task is not None and task not in TASK_REGISTRY:
        print(
            f"ERROR: Unknown --task '{task}'. Known tasks: {', '.join(TASK_NAMES)}"
        )
        sys.exit(1)

    if tag and not task:
        print(
            "ERROR: --tag applies to task runs (--task). For a full verify-paper "
            "report tag, use --mode <tag> instead."
        )
        sys.exit(1)

    # The task name replaces the mode as the judge-prompt note and the result
    # label, so task runs never collide with full-paper runs. The judge cache
    # additionally carries the tag, keeping one model's verdicts off another's.
    judge_mode = task or mode
    cache_tag = f"{task}-{tag}" if (task and tag) else judge_mode

    client = None
    if use_llm_judge:
        client = load_anthropic_client(api_key)
        if client is not None:
            print(f"LLM-as-a-judge: ENABLED (holistic, model={judge_model})\n")
        else:
            print("LLM-as-a-judge: UNAVAILABLE (using heuristic smoke test)\n")
    else:
        print(
            "LLM-as-a-judge: DISABLED — heuristic results are a smoke test only, "
            "NOT comparable to judged or published figures\n"
        )

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

    scope = f"task={task}" if task else f"mode={mode}"
    if tag:
        scope += f", tag={tag}"
    if limit:
        paper_ids = paper_ids[:limit]
        print(f"Evaluating {len(paper_ids)} papers (limit={limit}, {scope})\n")
    else:
        print(f"Evaluating all {len(paper_ids)} papers ({scope})\n")

    # Load data and build judge task list
    tasks: List[Dict[str, Any]] = []
    skipped_no_task_gt = 0
    borrowed = 0
    for paper_id in paper_ids:
        workspace_dir = ws_base / paper_id
        borrowed_report = False
        if task:
            # Prefer the ad-hoc system's standalone report; fall back to the
            # mode report so a full verify-paper run can also be scored on the
            # task alone.
            report_path = resolve_task_report_path(workspace_dir, task, tag)
            if report_path is None and not tag:
                # Only an untagged run may borrow: a tagged run names one
                # specific system, and substituting another would be a lie.
                report_path = resolve_report_path(workspace_dir, mode)
                borrowed_report = report_path is not None
            if report_path is None:
                globs = " / ".join(task_report_globs(task, tag))
                print(f"  SKIP {paper_id}: no '{task}' report ({globs})")
                continue
        else:
            report_path = resolve_report_path(workspace_dir, mode)
            if report_path is None:
                print(f"  SKIP {paper_id}: no verification report for mode '{mode}'")
                continue
        try:
            gt = load_json(workspace_dir / "ground_truth.json")
            report = load_json(report_path)
        except Exception as e:
            print(f"  ERROR {paper_id}: {e}")
            continue

        if task:
            gt = filter_ground_truth(gt, task)
            if not gt:
                # No task errors were injected here — scoring it would only
                # push precision down against a task this paper never tests.
                skipped_no_task_gt += 1
                print(f"  SKIP {paper_id}: no '{task}' ground-truth errors")
                continue
            predicted = task_predictions(report, task)
            if borrowed_report:
                # A whole-paper report standing in for the task's own: drop the
                # other verifiers' findings so precision stays comparable.
                predicted = filter_predictions(predicted, task)
                borrowed += 1
        else:
            predicted = report.get("predicted_errors", [])

        tasks.append(
            {
                "paper_id": paper_id,
                "gt": gt,
                "predicted": predicted,
                "title": report.get("title")
                or report.get("paper_name")
                or paper_id,
                "mode": judge_mode,
                "cache_path": judge_cache_path(workspace_dir, cache_tag, judge_model),
                "report": report,
            }
        )

    if task and borrowed:
        print(
            f"\nNOTE: {borrowed} papers have no standalone '{task}' report; their "
            f"'{mode}' verify-paper report was used instead, filtered to "
            f"{task} findings.\n"
        )

    if client is not None:
        _warm_judge_cache(client, judge_model, tasks, concurrency, overwrite_judge)

    per_paper: List[Dict[str, Any]] = []
    difficulty_totals: Dict[str, int] = Counter()
    difficulty_founds: Dict[str, int] = Counter()
    type_totals: Dict[str, int] = Counter()
    type_founds: Dict[str, int] = Counter()

    for i, t in enumerate(tasks, 1):
        alignment = align_paper(
            t["predicted"],
            t["gt"],
            client,
            judge_model,
            t["title"],
            judge_mode,
            t["cache_path"],
            overwrite_judge=False,  # cache already warmed/overwritten above
        )
        if alignment is None:
            print(f"  [{i}/{len(tasks)}] JUDGE FAILED {t['paper_id']} — skipping")
            continue
        metrics = compute_metrics(alignment)

        for diff, count in alignment["difficulty_gt_total"].items():
            difficulty_totals[diff] += count
        for diff, count in alignment["difficulty_gt_found"].items():
            difficulty_founds[diff] += count
        for ctype, count in alignment["type_gt_total"].items():
            type_totals[ctype] += count
        for ctype, count in alignment["type_gt_found"].items():
            type_founds[ctype] += count

        result = {
            "paper_id": t["paper_id"],
            "n_gt": alignment["n_gt"],
            "n_pred": alignment["n_pred"],
            "matched_gt": alignment["matched_gt_count"],
            "matched_pred": alignment["matched_pred_count"],
            "mode": judge_mode,
            "task": task or "all",
            "tag": tag or "",
            **metrics,
        }
        per_paper.append(result)

        print(
            f"  [{i}/{len(tasks)}] {t['paper_id']}: "
            f"GT={alignment['n_gt']} Pred={alignment['n_pred']} "
            f"GTmatched={alignment['matched_gt_count']} "
            f"PredMatched={alignment['matched_pred_count']} "
            f"P={metrics['precision']:.2f} R={metrics['recall']:.2f} F1={metrics['f1']:.2f}"
        )

    if not per_paper:
        print("\nNo papers evaluated.")
        return [], {}

    summary = {
        "total_papers": len(per_paper),
        "mode": judge_mode,
        "task": task or "all",
        "tag": tag or "",
        "papers_without_task_gt": skipped_no_task_gt,
        "judge_model": judge_model if client is not None else "heuristic",
        **summarize(per_paper),
        **{
            f"recall_{diff}": (difficulty_founds.get(diff, 0) / total if total else 0.0)
            for diff, total in difficulty_totals.items()
        },
        **{
            f"recall_type_{ctype}": (
                type_founds.get(ctype, 0) / total if total else 0.0
            )
            for ctype, total in type_totals.items()
        },
    }

    print(f"\n{'=' * 60}")
    if task:
        tag_note = f" [tag: {tag}]" if tag else ""
        print(f"TASK: {task} — {TASK_REGISTRY[task]['label']}{tag_note}")
    print(
        f"SUMMARY: {len(per_paper)} papers, {summary['total_gt']} GT errors, "
        f"{summary['total_pred']} predicted errors"
    )
    if task and skipped_no_task_gt:
        print(
            f"  ({skipped_no_task_gt} papers skipped: no '{task}' errors in "
            f"their ground truth)"
        )
    print(
        f"  Headline avg_f1_score (F1 of macro P/R, comparable to PaperAudit): "
        f"{summary['avg_f1_score']:.3f}"
    )
    print(
        f"  Macro: P={summary['macro_precision']:.3f} "
        f"R={summary['macro_recall']:.3f} F1={summary['macro_f1']:.3f}"
    )
    print(
        f"  Micro: P={summary['micro_precision']:.3f} "
        f"R={summary['micro_recall']:.3f} F1={summary['micro_f1']:.3f}"
    )
    diff_str = {
        k.replace("recall_", ""): f"{v:.3f}"
        for k, v in summary.items()
        if k.startswith("recall_") and not k.startswith("recall_type_")
    }
    if diff_str:
        print(f"  Difficulty recall: {diff_str}")

    return per_paper, summary


def evaluate_comparison(
    llm_name: str,
    limit: Optional[int] = None,
    api_key: Optional[str] = None,
    use_llm_judge: bool = True,
    workspace_base: Optional[Path] = None,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    concurrency: int = 8,
    overwrite_judge: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Compare exhaustive vs uncertainty verification modes.

    For each paper, loads both verification-report-exhaustive.json and
    verification-report-uncertainty.json, computes standard metrics for each,
    and derives efficiency metrics showing how much work the uncertainty mode
    saved (in terms of snippets verified) vs the accuracy trade-off.

    Returns (per_paper_results, summary).
    """
    ws_base = workspace_base or WORKSPACE_BASE

    client = None
    if use_llm_judge:
        client = load_anthropic_client(api_key)
        if client is not None:
            print(f"LLM-as-a-judge: ENABLED (holistic, model={judge_model})\n")
        else:
            print("LLM-as-a-judge: UNAVAILABLE (using heuristic smoke test)\n")
    else:
        print(
            "LLM-as-a-judge: DISABLED — heuristic results are a smoke test only, "
            "NOT comparable to judged or published figures\n"
        )

    if not ws_base.exists():
        print(f"ERROR: Workspace directory not found: {ws_base}")
        print("Run with --prepare first, then run compare_uncertainty.sh.")
        sys.exit(1)

    paper_ids = sorted(
        [
            d.name
            for d in ws_base.iterdir()
            if d.is_dir()
            and (d / "ground_truth.json").exists()
            and (d / "verification-report-exhaustive.json").exists()
            and (d / "verification-report-uncertainty.json").exists()
        ]
    )

    if not paper_ids:
        print(
            "ERROR: No papers with both verification-report-exhaustive.json "
            "and verification-report-uncertainty.json found."
        )
        print("Run compare_uncertainty.sh first to generate both reports.")
        sys.exit(1)

    if limit:
        paper_ids = paper_ids[:limit]
        print(f"Comparing {len(paper_ids)} papers (limit={limit})\n")
    else:
        print(f"Comparing all {len(paper_ids)} papers\n")

    # Build judge task list for both modes
    paper_data: List[Dict[str, Any]] = []
    tasks: List[Dict[str, Any]] = []
    for paper_id in paper_ids:
        workspace_dir = ws_base / paper_id
        try:
            gt = load_json(workspace_dir / "ground_truth.json")
            exh_report = load_json(workspace_dir / "verification-report-exhaustive.json")
            unc_report = load_json(
                workspace_dir / "verification-report-uncertainty.json"
            )
        except Exception as e:
            print(f"  ERROR {paper_id}: {e}")
            continue

        title = exh_report.get("title", paper_id)
        entry = {
            "paper_id": paper_id,
            "gt": gt,
            "title": title,
            "exh_report": exh_report,
            "unc_report": unc_report,
            "exh_cache": judge_cache_path(workspace_dir, "exhaustive", judge_model),
            "unc_cache": judge_cache_path(workspace_dir, "uncertainty", judge_model),
        }
        paper_data.append(entry)
        tasks.append(
            {
                "paper_id": paper_id,
                "gt": gt,
                "predicted": exh_report.get("predicted_errors", []),
                "title": title,
                "mode": "exhaustive",
                "cache_path": entry["exh_cache"],
            }
        )
        tasks.append(
            {
                "paper_id": paper_id,
                "gt": gt,
                "predicted": unc_report.get("predicted_errors", []),
                "title": title,
                "mode": "uncertainty",
                "cache_path": entry["unc_cache"],
            }
        )

    if client is not None:
        _warm_judge_cache(client, judge_model, tasks, concurrency, overwrite_judge)

    per_paper: List[Dict[str, Any]] = []

    for i, entry in enumerate(paper_data, 1):
        paper_id = entry["paper_id"]
        gt = entry["gt"]

        exh_predicted = entry["exh_report"].get("predicted_errors", [])
        exh_alignment = align_paper(
            exh_predicted, gt, client, judge_model, entry["title"],
            "exhaustive", entry["exh_cache"],
        )
        unc_predicted = entry["unc_report"].get("predicted_errors", [])
        unc_alignment = align_paper(
            unc_predicted, gt, client, judge_model, entry["title"],
            "uncertainty", entry["unc_cache"],
        )
        if exh_alignment is None or unc_alignment is None:
            print(f"  [{i}/{len(paper_data)}] JUDGE FAILED {paper_id} — skipping")
            continue

        exh_metrics = compute_metrics(exh_alignment)
        unc_metrics = compute_metrics(unc_alignment)

        # ── Efficiency metrics ──────────────────────────────────────────
        exh_stats = entry["exh_report"].get("statistics", {})
        unc_stats = entry["unc_report"].get("statistics", {})

        exh_snippets = exh_stats.get(
            "snippets_verified", exh_stats.get("total_snippets", 0)
        )
        unc_snippets = unc_stats.get(
            "snippets_verified", unc_stats.get("total_snippets", 0)
        )

        snippet_reduction = (
            1.0 - (unc_snippets / exh_snippets) if exh_snippets > 0 else 0.0
        )

        f1_delta = unc_metrics["f1"] - exh_metrics["f1"]

        # Efficiency score: F1 change per unit of work saved.
        if snippet_reduction > 0.001:
            efficiency_score = f1_delta / snippet_reduction
        elif abs(f1_delta) < 0.001:
            efficiency_score = 0.0
        else:
            efficiency_score = float("inf") if f1_delta > 0 else float("-inf")

        result = {
            "paper_id": paper_id,
            "total_gt": exh_alignment["n_gt"],
            # Exhaustive
            "exh_n_gt": exh_alignment["n_gt"],
            "exh_n_pred": exh_alignment["n_pred"],
            "exh_matched_gt": exh_alignment["matched_gt_count"],
            "exh_matched_pred": exh_alignment["matched_pred_count"],
            "exh_precision": exh_metrics["precision"],
            "exh_recall": exh_metrics["recall"],
            "exh_f1": exh_metrics["f1"],
            "exh_snippets_verified": exh_snippets,
            "exh_errors_detected": exh_stats.get("errors_detected", 0),
            # Uncertainty
            "unc_n_gt": unc_alignment["n_gt"],
            "unc_n_pred": unc_alignment["n_pred"],
            "unc_matched_gt": unc_alignment["matched_gt_count"],
            "unc_matched_pred": unc_alignment["matched_pred_count"],
            "unc_precision": unc_metrics["precision"],
            "unc_recall": unc_metrics["recall"],
            "unc_f1": unc_metrics["f1"],
            "unc_snippets_verified": unc_snippets,
            "unc_errors_detected": unc_stats.get("errors_detected", 0),
            # Efficiency
            "snippet_reduction_ratio": snippet_reduction,
            "f1_delta": f1_delta,
            "precision_delta": unc_metrics["precision"] - exh_metrics["precision"],
            "recall_delta": unc_metrics["recall"] - exh_metrics["recall"],
            "efficiency_score": efficiency_score,
        }
        per_paper.append(result)

        eff_str = (
            f"eff={efficiency_score:.3f}"
            if isinstance(efficiency_score, float)
            and abs(efficiency_score) != float("inf")
            else f"eff={efficiency_score}"
        )
        print(
            f"  [{i}/{len(paper_data)}] {paper_id}: "
            f"GT={exh_alignment['n_gt']} | "
            f"Exh: P={exh_metrics['precision']:.2f} R={exh_metrics['recall']:.2f} "
            f"F1={exh_metrics['f1']:.2f} (snippets={exh_snippets}) | "
            f"Unc: P={unc_metrics['precision']:.2f} R={unc_metrics['recall']:.2f} "
            f"F1={unc_metrics['f1']:.2f} (snippets={unc_snippets}) | "
            f"ΔF1={f1_delta:+.3f} save={snippet_reduction:.1%} {eff_str}"
        )

    # ── Aggregate summary ──────────────────────────────────────────────
    if not per_paper:
        print("\nNo papers evaluated.")
        return [], {}

    n = len(per_paper)

    exh_summary = summarize(per_paper, prefix="exh_")
    unc_summary = summarize(per_paper, prefix="unc_")

    finite_eff = [
        r["efficiency_score"]
        for r in per_paper
        if isinstance(r["efficiency_score"], float)
        and abs(r["efficiency_score"]) != float("inf")
    ]
    avg_efficiency = sum(finite_eff) / len(finite_eff) if finite_eff else 0.0

    avg_snippet_reduction = sum(r["snippet_reduction_ratio"] for r in per_paper) / n
    avg_f1_delta = sum(r["f1_delta"] for r in per_paper) / n
    total_exh_snippets = sum(r["exh_snippets_verified"] for r in per_paper)
    total_unc_snippets = sum(r["unc_snippets_verified"] for r in per_paper)

    summary = {
        "total_papers": n,
        "judge_model": judge_model if client is not None else "heuristic",
        **exh_summary,
        "exh_total_snippets_verified": total_exh_snippets,
        **unc_summary,
        "unc_total_snippets_verified": total_unc_snippets,
        # Efficiency
        "avg_snippet_reduction_ratio": avg_snippet_reduction,
        "avg_f1_delta": avg_f1_delta,
        "avg_efficiency_score": avg_efficiency,
        "total_snippets_saved": total_exh_snippets - total_unc_snippets,
    }

    print(f"\n{'=' * 70}")
    print(f"COMPARISON SUMMARY: {n} papers, {exh_summary['exh_total_gt']} GT errors")
    print(f"{'=' * 70}")
    print(
        f"  Exhaustive    — avg_f1_score={exh_summary['exh_avg_f1_score']:.3f}  "
        f"Macro: P={exh_summary['exh_macro_precision']:.3f} "
        f"R={exh_summary['exh_macro_recall']:.3f} "
        f"F1={exh_summary['exh_macro_f1']:.3f}  "
        f"Micro F1={exh_summary['exh_micro_f1']:.3f}  "
        f"Snippets: {total_exh_snippets}"
    )
    print(
        f"  Uncertainty   — avg_f1_score={unc_summary['unc_avg_f1_score']:.3f}  "
        f"Macro: P={unc_summary['unc_macro_precision']:.3f} "
        f"R={unc_summary['unc_macro_recall']:.3f} "
        f"F1={unc_summary['unc_macro_f1']:.3f}  "
        f"Micro F1={unc_summary['unc_micro_f1']:.3f}  "
        f"Snippets: {total_unc_snippets}"
    )
    print(f"  {'─' * 61}")
    print(f"  Efficiency    — Avg snippet reduction: {avg_snippet_reduction:.1%}")
    print(
        f"                  Total snippets saved:  {total_exh_snippets - total_unc_snippets}"
    )
    print(f"                  Avg F1 delta:          {avg_f1_delta:+.4f}")
    print(f"                  Avg efficiency score:  {avg_efficiency:+.4f}")
    print(f"                  (efficiency = ΔF1 / snippet_reduction; ")
    print(f"                   positive = better accuracy with less work)")

    return per_paper, summary


def discover_llm_tags(
    ws_base: Path, tags_filter: Optional[List[str]] = None
) -> Dict[str, Dict[str, str]]:
    """Find baseline report tags in the workspace and the model behind each.

    A baseline report is any verification-report-<tag>.json carrying a
    `baseline` block (written by scripts/run_baseline.py). Returns
    {tag: {"model": ..., "provider": ...}}. Reports written before providers
    were recorded get their provider inferred from the model name.
    """
    found: Dict[str, Dict[str, str]] = {}
    for paper_dir in sorted(d for d in ws_base.iterdir() if d.is_dir()):
        if not (paper_dir / "ground_truth.json").exists():
            continue
        for report_path in sorted(paper_dir.glob("verification-report-*.json")):
            tag = report_path.stem[len("verification-report-") :]
            if tag in found:
                continue
            if tags_filter and tag not in tags_filter:
                continue
            try:
                report = load_json(report_path)
            except Exception:
                continue
            baseline = report.get("baseline")
            if not isinstance(baseline, dict) or not baseline.get("model"):
                continue
            model = baseline["model"]
            provider = baseline.get("provider")
            if not provider:
                try:
                    provider = resolve_provider(model)[0]
                except ValueError:
                    provider = "?"
            found[tag] = {"model": model, "provider": provider}
    return found


def evaluate_llms(
    limit: Optional[int] = None,
    api_key: Optional[str] = None,
    use_llm_judge: bool = True,
    workspace_base: Optional[Path] = None,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    concurrency: int = 8,
    overwrite_judge: bool = False,
    tags_filter: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Score every single-call baseline run in the workspace and compare LLMs.

    Discovers verification-report-<tag>.json files written by run_baseline.py /
    compare_llms.py, judges each against ground truth with the same holistic
    protocol used everywhere else, and adds an API cost estimate per model.

    Returns one row per model/tag.
    """
    ws_base = workspace_base or WORKSPACE_BASE
    if not ws_base.exists():
        print(f"ERROR: Workspace directory not found: {ws_base}")
        sys.exit(1)

    discovered = discover_llm_tags(ws_base, tags_filter)
    tag_to_model = {tag: info["model"] for tag, info in discovered.items()}
    tag_to_provider = {tag: info["provider"] for tag, info in discovered.items()}
    if not tag_to_model:
        print(
            "ERROR: No single-call baseline reports found in "
            f"{ws_base}.\nRun scripts/compare_llms.py first."
        )
        sys.exit(1)

    client = None
    if use_llm_judge:
        client = load_anthropic_client(api_key)
        if client is not None:
            print(f"LLM-as-a-judge: ENABLED (holistic, model={judge_model})\n")
        else:
            print("LLM-as-a-judge: UNAVAILABLE (using heuristic smoke test)\n")
    else:
        print(
            "LLM-as-a-judge: DISABLED — heuristic results are a smoke test only, "
            "NOT comparable to judged or published figures\n"
        )

    pricing = load_pricing()
    print(f"Comparing {len(tag_to_model)} LLM run(s): {', '.join(sorted(tag_to_model))}\n")

    rows: List[Dict[str, Any]] = []

    for tag, model in sorted(tag_to_model.items(), key=lambda kv: kv[1]):
        provider = tag_to_provider.get(tag) or "?"
        print(f"\n{'=' * 70}")
        print(f"MODEL: {model}  [{provider}]  (tag={tag})")
        print(f"{'=' * 70}")

        paper_ids = sorted(
            d.name
            for d in ws_base.iterdir()
            if d.is_dir()
            and (d / "ground_truth.json").exists()
            and (d / f"verification-report-{tag}.json").exists()
        )
        if limit:
            paper_ids = paper_ids[:limit]

        tasks: List[Dict[str, Any]] = []
        for paper_id in paper_ids:
            workspace_dir = ws_base / paper_id
            try:
                gt = load_json(workspace_dir / "ground_truth.json")
                report = load_json(workspace_dir / f"verification-report-{tag}.json")
            except Exception as e:
                print(f"  ERROR {paper_id}: {e}")
                continue
            tasks.append(
                {
                    "paper_id": paper_id,
                    "gt": gt,
                    "predicted": report.get("predicted_errors", []),
                    "title": report.get("title", paper_id),
                    "mode": tag,
                    "cache_path": judge_cache_path(workspace_dir, tag, judge_model),
                    "report": report,
                }
            )

        if not tasks:
            print(f"  No scorable papers for tag '{tag}' — skipping.")
            continue

        if client is not None:
            _warm_judge_cache(client, judge_model, tasks, concurrency, overwrite_judge)

        per_paper: List[Dict[str, Any]] = []
        in_tokens = out_tokens = cache_read = cache_write = 0
        usage_papers = 0

        for t in tasks:
            alignment = align_paper(
                t["predicted"],
                t["gt"],
                client,
                judge_model,
                t["title"],
                tag,
                t["cache_path"],
            )
            if alignment is None:
                print(f"  JUDGE FAILED {t['paper_id']} — skipping")
                continue
            metrics = compute_metrics(alignment)
            per_paper.append(
                {
                    "paper_id": t["paper_id"],
                    "n_gt": alignment["n_gt"],
                    "n_pred": alignment["n_pred"],
                    "matched_gt": alignment["matched_gt_count"],
                    "matched_pred": alignment["matched_pred_count"],
                    **metrics,
                }
            )

            usage = (t["report"].get("baseline") or {}).get("usage") or {}
            if isinstance(usage.get("input_tokens"), int):
                in_tokens += usage["input_tokens"]
                out_tokens += usage.get("output_tokens") or 0
                cache_read += usage.get("cache_read_input_tokens") or 0
                cache_write += usage.get("cache_creation_input_tokens") or 0
                usage_papers += 1

        if not per_paper:
            print(f"  No papers scored for tag '{tag}' — skipping.")
            continue

        summary = summarize(per_paper)
        n_papers = len(per_paper)

        # Cost: only meaningful if the reports actually recorded token usage.
        cost: Optional[float] = None
        cost_note = ""
        if usage_papers == 0:
            cost_note = "no token usage recorded in reports"
        elif price_for(model, pricing) is None:
            cost_note = "model not in configs/model_pricing.json"
        else:
            cost = estimate_cost(
                model,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                pricing=pricing,
                cache_read_tokens=cache_read,
                cache_write_tokens=cache_write,
            )
            if usage_papers < n_papers:
                # Extrapolate so cost-per-paper stays comparable across models.
                cost = cost * n_papers / usage_papers
                cost_note = f"extrapolated from {usage_papers}/{n_papers} papers"

        found = summary["total_matched_gt"]
        row = {
            "model": model,
            "provider": provider,
            "tag": tag,
            "papers": n_papers,
            "total_gt": summary["total_gt"],
            "total_pred": summary["total_pred"],
            "total_matched_gt": found,
            "macro_precision": summary["macro_precision"],
            "macro_recall": summary["macro_recall"],
            "macro_f1": summary["macro_f1"],
            "avg_f1_score": summary["avg_f1_score"],
            "micro_precision": summary["micro_precision"],
            "micro_recall": summary["micro_recall"],
            "micro_f1": summary["micro_f1"],
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "cache_read_tokens": cache_read,
            "cache_write_tokens": cache_write,
            "est_cost_usd": round(cost, 4) if cost is not None else None,
            "cost_per_paper_usd": round(cost / n_papers, 4) if cost is not None else None,
            "cost_per_error_found_usd": (
                round(cost / found, 4) if cost is not None and found > 0 else None
            ),
            "cost_note": cost_note,
        }
        rows.append(row)

        cost_str = f"${cost:.2f}" if cost is not None else f"n/a ({cost_note})"
        print(
            f"  {n_papers} papers | avg_f1_score={summary['avg_f1_score']:.3f} "
            f"P={summary['macro_precision']:.3f} R={summary['macro_recall']:.3f} | "
            f"tokens in={in_tokens:,} out={out_tokens:,} | est. cost {cost_str}"
        )

    if not rows:
        print("\nNo LLM runs scored.")
        return []

    # ── Comparison table ────────────────────────────────────────────────
    rows.sort(key=lambda r: r["avg_f1_score"], reverse=True)

    print(f"\n{'=' * 100}")
    print("LLM COMPARISON (holistic judge; avg_f1_score = F1 of macro P/R)")
    print(f"{'=' * 100}")
    header = (
        f"{'Model':<24} {'Provider':<10} {'Papers':>6} {'avg_F1':>7} {'MacroP':>7} "
        f"{'MacroR':>7} {'MicroF1':>8} {'In tok':>11} {'Out tok':>9} {'Cost':>9} "
        f"{'$/paper':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        cost_s = f"${r['est_cost_usd']:.2f}" if r["est_cost_usd"] is not None else "n/a"
        per_paper_s = (
            f"${r['cost_per_paper_usd']:.4f}"
            if r["cost_per_paper_usd"] is not None
            else "n/a"
        )
        print(
            f"{r['model']:<24} {r['provider']:<10} {r['papers']:>6} "
            f"{r['avg_f1_score']:>7.3f} {r['macro_precision']:>7.3f} "
            f"{r['macro_recall']:>7.3f} {r['micro_f1']:>8.3f} "
            f"{r['input_tokens']:>11,} {r['output_tokens']:>9,} "
            f"{cost_s:>9} {per_paper_s:>9}"
        )
    notes = [f"{r['model']}: {r['cost_note']}" for r in rows if r["cost_note"]]
    if notes:
        print("\nCost notes: " + "; ".join(notes))
    print(
        "\nCost covers the detector LLM only (judge calls are billed separately). "
        "Prices from configs/model_pricing.json."
    )

    return rows


def write_llm_comparison_results(rows: List[Dict[str, Any]], output_dir: Path) -> None:
    """Write the per-model LLM comparison table to CSV."""
    if not rows:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "llm_comparison.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nLLM comparison saved to: {path}")


def write_comparison_results(
    llm_name: str,
    per_paper: List[Dict[str, Any]],
    summary: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Write per-paper comparison metrics and summary to CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    per_paper_path = output_dir / f"{llm_name}_comparison.csv"
    if per_paper:
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

        print(f"\nPer-paper comparison saved to: {per_paper_path}")

    summary_path = output_dir / f"{llm_name}_comparison_summary.csv"
    if summary:
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
            writer.writeheader()
            writer.writerow(summary)

        print(f"Comparison summary saved to: {summary_path}")


def write_results(
    llm_name: str,
    per_paper: List[Dict[str, Any]],
    summary: Dict[str, Any],
    output_dir: Path,
    stem: Optional[str] = None,
) -> None:
    """Write per-paper metrics and summary to CSV files.

    `stem` overrides the filename prefix — a tagged run uses <llm>-<tag> so
    several models' results coexist in one task folder.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = stem or llm_name

    per_paper_path = output_dir / f"{stem}_metrics.csv"
    if per_paper:
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

    summary_path = output_dir / f"{stem}_summary.csv"
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

  # Compare exhaustive vs uncertainty verification modes
  python scripts/evaluate_verifier.py --llm claude-sonnet-4-5-20250929 --evaluate-comparison --limit 5

  # Score only the mathematical errors, from the orchestrate-math reports
  python scripts/evaluate_verifier.py --llm claude-sonnet-4-5-20250929 --evaluate --task math

  # Score one tagged math run (batch-verify-math.sh --tag opus5)
  python scripts/evaluate_verifier.py --llm claude-sonnet-4-5-20250929 --evaluate --task math --tag opus5

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
        "--evaluate-comparison",
        action="store_true",
        help="Compare exhaustive vs uncertainty verification modes. "
        "Requires both verification-report-exhaustive.json and "
        "verification-report-uncertainty.json for each paper "
        "(generated by compare_uncertainty.sh).",
    )
    parser.add_argument(
        "--evaluate-llms",
        action="store_true",
        help="Compare single-call baseline runs from different LLMs. Discovers "
        "verification-report-<tag>.json files written by scripts/compare_llms.py, "
        "scores each with the holistic judge, and adds an API cost estimate.",
    )
    parser.add_argument(
        "--tags",
        type=str,
        default=None,
        help="--evaluate-llms only: comma-separated report tags to include "
        "(default: every baseline tag found in the workspace)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Directory for output CSV files (default: {RESULTS_DIR}, or "
        f"{TASK_SPECIFIC_OUTPUT_DIR}/<task> when --task is given)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Anthropic API key for LLM-as-a-judge "
        "(default: ANTHROPIC_API_KEY env var, then project .env)",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Disable LLM judge; use a Jaccard-overlap heuristic instead. "
        "Smoke test only — NOT comparable to judged results.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=DEFAULT_JUDGE_MODEL,
        help=f"Model used for holistic LLM judging (default: {DEFAULT_JUDGE_MODEL}). "
        "Any provider works — a bare name uses that provider's native API, and a "
        "'provider:model' prefix forces one (e.g. 'anthropic:deepseek-v4-pro' to "
        "reach a third-party model through an Anthropic-compatible proxy). "
        "Judge choice materially shifts scores: check it with scripts/judge_agreement.py.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Parallel judge calls (default: 8)",
    )
    parser.add_argument(
        "--overwrite-judge",
        action="store_true",
        help="Re-run judge calls even when a cached verdict exists",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="exhaustive",
        help="Which verification report to evaluate (default: exhaustive). "
        "Loads verification-report-<mode>.json, falling back to "
        "verification-report.json. Any tag works — e.g. 'uncertainty', or "
        "'baseline' for reports produced by scripts/run_baseline.py.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        choices=TASK_NAMES,
        help="Evaluate a single error family produced by an ad-hoc system "
        "instead of the whole verify-paper pipeline. Default: None — every "
        f"error is evaluated. Known tasks: {', '.join(TASK_NAMES)}. With a task "
        "set, predictions come from that task's standalone report (e.g. "
        "<paper>_verified_math.json from the orchestrate-math skill), the "
        "ground truth is filtered to the task's error categories, and results "
        f"are written to {TASK_SPECIFIC_OUTPUT_DIR}/<task>/. Use with --evaluate.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="--task only: read the tagged task report "
        "(<paper>_verified_math-<tag>.json, written by "
        "scripts/batch-verify-math.sh --tag <tag>) instead of the untagged one. "
        "Judge verdicts are cached per tag and the CSVs are named "
        "<llm>-<tag>_*.csv, so several models can be compared side by side. "
        "Composes with --skill and --llm-skill, which prepend their own "
        "segments. For a full verify-paper report tag, use --mode instead.",
    )
    parser.add_argument(
        "--skill",
        type=str,
        default=None,
        help="--task only: the skill that produced the report. Folded into the "
        "tag, so /orchestrate-math and /orchestrate-math-v2 reports over the "
        "same paper can never overwrite each other: --skill orchestrate-math-v2 "
        "reads <paper>_verified_math-orchestrate-math-v2.json, and with --tag "
        "run3 it reads ...-orchestrate-math-v2-run3.json. Pass the same "
        "--skill/--llm-skill/--tag trio you gave batch-verify-math.sh.",
    )
    parser.add_argument(
        "--llm-skill",
        type=str,
        default=None,
        help="--task only: the model the skill was driven with via its own "
        "--llm flag (orchestrate-math-v2's baseline arm; no effect for "
        "orchestrate-math, which has no --llm). Also folded into the tag, so "
        "the same skill run against different baseline models keeps separate "
        "reports.",
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

    if (
        not args.prepare
        and not args.evaluate
        and not args.evaluate_comparison
        and not args.evaluate_llms
    ):
        parser.print_help()
        print(
            "\nERROR: Specify at least one of --prepare, --evaluate, "
            "--evaluate-comparison, or --evaluate-llms."
        )
        sys.exit(1)

    exclusive = [args.evaluate, args.evaluate_comparison, args.evaluate_llms]
    if sum(bool(x) for x in exclusive) > 1:
        print(
            "ERROR: --evaluate, --evaluate-comparison, and --evaluate-llms are "
            "mutually exclusive. Use --evaluate for a single-mode evaluation, "
            "--evaluate-comparison for exhaustive-vs-uncertainty, or "
            "--evaluate-llms to compare baseline runs across models."
        )
        sys.exit(1)

    if (args.tag or args.skill or args.llm_skill) and not args.task:
        given = ", ".join(
            f"--{name}"
            for name, value in (
                ("tag", args.tag),
                ("skill", args.skill),
                ("llm-skill", args.llm_skill),
            )
            if value
        )
        print(
            f"ERROR: {given} applies to task runs (--task). For a full "
            "verify-paper report tag, use --mode <tag> instead."
        )
        sys.exit(1)

    # One composed tag drives the report filename, the judge cache and the CSV
    # stem. Computed by the same function batch-verify-math.sh uses to name the
    # report, so the two cannot drift apart.
    try:
        effective_tag = compose_task_tag(args.tag, args.skill, args.llm_skill)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    if args.task and not args.evaluate:
        print(
            "ERROR: --task only applies to --evaluate. Run the task's ad-hoc "
            "system first (e.g. ./scripts/batch-verify-math.sh for --task math), "
            "then score it with --evaluate --task <task>."
        )
        sys.exit(1)

    llm_name = args.llm
    synth_name = LLM_NAME_MAP.get(llm_name, llm_name)

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    elif args.task:
        output_dir = task_output_dir(args.task)
    else:
        output_dir = RESULTS_DIR
    paperaudit_base = Path(args.paperaudit_base)
    workspace_base = Path(args.workspace_dir)

    # --- Phase 1: Prepare ---
    if args.prepare:
        print(f"{'=' * 60}")
        print(f"PHASE 1: PREPARE (LLM: {synth_name})")
        print(f"  PaperAudit base: {paperaudit_base}")
        print(f"  Workspace:       {workspace_base}")
        print(f"{'=' * 60}\n")
        prepare_papers(
            synth_name,
            args.limit,
            paperaudit_base=paperaudit_base,
            workspace_base=workspace_base,
        )

    # --- Phase 2: Evaluate ---
    if args.evaluate:
        print(f"\n{'=' * 60}")
        print(f"PHASE 2: EVALUATE (LLM: {synth_name})")
        print(f"  Workspace: {workspace_base}")
        if args.task:
            print(f"  Task:      {args.task} — {TASK_REGISTRY[args.task]['label']}")
            if args.skill:
                print(f"  Skill:     {args.skill}")
            if args.llm_skill:
                print(f"  Skill LLM: {args.llm_skill}")
            if effective_tag:
                print(f"  Tag:       {effective_tag}")
        else:
            print(f"  Mode:      {args.mode}")
        print(f"  Judge:     {args.judge_model}")
        print(f"  Output:    {output_dir}")
        print(f"{'=' * 60}\n")

        per_paper, summary = evaluate_papers(
            synth_name,
            args.limit,
            args.api_key,
            use_llm_judge=not args.no_judge,
            mode=args.mode,
            workspace_base=workspace_base,
            judge_model=args.judge_model,
            concurrency=args.concurrency,
            overwrite_judge=args.overwrite_judge,
            task=args.task,
            tag=effective_tag,
        )

        result_stem = f"{synth_name}-{effective_tag}" if effective_tag else synth_name
        write_results(synth_name, per_paper, summary, output_dir, stem=result_stem)

    # --- Mode Comparison: Exhaustive vs Uncertainty ---
    if args.evaluate_comparison:
        print(f"\n{'=' * 60}")
        print(f"MODE COMPARISON: Exhaustive vs Uncertainty")
        print(f"  LLM:       {synth_name}")
        print(f"  Workspace: {workspace_base}")
        print(f"  Judge:     {args.judge_model}")
        print(f"{'=' * 60}\n")

        per_paper, summary = evaluate_comparison(
            synth_name,
            args.limit,
            args.api_key,
            use_llm_judge=not args.no_judge,
            workspace_base=workspace_base,
            judge_model=args.judge_model,
            concurrency=args.concurrency,
            overwrite_judge=args.overwrite_judge,
        )

        write_comparison_results(synth_name, per_paper, summary, output_dir)

    # --- LLM Comparison: single-call baseline across models ---
    if args.evaluate_llms:
        print(f"\n{'=' * 60}")
        print("LLM COMPARISON: single-call baseline across models")
        print(f"  Workspace: {workspace_base}")
        print(f"  Judge:     {args.judge_model}")
        print(f"{'=' * 60}\n")

        tags_filter = (
            [t.strip() for t in args.tags.split(",") if t.strip()]
            if args.tags
            else None
        )
        rows = evaluate_llms(
            limit=args.limit,
            api_key=args.api_key,
            use_llm_judge=not args.no_judge,
            workspace_base=workspace_base,
            judge_model=args.judge_model,
            concurrency=args.concurrency,
            overwrite_judge=args.overwrite_judge,
            tags_filter=tags_filter,
        )
        write_llm_comparison_results(rows, output_dir)


if __name__ == "__main__":
    main()
