#!/usr/bin/env bash
#
# compare_uncertainty.sh — Compare "exhaustive" vs "uncertainty" verification modes.
#
# For each prepared paper (up to --limit), runs verify-paper TWICE:
#   1. Exhaustive mode  → verification-report-exhaustive.json
#   2. Uncertainty mode → verification-report-uncertainty.json
#
# Unlike verify_papers.sh, this script does NOT skip papers that already have
# a verification-report.json — it always re-runs both modes so the comparison
# is based on fresh, consistent results.
#
# Usage:
#   ./scripts/compare_uncertainty.sh --llm claude-sonnet-4-5-20250929
#   ./scripts/compare_uncertainty.sh --llm claude-sonnet-4-5-20250929 --limit 5
#   ./scripts/compare_uncertainty.sh --llm gpt-5-2025-08-07 --limit 3 --model claude-opus-4-8
#   ./scripts/compare_uncertainty.sh --llm claude-sonnet-4-5-20250929 --strictness lenient
#   ./scripts/compare_uncertainty.sh --llm claude-sonnet-4-5-20250929 --workspace-dir data/paperaudit-workspace --limit 5
#
# Options:
#   --llm <name>       LLM name used to corrupt papers (required — matches prepare step)
#   --limit <N>        Max papers to verify (default: all prepared papers)
#   --model <model>    Claude model to use (default: claude-sonnet-4-5-20250929)
#   --strictness <s>   Error sensitivity: strict (default) or lenient
#   --workspace-dir    Workspace directory (default: data/spot-workspace)
#   --resume           Skip papers where both reports already exist and are valid
#   --dry-run          Print what would be done without actually running
#   --help             Show this help
#
# Prerequisites:
#   1. claude CLI installed and authenticated (claude login)
#   2. Run evaluate_verifier.py --prepare first to extract papers + ground truth
#   3. Run this script from the Wittgenstein/ project root (so .claude/skills are loaded)
#
# Next step after this script completes:
#   python scripts/evaluate_verifier.py --llm <name> --evaluate-comparison --limit <N>

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"            # Wittgenstein/ (where .claude/skills lives)
RESULTS_DIR="$PROJECT_DIR/outputs/spot-eval"

# Defaults
LLM_NAME=""
LIMIT=""
MODEL="claude-sonnet-4-5-20250929"
STRICTNESS="lenient"
WORKSPACE_DIR="$PROJECT_DIR/data/spot-workspace"
DRY_RUN=false
RESUME=false

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------

usage() {
    head -34 "$0" | tail -32
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --llm)
            LLM_NAME="$2"; shift 2 ;;
        --limit)
            LIMIT="$2"; shift 2 ;;
        --model)
            MODEL="$2"; shift 2 ;;
        --strictness)
            STRICTNESS="$2"
            if [[ "$STRICTNESS" != "strict" && "$STRICTNESS" != "lenient" ]]; then
                echo "ERROR: --strictness must be 'strict' or 'lenient', got: $STRICTNESS"
                usage
            fi
            shift 2 ;;
        --workspace-dir)
            WORKSPACE_DIR="$2"
            # Allow relative paths (relative to PROJECT_DIR)
            if [[ "$WORKSPACE_DIR" != /* ]]; then
                WORKSPACE_DIR="$PROJECT_DIR/$WORKSPACE_DIR"
            fi
            shift 2 ;;
        --dry-run)
            DRY_RUN=true; shift ;;
        --resume)
            RESUME=true; shift ;;
        --help|-h)
            usage ;;
        *)
            echo "ERROR: Unknown option: $1"
            usage ;;
    esac
done

if [[ -z "$LLM_NAME" ]]; then
    echo "ERROR: --llm is required (e.g., --llm claude-sonnet-4-5-20250929)"
    usage
fi

# ---------------------------------------------------------------------------
# Discover papers to verify
# ---------------------------------------------------------------------------

PAPER_DIRS=()
SKIPPED_COUNT=0
if [[ -d "$WORKSPACE_DIR" ]]; then
    for d in "$WORKSPACE_DIR"/*/; do
        paper_id=$(basename "$d")
        # Skip non-paper directories
        [[ "$paper_id" == "*" ]] && continue

        gt_file="$d/ground_truth.json"
        paper_file="$d/paper.md"
        exh_report="$d/verification-report-exhaustive.json"
        unc_report="$d/verification-report-uncertainty.json"

        # Only include papers that were prepared (have ground_truth + paper.md).
        if [[ ! -f "$gt_file" || ! -f "$paper_file" ]]; then
            continue
        fi

        # In --resume mode, skip papers where BOTH reports already exist and
        # are valid JSON (i.e. completed successfully in a previous run).
        if [[ "$RESUME" == true && -f "$exh_report" && -f "$unc_report" ]]; then
            if python3 -c "
import json
with open('$exh_report') as f: json.load(f)
with open('$unc_report') as f: json.load(f)
" 2>/dev/null; then
                SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
                continue
            fi
        fi

        PAPER_DIRS+=("$paper_id")
    done
fi

# Sort for deterministic ordering
if [[ ${#PAPER_DIRS[@]} -gt 0 ]]; then
    IFS=$'\n' PAPER_DIRS=($(sort <<<"${PAPER_DIRS[*]}"))
    unset IFS
fi

TOTAL=${#PAPER_DIRS[@]}

if [[ "$TOTAL" -eq 0 ]]; then
    echo "No papers to verify."
    echo "Run 'python scripts/evaluate_verifier.py --llm $LLM_NAME --prepare' first."
    exit 0
fi

# Apply limit
if [[ -n "$LIMIT" && "$LIMIT" -lt "$TOTAL" ]]; then
    PAPER_DIRS=("${PAPER_DIRS[@]:0:$LIMIT}")
    echo "Comparing ${#PAPER_DIRS[@]} of $TOTAL prepared papers (limit=$LIMIT)"
else
    echo "Comparing all $TOTAL prepared papers"
fi
if [[ "$RESUME" == true && "$SKIPPED_COUNT" -gt 0 ]]; then
    echo "  ($SKIPPED_COUNT already completed — skipped via --resume)"
fi
echo "Each paper will be verified twice: exhaustive mode + uncertainty mode"
echo ""

# ---------------------------------------------------------------------------
# Verify each paper in both modes
# ---------------------------------------------------------------------------

SUCCESS=0
FAILED=0
declare -a FAILED_PAPERS=()

for paper_id in "${PAPER_DIRS[@]}"; do
    paper_path="$WORKSPACE_DIR/$paper_id/paper.md"

    # Report paths for each mode
    exhaustive_report="$WORKSPACE_DIR/$paper_id/verification-report-exhaustive.json"
    uncertainty_report="$WORKSPACE_DIR/$paper_id/verification-report-uncertainty.json"

    # Log paths for each mode
    exhaustive_log="$WORKSPACE_DIR/$paper_id/verify-exhaustive.log"
    uncertainty_log="$WORKSPACE_DIR/$paper_id/verify-uncertainty.log"

    paper_num=$((SUCCESS + FAILED + 1))
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[$paper_num/${#PAPER_DIRS[@]}] Comparing: $paper_id"
    echo "  Paper: $paper_path"
    echo ""

    paper_failed=false

    # -----------------------------------------------------------------------
    # Mode 1: Exhaustive
    # -----------------------------------------------------------------------
    echo "  ── Exhaustive mode ──────────────────────────────────────────"
    echo "  Report: $exhaustive_report"

    if [[ "$DRY_RUN" == true ]]; then
        echo "  [DRY RUN] Would run: claude -p ... (exhaustive mode)"
    else
        read -r -d '' PROMPT_EXHAUSTIVE <<EOF || true
You are a scientific paper verification system. Your task is to verify the paper stored at the path below using the verify-paper skill.

Paper path: ${paper_path}

Verification settings:
- Mode: exhaustive
- Error sensitivity (strictness): ${STRICTNESS}

Steps:
1. Read the paper file at the path above.
2. Use the verify-paper skill (Skill tool with skill="verify-paper") to run the full verification pipeline on this paper. Use the verification settings above — when invoking the skill, pass mode="exhaustive" and strictness="${STRICTNESS}" parameters.
3. After the verification pipeline completes, write the final verification report as a JSON file to: ${exhaustive_report}

The report must follow this structure:
{
  "paper_id": "...",
  "title": "...",
  "paper_category": "...",
  "predicted_errors": [
    {
      "error_category": "...",
      "error_location": "...",
      "confidence": 0.0,
      "supporting_evidence": "...",
      "verifier_name": "...",
      "snippet_id": "..."
    }
  ],
  "statistics": {
    "total_snippets": 0,
    "snippets_verified": 0,
    "errors_detected": 0,
    "verifier_usage": {}
  },
  "verdict": "PASS | NEEDS_REVIEW | NEEDS_CORRECTION",
  "summary": "..."
}

Important: Write the complete JSON report to the exact path specified. Do not output the report inline — use the Write tool to save it.
EOF

        set +e
        claude -p \
            --model "$MODEL" \
            --dangerously-skip-permissions \
            --output-format text \
            --verbose \
            "$PROMPT_EXHAUSTIVE" \
            > "$exhaustive_log" 2>&1
        EXHAUSTIVE_EXIT=$?
        set -e

        if [[ $EXHAUSTIVE_EXIT -eq 0 && -f "$exhaustive_report" ]]; then
            if python3 -c "
import json
with open('$exhaustive_report') as f:
    r = json.load(f)
assert 'predicted_errors' in r
assert 'verdict' in r
print(f'  ✓ Exhaustive OK: {len(r[\"predicted_errors\"])} errors detected, verdict={r[\"verdict\"]}')
" 2>/dev/null; then
                echo "  ✓ Exhaustive mode complete"
            else
                paper_failed=true
                echo "  ✗ Exhaustive report exists but is invalid JSON"
            fi
        else
            paper_failed=true
            echo "  ✗ Exhaustive mode failed (exit code: $EXHAUSTIVE_EXIT)"
            echo "  Log saved to: $exhaustive_log"
        fi
    fi
    echo ""

    # -----------------------------------------------------------------------
    # Mode 2: Uncertainty
    # -----------------------------------------------------------------------
    echo "  ── Uncertainty mode ─────────────────────────────────────────"
    echo "  Report: $uncertainty_report"

    if [[ "$DRY_RUN" == true ]]; then
        echo "  [DRY RUN] Would run: claude -p ... (uncertainty mode)"
    else
        read -r -d '' PROMPT_UNCERTAINTY <<EOF || true
You are a scientific paper verification system. Your task is to verify the paper stored at the path below using the verify-paper skill.

Paper path: ${paper_path}

Verification settings:
- Mode: uncertainty
- Error sensitivity (strictness): ${STRICTNESS}

Steps:
1. Read the paper file at the path above.
2. Use the verify-paper skill (Skill tool with skill="verify-paper") to run the full verification pipeline on this paper. Use the verification settings above — when invoking the skill, pass mode="uncertainty" and strictness="${STRICTNESS}" parameters.
3. After the verification pipeline completes, write the final verification report as a JSON file to: ${uncertainty_report}

The report must follow this structure:
{
  "paper_id": "...",
  "title": "...",
  "paper_category": "...",
  "predicted_errors": [
    {
      "error_category": "...",
      "error_location": "...",
      "confidence": 0.0,
      "supporting_evidence": "...",
      "verifier_name": "...",
      "snippet_id": "..."
    }
  ],
  "statistics": {
    "total_snippets": 0,
    "snippets_verified": 0,
    "errors_detected": 0,
    "verifier_usage": {}
  },
  "verdict": "PASS | NEEDS_REVIEW | NEEDS_CORRECTION",
  "summary": "..."
}

Important: Write the complete JSON report to the exact path specified. Do not output the report inline — use the Write tool to save it.
EOF

        set +e
        claude -p \
            --model "$MODEL" \
            --dangerously-skip-permissions \
            --output-format text \
            --verbose \
            "$PROMPT_UNCERTAINTY" \
            > "$uncertainty_log" 2>&1
        UNCERTAINTY_EXIT=$?
        set -e

        if [[ $UNCERTAINTY_EXIT -eq 0 && -f "$uncertainty_report" ]]; then
            if python3 -c "
import json
with open('$uncertainty_report') as f:
    r = json.load(f)
assert 'predicted_errors' in r
assert 'verdict' in r
print(f'  ✓ Uncertainty OK: {len(r[\"predicted_errors\"])} errors detected, verdict={r[\"verdict\"]}')
" 2>/dev/null; then
                echo "  ✓ Uncertainty mode complete"
            else
                paper_failed=true
                echo "  ✗ Uncertainty report exists but is invalid JSON"
            fi
        else
            paper_failed=true
            echo "  ✗ Uncertainty mode failed (exit code: $UNCERTAINTY_EXIT)"
            echo "  Log saved to: $uncertainty_log"
        fi
    fi
    echo ""

    if [[ "$paper_failed" == true ]]; then
        FAILED=$((FAILED + 1))
        FAILED_PAPERS+=("$paper_id")
    else
        SUCCESS=$((SUCCESS + 1))
    fi
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "DONE: $SUCCESS succeeded, $FAILED failed"
echo ""

if [[ ${#FAILED_PAPERS[@]} -gt 0 ]]; then
    echo "Failed papers:"
    for p in "${FAILED_PAPERS[@]}"; do
        echo "  - $p"
    done
    echo ""
fi

if [[ "$SUCCESS" -gt 0 ]]; then
    echo "Next step: evaluate the comparison results:"
    echo "  python scripts/evaluate_verifier.py --llm $LLM_NAME --evaluate-comparison"
    if [[ -n "$LIMIT" ]]; then
        echo "  python scripts/evaluate_verifier.py --llm $LLM_NAME --evaluate-comparison --limit $LIMIT"
    fi
fi
