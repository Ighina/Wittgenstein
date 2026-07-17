#!/usr/bin/env bash
#
# verify_papers.sh — Automate verify-paper skill on papers prepared by evaluate_verifier.py.
#
# For each prepared paper (up to --limit), invokes Claude Code in print mode
# with the verify-paper skill to produce a verification-report.json.
#
# Usage:
#   ./scripts/verify_papers.sh --llm claude-sonnet-4-5-20250929
#   ./scripts/verify_papers.sh --llm claude-sonnet-4-5-20250929 --limit 5
#   ./scripts/verify_papers.sh --llm gpt-5-2025-08-07 --limit 3 --model claude-opus-4-8
#   ./scripts/verify_papers.sh --llm claude-sonnet-4-5-20250929 --mode uncertainty --strictness lenient
#
# Options:
#   --llm <name>       LLM name used to corrupt papers (required — matches prepare step)
#   --limit <N>        Max papers to verify (default: all prepared papers)
#   --model <model>    Claude model to use (default: claude-sonnet-4-5-20250929)
#   --mode <mode>      Verification mode: exhaustive (default) or uncertainty
#   --strictness <s>   Error sensitivity: strict (default) or lenient
#   --dry-run          Print what would be done without actually running
#   --help             Show this help
#
# Prerequisites:
#   1. claude CLI installed and authenticated (claude login)
#   2. Run evaluate_verifier.py --prepare first to extract papers + ground truth
#   3. Run this script from the Wittgenstein/ project root (so .claude/skills are loaded)

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"            # Wittgenstein/ (where .claude/skills lives)
WORKSPACE_DIR="$PROJECT_DIR/outputs/paperaudit-workspace"
RESULTS_DIR="$PROJECT_DIR/outputs/paperaudit-eval"

# Defaults
LLM_NAME=""
LIMIT=""
MODEL="claude-sonnet-4-5-20250929"
MODE="exhaustive"
STRICTNESS="strict"
DRY_RUN=false

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------

usage() {
    head -22 "$0" | tail -20
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
        --mode)
            MODE="$2"
            if [[ "$MODE" != "exhaustive" && "$MODE" != "uncertainty" ]]; then
                echo "ERROR: --mode must be 'exhaustive' or 'uncertainty', got: $MODE"
                usage
            fi
            shift 2 ;;
        --strictness)
            STRICTNESS="$2"
            if [[ "$STRICTNESS" != "strict" && "$STRICTNESS" != "lenient" ]]; then
                echo "ERROR: --strictness must be 'strict' or 'lenient', got: $STRICTNESS"
                usage
            fi
            shift 2 ;;
        --dry-run)
            DRY_RUN=true; shift ;;
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
if [[ -d "$WORKSPACE_DIR" ]]; then
    for d in "$WORKSPACE_DIR"/*/; do
        paper_id=$(basename "$d")
        # Skip non-paper directories
        [[ "$paper_id" == "*" ]] && continue

        gt_file="$d/ground_truth.json"
        paper_file="$d/paper.md"
        report_file="$d/verification-report.json"

        # Only include papers that were prepared (have ground_truth + paper.md)
        # and have NOT yet been verified (no verification-report.json)
        if [[ -f "$gt_file" && -f "$paper_file" && ! -f "$report_file" ]]; then
            PAPER_DIRS+=("$paper_id")
        fi
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
    echo "Verifying ${#PAPER_DIRS[@]} of $TOTAL prepared papers (limit=$LIMIT)"
else
    echo "Verifying all $TOTAL prepared papers"
fi
echo ""

# ---------------------------------------------------------------------------
# Verify each paper
# ---------------------------------------------------------------------------

SUCCESS=0
FAILED=0
SKIPPED=0
declare -a FAILED_PAPERS=()

for paper_id in "${PAPER_DIRS[@]}"; do
    paper_path="$WORKSPACE_DIR/$paper_id/paper.md"
    report_path="$WORKSPACE_DIR/$paper_id/verification-report.json"
    log_path="$WORKSPACE_DIR/$paper_id/verify.log"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[$((SUCCESS + FAILED + 1))/${#PAPER_DIRS[@]}] Verifying: $paper_id"
    echo "  Paper:  $paper_path"
    echo "  Report: $report_path"
    echo ""

    if [[ "$DRY_RUN" == true ]]; then
        echo "  [DRY RUN] Would run: claude -p ..."
        echo ""
        continue
    fi

    # Construct the prompt for Claude Code.
    # The verify-paper skill is loaded from .claude/skills/ (project-level).
    # We instruct Claude to:
    #   1. Read the paper markdown
    #   2. Invoke the verify-paper skill
    #   3. Write the structured verification report to the output path
    read -r -d '' PROMPT <<EOF || true
You are a scientific paper verification system. Your task is to verify the paper stored at the path below using the verify-paper skill.

Paper path: ${paper_path}

Verification settings:
- Mode: ${MODE}
- Error sensitivity (strictness): ${STRICTNESS}

Steps:
1. Read the paper file at the path above.
2. Use the verify-paper skill (Skill tool with skill="verify-paper") to run the full verification pipeline on this paper. Use the verification settings above — when invoking the skill, pass the mode and strictness parameters.
3. After the verification pipeline completes, write the final verification report as a JSON file to: ${report_path}

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

    # Run Claude Code in print mode (skills loaded from PROJECT_DIR/.claude/skills)
    set +e
    claude -p \
        --model "$MODEL" \
        --dangerously-skip-permissions \
        --output-format text \
        --verbose \
        "$PROMPT" \
        > "$log_path" 2>&1
    EXIT_CODE=$?
    set -e

    if [[ $EXIT_CODE -eq 0 && -f "$report_path" ]]; then
        # Quick validation: check it's valid JSON with predicted_errors
        if python3 -c "
import json
with open('$report_path') as f:
    r = json.load(f)
assert 'predicted_errors' in r
assert 'verdict' in r
print(f'  ✓ OK: {len(r[\"predicted_errors\"])} errors detected, verdict={r[\"verdict\"]}')
" 2>/dev/null; then
            SUCCESS=$((SUCCESS + 1))
            echo "  ✓ Verification complete"
        else
            FAILED=$((FAILED + 1))
            FAILED_PAPERS+=("$paper_id")
            echo "  ✗ Report file exists but is invalid JSON"
        fi
    else
        FAILED=$((FAILED + 1))
        FAILED_PAPERS+=("$paper_id")
        echo "  ✗ Verification failed (exit code: $EXIT_CODE)"
        echo "  Log saved to: $log_path"
    fi
    echo ""
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "DONE: $SUCCESS succeeded, $FAILED failed, $SKIPPED skipped"
echo ""

if [[ ${#FAILED_PAPERS[@]} -gt 0 ]]; then
    echo "Failed papers:"
    for p in "${FAILED_PAPERS[@]}"; do
        echo "  - $p"
    done
    echo ""
fi

if [[ "$SUCCESS" -gt 0 ]]; then
    echo "Next step: evaluate the results:"
    echo "  python scripts/evaluate_verifier.py --llm $LLM_NAME --evaluate --no-judge"
    if [[ -n "$LIMIT" ]]; then
        echo "  python scripts/evaluate_verifier.py --llm $LLM_NAME --evaluate --no-judge --limit $LIMIT"
    fi
fi
