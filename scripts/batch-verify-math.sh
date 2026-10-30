#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# batch-verify-math.sh — Run a mathematical-verification skill over every paper
# of the PaperAudit or SPOT workspace, producing one standalone mathematical
# verification report per paper.
#
# --skill picks the skill (default: orchestrate-math). --llm-skill drives a
# skill's own --llm flag, where it has one (orchestrate-math-v2's baseline arm).
#
# This is the math-only sibling of batch-verify.sh: instead of the full
# verify-paper pipeline it runs the orchestrate-math skill, which extracts every
# self-contained equation, theorem and proof, routes each to verify-math or
# verify-math-lean, and writes <paper_name>_verified_math.json NEXT TO THE PAPER
# (so data/<ds>-workspace/<paper_id>/paper_verified_math.json).
#
# The report name carries the producing skill, its --llm-skill and any --tag, so
# runs never overwrite each other:
#
#   <paper>_verified_math[-<skill>][-<llm-skill>][-<tag>].json
#
# The skill segment is omitted for the default orchestrate-math, keeping existing
# untagged reports exactly where they were. The name is computed by
# evaluate_verifier.compose_task_tag() — the SAME function the scorer uses to
# find the report — so the writer and the reader cannot drift apart.
#
# Score the results with the task-specific evaluation, passing the same trio:
#   python scripts/evaluate_verifier.py --llm <llm> --evaluate --task math \
#       --workspace-dir data/paperaudit-workspace \
#       [--skill <skill>] [--llm-skill <model>] [--tag <tag>]
# which writes to task_specific_output/math/.
#
# Usage:
#   ./scripts/batch-verify-math.sh [OPTIONS]
#
# Options:
#   --dataset NAME    paperaudit|spot — which workspace to walk (default: paperaudit)
#   --workspace DIR   Explicit workspace directory (overrides --dataset)
#   --papers N        Number of papers to verify (default: all)
#   --offset N        Skip first N papers (default: 0)
#   --paper-id ID     Verify a single paper by workspace directory name
#   --paper-list FILE Verify exactly the workspace directory names listed in
#                     FILE, one per line (# comments and blanks ignored). Use it
#                     to run a scored subset — e.g. SPOT's 37 equation/proof
#                     papers — instead of a whole workspace. Honours --force the
#                     same way as a full walk, and keeps the file's own order
#                     after de-duplication.
#   --model MODEL     Claude model to drive the skill (default: claude-sonnet-4-5-20250929)
#   --skill NAME      Verification skill to run (default: orchestrate-math).
#                     Folded into the report name unless it is the default.
#   --llm-skill MODEL Model for the skill's own --llm flag, where it has one
#                     (orchestrate-math-v2 only). Folded into the report name.
#                     Ignored, with a warning, for skills that take no --llm.
#   --tag TAG         Extra tag appended to the report name and stamped into
#                     the report as "tag" (default: untagged)
#   --force           Re-verify papers that already have a math report
#   --keep-lean       Keep each paper's Lean project instead of deleting it
#   --dry-run         Print what would be done without actually verifying
#   --list            List the papers that would be verified
#   --help            Show this help message
#
# Prerequisites:
#   - Claude Code CLI installed and authenticated (claude login)
#   - Run from the project root so .claude/skills/orchestrate-math is loaded
#   - Papers prepared into the workspace:
#       python scripts/evaluate_verifier.py --llm <llm> --prepare   (PaperAudit)
#       python scripts/download_spot.py                             (SPOT)
#   - Lean toolchain + lean-lsp MCP for the formal side (docs/math-verification.md)
#
# Examples:
#   # Math-verify the first 5 PaperAudit papers
#   ./scripts/batch-verify-math.sh --papers 5
#
#   # Math-verify the whole SPOT set
#   ./scripts/batch-verify-math.sh --dataset spot
#
#   # See what would run
#   ./scripts/batch-verify-math.sh --dataset spot --dry-run
#
#   # Two models, side by side, then compare
#   ./scripts/batch-verify-math.sh --model claude-opus-5 --tag opus5
#   ./scripts/batch-verify-math.sh --model claude-sonnet-4-5-20250929 --tag sonnet45
#   python scripts/evaluate_verifier.py --llm <llm> --evaluate --task math --tag opus5
#   python scripts/evaluate_verifier.py --llm <llm> --evaluate --task math --tag sonnet45
#
#   # v1 vs v2 on the same papers — different report names, no overwrite
#   ./scripts/batch-verify-math.sh --skill orchestrate-math
#   ./scripts/batch-verify-math.sh --skill orchestrate-math-v2
#   python scripts/evaluate_verifier.py --llm <llm> --evaluate --task math
#   python scripts/evaluate_verifier.py --llm <llm> --evaluate --task math --skill orchestrate-math-v2
#
#   # v2 with its baseline arm on an external model
#   ./scripts/batch-verify-math.sh --skill orchestrate-math-v2 --llm-skill gpt-5-6-terra
#   python scripts/evaluate_verifier.py --llm <llm> --evaluate --task math \
#       --skill orchestrate-math-v2 --llm-skill gpt-5-6-terra
# ---------------------------------------------------------------------------

set -euo pipefail

# ---- Resolve project root ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---- Default configuration ----
DATASET="paperaudit"
WORKSPACE_DIR=""
MAX_PAPERS=""
OFFSET=0
PAPER_ID=""
PAPER_LIST=""
MODEL="claude-sonnet-4-5-20250929"
SKILL="orchestrate-math"
DEFAULT_SKILL="orchestrate-math"
LLM_SKILL=""
# Skills that accept their own --llm flag. Anything else gets --llm-skill
# ignored (with a warning) rather than silently renaming an identical report.
SKILLS_WITH_LLM=("orchestrate-math-v2" "orchestrate-math-v3")
TAG=""
FORCE=false
KEEP_LEAN=false
DRY_RUN=false
LIST_ONLY=false

# ---- Parse arguments ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            DATASET="$2"
            if [[ "$DATASET" != "paperaudit" && "$DATASET" != "spot" ]]; then
                echo "ERROR: --dataset must be 'paperaudit' or 'spot', got: $DATASET"
                exit 1
            fi
            shift 2
            ;;
        --workspace)
            WORKSPACE_DIR="$2"; shift 2 ;;
        --papers)
            MAX_PAPERS="$2"; shift 2 ;;
        --offset)
            OFFSET="$2"; shift 2 ;;
        --paper-id)
            PAPER_ID="$2"; shift 2 ;;
        --paper-list)
            PAPER_LIST="$2"
            if [[ ! -f "$PAPER_LIST" ]]; then
                echo "ERROR: --paper-list file not found: $PAPER_LIST"
                exit 1
            fi
            shift 2 ;;
        --model)
            MODEL="$2"; shift 2 ;;
        --skill)
            SKILL="$2"
            if [[ ! -f "$PROJECT_ROOT/.claude/skills/$SKILL/SKILL.md" ]]; then
                echo "ERROR: no such skill: .claude/skills/$SKILL/SKILL.md"
                echo "Available:"
                for s in "$PROJECT_ROOT"/.claude/skills/*/SKILL.md; do
                    echo "  - $(basename "$(dirname "$s")")"
                done
                exit 1
            fi
            shift 2 ;;
        --llm-skill)
            LLM_SKILL="$2"
            if [[ ! "$LLM_SKILL" =~ ^[A-Za-z0-9._:-]+$ ]]; then
                echo "ERROR: --llm-skill must be a model name ([A-Za-z0-9._:-]), got: $LLM_SKILL"
                exit 1
            fi
            shift 2 ;;
        --tag)
            TAG="$2"
            if [[ ! "$TAG" =~ ^[A-Za-z0-9._-]+$ ]]; then
                echo "ERROR: --tag must be filename-safe ([A-Za-z0-9._-]), got: $TAG"
                exit 1
            fi
            shift 2 ;;
        --force)
            FORCE=true; shift ;;
        --keep-lean)
            KEEP_LEAN=true; shift ;;
        --dry-run)
            DRY_RUN=true; shift ;;
        --list)
            LIST_ONLY=true; shift ;;
        --help|-h)
            # Print the leading comment block, stopping at the first
            # non-comment line — no hard-coded line count to drift.
            awk 'NR>1 && /^#/ {sub(/^#[ ]?/, ""); print; next} NR>1 {exit}' "$0"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# ---- Check prerequisites ----
if ! command -v claude &> /dev/null; then
    echo "ERROR: Claude Code CLI ('claude') not found in PATH."
    echo "Install it from: https://claude.ai/code"
    exit 1
fi

if [[ -z "$WORKSPACE_DIR" ]]; then
    if [[ "$DATASET" == "spot" ]]; then
        WORKSPACE_DIR="$PROJECT_ROOT/data/spot-workspace"
    else
        WORKSPACE_DIR="$PROJECT_ROOT/data/paperaudit-workspace"
    fi
fi

if [[ ! -d "$WORKSPACE_DIR" ]]; then
    echo "ERROR: Workspace directory not found: $WORKSPACE_DIR"
    echo "Prepare the papers first:"
    echo "  PaperAudit: python scripts/evaluate_verifier.py --llm <llm> --prepare"
    echo "  SPOT:       python scripts/download_spot.py"
    exit 1
fi

# --llm-skill only means something for skills that have their own --llm. For any
# other skill it would rename an otherwise identical report, so drop it here
# rather than let it reach the prompt or the filename.
if [[ -n "$LLM_SKILL" ]]; then
    SUPPORTED=false
    for s in "${SKILLS_WITH_LLM[@]}"; do
        [[ "$s" == "$SKILL" ]] && SUPPORTED=true && break
    done
    if [[ "$SUPPORTED" == false ]]; then
        echo "WARNING: --llm-skill '$LLM_SKILL' ignored: skill '$SKILL' has no --llm flag." >&2
        echo "         (Skills that accept one: ${SKILLS_WITH_LLM[*]})" >&2
        LLM_SKILL=""
    fi
fi

# The skill derives its report name from the paper filename:
# paper.md -> paper_verified_math.json, written in the paper's directory.
# The skill, its --llm-skill and any --tag are folded into one tag segment so
# runs never overwrite each other. compose_task_tag() is the SAME function
# evaluate_verifier.py uses to FIND the report — calling it here rather than
# rebuilding the rule in bash is what guarantees the two agree. The default
# skill contributes no segment, so existing untagged reports keep their path.
SKILL_SEGMENT="$SKILL"
[[ "$SKILL" == "$DEFAULT_SKILL" ]] && SKILL_SEGMENT=""

EFFECTIVE_TAG="$(python3 - "$PROJECT_ROOT" "$SKILL_SEGMENT" "$LLM_SKILL" "$TAG" <<'PY'
import sys
sys.path.insert(0, f"{sys.argv[1]}/scripts")
from evaluate_verifier import compose_task_tag

skill, llm_skill, tag = (a or None for a in sys.argv[2:5])
try:
    print(compose_task_tag(tag, skill, llm_skill) or "")
except ValueError as e:
    print(f"__ERROR__{e}")
PY
)" || { echo "ERROR: could not compute the report tag (is python3 on PATH?)"; exit 1; }

if [[ "$EFFECTIVE_TAG" == __ERROR__* ]]; then
    echo "ERROR: ${EFFECTIVE_TAG#__ERROR__}"
    exit 1
fi

if [[ -n "$EFFECTIVE_TAG" ]]; then
    REPORT_BASENAME="paper_verified_math-${EFFECTIVE_TAG}.json"
else
    REPORT_BASENAME="paper_verified_math.json"
fi

# ---- Discover papers ----
# A paper is eligible when it has been prepared (paper.md + ground_truth.json)
# and, unless --force, has no math report yet.
CANDIDATE_IDS=()
if [[ -n "$PAPER_LIST" ]]; then
    # The file's order is the run order; duplicates are dropped on first sight
    # so a hand-edited list cannot make a paper run twice. The seen-set is a
    # newline-delimited string, not an associative array: macOS ships bash 3.2,
    # where `declare -A` does not exist and the script would die on line 1 of
    # the list.
    SEEN=$'\n'
    while IFS= read -r line || [[ -n "$line" ]]; do
        line="${line%%#*}"
        # Trim in pure bash, NOT via xargs: these directory names contain
        # apostrophes ("A_form_of_refined_Roth's_theorem") and dollar signs
        # ("...$\mathbb{Z}_p$-extensions"). xargs treats a quote as a quoting
        # character, dies on the unmatched one, and `|| true` turns that into an
        # empty line -- so the paper is silently skipped. Two of SPOT's 37 went
        # missing exactly this way.
        line="${line#"${line%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"
        [[ -z "$line" ]] && continue
        case "$SEEN" in
            *$'\n'"$line"$'\n'*) continue ;;
        esac
        SEEN="${SEEN}${line}"$'\n'
        if [[ ! -f "$WORKSPACE_DIR/$line/paper.md" ]]; then
            echo "ERROR: --paper-list entry has no paper.md: $WORKSPACE_DIR/$line/paper.md"
            exit 1
        fi
        CANDIDATE_IDS+=("$line")
    done < "$PAPER_LIST"
    if [[ ${#CANDIDATE_IDS[@]} -eq 0 ]]; then
        echo "ERROR: --paper-list $PAPER_LIST listed no papers"
        exit 1
    fi
else
    for d in "$WORKSPACE_DIR"/*/; do
        [[ -d "$d" ]] || continue
        CANDIDATE_IDS+=("$(basename "$d")")
    done
    # Deterministic ordering so --offset/--papers slice reproducibly
    if [[ ${#CANDIDATE_IDS[@]} -gt 0 ]]; then
        IFS=$'\n' CANDIDATE_IDS=($(sort <<<"${CANDIDATE_IDS[*]}"))
        unset IFS
    fi
fi

ALL_PAPERS=()
for paper_id in "${CANDIDATE_IDS[@]}"; do
    d="$WORKSPACE_DIR/$paper_id"
    [[ -d "$d" ]] || continue

    [[ -f "$d/paper.md" ]] || continue
    [[ -f "$d/ground_truth.json" ]] || continue

    if [[ "$FORCE" == false && -f "$d/$REPORT_BASENAME" ]]; then
        continue
    fi

    ALL_PAPERS+=("$paper_id")
done

# ---- Single paper mode ----
if [[ -n "$PAPER_ID" ]]; then
    if [[ ! -f "$WORKSPACE_DIR/$PAPER_ID/paper.md" ]]; then
        echo "ERROR: Paper not found: $WORKSPACE_DIR/$PAPER_ID/paper.md"
        exit 1
    fi
    PAPERS=("$PAPER_ID")
else
    # Apply offset + limit
    TOTAL_AVAILABLE=${#ALL_PAPERS[@]}
    if [[ "$TOTAL_AVAILABLE" -eq 0 ]]; then
        echo "No papers to math-verify in $WORKSPACE_DIR."
        if [[ "$FORCE" == false ]]; then
            echo "(Papers with an existing $REPORT_BASENAME are skipped — use --force to redo them.)"
        fi
        exit 0
    fi
    PAPERS=("${ALL_PAPERS[@]:$OFFSET}")
    if [[ -n "$MAX_PAPERS" && "$MAX_PAPERS" -lt ${#PAPERS[@]} ]]; then
        PAPERS=("${PAPERS[@]:0:$MAX_PAPERS}")
    fi
fi

TOTAL=${#PAPERS[@]}

# ---- List mode ----
if $LIST_ONLY; then
    echo "📚 Papers to math-verify ($DATASET):"
    echo "====================================="
    for i in "${!PAPERS[@]}"; do
        echo "  $((i + 1)). ${PAPERS[$i]}"
    done
    echo ""
    echo "Total: $TOTAL papers"
    echo "Workspace: $WORKSPACE_DIR"
    exit 0
fi

# ---- Dry-run mode ----
if $DRY_RUN; then
    echo "🔍 DRY RUN — would math-verify the following papers:"
    echo "===================================================="
    for i in "${!PAPERS[@]}"; do
        paper_id="${PAPERS[$i]}"
        echo "  $((i + 1)). $paper_id"
        echo "     Paper:  $WORKSPACE_DIR/$paper_id/paper.md"
        echo "     Report: $WORKSPACE_DIR/$paper_id/$REPORT_BASENAME"
    done
    echo ""
    echo "Total: $TOTAL papers"
    echo "Model: $MODEL"
    echo "Skill: /$SKILL"
    echo "Skill LLM: ${LLM_SKILL:-<skill default>}"
    echo "Tag:   ${EFFECTIVE_TAG:-<none>}"
    exit 0
fi

# ---- Batch mode ----
echo "🚀 Batch Mathematical Verification (/$SKILL)"
echo "======================================================"
echo "  Dataset:    $DATASET"
echo "  Workspace:  $WORKSPACE_DIR"
echo "  Model:      $MODEL"
echo "  Skill:      /$SKILL"
echo "  Skill LLM:  ${LLM_SKILL:-<skill default>}"
echo "  Tag:        ${EFFECTIVE_TAG:-<none>}"
echo "  Papers:     $TOTAL"
echo "  Keep Lean:  $KEEP_LEAN"
echo ""

SUCCESS=0
FAILED=0
START_TIME=$(date +%s)
declare -a FAILED_PAPERS=()

for i in "${!PAPERS[@]}"; do
    paper_id="${PAPERS[$i]}"
    NUM=$((i + 1))

    PAPER_PATH="$WORKSPACE_DIR/$paper_id/paper.md"
    REPORT_PATH="$WORKSPACE_DIR/$paper_id/$REPORT_BASENAME"
    LOG_PATH="$WORKSPACE_DIR/$paper_id/verify-math.log"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[$NUM/$TOTAL] Math-verifying: $paper_id"
    echo "  Paper:  $PAPER_PATH"
    echo "  Report: $REPORT_PATH"
    echo ""

    TAG_NOTE=""
    if [[ -n "$EFFECTIVE_TAG" ]]; then
        TAG_NOTE="
This run is tagged '${EFFECTIVE_TAG}'. Add a top-level \"tag\": \"${EFFECTIVE_TAG}\" field to
the report so it can be told apart from other runs over the same paper."
    fi

    LLM_NOTE=""
    if [[ -n "$LLM_SKILL" ]]; then
        LLM_NOTE="
Pass --llm ${LLM_SKILL} to the skill: its baseline arm must run through
scripts/run_baseline.py against that model, not inline. If that model's API key
is missing, stop and report it — do not fall back to running the arm yourself."
    fi

    KEEP_LEAN_NOTE=""
    if [[ "$KEEP_LEAN" == true ]]; then
        KEEP_LEAN_NOTE="
Pass --keep-lean to every /verify-math-lean call, including the last one, and set \"lean_project_kept\": true in the report. Do not delete the Lean project."
    fi

    read -r -d '' PROMPT <<EOF || true
Run the ${SKILL} skill (Skill tool with skill="${SKILL}") on the paper below.

Paper path: ${PAPER_PATH}

Follow the skill exactly, including its own step order and its rules for the
Lean project lifecycle, and write the report to:

  ${REPORT_PATH}

The report must follow THE SKILL'S OWN JSON schema, whatever it specifies —
paper_name, paper_path, totals, results, findings, verdict, verdict_confidence
and summary at minimum. The skills differ on what a results[] entry is and on
the confidence at which a result is promoted to findings[]; follow the schema in
the skill you are running and do not substitute another skill's rule.
Report a Lean FAILED as an unsettled claim, never as a finding.${LLM_NOTE}${TAG_NOTE}${KEEP_LEAN_NOTE}

THE RUN IS NOT COMPLETE UNTIL THAT JSON FILE EXISTS. A verifier that fails is a
normal, reportable outcome, never a reason to stop: record it (Lean FAILED /
NOT_ATTEMPTED, or an UNVERIFIABLE snippet) and carry on to the next snippet. If
a sub-skill returns a summary saying it gave up, that is ITS verdict for ONE
snippet, not the end of your run -- resume the skill's step order and finish.
Even if every verifier fails, still write the report, with the failures recorded
and the verdict set to INCONCLUSIVE.

Write the complete JSON report to the exact path above with the Write tool. Do
not print the report inline.
EOF

    set +e
    claude -p \
        --model "$MODEL" \
        --dangerously-skip-permissions \
        --output-format text \
        --verbose \
        "$PROMPT" \
        > "$LOG_PATH" 2>&1
    EXIT_CODE=$?
    set -e

    # The report path is passed as argv[1], NOT interpolated into the source:
    # a paper title containing an apostrophe (e.g. "Roth's theorem") would
    # otherwise close the Python string literal and the validator would die
    # with a SyntaxError, reporting a perfectly good report as invalid.
    if [[ $EXIT_CODE -eq 0 && -f "$REPORT_PATH" ]] && python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    r = json.load(f)
assert isinstance(r.get('findings'), list), 'missing findings[]'
assert isinstance(r.get('results'), list), 'missing results[]'
assert r.get('verdict'), 'missing verdict'
tag = r.get('tag')
print(f\"  ✓ OK: {len(r['results'])} snippets checked, \"
      f\"{len(r['findings'])} findings, verdict={r['verdict']}\"
      + (f\", tag={tag}\" if tag else ''))
" "$REPORT_PATH" 2>/dev/null; then
        SUCCESS=$((SUCCESS + 1))
    else
        FAILED=$((FAILED + 1))
        FAILED_PAPERS+=("$paper_id")
        if [[ $EXIT_CODE -ne 0 ]]; then
            echo "  ✗ orchestrate-math failed (exit code: $EXIT_CODE)"
        elif [[ ! -f "$REPORT_PATH" ]]; then
            echo "  ✗ No report written to $REPORT_PATH"
        else
            echo "  ✗ Report exists but is not a valid math report"
        fi
        echo "  Log saved to: $LOG_PATH"
    fi

    echo ""

    # Sleep briefly between papers to avoid rate limiting
    sleep 2
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "Batch mathematical verification complete!"
echo "  Total papers:  $TOTAL"
echo "  Verified:      $SUCCESS"
echo "  Failed:        $FAILED"
echo "  Duration:      ${DURATION}s ($(( DURATION / 60 ))m $(( DURATION % 60 ))s)"
echo "  Reports:       $WORKSPACE_DIR/<paper_id>/$REPORT_BASENAME"
echo "═══════════════════════════════════════════════════════════════════"

if [[ ${#FAILED_PAPERS[@]} -gt 0 ]]; then
    echo ""
    echo "Failed papers:"
    for p in "${FAILED_PAPERS[@]}"; do
        echo "  - $p"
    done
fi

if [[ "$SUCCESS" -gt 0 ]]; then
    echo ""
    echo "Next step: score only the mathematical errors:"
    echo "  python scripts/evaluate_verifier.py --llm <llm> --evaluate --task math \\"
    echo "      --workspace-dir $WORKSPACE_DIR${SKILL_SEGMENT:+ --skill $SKILL}${LLM_SKILL:+ --llm-skill $LLM_SKILL}${TAG:+ --tag $TAG}"
    echo "Results are written to: $PROJECT_ROOT/task_specific_output/math/"
fi
