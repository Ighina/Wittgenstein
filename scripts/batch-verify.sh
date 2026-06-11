#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# batch-verify.sh — Programmatically verify papers from the parquet dataset
# using Claude Code with the /verify-paper skill.
#
# Usage:
#   ./scripts/batch-verify.sh [OPTIONS]
#
# Options:
#   --papers N        Number of papers to verify (default: all)
#   --offset N        Skip first N papers (default: 0)
#   --output DIR      Output directory for results (default: outputs/claude-batch)
#   --mode MODE       Orchestration mode: exhaustive|uncertainty (default: exhaustive)
#   --threshold FLOAT Uncertainty threshold for uncertainty mode (default: 0.30)
#   --dry-run         Print what would be done without actually verifying
#   --paper-id ID     Verify a single paper by DOI/arXiv ID
#   --list            List available papers in the dataset
#   --help            Show this help message
#
# Prerequisites:
#   - Claude Code CLI must be installed and configured
#   - The Paperena MCP server must be configured in .claude/settings.json
#   - Python 3.11+ with pandas and pyarrow must be available
#
# Examples:
#   # Verify first 5 papers
#   ./scripts/batch-verify.sh --papers 5
#
#   # Verify a specific paper
#   ./scripts/batch-verify.sh --paper-id 10.1038/s41586-020-2649-2
#
#   # List available papers
#   ./scripts/batch-verify.sh --list
#
#   # Dry run — see what papers would be verified
#   ./scripts/batch-verify.sh --papers 3 --dry-run
# ---------------------------------------------------------------------------

set -euo pipefail

# ---- Resolve project root ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---- Default configuration ----
MAX_PAPERS=""
OFFSET=0
OUTPUT_DIR="$PROJECT_ROOT/outputs/claude-batch"
MODE="exhaustive"
THRESHOLD="0.30"
DRY_RUN=false
PAPER_ID=""
LIST_ONLY=false
PARQUET_PATH="$PROJECT_ROOT/data/train-00000-of-00001.parquet"

# ---- Parse arguments ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --papers)
            MAX_PAPERS="$2"
            shift 2
            ;;
        --offset)
            OFFSET="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --paper-id)
            PAPER_ID="$2"
            shift 2
            ;;
        --list)
            LIST_ONLY=true
            shift
            ;;
        --help)
            head -40 "$0" | grep '^#' | sed 's/^# \?//'
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

if [ ! -f "$PARQUET_PATH" ]; then
    echo "ERROR: Parquet file not found: $PARQUET_PATH"
    echo "Download the dataset first. See README.md for instructions."
    exit 1
fi

# ---- List papers mode ----
if $LIST_ONLY; then
    echo "📚 Papers in dataset:"
    echo "====================="
    python3 -c "
import pandas as pd
df = pd.read_parquet('$PARQUET_PATH')
print(f'Total papers: {len(df)}')
print(f'Columns: {list(df.columns)}')
print()
for i, (_, row) in enumerate(df.head(20).iterrows()):
    paper_id = str(row['doi/arxiv_id'])
    title = str(row.get('title', 'N/A'))[:120]
    cat = str(row.get('paper_category', 'N/A'))
    err_cat = str(row.get('error_category', 'N/A'))
    print(f'{i+1:3d}. [{cat}] {paper_id}')
    print(f'     Title: {title}')
    print(f'     Error: {err_cat}')
    print()
"
    exit 0
fi

# ---- Python helper to list paper IDs ----
get_paper_ids() {
    python3 -c "
import pandas as pd
df = pd.read_parquet('$PARQUET_PATH')
offset = $OFFSET
limit = ${MAX_PAPERS:-len(df)}
subset = df.iloc[offset:offset + limit]
for _, row in subset.iterrows():
    paper_id = str(row['doi/arxiv_id'])
    title = str(row.get('title', 'N/A'))[:100]
    cat = str(row.get('paper_category', 'N/A'))
    print(f'{paper_id}|{title}|{cat}')
"
}

# ---- Dry-run mode ----
if $DRY_RUN; then
    echo "🔍 DRY RUN — would verify the following papers:"
    echo "==============================================="
    count=0
    while IFS='|' read -r paper_id title cat; do
        count=$((count + 1))
        echo "  $count. [$cat] $paper_id"
        echo "     $title"
    done < <(get_paper_ids)
    echo ""
    echo "Total: $count papers"
    echo "Mode: $MODE"
    echo "Output: $OUTPUT_DIR"
    exit 0
fi

# ---- Single paper mode ----
if [ -n "$PAPER_ID" ]; then
    echo "🔬 Verifying single paper: $PAPER_ID"
    echo "==============================================="

    # Get paper details
    PAPER_INFO=$(python3 -c "
import pandas as pd
df = pd.read_parquet('$PARQUET_PATH')
row = df[df['doi/arxiv_id'].astype(str) == '$PAPER_ID']
if len(row) == 0:
    print('NOT_FOUND')
else:
    r = row.iloc[0]
    print(f\"{r['title']}|{r['paper_category']}|{r.get('error_category','')}|{r.get('error_location','')}\")
")

    if [ "$PAPER_INFO" = "NOT_FOUND" ]; then
        echo "ERROR: Paper not found: $PAPER_ID"
        exit 1
    fi

    IFS='|' read -r TITLE CATEGORY GT_ERROR GT_LOCATION <<< "$PAPER_INFO"

    echo ""
    echo "  Title:     $TITLE"
    echo "  Category:  $CATEGORY"
    echo "  GT Error:  $GT_ERROR"
    echo "  GT Loc:    $GT_LOCATION"
    echo ""

    # Build the prompt for Claude Code
    PROMPT="/verify-paper
Verify the paper with ID '$PAPER_ID' from the dataset at '$PARQUET_PATH'.
Use orchestration mode: $MODE.
Use the segment_paper MCP tool to parse and segment the paper.
Then route each snippet to the appropriate specialist verifier skill.
Finally, aggregate all findings and produce a verification report with the verdict.
Include comparison with the ground truth annotation."

    mkdir -p "$OUTPUT_DIR"

    if $DRY_RUN; then
        echo "Would run: claude -p \"$PROMPT\""
    else
        echo "Running verification..."
        claude -p "$PROMPT" --print 2>&1 | tee "$OUTPUT_DIR/${PAPER_ID//\//_}.md"
        echo ""
        echo "Result saved to: $OUTPUT_DIR/${PAPER_ID//\//_}.md"
    fi

    exit 0
fi

# ---- Batch mode ----
echo "🚀 Batch Paper Verification"
echo "============================"
echo "  Dataset:    $PARQUET_PATH"
echo "  Mode:       $MODE"
echo "  Threshold:  $THRESHOLD"
echo "  Output:     $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"

# Collect paper list
PAPERS=()
while IFS='|' read -r paper_id title cat; do
    PAPERS+=("$paper_id|$title|$cat")
done < <(get_paper_ids)

TOTAL=${#PAPERS[@]}
echo "Papers to verify: $TOTAL"
echo ""

RESULTS_JSON="$OUTPUT_DIR/batch_results.json"
echo '{"results": []}' > "$RESULTS_JSON"

VERIFIED=0
FAILED=0
START_TIME=$(date +%s)

for i in "${!PAPERS[@]}"; do
    IFS='|' read -r paper_id title cat <<< "${PAPERS[$i]}"
    NUM=$((i + 1))

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[$NUM/$TOTAL] Verifying: $paper_id"
    echo "  Title: $title"
    echo "  Category: $cat"
    echo ""

    PROMPT="/verify-paper
Verify the paper with ID '$paper_id' from the dataset at '$PARQUET_PATH'.
Use orchestration mode: $MODE.
If using uncertainty mode, use threshold: $THRESHOLD.
Use the segment_paper MCP tool to parse and segment the paper.
Then route each snippet to the appropriate specialist verifier skill.
Finally, aggregate all findings and produce a verification report with the verdict."

    OUTPUT_FILE="$OUTPUT_DIR/${paper_id//\//_}.md"

    if claude -p "$PROMPT" --print > "$OUTPUT_FILE" 2>&1; then
        VERIFIED=$((VERIFIED + 1))
        echo "  ✅ Verification complete → $OUTPUT_FILE"
    else
        FAILED=$((FAILED + 1))
        echo "  ❌ Verification failed for $paper_id"
        echo "# Verification failed for $paper_id" > "$OUTPUT_FILE"
    fi

    echo ""

    # Sleep briefly between papers to avoid rate limiting
    sleep 2
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "Batch verification complete!"
echo "  Total papers:  $TOTAL"
echo "  Verified:      $VERIFIED"
echo "  Failed:        $FAILED"
echo "  Duration:      ${DURATION}s ($(( DURATION / 60 ))m $(( DURATION % 60 ))s)"
echo "  Results:       $OUTPUT_DIR"
echo "═══════════════════════════════════════════════════════════════════"

# Aggregate summary
echo ""
echo "Individual results are in: $OUTPUT_DIR/"
echo "To evaluate against ground truth, use the existing Python evaluation:"
echo "  python3 main.py evaluate --predictions outputs/claude-batch/..."
