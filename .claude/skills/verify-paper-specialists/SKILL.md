---
name: verify-paper-specialists
description: Verify an entire scientific paper for errors using specialist fine-tuned models instead of generic LLM verifiers. Parses, segments, routes to specialist models via MCP tools, and aggregates findings.
---

# Verify Paper (Specialist Edition)

You are the orchestrator for the **specialist-model** variant of the Paperena scientific paper verification pipeline. Your job is to verify an entire paper end-to-end: parse it, segment it into verification snippets, route each snippet to the appropriate **specialist fine-tuned model** via MCP tools, and aggregate the findings into a comprehensive report.

## Key Difference from `/verify-paper`

The standard `/verify-paper` skill invokes other Claude Code skills (like `/verify-math`, `/verify-text`, etc.), which use a **generic large LLM** with a specialised prompt. **This skill** instead calls **specialist fine-tuned models** through the Paperena MCP server — smaller models trained specifically for math verification, text consistency, citation checking, etc.

| Aspect | `/verify-paper` (standard) | `/verify-paper-specialists` (this skill) |
|---|---|---|
| Math verification | Skill → generic LLM + SymPy sandbox | MCP `run_specialist("math", ...)` → fine-tuned math model |
| Text verification | Skill → generic LLM with text prompt | MCP `run_specialist("text", ...)` → fine-tuned text model |
| Citation verification | Skill → generic LLM + web search | MCP `run_specialist("citation", ...)` → fine-tuned citation model |
| Vision verification | Skill → generic multimodal LLM | MCP `run_specialist("vision", ...)` → fine-tuned vision model |
| Statistical verification | Skill → generic LLM | MCP `run_specialist("statistical", ...)` → fine-tuned stats model |
| Cross-section consistency | Skill → generic LLM | MCP `run_specialist("cross-section-consistency", ...)` |
| Triage | Skill → generic LLM | MCP `run_triage_specialist(...)` → fine-tuned triage model |

## When to Use This Skill

Invoke this skill to verify a complete scientific paper using specialist models. The paper can be:
1. **Markdown file**: A Markdown file input including sections, tables, equations, theorems, etc.
2. **LaTeX input**: A .tex file to be read including same as above.

Use this skill when you want higher accuracy from task-specific models or want to benchmark specialist vs. generalist performance.

## Available MCP Tools

This skill relies on the following MCP tools exposed by the Paperena MCP server:

| Tool | Purpose |
|---|---|
| `run_specialist(specialist_type, text, snippet_id, location, snippet_type, model?, metadata?)` | Invoke any specialist verifier |
| `run_triage_specialist(text, snippet_id, location, snippet_type, model?)` | Run the triage specialist (convenience wrapper) |
| `list_available_specialists()` | List all registered specialists and their default models |
| `segment_paper(paper_id, title, paper_category, paper_content)` | Segment a parsed paper into verification snippets |
| `parse_paper(paper_id, title, paper_category, paper_content)` | Parse raw paper content into structured form |

## Orchestration Modes

### Exhaustive Mode (default)
Every snippet is routed to a specialist model by type:

| Snippet Type | Specialist | MCP Call |
|---|---|---|
| `EQUATION` | `math` | `run_specialist("math", ...)` |
| `FIGURE` | `vision` | `run_specialist("vision", ...)` |
| `TABLE` | `vision` | `run_specialist("vision", ...)` |
| `SECTION`, `SUBSECTION`, `PARAGRAPH` | `text` | `run_specialist("text", ...)` |
| `THEOREM`, `LEMMA`, `PROPOSITION`, `COROLLARY` | `text` | `run_specialist("text", ...)` |
| `ALGORITHM`, `APPENDIX` | `text` | `run_specialist("text", ...)` |

Additionally, for any text snippet:
- If it contains citations/references → also invoke `run_specialist("citation", ...)`
- If it contains significant numeric content → also invoke `run_specialist("statistical", ...)`

### Uncertainty Mode
1. First, run `run_triage_specialist(...)` on EVERY snippet to produce an uncertainty map.
2. Select only snippets with uncertainty ≥ threshold (default 0.30).
3. Route those high-uncertainty snippets to the appropriate specialist model based on the triage `route` field:
   - `math` → `run_specialist("math", ...)`
   - `proof` → `run_specialist("text", ...)`
   - `statistics` → `run_specialist("statistical", ...)`
   - `citation` → `run_specialist("citation", ...)`
   - `vision` → `run_specialist("vision", ...)`
   - `text` → `run_specialist("text", ...)`
   - `none` → skip specialist verification
4. Low-uncertainty snippets are recorded as NO_ERROR with confidence = 1.0 - uncertainty.

## Pipeline Steps

### Step 1: Obtain the paper

**From Markdown input:**
Use the provided markdown file location and read it to pass it as-is to the next step.

**From LaTeX file:**
Use the provided path and read the .tex file. Convert the LaTeX to markdown first, then pass to the next step.

### Step 2: Parse and segment the paper

First, use the MCP tool `parse_paper` to get the structured paper representation. Then use `segment_paper` to break it into verification snippets.

If the paper is not from the dataset (plain markdown/text input), invoke the `/segment_paper` skill to segment it into typed snippets.

The output is snippets grouped by type: sections, equations, figures, tables, theorems.

### Step 3: Route snippets to specialists (Exhaustive mode)

For each snippet, call the appropriate MCP specialist tool. Here is the precise mapping:

**Equations → Math Specialist:**
```
mcp__paperena__run_specialist(
    specialist_type="math",
    text=<equation_latex>,
    snippet_id=<snippet_id>,
    location=<location>,
    snippet_type="EQUATION"
)
```

**Figures/Tables → Vision Specialist:**
```
mcp__paperena__run_specialist(
    specialist_type="vision",
    text=<caption_and_description>,
    snippet_id=<snippet_id>,
    location=<location>,
    snippet_type=<"FIGURE"|"TABLE">,
    metadata={"caption": <caption>, "content_type": <"figure"|"table">}
)
```

**Text snippets (Sections, Theorems, Lemmas, Algorithms, etc.) → Text Specialist:**
```
mcp__paperena__run_specialist(
    specialist_type="text",
    text=<snippet_content>,
    snippet_id=<snippet_id>,
    location=<location>,
    snippet_type=<snippet_type>
)
```

**Citation-heavy text → Citation Specialist (additional):**
```
mcp__paperena__run_specialist(
    specialist_type="citation",
    text=<snippet_content>,
    snippet_id=<snippet_id>,
    location=<location>,
    snippet_type=<snippet_type>
)
```

**Numeric/statistical text → Statistical Specialist (additional):**
```
mcp__paperena__run_specialist(
    specialist_type="statistical",
    text=<snippet_content>,
    snippet_id=<snippet_id>,
    location=<location>,
    snippet_type=<snippet_type>
)
```

**Cross-section consistency → Cross-Section Specialist:**
After all individual snippets are verified, concatenate all section text and call:
```
mcp__paperena__run_specialist(
    specialist_type="cross-section-consistency",
    text=<all_sections_concatenated>,
    snippet_id=<paper_id>::FULL_PAPER::0,
    location="Full paper",
    snippet_type="FULL_PAPER"
)
```

### Step 3 (Alternative): Uncertainty mode

1. For EVERY snippet, call `run_triage_specialist`:
   ```
   mcp__paperena__run_triage_specialist(
       text=<snippet_content>,
       snippet_id=<snippet_id>,
       location=<location>,
       snippet_type=<snippet_type>
   )
   ```
2. Build the uncertainty map: for each snippet, record its uncertainty score and suggested route.
3. Filter snippets: keep only those with uncertainty ≥ the configured threshold (default 0.30).
4. For selected snippets, invoke the specialist suggested by the triage route using `run_specialist`.
5. For unselected snippets, record them as NO_ERROR with confidence = 1.0 - uncertainty.

### Step 4: Aggregate findings

After all specialist invocations return:

1. Collect all verification results.
2. Filter to only findings where `error_detected = true` AND `confidence ≥ threshold`:
   - Math/statistical: 0.7
   - Text/citation: 0.8
   - Vision: 0.75
   - Cross-section: 0.85
3. Sort predicted errors by confidence (highest first).
4. Produce the final paper-level report.

### Step 5: Generate the report

Output a comprehensive verification report:

```json
{
  "paper_id": "<id>",
  "title": "<title>",
  "paper_category": "<category>",
  "mode": "specialists",
  "predicted_errors": [
    {
      "error_category": "Equation / proof",
      "error_location": "Equation 7",
      "confidence": 0.90,
      "supporting_evidence": "The equation simplifies to a non-zero residual...",
      "verifier_name": "math",
      "snippet_id": "<paper_id>_eq_6"
    }
  ],
  "statistics": {
    "total_snippets": 45,
    "snippets_verified": 45,
    "errors_detected": 1,
    "specialist_usage": {
      "math": 12,
      "vision": 6,
      "text": 25,
      "statistical": 2,
      "citation": 8,
      "cross-section-consistency": 1,
      "triage": 0
    }
  },
  "verdict": "PASS | NEEDS_REVIEW | NEEDS_CORRECTION",
  "summary": "Human-readable summary of findings...",
  "uncertainty_map": [...]
}
```

## Verdict Guidelines

- **PASS**: No errors detected, or only low-confidence (<0.7) findings.
- **NEEDS_REVIEW**: One finding with medium confidence (0.7-0.85).
- **NEEDS_CORRECTION**: One or more findings with high confidence (≥0.85).

## Configuration Reference

| Parameter | Default | Description |
|---|---|---|
| `strictness` | `strict` | Error sensitivity: `strict` (only erratum-worthy) or `lenient` (broader) |
| `mode` | `exhaustive` | Orchestration: `exhaustive` or `uncertainty` |
| `uncertainty_threshold` | `0.30` | In uncertainty mode, escalate snippets at/above this score |
| `model` | (specialist default) | Optional model override for all specialists |
| `math_model` | `ollama/deepseek-r1:latest` | Reasoning model (R1); chain-of-thought kept off the JSON channel |
| `text_model` | `hf/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | Local NLI cross-encoder for cross-sentence contradiction detection |
| `citation_model` | `ollama/qwen3:latest` | Internal attribution/novelty consistency (no web access → externally-dependent checks are UNVERIFIABLE) |
| `vision_model` | `ollama/granite3.2-vision:latest` | Multimodal; figures sent as images alongside captions |
| `statistical_model` | `statcheck/scipy` | **Deterministic** scipy engine (no LLM): statcheck decision errors, impossible bounds, CI checks |
| `cross_section_model` | `ollama/qwen3:latest` | Whole-paper reasoning at 16k context |
| `triage_model` | `ollama/qwen3:latest` | Uncertainty scoring + routing (thinking disabled) |

## Backend Requirements

The specialists are backed by **out-of-the-box models**, not proprietary
fine-tunes. Two backends are used:

1. **Ollama** (`math`, `citation`, `vision`, `cross-section-consistency`,
   `triage`) — a local Ollama server must be running (`ollama serve`) with the
   models pulled (`ollama pull deepseek-r1`, `qwen3`, `granite3.2-vision`).
   Any model name prefixed `ollama/` is routed to `http://localhost:11434`
   (override with the `OLLAMA_HOST` env var). Reasoning models keep their
   chain-of-thought in a separate channel; JSON mode is auto-enabled.
2. **HuggingFace transformers** (`text`) — the DeBERTa-v3 NLI model
   (prefix `hf/`) runs locally on Apple MPS / CUDA / CPU. Install the extra:
   `pip install -e '.[specialists]'`.

The `statistical` specialist needs **no model** — it is a deterministic scipy
statcheck reimplementation. To force any specialist onto a generic LLM prompt
instead of its native backend, pass an `ollama/...` model override (the `text`
and `statistical` specialists honor this and fall back to the prompt path).

## Specialist Model Architecture

Each specialist wraps the most suitable **out-of-the-box** model for its task (a local Ollama LLM, a HuggingFace NLI cross-encoder, or a deterministic scipy engine) behind a uniform interface. They are accessed through the MCP server which handles model loading and invocation. The specialist class hierarchy is:

```
Specialist (ABC)                          # src/verifiers/specialists.py
├── MathSpecialist                        # Equations, derivations, proof steps
├── TextSpecialist                        # Prose, theorems, lemmas, algorithms
├── CitationSpecialist                    # Attribution, novelty, references
├── VisionSpecialist                      # Figures and tables
├── StatisticalSpecialist                 # Test statistics, p-values, sample sizes
├── CrossSectionConsistencySpecialist     # Macro-level section alignment
└── TriageSpecialist                      # Uncertainty estimation & routing
```

All specialists return a uniform `SpecialistResult` with these common fields:
- `snippet_id` — unique identifier
- `specialist_type` — which specialist produced this result
- `location` — human-readable location of the error
- `status` — ERROR_DETECTED | NO_ERROR | UNVERIFIABLE | SKIPPED
- `error_detected` — boolean
- `confidence` — 0.0 to 1.0
- `reasoning` — detailed explanation
- `predicted_error_category` — error type or null
- `execution_time_ms` — latency measurement
- Plus specialist-specific extra fields (e.g., `contradiction_locations`, `sympy_code`, `uncertainty`, `route`)

## Example: Verifying a Paper End-to-End

```
1. Read paper markdown → paper_text
2. Call /segment_paper skill → snippets[]
3. For each snippet:
   a. Determine specialist_type from snippet_type
   b. Call mcp__paperena__run_specialist(specialist_type, text, snippet_id, location, ...)
   c. Collect SpecialistResult
4. Call mcp__paperena__run_specialist("cross-section-consistency", full_text, ...)
5. Aggregate, filter by confidence threshold, sort by confidence
6. Output the final JSON report
```

## Specialist-Specific Output Fields

In addition to the common fields, each specialist may return extra fields:

### Math Specialist extras
- `snippet_type`: "EQUATION"
- `contradiction_locations`: []

### Text Specialist extras
- `snippet_type`: the original snippet type
- `contradiction_locations`: list of `"location: “sentence A” ↔ “sentence B”"` strings for each flagged contradiction
- `n_sentence_pairs_checked`: number of sentence pairs the NLI model scored

### Citation Specialist extras
- `snippet_type`: the original snippet type

### Vision Specialist extras
- `content_type`: "figure" or "table"

### Statistical Specialist extras
- `snippet_type`: the original snippet type
- `contradiction_locations`: []

### Cross-Section Consistency Specialist extras
- `snippet_type`: "FULL_PAPER"
- `contradiction_locations`: list of section names that contradict each other

### Triage Specialist extras
- `uncertainty`: float [0, 1] — estimated error probability
- `route`: string — suggested specialist for escalation
- `reason`: string — one-sentence explanation
