# API Reference

Complete reference for all public modules, classes, and functions in the Paperena Verification pipeline.

---

## `src.models` — Data Models

All inter-layer communication uses Pydantic models defined here. Every model supports `.model_dump()` and `.model_dump_json()`.

### Dataset Exploration

#### `PaperContentSchemaReport`
Produced by `analyze_dataset_schema()`.

| Field | Type | Description |
|-------|------|-------------|
| `total_rows` | `int` | Number of rows in the dataset |
| `total_columns` | `int` | Number of columns |
| `column_names` | `list[str]` | All column names |
| `column_dtypes` | `dict[str, str]` | Column name → dtype string |
| `content_types` | `list[str]` | Unique values of `type` in paper_content |
| `keys_found` | `list[str]` | All keys found in paper_content items |
| `text_item_count` | `int` | Count of `type: "text"` items |
| `image_item_count` | `int` | Count of `type: "image_url"` items |
| `rows_with_images` | `int` | Papers containing at least 1 image |
| `rows_with_local_content` | `int` | Papers with non-null `error_local_content` |
| `sample_content_items` | `list[dict]` | Sampled paper_content items |
| `error_categories` | `list[dict]` | `[{category, count}, …]` |
| `error_locations_sample` | `list[str]` | Unique error_location values |
| `error_severities` | `list[dict]` | `[{severity, count}, …]` |
| `paper_categories` | `list[dict]` | `[{category, count}, …]` |
| `generated_at` | `str` | ISO timestamp |

---

### Paper Parsing

#### `NormalizedPaper`
Fully parsed paper representation. Produced by `parse_paper_content()`.

| Field | Type | Description |
|-------|------|-------------|
| `paper_id` | `str` | DOI/arXiv identifier |
| `title` | `str` | Paper title |
| `paper_category` | `str` | Scientific field |
| `sections` | `list[PaperSection]` | Extracted sections |
| `equations` | `list[EquationBlock]` | Extracted LaTeX equations |
| `images` | `list[ImageBlock]` | Extracted figures (with decoded file paths) |
| `tables` | `list[TableBlock]` | Extracted markdown tables |
| `theorems` | `list[TheoremBlock]` | Extracted theorem/lemma environments |
| `tagged_full_text` | `str` | Full text with `[IMAGE:FIGURE_N]` inline tags |
| `raw_items` | `list[RawContentItem]` | Original content items |
| `parse_timestamp` | `str` | ISO timestamp |

#### `PaperSection`

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique section identifier |
| `section_title` | `str` | e.g., "2. Some results" |
| `section_level` | `int` | Header depth (1–4) |
| `content` | `str` | Section body text |
| `start_index` | `int` | Character offset in full text |
| `end_index` | `int` | Character offset in full text |

#### `EquationBlock`

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique equation identifier |
| `equation_label` | `str` or `None` | e.g., "Equation 1", from `\label{…}` |
| `latex` | `str` | Raw LaTeX source |
| `display_mode` | `bool` | True for `\[…\]`, False for `\(…\)` |
| `context_before` | `str` or `None` | Surrounding text before equation |
| `context_after` | `str` or `None` | Surrounding text after equation |

#### `ImageBlock`

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique image identifier |
| `caption` | `str` or `None` | Detected figure caption |
| `image_path` | `str` or `None` | Path to decoded temp file |
| `base64_data` | `str` or `None` | Raw base64 string |
| `context_before` | `str` or `None` | Surrounding text |
| `context_after` | `str` or `None` | Surrounding text |

#### `TableBlock`

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique table identifier |
| `caption` | `str` or `None` | Detected table caption |
| `raw_content` | `str` | Original markdown table text |
| `rows` | `list[list[str]]` or `None` | Parsed rows and columns |
| `context_before` | `str` or `None` | Surrounding text |
| `context_after` | `str` or `None` | Surrounding text |

#### `TheoremBlock`

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique theorem identifier |
| `theorem_type` | `str` | "theorem", "lemma", "proposition", "corollary" |
| `label` | `str` or `None` | e.g., "**Theorem 1.1.**" |
| `statement` | `str` | The theorem statement |
| `proof` | `str` or `None` | The proof text (if found) |
| `context_before` | `str` or `None` | Surrounding text |
| `context_after` | `str` or `None` | Surrounding text |

---

### Location Parsing

#### `LocationReference`
Structured error location. Produced by `parse_error_location()`.

| Field | Type | Description |
|-------|------|-------------|
| `raw` | `str` | Original string (e.g., "Lemma 3,4") |
| `location_type` | `LocationType` | Enum: equation, figure, lemma, theorem, … |
| `identifier` | `str` | e.g., "3,4" |
| `identifiers` | `list[str]` | Split: `["3", "4"]` |
| `is_range` | `bool` | True if multiple identifiers |
| `normalized` | `str` | Canonical: "lemma 3,4" |

#### `LocationType` (enum)

| Value | Examples |
|-------|----------|
| `equation` | "Equation 6", "Eq. (12)" |
| `figure` | "Fig 5", "Figure 2d" |
| `table` | "Table 2", "Table. 1" |
| `section` | "Section 4.2.3", "Sec 3.1" |
| `theorem` | "Theorem 1.1", "Theorems 1.2, 1.3" |
| `lemma` | "Lemma 3,4", "Lemma 4.2" |
| `proposition` | "Proposition 2", "Proposition 3.9" |
| `corollary` | "1.10. Corollary" |
| `claim` | "Claim 3", "Claim 7" |
| `algorithm` | "Algorithm 1" |
| `appendix` | "Appendix B" |
| `page` | "Page 4" |
| `overall` | "Overall", "Overview" |
| `unknown` | Unrecognized format |

---

### Segmentation

#### `VerificationSnippet`
A single verifiable unit. Produced by `segment_paper()`.

| Field | Type | Description |
|-------|------|-------------|
| `snippet_id` | `str` | Unique identifier |
| `snippet_type` | `SnippetType` | SECTION, EQUATION, FIGURE, TABLE, THEOREM, … |
| `paper_id` | `str` | Parent paper identifier |
| `location` | `str` | Human-readable descriptor |
| `content` | `str` | The content to verify |
| `image_path` | `str` or `None` | Path for vision verifier |
| `location_ref` | `LocationReference` or `None` | Structured reference for alignment |
| `metadata` | `dict` | Extra data (latex, caption, theorem_type, …) |
| `content_length` | `int` | Character count |
| `estimated_tokens` | `int` | Rough token estimate (~4 chars/token) |

#### `SnippetType` (enum)

| Value | Routed To |
|-------|-----------|
| `SECTION`, `SUBSECTION` | `text` verifier |
| `EQUATION` | `math_equation` verifier |
| `FIGURE`, `TABLE` | `vision` verifier |
| `THEOREM`, `LEMMA`, `PROPOSITION`, `COROLLARY`, `ALGORITHM`, `APPENDIX`, `PARAGRAPH` | `text` verifier |

---

### Verification Results

#### `EquationVerificationResult`
Produced by `MathEquationVerifier`.

| Field | Type | Description |
|-------|------|-------------|
| `snippet_id` | `str` | Source snippet |
| `verifier_name` | `str` | `"math_equation"` |
| `status` | `VerificationStatus` | VALID, INVALID, MALFORMED, UNVERIFIABLE |
| `error_detected` | `bool` | Status == INVALID |
| `confidence` | `float` | 0.0–1.0 |
| `reasoning` | `str` | Interpretation of execution output |
| `predicted_error_category` | `str` or `None` | "Equation / proof" if INVALID |
| `sympy_code` | `str` or `None` | The generated SymPy code |
| `execution_output` | `str` or `None` | stdout from sandbox |
| `execution_error` | `str` or `None` | stderr from sandbox |
| `return_code` | `int` or `None` | Subprocess return code |
| `execution_time_ms` | `float` | Wall-clock time |

#### `VisionVerificationResult`
Produced by `VisionVerifier`.

| Field | Type | Description |
|-------|------|-------------|
| `snippet_id` | `str` | Source snippet |
| `verifier_name` | `str` | `"vision"` |
| `status` | `VerificationStatus` | ERROR_DETECTED, NO_ERROR, UNVERIFIABLE |
| `error_detected` | `bool` | From LLM response |
| `confidence` | `float` | From LLM response |
| `reasoning` | `str` | From LLM response |
| `predicted_error_category` | `str` or `None` | From LLM response |
| `content_type` | `str` | `"figure"` or `"table"` |
| `image_path` | `str` or `None` | Path to image file |
| `caption_text` | `str` or `None` | Detected caption |
| `execution_time_ms` | `float` | Wall-clock time |

#### `TextVerificationResult`
Produced by `TextVerifier`.

| Field | Type | Description |
|-------|------|-------------|
| `snippet_id` | `str` | Source snippet |
| `verifier_name` | `str` | `"text"` |
| `status` | `VerificationStatus` | ERROR_DETECTED, NO_ERROR, UNVERIFIABLE |
| `error_detected` | `bool` | From LLM response |
| `confidence` | `float` | From LLM response |
| `reasoning` | `str` | From LLM response |
| `predicted_error_category` | `str` or `None` | From LLM response |
| `snippet_type` | `str` | Original snippet type |
| `contradiction_locations` | `list[str]` | Specific spots where contradictions found |
| `execution_time_ms` | `float` | Wall-clock time |

#### `StatisticalVerificationResult`
Produced by `StatisticalVerifier`. Extends the base result with:

| Field | Type | Description |
|-------|------|-------------|
| `verifier_name` | `str` | `"statistical"` |
| `status` | `VerificationStatus` | INVALID (numeric contradiction), VALID, UNVERIFIABLE |
| `predicted_error_category` | `str` or `None` | `"Statistical reporting"` if INVALID |
| `checks` | `list[dict]` | Per-check detail: `{description, expr, expected, computed, tolerance, passed, error}` |

#### `CitationVerificationResult`
Produced by `CitationVerifier`. Base fields plus `snippet_type: str`. Flags
novelty over-claims / attribution mismatches decidable from the excerpt alone;
`status` is ERROR_DETECTED / NO_ERROR / UNVERIFIABLE.

#### `TriageResult`
Produced by `TriageVerifier.triage()` (uncertainty mode). Not an error verdict —
an uncertainty estimate used for routing.

| Field | Type | Description |
|-------|------|-------------|
| `snippet_id` | `str` | Source snippet |
| `snippet_type` | `str` | Snippet type value |
| `location` | `str` | Human-readable location |
| `uncertainty` | `float` | `[0,1]` estimated error likelihood |
| `suggested_route` | `str` | `math`/`proof`/`statistics`/`citation`/`vision`/`text`/`none` |
| `reason` | `str` | One-line justification |
| `selected` | `bool` | Whether escalated to a specialist |
| `routed_to` | `str` or `None` | Registered verifier it was routed to |
| `execution_time_ms` | `float` | Wall-clock time |

#### `VerificationStatus` (enum)

| Value | Meaning |
|-------|---------|
| `VALID` | Verified consistent (e.g. equation residual reduces to zero) |
| `INVALID` | A **provable** contradiction (math: a non-zero *numeric* residual, or a claimed identity that fails numeric sampling at every point; statistical: a recomputed value disagrees beyond tolerance) |
| `MALFORMED` | Code crashed or produced unparseable output |
| `UNVERIFIABLE` | Cannot be soundly decided (definitions, constrained/operator equations, no numeric claim, …) |
| `ERROR_DETECTED` | LLM found a potential issue (vision/text/citation) |
| `NO_ERROR` | LLM found no issues (vision/text/citation) |
| `SKIPPED` | Verifier decided not to process this snippet |

> **Conservative math/statistical verdicts.** A residual that merely *fails to
> simplify to zero* but still contains free symbols is **UNVERIFIABLE**, not
> INVALID — most such equations are definitions or constrained relations, not
> falsifiable identities. INVALID is reserved for deterministic contradictions.
> See `docs/UNCERTAINTY_ORCHESTRATION.md` §6 and `src/verifiers/math_verifier.py`.

---

### Predictions & Alignment

#### `PaperPrediction`
Paper-level prediction. Produced by `VerificationOrchestrator.run()`.

| Field | Type | Description |
|-------|------|-------------|
| `paper_id` | `str` | Paper identifier |
| `title` | `str` | Paper title |
| `paper_category` | `str` | Scientific field |
| `predicted_errors` | `list[PredictedError]` | Aggregated findings |
| `total_snippets` | `int` | Snippets generated |
| `snippets_verified` | `int` | Snippets actually verified |
| `errors_detected` | `int` | Count of errors detected |
| `verifier_usage` | `dict[str, int]` | Verifier name → snippet count |
| `raw_results` | `list[dict]` | All verifier results (for debugging) |
| `uncertainty_map` | `list[dict]` | Serialized `TriageResult`s (uncertainty mode; empty otherwise) |
| `generation_timestamp` | `str` | ISO timestamp |

#### `PredictedError`

| Field | Type | Description |
|-------|------|-------------|
| `error_category` | `str` | Predicted category |
| `error_location` | `str` | Predicted location |
| `confidence` | `float` | 0.0–1.0 |
| `supporting_evidence` | `str` | Reasoning from verifier |
| `verifier_name` | `str` | Which verifier found this |
| `snippet_id` | `str` | Source snippet |

#### `AlignedPrediction`
Prediction matched to ground truth. Produced by `match_predictions_to_ground_truth()`.

| Field | Type | Description |
|-------|------|-------------|
| `paper_id` | `str` | Paper identifier |
| `predicted` | `PredictedError` | The prediction |
| `matched_ground_truth` | `bool` | Whether a GT match was found |
| `ground_truth_category` | `str` or `None` | Matched GT category |
| `ground_truth_location` | `str` or `None` | Matched GT location |
| `ground_truth_severity` | `str` or `None` | Matched GT severity |
| `ground_truth_annotation` | `str` or `None` | Matched GT annotation |
| `match_quality` | `float` | Fuzzy match score 0.0–1.0 |
| `is_true_positive` | `bool` | Prediction matched GT |
| `is_false_positive` | `bool` | Prediction with no GT match |

---

### Evaluation

#### `EvaluationMetrics`
Complete metrics for a pipeline run. Produced by `evaluate_predictions()`.

| Field | Type | Description |
|-------|------|-------------|
| `true_positives` | `int` | Correctly predicted errors |
| `true_negatives` | `int` | — (all papers have errors in this dataset) |
| `false_positives` | `int` | Incorrectly predicted errors |
| `false_negatives` | `int` | Missed errors |
| `accuracy` | `float` | Overall accuracy |
| `precision` | `float` | Overall precision |
| `recall` | `float` | Overall recall |
| `f1_score` | `float` | Overall F1 |
| `by_error_category` | `list[CategoryMetrics]` | Per error-type breakdown |
| `by_error_severity` | `list[CategoryMetrics]` | Per severity breakdown |
| `by_paper_category` | `list[CategoryMetrics]` | Per field breakdown |
| `total_papers` | `int` | Papers evaluated |
| `total_ground_truth_errors` | `int` | GT errors total |
| `total_predictions` | `int` | Aligned predictions |
| `matched_predictions` | `int` | TP + FP |
| `computed_at` | `str` | ISO timestamp |

#### `CategoryMetrics`

| Field | Type | Description |
|-------|------|-------------|
| `category_name` | `str` | The category value |
| `true_positives` | `int` | TP for this category |
| `false_positives` | `int` | FP for this category |
| `false_negatives` | `int` | FN for this category |
| `true_negatives` | `int` | TN for this category |
| `precision` | `float` | Category precision |
| `recall` | `float` | Category recall |
| `f1_score` | `float` | Category F1 |
| `accuracy` | `float` | Category accuracy |
| `support` | `int` | TP + FN |

---

## `src.parser` — Parser Layer

### `analyze_dataset_schema(parquet_path, sample_rows=5) → PaperContentSchemaReport`

Loads and inspects the parquet dataset. Analyzes the structure of `paper_content` items, catalogs error categories and locations, and produces a comprehensive schema report.

### `parse_paper_content(paper_id, title, paper_category, paper_content, decode_images=True, image_output_dir=None) → NormalizedPaper`

Transforms raw `paper_content` list into a structured `NormalizedPaper`. Extracts sections, equations, images, tables, and theorems. Optionally decodes base64 images to temp files.

### `parse_error_location(raw: str) → LocationReference`

Parses a human-written error location string (e.g., "Lemma 3,4", "Fig 5") into a structured `LocationReference`. Handles 25+ format variants.

### `fuzzy_match_locations(loc_a, loc_b) → float`

Computes a fuzzy match score (0.0–1.0) between two location strings or `LocationReference` objects. Handles equivalence classes like "Equation 7" ↔ "Eq. (7)".

---

## `src.segmentation` — Segmentation Layer

### `segment_paper(paper: NormalizedPaper, config: SegmentationConfig | None = None) → list[VerificationSnippet]`

Splits a normalized paper into compact verification snippets. Produces SECTION, EQUATION, FIGURE, TABLE, THEOREM, LEMMA, PROPOSITION, and other snippet types. Long sections are chunked with configurable overlap.

---

## `src.orchestrator` — Orchestration Layer

### `VerificationOrchestrator`

```python
class VerificationOrchestrator:
    def __init__(self, config=None, registry=None)
    def run(self, paper: NormalizedPaper, progress=None) → PaperPrediction
```

Coordinates the full pipeline: segment → route → verify → aggregate. Caches verifier instances for reuse. Accepts an optional Rich progress bar.

### `UncertaintyOrchestrator(VerificationOrchestrator)`

```python
class UncertaintyOrchestrator(VerificationOrchestrator):
    def run(self, paper: NormalizedPaper, progress=None) → PaperPrediction
```

Triage-first orchestrator (`orchestration_mode="uncertainty"`): scores every
snippet's error likelihood, builds an uncertainty map, and runs specialized
verifiers only on snippets above `uncertainty_threshold` (optionally capped by
`uncertainty_budget`). Populates `PaperPrediction.uncertainty_map`. Reuses the
parent's verifier cache, aggregation, and thresholding. See
`docs/UNCERTAINTY_ORCHESTRATION.md`.

### `select_verifier_name(snippet, config=None) → str`

Looks up the verifier name for a snippet type in the routing table.

### `select_verifier(snippet, registry, config=None) → Type[BaseVerifier]`

Returns the verifier class for a snippet.

### `resolve_route_to_verifier(route, snippet, config=None) → str | None`

Maps a triage `route` label to a concrete, registered verifier via
`config.triage_route_map`. Returns `None` for the `"none"` route; falls back to
type-based routing for unknown routes.

### `create_default_registry() → VerifierRegistry`

Creates a registry pre-populated with the six standard verifiers: `math_equation`,
`vision`, `text`, `statistical`, `citation`, and `triage`.

---

## `src.verifiers` — Verifier Layer

### `BaseVerifier` (abstract)

```python
class BaseVerifier(ABC):
    name: str = "base"

    def __init__(self, config=None)
    def verify(self, snippet: VerificationSnippet) → BaseVerificationResult  # abstract
    def can_verify(self, snippet: VerificationSnippet) → bool
    def _call_llm(self, prompt, system_prompt="", image_path=None) → str
    def _call_llm_json(self, prompt, system_prompt="", image_path=None) → dict
    def _make_result(self, snippet_id, status, error_detected, confidence, …) → BaseVerificationResult
    def _analyze_in_chunks(self, content, analyze_chunk) → tuple[dict | None, int, int]
```

`_analyze_in_chunks(content, analyze_chunk)` splits `content` into overlapping
chunks (`verify_chunk_chars`/`verify_chunk_overlap`), calls `analyze_chunk(chunk)`
on each, and aggregates: an error finding wins (highest confidence, prefixed
`[chunk i/n]`); a chunk whose call raises is counted as failed and skipped; if
all fail it returns `(None, n_chunks, n_failed)`. Used by the text and citation
verifiers to stay robust on long, dense inputs. Returns
`(chosen_finding | None, n_chunks, n_failed)`.

### `VerifierRegistry`

```python
class VerifierRegistry:
    def register(self, name: str, verifier_cls: Type[BaseVerifier])
    def unregister(self, name: str)
    def get(self, name: str) → Type[BaseVerifier]
    def has(self, name: str) → bool
    def list_verifiers(self) → list[str]
```

### `MathEquationVerifier(BaseVerifier)`
`name = "math_equation"`

Converts LaTeX to SymPy via LLM, executes in sandbox, interprets result.

### `VisionVerifier(BaseVerifier)`
`name = "vision"`

Sends figures/tables to multimodal LLM with specialized prompts for each content type.

### `TextVerifier(BaseVerifier)`
`name = "text"`

Analyzes text for logical contradictions, unsupported claims, and internal inconsistencies. Chunks long snippets via `_analyze_in_chunks`.

### `StatisticalVerifier(BaseVerifier)`
`name = "statistical"`

Deterministic numeric verification. An LLM extracts *closed* arithmetic claims
(numbers only) from the text; Python recomputes them with `safe_eval` and flags
INVALID only on a contradiction beyond tolerance. Optional unit checks via `pint`.

### `CitationVerifier(BaseVerifier)`
`name = "citation"`

Flags novelty over-claims / attribution mismatches decidable from the excerpt
alone. Chunks long snippets via `_analyze_in_chunks`.

### `TriageVerifier(BaseVerifier)`
`name = "triage"`

```python
def triage(self, snippet: VerificationSnippet) → TriageResult
```

The general "where is uncertainty concentrated?" pass for uncertainty mode. One
cheap LLM call per snippet → an `uncertainty` score and a `suggested_route`.
Fails open (moderate uncertainty + type-based route) on error. `verify()` is not
implemented — drive it via `triage()` / `UncertaintyOrchestrator`.

---

## `src.baseline` — Baseline Verifiers

### `SingleCallBaseline`

```python
class SingleCallBaseline:
    def __init__(self, config=None, max_input_chars=60000)
    def run(self, paper: NormalizedPaper, progress=None) → PaperPrediction
```

One-LLM-call whole-paper error detector with orchestrator-compatible output (same
`PaperPrediction`/`PredictedError` shape, so the existing evaluation scores it
unchanged). Degrades its input budget on failure (a heavy whole-paper call can
drop the connection). Used by `scripts/baseline_comparison.py`.

---

## `src.evaluation` — Evaluation Layer

### `match_predictions_to_ground_truth(predictions, ground_truth_df, config=None, match_threshold=0.6) → list[AlignedPrediction]`

Aligns predictions against ground-truth annotations. With `config.use_llm_judge`
and a non-mock provider, an LLM judge decides matches semantically; otherwise a
fuzzy location matcher is used. Assigns TP / FP / FN labels.

### `evaluate_predictions(aligned, ground_truth_df) → EvaluationMetrics`

Computes binary metrics (accuracy, precision, recall, F1) and per-category breakdowns using scikit-learn.

### `generate_classification_report(aligned) → str`

Returns a sklearn-style classification report string.

### `generate_confusion_matrix(aligned) → list[list[int]]`

Returns a 2×2 confusion matrix as a list of lists.

---

## `src.reporting` — Reporting Layer

### `generate_report(metrics, aligned_predictions, predictions, schema_report, output_dir, ground_truth_df=None) → Path`

Writes all output files:
- `metrics.json` — all metrics in JSON
- `predictions.json` — predictions + alignments
- `confusion_matrix.csv` — sklearn confusion matrix
- `run_summary.md` — comprehensive markdown report

---

## `src.utils` — Utilities

### `llm_call(prompt, system_prompt="", model=None, image_path=None, max_tokens=None, temperature=0.0, config=None) → str`

Central LLM entry point. Routes to `mock`, `anthropic`, `openai`, or `deepseek`
backend based on `config.provider`. When `max_tokens` is `None` it is resolved
from `config.max_tokens`. Retries transient failures **and empty completions**
with exponential backoff (`config.max_retries` / `retry_backoff_seconds`).

### `parse_json_response(response: str) → dict`

Parses LLM response as JSON, handling markdown code blocks and embedded JSON.

### `run_sympy_sandbox(code, python_executable="python3", timeout_seconds=10, max_output_bytes=65536) → tuple[str, str, int]`

Executes SymPy code in a sandboxed subprocess. Returns (stdout, stderr, returncode).

### `chunk_text(text, max_chars=2000, overlap=200) → list[str]`

Boundary-aware splitter (paragraphs → sentences → hard split) with trailing-context
overlap. Always returns at least one chunk. Used by `BaseVerifier._analyze_in_chunks`.

### `safe_eval(expr: str) → float`

Evaluates a *closed* numeric arithmetic expression (literal numbers + a whitelist
of math functions/constants only). Raises `ArithmeticEvalError` for anything
outside that grammar — no names, attributes, or calls. Used by the statistical
verifier so a malformed expression can never fabricate a contradiction.

### `setup_logging(log_level="INFO", log_file=None, serialize=False)`

Configures Loguru with Rich-compatible console formatting and optional file output.

---

## `mcp-server` — MCP Server Tools

The MCP server (`mcp-server/server.py`) exposes 9 tools via the Model Context Protocol (JSON-RPC over stdio). These are the deterministic building blocks that Claude Code skills invoke. All tools are configured in `.claude/settings.json`.

### `parse_paper(paper_id, title, paper_category, paper_content, decode_images=False) → dict`

Parse raw `paper_content` (list of dicts with `type`/`text`/`image_url`) into a structured paper. Returns sections, equations, images, tables, theorems, and summary counts. Each section/equation/table/theorem includes an ID, label, and content preview (truncated to 500 chars).

### `segment_paper(paper_id, title, paper_category, paper_content) → dict`

Parse + segment a paper in one call. Returns the paper ID, title, total snippet count, and a list of snippets — each with `snippet_id`, `snippet_type` (SECTION, EQUATION, FIGURE, TABLE, THEOREM, LEMMA, etc.), `location`, `content` (truncated to 3000 chars), `content_length`, and `metadata`. This is the primary tool used by `/verify-paper` to get verifiable units.

### `run_sympy_check(latex, equation_context="") → dict`

Verify a LaTeX equation via the SymPy sandbox. Returns a note directing the caller to generate SymPy code first (using the math-verifier conventions), then call `run_sympy_sandbox_exec`. The LLM→code generation step happens in the `/verify-math` skill.

### `run_sympy_sandbox_exec(sympy_code, harness="", timeout_seconds=10) → dict`

Execute SymPy code in a sandboxed subprocess. Prepends the standard Paperena verdict harness (unless a custom one is provided). Returns `stdout`, `stderr`, `returncode`, and a parsed `verdict` object extracted from the `VERDICT:` line in stdout.

| Return field | Type | Description |
|-------------|------|-------------|
| `stdout` | `str` | Sandbox stdout (truncated to 2000 chars) |
| `stderr` | `str` | Sandbox stderr (truncated to 1000 chars) |
| `returncode` | `int` | Process exit code (-1 on error) |
| `verdict` | `dict` or `None` | Parsed verdict: `{verdict, residual?, reason?}` |
| `success` | `bool` | True if returncode == 0 and verdict was parsed |

### `safe_arithmetic_eval(expression: str) → dict`

Safely evaluate a closed arithmetic expression. Supports `+`, `-`, `*`, `/`, `**`, `%`, `sqrt`, `log`, `ln`, `log10`, `log2`, `exp`, `abs`, `round`, `min`, `max`, `sum`, `mean`, `floor`, `ceil`, `pow`, `pi`, `e`. Returns `{expression, computed, success}` or `{expression, computed: null, success: false, error}` on failure. Used by the statistical verifier to recompute reported numbers.

### `check_numeric_claim(expression, expected, tolerance=0.01) → dict`

Evaluate an arithmetic expression and compare to an expected value within a relative tolerance. Returns `{expression, expected, tolerance, computed, relative_error, passed}`.

### `get_paper_from_dataset(paper_id, parquet_path="data/train-00000-of-00001.parquet") → dict`

Fetch a single paper by DOI/arXiv ID. Returns metadata (`paper_id`, `title`, `paper_category`, `error_category`, `error_location`, `error_severity`), `paper_content` (with large text/images truncated), and a `has_ground_truth` flag. Content items over 3000 chars are truncated to keep MCP responses manageable.

### `list_papers_in_dataset(parquet_path="data/train-00000-of-00001.parquet", max_papers=50, offset=0) → dict`

List paper IDs and metadata from the dataset. Returns `{total_papers, offset, returned, papers: [{paper_id, title, paper_category, error_category, error_severity}]}`.

### `analyze_dataset_schema(parquet_path="data/train-00000-of-00001.parquet", sample_rows=5) → dict`

Analyze the dataset schema and return a full `PaperContentSchemaReport` as a dict. Includes column info, content types, error categories, paper categories, and sample content items.
