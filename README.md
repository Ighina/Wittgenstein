# Paperena Verification — Automated Scientific Paper Verification Pipeline

A production-quality, modular Python pipeline that automatically verifies scientific papers against a ground-truth dataset of human-annotated errors. Designed for extensibility: new verifiers, parsers, and evaluation strategies can be added without modifying the orchestrator.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Pipeline Overview](#pipeline-overview)
3. [Architecture](#architecture)
4. [CLI Commands](#cli-commands)
5. [Dataset](#dataset)
6. [Configuration](#configuration)
7. [Extending the Pipeline](#extending-the-pipeline)
8. [Project Structure](#project-structure)
9. [Testing](#testing)
10. [Dependencies](#dependencies)
11. [Further Documentation](#further-documentation)

---

## Quick Start

### Prerequisites

- Python 3.11 or later
- A parquet dataset file in the format described under [Dataset](#dataset)

### Installation

```bash
# Clone and enter the project
cd PaperenaVerification

# Install the package and dependencies
pip install -e ".[dev]"
```

### First Run

```bash
# 1. Inspect the dataset
python main.py analyze

# 2. Verify a single paper (mock LLM — no API key needed)
python main.py verify-one "2405.01133v3"

# 3. Run the full pipeline on a subset
python main.py verify --max-papers 5

# 4. Evaluate saved predictions
python main.py evaluate
```

### Running with a Real LLM

The default provider is **DeepSeek** (`deepseek-v4-pro`); just export the key:

```bash
export DEEPSEEK_API_KEY="sk-..."
python main.py verify-one "2405.01133v3"
```

To use another backend, set the provider/model (and its key):

```python
config = PipelineConfig(
    llm=LLMConfig(provider="anthropic", model="claude-opus-4-8")  # or provider="openai"
)
```

```bash
export ANTHROPIC_API_KEY="sk-ant-..."   # or OPENAI_API_KEY
```

> The mock backend (`provider="mock"`) needs no key and returns deterministic
> responses for every verifier — used by the offline test suite.

---

## Pipeline Overview

The system processes papers through a linear chain of phases, each implemented as an independent, swappable module:

```
Parquet Dataset
      │
      ▼
┌─────────────────┐
│ Phase 1         │   Dataset Exploration
│ Schema Analyzer │   Inspects schema, infers content types, produces report
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Phase 2         │   Parsing
│ Content Parser  │   Raw paper_content → NormalizedPaper (Pydantic models)
│ Location Parser │   "Lemma 3,4" → LocationReference(type=lemma, id="3,4")
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Phase 3         │   Segmentation
│ Segmenter       │   NormalizedPaper → List[VerificationSnippet]
│                 │   Produces: SECTION, EQUATION, FIGURE, TABLE, THEOREM, …
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Phase 4         │   Routing
│ Router          │   Maps snippet type → verifier name via config
│ Registry        │   Plugin registry: name → verifier class
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│ Phase 5–8        Verification (3 parallel verifiers) │
│                                                      │
│  MathEquationVerifier    VisionVerifier    TextVerifier│
│  LaTeX → LLM → SymPy     Figure/Table      Logical   │
│  → sandbox execution     → multimodal LLM  consistency│
└────────┬────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│ Phase 9         │   Aggregation
│ Aggregator      │   Findings → paper-level predictions
│                 │   Filters by confidence threshold per verifier
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Phase 10        │   Evaluation
│ Alignment       │   Fuzzy-matches predictions ↔ ground truth
│ Metrics         │   Binary + per-category + sklearn report
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Phase 11        │   Reporting
│ Reporter        │   metrics.json, predictions.json, CM.csv, run_summary.md
└─────────────────┘
```

---

## Architecture

The system is built around six independent layers, each in its own package:

| Layer | Package | Responsibility |
|-------|---------|---------------|
| **Parser** | `src.parser` | Load, inspect, and normalize raw data |
| **Segmentation** | `src.segmentation` | Split papers into verifiable units |
| **Orchestration** | `src.orchestrator` | Coordinate the pipeline and route snippets |
| **Verification** | `src.verifiers` | Specialized error-detection modules |
| **Evaluation** | `src.evaluation` | Align predictions to ground truth, compute metrics |
| **Reporting** | `src.reporting` | Generate output files and summaries |

All layers communicate through Pydantic models defined in `src/models.py`. No layer directly imports from another layer's internals — only from `src.models` and the public API of other packages.

### Key Design Decisions

1. **LLM abstraction** — All LLM calls go through `src/utils/llm.py:llm_call()`. Backend (mock, Anthropic, OpenAI, DeepSeek) is configured in `PipelineConfig.llm.provider`. The mock backend returns deterministic responses based on prompt keywords, enabling full pipeline testing without API keys.

2. **Plugin architecture** — Verifiers extend `BaseVerifier` and register with `VerifierRegistry`. Adding a verifier = one new file + one registry call + one routing config line. The orchestrator never needs modification.

3. **Fuzzy location matching** — Error locations are parsed into structured `LocationReference` objects via 25+ regex patterns. Alignment uses normalized forms ("Equation 7" ↔ "Eq. (7)") plus Jaccard overlap of multi-reference identifiers.

4. **Safe execution** — SymPy code generated by the LLM runs in a subprocess with a 10-second timeout, in a temporary directory. No network access, no filesystem side effects.

5. **Inline tagging** — Images, tables, and equations are tagged in text as `[IMAGE:FIGURE_3]`, `[TABLE:TABLE_1]`, and `[EQUATION:EQ_5]` to preserve location context throughout verification.

6. **Concurrent verification** — Within a paper, snippets are verified in parallel across a thread pool (`LLMConfig.num_workers`, default 8), since each `verify()` blocks on a network LLM call. Verifiers are stateless and pre-instantiated before fan-out, and `llm_call()` builds a fresh client per call, so the path is thread-safe. Transient API failures are absorbed by a bounded exponential-backoff retry in `llm_call()`. Aggregation sorts findings by confidence, so output is identical regardless of worker count; `num_workers=1` forces a deterministic sequential run.

7. **Tunable strictness** — `PipelineConfig.strictness` selects between `strict` prompts (flag only erratum/retraction-worthy errors) and the original `lenient` prompts, and adjusts default confidence thresholds. This curbs over-identification of errors when a paper is split into hundreds or thousands of independently-judged snippets.

8. **Two orchestration modes** — `config.orchestration_mode` chooses `exhaustive` (route every snippet by type) or `uncertainty` (a cheap triage pass scores each snippet, and specialists run only where error density is high). The uncertainty mode routes effort by *expected error density* rather than document structure — see **[docs/UNCERTAINTY_ORCHESTRATION.md](docs/UNCERTAINTY_ORCHESTRATION.md)**.

9. **Conservative, deterministic numeric verdicts** — the math verifier only flags `INVALID` on a *provable* contradiction (a non-zero numeric residual, or a claimed identity that fails numeric sampling everywhere); definitions and constrained equations are `UNVERIFIABLE`. The statistical verifier recomputes reported numbers with a sandbox-free safe arithmetic evaluator. Both avoid the false positives that naive symbolic checking produces.

10. **Six verifiers, pluggable routing** — `math_equation`, `vision`, `text`, `statistical`, `citation`, and `triage`. Long text snippets are chunked and verified piece-by-piece for robustness on dense proofs.

---

## CLI Commands

### `python main.py analyze`

Inspect the dataset schema. Displays a Rich table with content types, keys found, error categories, severities, and paper categories.

```
Options:
  --parquet, -p      Path to parquet file (default: data/train-00000-of-00001.parquet)
  --output, -o       Save report to JSON file
  --sample-rows, -n  Number of rows to sample (default: 5)
```

### `python main.py verify`

Run the full pipeline: parse → segment → verify → evaluate → report.

```
Options:
  --parquet, -p            Path to parquet file
  --output, -o             Output directory (default: outputs/)
  --max-papers, -n         Limit number of papers (useful for testing)
  --decode-images          Decode base64 images to files for vision verification
  --no-decode-images       Skip image decoding (faster)
  --skip-evaluation        Verify only, skip metrics computation
  --workers, -w            Concurrent API workers per paper (default: 8; 1 = sequential)
  --strictness             Error sensitivity: "strict" (default) or "lenient" (see below)
  --mode                   Orchestration: "exhaustive" (default) or "uncertainty"
  --uncertainty-threshold  In uncertainty mode, escalate snippets ≥ this score (default 0.30)
```

> **Performance:** snippets within a paper are verified concurrently across `--workers` threads
> (LLM calls are network-bound). The default of `8` gives roughly an 8× speedup on the API-bound
> work versus the old sequential behavior. Use `--workers 1` for fully deterministic, sequential
> runs when debugging.

> **Strictness:** `strict` (default) only flags **critical** errors — mistakes serious enough to
> warrant an erratum or retraction — and ignores typos, style, missing citations, and minor
> issues. `lenient` reproduces the original, broader prompts and lower confidence thresholds.
> Strict mode also raises confidence thresholds (text 0.8, vision 0.75).

### `python main.py verify-one <PAPER_ID>`

Verify a single paper by its DOI/arXiv ID. Prints parsed structure, verification results, and ground-truth comparison.

```
Arguments:
  PAPER_ID    Paper identifier (e.g., "2405.01133v3")

Options:
  --parquet, -p               Path to parquet file
  --decode-images / --no-decode-images
  --workers, -w               Concurrent API workers (default: 8; 1 = sequential)
  --strictness                "strict" (default) or "lenient"
  --mode                      "exhaustive" (default) or "uncertainty"
  --uncertainty-threshold     In uncertainty mode, escalation threshold (default 0.30)
```

In `--mode uncertainty`, `verify-one` also prints the per-region **uncertainty
map** and which snippets were escalated to which specialist.

### `python main.py evaluate`

Compute metrics from previously-saved predictions (after a `--skip-evaluation` run).

```
Options:
  --predictions, -p  Path to raw_predictions.json (default: outputs/raw_predictions.json)
  --parquet, -d       Path to ground-truth parquet file
  --output, -o        Output directory (default: outputs/)
```

### Analysis scripts

```bash
# Recall/cost tradeoff of uncertainty mode (triages once, reuses specialist
# results across thresholds). Writes outputs_new/threshold_sweep.json.
python scripts/threshold_sweep.py 2405.01133v3 2402.10307v2

# Single-call baseline vs the orchestrated pipeline on the same papers, scored
# with the same alignment + metrics. Writes outputs_new/baseline_comparison.json.
python scripts/baseline_comparison.py 2405.01133v3 2402.10307v2 --mode uncertainty
python scripts/baseline_comparison.py --provider mock          # offline dry run
```

---

## Dataset

The pipeline expects a parquet file with the following schema:

| Column | Type | Description |
|--------|------|-------------|
| `doi/arxiv_id` | `str` | Unique paper identifier |
| `title` | `str` | Paper title |
| `paper_category` | `str` | Scientific field (Mathematics, Biology, …) |
| `error_category` | `str` | Type of annotated error (Equation / proof, Figure duplication, …) |
| `error_location` | `str` | Human-written location (Lemma 3,4, Fig 5, Section 4.2.3, …) |
| `error_severity` | `str` | `errata` or `retract` |
| `error_annotation` | `str` | Detailed description of the error |
| `paper_content` | `List[Dict]` | Content items (see below) |
| `error_local_content` | `List[Dict]` or `None` | Local context around the error (25/68 rows) |

### `paper_content` Structure

Each item is a dictionary with three keys:

```python
{
    "type": "text",           # or "image_url"
    "text": "ABSTRACT. …",    # Present when type == "text"
    "image_url": None         # Present when type == "image_url":
                              # {"url": "data:image/jpeg;base64,..."}
}
```

- **Text items** (967 total): Contain the paper's prose, LaTeX math (`\(inline\)`, `\[display\]`), and markdown formatting.
- **Image items** (826 total): Base64-encoded JPEG figures interleaved with text items.
- **Content type sequence**: Text and image items alternate throughout the paper. Sections, theorems, and equations are embedded in text — there are no explicit structural markers beyond markdown headers and `**Theorem N.**`-style formatting.

---

## Configuration

All configuration lives in `src/config.py` as Python dataclasses. The global default instance is accessible as `src.config.default_config`.

### Configuration Classes

```python
@dataclass
class PathsConfig:
    data_dir: Path          # Input data directory
    output_dir: Path        # Output directory for reports
    parquet_file: str       # Default parquet filename

@dataclass
class LLMConfig:
    provider: str               # default "deepseek" ("mock", "anthropic", "openai", "deepseek")
    model: str                  # default "deepseek-v4-pro"
    api_key_env: str            # default "DEEPSEEK_API_KEY"
    max_tokens: int             # OUTPUT-token budget, default 8192 — see note
    temperature: float          # Sampling temperature (0.0 = deterministic)
    timeout_seconds: int        # API timeout
    num_workers: int            # Concurrent snippet verifications per paper (default 8)
    max_retries: int            # Retry attempts on transient API failures (default 3)
    retry_backoff_seconds: float # Base for exponential backoff between retries (default 2.0)

@dataclass
class SandboxConfig:
    timeout_seconds: int    # Max SymPy execution time
    max_output_bytes: int   # Max bytes to capture from stdout/stderr
    python_executable: str  # Python interpreter path

@dataclass
class SegmentationConfig:
    max_snippet_chars: int  # Max characters per snippet
    max_section_chars: int  # Threshold for splitting sections
    overlap_chars: int      # Overlap between split chunks

@dataclass
class VerifierConfig:
    enabled: bool           # Toggle verifier on/off
    confidence_threshold: float  # Minimum confidence to report a finding

@dataclass
class PipelineConfig:
    paths: PathsConfig
    llm: LLMConfig
    sandbox: SandboxConfig
    segmentation: SegmentationConfig
    verifiers: dict[str, VerifierConfig]
    verifier_routing: dict[str, str]
    strictness: str             # "strict" (default) or "lenient" — see note below
    orchestration_mode: str     # "exhaustive" (default) or "uncertainty"
    uncertainty_threshold: float  # escalation threshold in uncertainty mode (0.30)
    uncertainty_budget: int | None  # optional cap on specialist calls per paper
    triage_route_map: dict[str, str]  # triage route → verifier name
    verify_chunk_chars: int     # chunk text-verifier input over this length (2000)
    verify_chunk_overlap: int   # overlap between chunks (200)
    use_llm_judge: bool         # LLM judge for prediction↔ground-truth matching
```

> **`max_tokens` & reasoning models.** `max_tokens` is the *output* budget. With a
> reasoning model (e.g. `deepseek-v4-pro`) a too-small value is consumed by
> chain-of-thought and the response comes back **empty**, so the default is
> `8192` and `llm_call` resolves it from the config (treating empty responses as
> retryable). See [docs/CONFIGURATION.md](docs/CONFIGURATION.md).

> **`strictness`** controls both the verifier prompts and the default confidence thresholds.
> `strict` (the default) instructs the text/vision verifiers to flag only erratum/retraction-worthy
> errors and uses higher thresholds (text 0.8, vision 0.75, math 0.7); `lenient` restores the
> original broader prompts and thresholds (text 0.5, vision 0.6, math 0.7). When `verifiers` is
> supplied explicitly, those thresholds take precedence over the strictness defaults. The math
> verifier's decision is deterministic (SymPy decides validity, not the LLM), so its prompt is
> unaffected by strictness — only its threshold applies.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PAPERENA_DATA_DIR` | `data` | Input data directory |
| `PAPERENA_OUTPUT_DIR` | `outputs` | Output directory |
| `DEEPSEEK_API_KEY` | — | DeepSeek API key (default provider) |
| `ANTHROPIC_API_KEY` | — | Anthropic API key (when provider = "anthropic") |
| `OPENAI_API_KEY` | — | OpenAI API key (when provider = "openai") |

### Creating Custom Configurations

```python
from src.config import PipelineConfig, LLMConfig, VerifierConfig

config = PipelineConfig.from_dict({
    "strictness": "strict",          # "strict" (default) or "lenient"
    "llm": {
        "provider": "anthropic",
        "model": "claude-opus-4-8",
        "max_tokens": 8192,
        "num_workers": 8,            # concurrent snippet verifications per paper
    },
    "verifiers": {
        "math_equation": {"confidence_threshold": 0.8},
        "text": {"confidence_threshold": 0.6},
    },
    "verifier_routing": {
        "EQUATION": "math_equation",
        "FIGURE": "vision",
        "TABLE": "vision",
        "SECTION": "text",
        "THEOREM": "text",
    },
})
```

---

## Extending the Pipeline

### Adding a New Verifier

The system uses a plugin architecture. To add a new verifier:

**Step 1**: Create a new class in `src/verifiers/` extending `BaseVerifier`:

```python
# src/verifiers/citation_verifier.py
from src.verifiers.base import BaseVerifier
from src.models import VerificationSnippet, BaseVerificationResult, VerificationStatus

class CitationVerifier(BaseVerifier):
    name = "citation"

    def verify(self, snippet: VerificationSnippet) -> BaseVerificationResult:
        start_time = time.monotonic()
        # Your verification logic here
        result = self._call_llm_json(
            prompt=f"Check citations in: {snippet.content[:2000]}",
            system_prompt="You are a citation-checking expert. …",
        )
        return BaseVerificationResult(
            snippet_id=snippet.snippet_id,
            verifier_name=self.name,
            status=VerificationStatus.ERROR_DETECTED if result.get("error_detected") else VerificationStatus.NO_ERROR,
            error_detected=result.get("error_detected", False),
            confidence=result.get("confidence", 0.0),
            reasoning=result.get("reasoning", ""),
            predicted_error_category=result.get("predicted_error_category"),
            execution_time_ms=(time.monotonic() - start_time) * 1000,
        )

    def can_verify(self, snippet: VerificationSnippet) -> bool:
        return "references" in snippet.content.lower() or "bibliography" in snippet.content.lower()
```

**Step 2**: Register it in `src/orchestrator/router.py`:

```python
from src.verifiers.citation_verifier import CitationVerifier

def create_default_registry():
    registry = VerifierRegistry()
    registry.register("math_equation", MathEquationVerifier)
    registry.register("vision", VisionVerifier)
    registry.register("text", TextVerifier)
    registry.register("citation", CitationVerifier)  # <-- add this
    return registry
```

**Step 3**: Add routing in `src/config.py`:

```python
verifier_routing: dict[str, str] = field(default_factory=lambda: {
    "EQUATION": "math_equation",
    "FIGURE": "vision",
    "TABLE": "vision",
    "SECTION": "text",
    "THEOREM": "text",
    "CITATION": "citation",   # <-- add this
    …
})
```

No changes to the orchestrator, router logic, or CLI are needed.

### Adding a New Snippet Type

1. Add the type to `SnippetType` enum in `src/models.py`.
2. Update `src/segmentation/segmenter.py` to produce snippets of the new type.
3. Add a routing entry in `src/config.py`.
4. Optionally, add a new verifier to handle the type.

### Adding a New Evaluation Metric

1. Add the metric fields to `EvaluationMetrics` or `CategoryMetrics` in `src/models.py`.
2. Implement the computation in `src/evaluation/metrics.py`.
3. Update `src/reporting/reporter.py` to include the new metric in reports.

---

## Project Structure

```
PaperenaVerification/
│
├── data/                              # Input parquet dataset
│   └── train-00000-of-00001.parquet
│
├── outputs/                           # Generated reports (gitignored)
│   ├── metrics.json                   # All computed metrics
│   ├── predictions.json               # Predictions + alignments
│   ├── confusion_matrix.csv           # sklearn confusion matrix
│   ├── raw_predictions.json           # Pre-alignment predictions
│   └── run_summary.md                 # Comprehensive markdown report
│
├── docs/                              # Extended documentation
│   ├── ARCHITECTURE.md
│   ├── API_REFERENCE.md
│   ├── CONFIGURATION.md
│   ├── EXTENDING.md
│   └── UNCERTAINTY_ORCHESTRATION.md   # Uncertainty mode, new verifiers, chunking, baseline
│
├── scripts/                          # Analysis tooling
│   ├── threshold_sweep.py            #   recall/cost sweep for uncertainty mode
│   └── baseline_comparison.py        #   single-call baseline vs orchestrated pipeline
│
├── src/                               # Source code (5,200+ lines)
│   ├── __init__.py                    # Package version
│   ├── config.py                      # All configuration dataclasses
│   ├── models.py                      # 30+ Pydantic models across all phases
│   │
│   ├── parser/                        # Phase 1–2: data loading & parsing
│   │   ├── schema_analyzer.py         #   analyze_dataset_schema()
│   │   ├── content_parser.py          #   parse_paper_content()
│   │   └── location_parser.py         #   parse_error_location(), fuzzy_match_locations()
│   │
│   ├── segmentation/                  # Phase 3: paper → snippets
│   │   └── segmenter.py               #   segment_paper()
│   │
│   ├── orchestrator/                  # Phase 4–5: coordination & routing
│   │   ├── orchestrator.py            #   VerificationOrchestrator
│   │   └── router.py                  #   select_verifier(), create_default_registry()
│   │
│   ├── verifiers/                     # Phase 6–8: verification modules
│   │   ├── base.py                    #   Abstract BaseVerifier (+ chunk aggregation)
│   │   ├── registry.py                #   VerifierRegistry (plugin system)
│   │   ├── math_verifier.py           #   MathEquationVerifier (conservative SymPy)
│   │   ├── vision_verifier.py         #   VisionVerifier (figure + table)
│   │   ├── text_verifier.py           #   TextVerifier (chunked)
│   │   ├── statistical_verifier.py    #   StatisticalVerifier (deterministic numeric)
│   │   ├── citation_verifier.py       #   CitationVerifier (attribution/novelty)
│   │   └── triage_verifier.py         #   TriageVerifier (uncertainty scoring)
│   │
│   ├── orchestrator/                  #   …also uncertainty_orchestrator.py
│   │   └── uncertainty_orchestrator.py #   UncertaintyOrchestrator
│   │
│   ├── baseline/                      # Comparison baselines
│   │   └── single_call_baseline.py    #   SingleCallBaseline (whole paper, one call)
│   │
│   ├── evaluation/                    # Phase 10–11: alignment & metrics
│   │   ├── alignment.py               #   match_predictions_to_ground_truth()
│   │   └── metrics.py                 #   evaluate_predictions(), generate_confusion_matrix()
│   │
│   ├── reporting/                     # Phase 12: output generation
│   │   └── reporter.py                #   generate_report()
│   │
│   └── utils/                         # Shared utilities
│       ├── logging.py                 #   Loguru setup (setup_logging, get_logger)
│       ├── llm.py                     #   LLM abstraction (mock, anthropic, openai, deepseek)
│       ├── sandbox.py                 #   Subprocess sandbox for SymPy
│       ├── chunking.py                #   Boundary-aware text chunking
│       └── safe_arithmetic.py         #   Injection-safe numeric evaluator
│
├── tests/                             # 100+ tests
│   ├── test_parser.py                 #   location parsing + fuzzy matching + schema
│   ├── test_segmentation.py           #   snippet generation + chunking
│   ├── test_verifiers.py              #   registry + verifiers + math conservatism
│   ├── test_uncertainty.py            #   triage + uncertainty orchestration + routing
│   ├── test_statistical.py            #   statistical verifier + citation + safe_eval
│   ├── test_chunking.py               #   chunk_text + chunk aggregation
│   ├── test_baseline.py               #   single-call baseline
│   ├── test_evaluation.py             #   alignment + metrics
│   └── test_integration.py            #   end-to-end pipeline
│
├── main.py                            # CLI entry point (typer + Rich)
├── pyproject.toml                     # Dependencies & build configuration
└── README.md                          # This file
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_parser.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run integration tests only
pytest tests/test_integration.py -v
```

### Test Summary

| File | Coverage |
|------|----------|
| `test_parser.py` | Location parsing (25 formats), fuzzy matching, schema analysis |
| `test_segmentation.py` | Section/equation/figure/theorem snippets, chunking |
| `test_verifiers.py` | Registry CRUD, verifier behavior, **math conservatism** (INVALID only on provable contradictions) |
| `test_uncertainty.py` | Triage scoring, route resolution, escalation-by-uncertainty, budget cap |
| `test_statistical.py` | Statistical verifier, citation verifier, `safe_eval` injection-safety |
| `test_chunking.py` | `chunk_text` boundaries, chunk aggregation, failure tolerance |
| `test_baseline.py` | Single-call baseline output shape, truncation, fallback |
| `test_evaluation.py` | Alignment matching, metric computation, category breakdowns |
| `test_integration.py` | Parse→segment, parse→verify, multi-paper |
| **Total** | **117 tests collected** (mock/offline subset deterministic; verifier tests that hit a live provider need an API key) |

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `pydantic` | ≥2.0 | Data models with validation |
| `pandas` | ≥2.0 | Parquet I/O, DataFrame operations |
| `pyarrow` | ≥12.0 | Parquet backend for pandas |
| `sympy` | ≥1.12 | Symbolic equation verification |
| `scikit-learn` | ≥1.3 | Classification metrics, confusion matrix |
| `rich` | ≥13.0 | Console output, progress bars, tables |
| `loguru` | ≥0.7 | Structured logging |
| `pillow` | ≥10.0 | Image decoding for vision verifier |
| `numpy` | ≥1.24 | Numerical operations |
| `scipy` | ≥1.10 | Numerical support for the statistical verifier |
| `typer` | ≥0.9 | CLI framework |
| `openai` | (runtime) | OpenAI-compatible client for the OpenAI/DeepSeek backends |

Optional extras (`pip install '.[units]'`):

| Package | Purpose |
|---------|---------|
| `pint` | Unit-conversion checks in the statistical verifier (skipped if absent) |

---

## Further Documentation

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) — Deep dive into the design, data flow, and patterns
- [UNCERTAINTY_ORCHESTRATION.md](docs/UNCERTAINTY_ORCHESTRATION.md) — Uncertainty-driven orchestration, triage, statistical/citation verifiers, chunking, threshold sweep, and baseline comparison
- [API_REFERENCE.md](docs/API_REFERENCE.md) — Complete API reference for all modules
- [CONFIGURATION.md](docs/CONFIGURATION.md) — Detailed configuration guide with examples
- [EXTENDING.md](docs/EXTENDING.md) — Step-by-step guide to adding verifiers, parsers, and metrics
