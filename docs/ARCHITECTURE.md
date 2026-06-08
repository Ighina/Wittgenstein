# Architecture Deep Dive

This document describes the design principles, data flow, and key patterns used in the Paperena Verification pipeline.

---

## Design Principles

### 1. Layered Architecture

Each layer depends only on the layer below it and on shared models. No layer imports from another layer's internals.

```
┌──────────────────────────────────────────────┐
│  CLI (main.py)                                │
│  Orchestrates user commands                    │
└──────────────────┬───────────────────────────┘
                   │ imports
┌──────────────────▼───────────────────────────┐
│  Orchestrator (src/orchestrator)              │
│  Coordinates pipeline, routes snippets        │
└──────┬──────────┬──────────┬─────────────────┘
       │          │          │ imports
┌──────▼──┐ ┌─────▼────┐ ┌──▼──────────────┐
│ Parser  │ │Segmenter │ │ Verifiers        │
│ Layer   │ │ Layer    │ │ Layer            │
└────┬────┘ └──────────┘ └──┬──────────────┘
     │                       │ imports
┌────▼───────────────────────▼─────────────────┐
│  Evaluation Layer                             │
│  Alignment + Metrics                          │
└───────────────────┬──────────────────────────┘
                    │ imports
┌───────────────────▼──────────────────────────┐
│  Reporting Layer                              │
│  Generates output files                       │
└──────────────────────────────────────────────┘
```

All layers share `src/models.py` (Pydantic models) and `src/config.py` (configuration).

### 2. Model-Driven Communication

Every inter-layer boundary uses Pydantic models. This provides:

- **Type safety** — mypy/pyright catch mismatches at dev time
- **Validation** — malformed data is caught at the boundary, not deep in the pipeline
- **Serialization** — `.model_dump()` and `.model_dump_json()` enable clean output
- **Documentation** — field descriptions serve as API docs

### 3. Plugin Architecture for Verifiers

Verifiers are the extension point. The pattern:

```
VerifierRegistry (dict[str, Type[BaseVerifier]])
       │
       │ register("name", VerifierClass)
       ▼
Router (snippet_type → verifier_name)
       │
       │ registry.get(name)
       ▼
BaseVerifier.verify(snippet) → BaseVerificationResult
```

Adding a verifier touches three files:
1. New verifier class (1 file)
2. Registry registration (1 line)
3. Routing config (1 line)

The orchestrator never changes. The default registry ships six verifiers:
`math_equation`, `vision`, `text`, `statistical`, `citation`, and `triage`.

### 4. Two Orchestration Modes

The pipeline supports two orchestration strategies, selected by
`config.orchestration_mode`:

- **`exhaustive`** (default, `VerificationOrchestrator`) — route **every**
  snippet to a verifier by *type* and verify all of them.
- **`uncertainty`** (`UncertaintyOrchestrator`) — a cheap **triage** pass scores
  every snippet's error likelihood (an *uncertainty map*); specialized verifiers
  run **only** where the score exceeds a threshold. Routing emerges from expected
  error density rather than document structure, which attacks the sparse-error
  problem (most snippets are correct) and shrinks the false-positive surface.

Both share the same verifier registry, confidence thresholding, and aggregation —
`UncertaintyOrchestrator` subclasses `VerificationOrchestrator`. The uncertainty
mode is documented in full in **`docs/UNCERTAINTY_ORCHESTRATION.md`** (triage
verifier, route resolution, statistical/citation verifiers, chunking, threshold
sweep, baseline comparison).

---

## Data Flow: Paper Through the Pipeline

### Step-by-Step Trace

#### 1. Raw → NormalizedPaper

```
Raw parquet row:
{
  "doi/arxiv_id": "2405.01133v3",
  "title": "A missing theorem on dual spaces",
  "paper_category": "Mathematics",
  "paper_content": [
    {"type": "text", "text": "ABSTRACT. We answer …", "image_url": None},
    {"type": "text", "text": "## 2. Some results …", "image_url": None},
    {"type": "image_url", "text": None, "image_url": {"url": "data:image/jpeg;base64,…"}},
    …
  ]
}

       │ parse_paper_content()
       ▼

NormalizedPaper(
    paper_id="2405.01133v3",
    title="A missing theorem on dual spaces",
    paper_category="Mathematics",
    sections=[
        PaperSection(id="…_sec_0", section_title="2. Some results …", content="…"),
        …
    ],
    equations=[
        EquationBlock(id="…_eq_0", latex="Y^{**} \\cong E^{\\dagger\\dagger}", …),
        …  # 114 equations extracted
    ],
    images=[ImageBlock(id="…_img_0", caption="Figure 1", base64_data="…"), …],
    tables=[],
    theorems=[
        TheoremBlock(id="…_thm_0", theorem_type="theorem", label="**Theorem 1.1.**", …),
        …
    ],
    tagged_full_text="As shown in [IMAGE:FIGURE_1]…",
)
```

Key transformations:
- Sections extracted via regex on markdown headers (`## N. Title`)
- Equations extracted from `\[…\]`, `\(…\)`, `\begin{equation}…\end{equation}`
- Images decoded from base64 data URIs (optional)
- Theorems/lemmas identified via `**Theorem N.N.**` patterns
- Inline tags inserted: `[IMAGE:FIGURE_N]`, `[TABLE:TABLE_N]`, `[EQUATION:EQ_N]`

#### 2. NormalizedPaper → VerificationSnippets

```
NormalizedPaper
       │ segment_paper()
       ▼
[
    VerificationSnippet(
        snippet_id="2405.01133v3_sec_0",
        snippet_type=SECTION,
        location="Section 2. Some results on real operator spaces",
        content="…",    # ~2000 chars from the section
    ),
    VerificationSnippet(
        snippet_id="2405.01133v3_eq_0",
        snippet_type=EQUATION,
        location="Equation 1",
        content="Context: …\nEquation:\nY^{**} \\cong E^{\\dagger\\dagger}",
        metadata={"latex": "Y^{**} \\cong E^{\\dagger\\dagger}", …},
    ),
    VerificationSnippet(
        snippet_id="2405.01133v3_thm_0",
        snippet_type=THEOREM,
        location="**Theorem 1.1.**",
        content="Statement: If X is a real Banach space …",
    ),
    …  # 125 snippets total from this paper
]
```

Key transformations:
- Long sections split into overlapping chunks (configurable size)
- Each equation, image, theorem becomes its own snippet
- Snippets carry context (text before/after) for LLM comprehension
- Location descriptors enable ground-truth alignment later

#### 3. VerificationSnippet → VerificationResult

```
VerificationSnippet
       │ select_verifier_name()  →  "math_equation" | "vision" | "text"
       │ registry.get(name)      →  MathEquationVerifier | VisionVerifier | TextVerifier
       │ verifier.verify(snippet)
       ▼
EquationVerificationResult(
    snippet_id="2405.01133v3_eq_0",
    verifier_name="math_equation",
    status=VALID,
    error_detected=False,
    confidence=0.85,
    sympy_code="from sympy import *\nx = symbols('x')\nlhs = …\nprint(simplify(lhs-rhs))",
    execution_output="0",
    return_code=0,
)
```

Math verifier workflow:
1. LLM prompt: "Convert this LaTeX into SymPy code: `Y^{**} \\cong E^{\\dagger\\dagger}`"
2. LLM response: `{"sympy_code": "from sympy import *\n…", "unverifiable": false}`
3. Code written to temp file → `subprocess.run()` with 10s timeout
4. Output parsed: `"0"` → VALID, non-zero → INVALID, exception → MALFORMED

Vision verifier:
1. System prompt: "You are a scientific figure analysis expert…"
2. Image sent alongside text prompt (multimodal API call)
3. Response parsed as JSON: `{"error_detected": bool, "confidence": float, …}`

Text verifier:
1. System prompt: "You are a scientific text analysis expert…"
2. Snippet content sent with "Check for logical contradictions, unsupported claims…"
3. Response parsed as JSON

#### 4. Results → Predictions → Metrics

```
[VerificationResult, VerificationResult, …]
       │ _aggregate_findings()
       │   - Filter error_detected=True
       │   - Filter confidence >= threshold (per verifier)
       │   - Sort by confidence
       ▼
PaperPrediction(
    paper_id="2405.01133v3",
    predicted_errors=[
        PredictedError(
            error_category="Equation / proof",
            error_location="Lemma 3",
            confidence=0.78,
            supporting_evidence="Detected potential gap between theorem…",
        ),
        …
    ],
)
       │ match_predictions_to_ground_truth()
       │   - Fuzzy location matching
       │   - Category bonus
       │   - Assign TP / FP / FN
       ▼
[AlignedPrediction, AlignedPrediction, …]
       │ evaluate_predictions()
       ▼
EvaluationMetrics(
    accuracy=0.XX,
    precision=0.XX,
    recall=0.XX,
    f1_score=0.XX,
    by_error_category=[CategoryMetrics(…), …],
    by_error_severity=[CategoryMetrics(…), …],
    by_paper_category=[CategoryMetrics(…), …],
)
```

---

## Key Patterns

### Pattern 1: Location Parsing & Fuzzy Matching

**Problem**: Error locations in the dataset use wildly different formats:
```
"Lemma 3,4", "Equation 6", "Fig 5", "Fig. 4", "Sec 4.2.3",
"Section. 3.1.2", "Theorems 1.2, 1.3", "Fig1, Fig2",
"1.10. Corollary", "Appendix B", "Introduction", …
```

**Solution**: Multi-stage normalization:

1. **Pattern matching** — 25+ compiled regex patterns ordered most-specific-first.
2. **Type extraction** — Patterns map to `LocationType` enum: `equation`, `figure`, `lemma`, etc.
3. **Identifier splitting** — "3,4" → `["3", "4"]`; "1.2, 1.3" → `["1.2", "1.3"]`.
4. **Normalization** — `"section 4.2.3"` (canonical form, always `<type> <ids>`).
5. **Fuzzy comparison** — Jaccard overlap of identifier sets + string similarity fallback.

```python
# Equivalence classes:
"Equation 7"  ≡  "Eq. (7)"  ≡  "equation 7"      # score: 1.0
"Fig 5"       ≡  "Figure 5"  ≡  "FIGURE 5"        # score: 1.0
"Section 3.1" ≡  "Sec 3.1"   ≡  "§3.1"            # score: 0.85–1.0
"Lemma 3,4"   ∩  "Lemma 3"                         # score: 0.75 (partial overlap)
```

### Pattern 2: LLM Abstraction

**Problem**: The pipeline must work with mock LLM for testing, Anthropic Claude for production, and potentially OpenAI or local models.

**Solution**: Single entry point in `src/utils/llm.py:llm_call()`:

```python
def llm_call(
    prompt: str,
    system_prompt: str = "",
    model: str | None = None,
    image_path: str | None = None,
    config: LLMConfig | None = None,
) -> str:
    if config.provider == "mock":
        return _mock_llm_call(prompt, system_prompt, image_path, …)
    elif config.provider == "anthropic":
        return _anthropic_call(prompt, system_prompt, model, image_path, …)
    elif config.provider == "openai":
        return _openai_call(prompt, system_prompt, model, image_path, …)
```

The mock backend uses prompt keyword detection to return deterministic, plausible responses — enabling full pipeline testing without API costs.

### Pattern 3: Safe SymPy Sandbox

**Problem**: LLM-generated Python code must be executed safely.

**Solution**: Write code to temp file → execute in isolated subprocess with `subprocess.run()` constraints:

- **Timeout**: 10 seconds (configurable)
- **Output limit**: 64KB capture buffer
- **Isolation**: Runs in temp directory, no network, no persistent files
- **Cleanup**: Temp file deleted on completion or error
- **Error handling**: Timeout → `UNVERIFIABLE`; non-zero exit → `MALFORMED`

### Pattern 4: Confidence Thresholds Per Verifier

Different verifiers have different reliability profiles, so confidence thresholds are configurable per verifier:

```python
verifiers = {
    "math_equation": VerifierConfig(confidence_threshold=0.7),  # SymPy is precise
    "vision": VerifierConfig(confidence_threshold=0.6),         # Vision is noisier
    "text": VerifierConfig(confidence_threshold=0.5),           # Text is broad
}
```

Findings below threshold are excluded from predictions but preserved in `raw_results` for debugging.

---

## Thread Safety

The pipeline processes papers sequentially. Verifier instances are cached in the orchestrator (`_verifier_instances` dict) for reuse across snippets within a paper. The design is single-threaded and safe:

- Each `verify()` call creates its own temp files (sandbox) and API calls (LLM)
- No shared mutable state between verifier instances
- The registry is read-only at runtime (written only at startup)

---

## Error Handling

Errors at each layer are handled locally with logging:

| Layer | Error Strategy |
|-------|---------------|
| Parser | Log + raise (malformed data) |
| Segmenter | Log + skip malformed elements |
| Verifier | Catch exceptions → `SKIPPED` status with reasoning |
| LLM Call | Timeout/SyntaxError → `UNVERIFIABLE` status |
| Sandbox | Timeout → `SandboxTimeoutError`; crash → `MALFORMED` |
| Orchestrator | Catch exceptions → empty `PaperPrediction` with paper_id |
| Evaluation | Graceful with `zero_division=0` |
| CLI | Rich-styled error messages + non-zero exit code |
