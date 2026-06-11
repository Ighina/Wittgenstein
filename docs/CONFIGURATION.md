# Configuration Guide

The Paperena Verification pipeline is fully configurable through Python dataclasses in `src/config.py`. Every aspect — from file paths to LLM backend to verifier thresholds — can be adjusted without modifying source code.

---

## Quick Configuration

### Using the Default Config

```python
from src.config import default_config

# The default config uses the DeepSeek backend, standard paths, strict
# thresholds, and exhaustive orchestration. Set provider="mock" for offline use.
print(default_config.llm.provider)          # "deepseek"
print(default_config.orchestration_mode)    # "exhaustive"
```

### Overriding via Environment Variables

Two paths are controllable via environment variables:

```bash
export PAPERENA_DATA_DIR="my_data"
export PAPERENA_OUTPUT_DIR="my_outputs"
python main.py verify
```

LLM API keys use provider-specific variables:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
# or
export OPENAI_API_KEY="sk-..."
```

### Overriding Programmatically

```python
from src.config import PipelineConfig, LLMConfig, SandboxConfig

config = PipelineConfig(
    llm=LLMConfig(
        provider="anthropic",
        model="claude-opus-4-8",
        max_tokens=8192,
        temperature=0.0,
    ),
    sandbox=SandboxConfig(
        timeout_seconds=30,
        python_executable="/usr/local/bin/python3.12",
    ),
)

# Use this config with the orchestrator
from src.orchestrator.orchestrator import VerificationOrchestrator
orchestrator = VerificationOrchestrator(config=config)
```

### Loading from a Dictionary

```python
config = PipelineConfig.from_dict({
    "llm": {
        "provider": "openai",
        "model": "gpt-4o",
        "max_tokens": 8192,
    },
    "verifiers": {
        "math_equation": {"confidence_threshold": 0.8},
        "vision": {"enabled": True, "confidence_threshold": 0.65},
        "text": {"confidence_threshold": 0.55},
    },
    "verifier_routing": {
        "EQUATION": "math_equation",
        "FIGURE": "vision",
        "TABLE": "vision",
        "SECTION": "text",
        "THEOREM": "text",
        "LEMMA": "text",
    },
})
```

This format is also suitable for loading from JSON or YAML files:

```python
import json
with open("pipeline_config.json") as f:
    config = PipelineConfig.from_dict(json.load(f))
```

---

## Reference: All Configuration Options

### `PathsConfig`

Controls where the pipeline reads input and writes output.

```python
@dataclass
class PathsConfig:
    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
    parquet_file: str = "train-00000-of-00001.parquet"
```

| Option | Default | Env Variable | Description |
|--------|---------|-------------|-------------|
| `data_dir` | `data/` | `PAPERENA_DATA_DIR` | Directory containing the parquet file |
| `output_dir` | `outputs/` | `PAPERENA_OUTPUT_DIR` | Directory for generated reports |
| `parquet_file` | `train-00000-of-00001.parquet` | — | Filename within data_dir |

### `LLMConfig`

Controls the LLM backend used by all verifiers.

```python
@dataclass
class LLMConfig:
    provider: str = "deepseek"        # "mock", "anthropic", "openai", "deepseek"
    model: str = "deepseek-v4-pro"
    api_key_env: str = "DEEPSEEK_API_KEY"
    max_tokens: int = 8192
    temperature: float = 0.0
    timeout_seconds: int = 120
    num_workers: int = 8              # snippets verified concurrently per paper
    max_retries: int = 3              # retries for transient/empty responses
    retry_backoff_seconds: float = 2.0
```

| Option | Default | Description |
|--------|---------|-------------|
| `provider` | `"deepseek"` | LLM backend. `"mock"` returns deterministic responses. |
| `model` | `"deepseek-v4-pro"` | Model name (passed to the API) |
| `api_key_env` | `"DEEPSEEK_API_KEY"` | Environment variable for API key |
| `max_tokens` | `8192` | **Output**-token budget. See the warning below. |
| `temperature` | `0.0` | Sampling temperature (0.0 = deterministic) |
| `timeout_seconds` | `120` | API call timeout |
| `num_workers` | `8` | Concurrent snippet verifications per paper (1 = sequential) |
| `max_retries` | `3` | Retries on transient errors **and empty completions** |
| `retry_backoff_seconds` | `2.0` | Base for exponential backoff between retries |

> ⚠️ **`max_tokens` and reasoning models.** `max_tokens` caps the *output*. With a
> reasoning-style model (e.g. `deepseek-v4-pro`), chain-of-thought consumes this
> budget; if it is too small, `message.content` comes back **empty**. (Verified:
> a trivial prompt returns empty at `max_tokens=20` but `"OK"` at `2000`/`8192`.)
> The default is therefore `8192`. `llm_call` resolves `max_tokens` from this
> config when a caller doesn't pass one explicitly, and treats an empty response
> as a retryable error. Don't lower `max_tokens` below a few thousand.

#### Provider Details

| Provider | Required Env Var | Python Package |
|----------|-----------------|----------------|
| `mock` | None | (built-in) |
| `anthropic` | `ANTHROPIC_API_KEY` | `anthropic` (pip install) |
| `openai` | `OPENAI_API_KEY` | `openai` (pip install) |
| `deepseek` | `DEEPSEEK_API_KEY` | `openai` (OpenAI-compatible client) |

#### Mock Backend Behavior

The mock backend returns deterministic responses keyed off prompt/system-prompt
content, so the whole pipeline (including the new verifiers) runs offline:

- **triage system prompt** → uncertainty score + route (equations score high)
- **sympy / latex** → SymPy `report(...)` script for the math verifier
- **"checkable numeric" system prompt** → a statistical extraction (`checks`)
- **"novelty / attribution" system prompt** → citation finding
- **"single-call whole-paper review" system prompt** → baseline finding
- **figure / image** + **duplicat / inconsist** → vision findings
- **contradiction / logical** + **theorem / proof** → theorem-proof gap
- **Other** → no-error response

### `SandboxConfig`

Controls the SymPy subprocess execution environment.

```python
@dataclass
class SandboxConfig:
    timeout_seconds: int = 10
    max_output_bytes: int = 65536
    python_executable: str = "python3"
```

| Option | Default | Description |
|--------|---------|-------------|
| `timeout_seconds` | `10` | Maximum execution time before killing the process |
| `max_output_bytes` | `65536` | Maximum bytes to read from stdout/stderr |
| `python_executable` | `"python3"` | Python interpreter to use |

### `SegmentationConfig`

Controls how papers are split into verification snippets.

```python
@dataclass
class SegmentationConfig:
    max_snippet_chars: int = 4000
    max_section_chars: int = 8000
    overlap_chars: int = 200
```

| Option | Default | Description |
|--------|---------|-------------|
| `max_snippet_chars` | `4000` | Maximum characters per chunked snippet |
| `max_section_chars` | `8000` | Threshold for splitting a section into chunks |
| `overlap_chars` | `200` | Overlap between consecutive chunks |

### `VerifierConfig`

Per-verifier settings.

```python
@dataclass
class VerifierConfig:
    enabled: bool = True
    confidence_threshold: float = 0.6
```

| Option | Default | Description |
|--------|---------|-------------|
| `enabled` | `True` | Set to `False` to skip this verifier entirely |
| `confidence_threshold` | `0.6` | Minimum confidence to report a finding |

### `PipelineConfig`

Top-level configuration composing all of the above.

```python
@dataclass
class PipelineConfig:
    paths: PathsConfig
    llm: LLMConfig
    sandbox: SandboxConfig
    segmentation: SegmentationConfig
    verifiers: dict[str, VerifierConfig]
    verifier_routing: dict[str, str]

    # Error-detection strictness and evaluation
    strictness: str = "strict"          # "strict" | "lenient"
    use_llm_judge: bool = True          # LLM judge for prediction↔GT matching
    judge_model: str | None = None      # None → reuse llm.model

    # Orchestration mode (see docs/UNCERTAINTY_ORCHESTRATION.md)
    orchestration_mode: str = "exhaustive"   # "exhaustive" | "uncertainty"
    uncertainty_threshold: float = 0.30      # escalate snippets ≥ this
    uncertainty_budget: int | None = None    # optional cap on specialist calls
    triage_route_map: dict[str, str]         # semantic route → verifier name

    # Chunking for text-based verifiers
    verify_chunk_chars: int = 2000      # chunk content longer than this
    verify_chunk_overlap: int = 200     # overlap between chunks
```

| Option | Type | Description |
|--------|------|-------------|
| `paths` | `PathsConfig` | File system paths |
| `llm` | `LLMConfig` | LLM backend |
| `sandbox` | `SandboxConfig` | SymPy execution |
| `segmentation` | `SegmentationConfig` | Snippet sizing |
| `verifiers` | `dict[str, VerifierConfig]` | Per-verifier settings keyed by name |
| `verifier_routing` | `dict[str, str]` | SnippetType → verifier name mapping (exhaustive mode) |
| `strictness` | `str` | `"strict"` (only erratum/retraction-worthy errors, higher thresholds) or `"lenient"` |
| `use_llm_judge` | `bool` | Use an LLM judge to match predictions to ground truth (vs. fuzzy location matching) |
| `judge_model` | `str \| None` | Judge model; `None` reuses `llm.model` |
| `orchestration_mode` | `str` | `"exhaustive"` (route every snippet by type) or `"uncertainty"` (triage-first) |
| `uncertainty_threshold` | `float` | In uncertainty mode, escalate snippets with `uncertainty ≥ threshold` |
| `uncertainty_budget` | `int \| None` | Optional hard cap on specialist calls per paper (keeps the top-K) |
| `triage_route_map` | `dict[str, str]` | Triage route label → registered verifier name |
| `verify_chunk_chars` | `int` | Text/citation verifiers chunk content longer than this |
| `verify_chunk_overlap` | `int` | Character overlap between chunks |

### Default verifier set

`PipelineConfig.__post_init__` populates `verifiers` based on `strictness`. Under
`"strict"` (default): `math_equation` 0.7, `vision` 0.75, `text` 0.8,
`statistical` 0.8, `citation` 0.8, `triage` 0.0 (triage never produces a final
finding, so it is never thresholded out).

---

## Verifier Routing Table

The routing table in `PipelineConfig.verifier_routing` maps `SnippetType` values to verifier names:

```python
# Default routing
verifier_routing = {
    "EQUATION": "math_equation",    # → MathEquationVerifier
    "FIGURE": "vision",             # → VisionVerifier
    "TABLE": "vision",              # → VisionVerifier
    "SECTION": "text",              # → TextVerifier
    "SUBSECTION": "text",           # → TextVerifier
    "THEOREM": "text",              # → TextVerifier
    "LEMMA": "text",                # → TextVerifier
    "PROPOSITION": "text",          # → TextVerifier
    "ALGORITHM": "text",            # → TextVerifier
    "APPENDIX": "text",             # → TextVerifier
    "PARAGRAPH": "text",            # → TextVerifier
}
```

To route a snippet type to a custom verifier, add or change the entry:

```python
verifier_routing = {
    …
    "CITATION": "citation",         # → Custom CitationVerifier
    "EQUATION": "math_equation",    # Unchanged
    …
}
```

Unknown snippet types fall back to `"text"`.

### Triage Route Map (uncertainty mode)

In `orchestration_mode="uncertainty"`, routing is driven by the triage verifier's
*suggested route* (not the snippet type). `triage_route_map` maps each semantic
route to a registered verifier:

```python
triage_route_map = {
    "math": "math_equation", "equation": "math_equation",
    "proof": "text",
    "statistics": "statistical", "statistical": "statistical", "numeric": "statistical",
    "citation": "citation", "reference": "citation",
    "vision": "vision", "figure": "vision", "table": "vision",
    "text": "text", "logic": "text",
    "none": "",            # skip — accepted as low-risk
}
```

Unknown routes fall back to type-based routing. See
`docs/UNCERTAINTY_ORCHESTRATION.md` for the full design.

---

## Common Configuration Scenarios

### Scenario 1: Fast Testing (Mock LLM)

```python
from src.config import PipelineConfig, LLMConfig

config = PipelineConfig(
    llm=LLMConfig(provider="mock"),
)
```

No API keys needed. Deterministic responses. Fast.

### Scenario 2: Production with Anthropic Claude

```python
config = PipelineConfig(
    llm=LLMConfig(
        provider="anthropic",
        model="claude-opus-4-8",
        max_tokens=8192,
        timeout_seconds=180,
    ),
    sandbox=SandboxConfig(timeout_seconds=30),
    verifiers={
        "math_equation": VerifierConfig(confidence_threshold=0.75),
        "vision": VerifierConfig(confidence_threshold=0.65),
        "text": VerifierConfig(confidence_threshold=0.55),
    },
)
```

### Scenario 3: Equation-Only Verification

```python
# Disable vision and text verifiers
config = PipelineConfig(
    verifiers={
        "math_equation": VerifierConfig(enabled=True),
        "vision": VerifierConfig(enabled=False),
        "text": VerifierConfig(enabled=False),
    },
)
```

### Scenario 4: Custom Snippet Size for Large Papers

```python
config = PipelineConfig(
    segmentation=SegmentationConfig(
        max_snippet_chars=8000,   # Larger snippets
        max_section_chars=16000,  # Fewer splits
        overlap_chars=400,
    ),
)
```

### Scenario 5: Low-Confidence Flagging (Sensitive Mode)

```python
# Lower thresholds to catch more potential issues
config = PipelineConfig(
    verifiers={
        "math_equation": VerifierConfig(confidence_threshold=0.5),
        "vision": VerifierConfig(confidence_threshold=0.4),
        "text": VerifierConfig(confidence_threshold=0.3),
    },
)
```

### Scenario 6: Uncertainty-Driven Orchestration

Triage every snippet, then run specialists only where error density is high:

```python
config = PipelineConfig(
    orchestration_mode="uncertainty",
    uncertainty_threshold=0.30,   # raise to spend fewer specialist calls
    uncertainty_budget=10,        # optional hard cap per paper
)
```

CLI equivalent:

```bash
python main.py verify-one 2405.01133v3 --mode uncertainty --uncertainty-threshold 0.3
```

See `docs/UNCERTAINTY_ORCHESTRATION.md` for the pipeline, the triage verifier,
the statistical/citation verifiers, chunking, and the threshold-sweep tool.

### Scenario 7: Loading Config from JSON File

```json
{
    "llm": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "max_tokens": 8192
    },
    "segmentation": {
        "max_snippet_chars": 5000
    },
    "verifiers": {
        "math_equation": {"confidence_threshold": 0.8},
        "vision": {"enabled": true, "confidence_threshold": 0.7},
        "text": {"confidence_threshold": 0.6}
    }
}
```

```python
import json
from src.config import PipelineConfig

with open("pipeline_config.json") as f:
    config = PipelineConfig.from_dict(json.load(f))
```

---

## Verifying Configuration

To check the active configuration:

```python
from src.config import default_config

c = default_config
print(f"LLM provider: {c.llm.provider}")
print(f"LLM model: {c.llm.model}")
print(f"Sandbox timeout: {c.sandbox.timeout_seconds}s")
print(f"Max snippet: {c.segmentation.max_snippet_chars} chars")
print(f"Verifiers: {list(c.verifiers.keys())}")
print(f"Routing: {c.verifier_routing}")
for name, vc in c.verifiers.items():
    print(f"  {name}: enabled={vc.enabled}, threshold={vc.confidence_threshold}")
```

---

## Claude Code Configuration (`.claude/settings.json`)

When driving the pipeline through Claude Code skills, configuration lives in `.claude/settings.json` rather than Python dataclasses. This file tells Claude Code about the MCP server and skill locations.

### MCP Server Registration

```json
{
  "mcpServers": {
    "paperena": {
      "command": "python3",
      "args": ["server.py"],
      "cwd": "mcp-server",
      "env": {
        "PYTHONPATH": ".."
      }
    }
  }
}
```

This registers the Paperena MCP server, making all 9 tools (`parse_paper`, `segment_paper`, `run_sympy_sandbox_exec`, `safe_arithmetic_eval`, `check_numeric_claim`, `get_paper_from_dataset`, `list_papers_in_dataset`, `analyze_dataset_schema`) available to Claude Code skills.

### Skills Directory

```json
{
  "skillsDirectory": ".claude/skills"
}
```

Points Claude Code to the directory containing the 7 verification skills. Each skill is a subdirectory with a `SKILL.md` file.

### Permissions

The settings file also controls which tool calls are allowed without prompting:

```json
{
  "permissions": {
    "allow": [
      "mcp__paperena__*",
      "Bash(python3:*)",
      "Skill(verify-*)"
    ]
  }
}
```

- `mcp__paperena__*` — Allow all Paperena MCP tools without prompting
- `Skill(verify-*)` — Allow invoking any verification skill

### Scenario 8: Batch Verification via Claude Code

```bash
# Verify 5 papers with Claude Code skills
python scripts/batch_verify.py --papers 5 --mode uncertainty

# Evaluate results after batch run
python scripts/batch_verify.py --papers 5 --mode exhaustive --evaluate

# List available papers
python scripts/batch_verify.py --list
```

The batch scripts use Claude Code CLI under the hood, so they inherit the MCP server and skill configuration from `.claude/settings.json`. The Python wrapper (`batch_verify.py`) adds progress bars and optional post-batch evaluation against ground truth.
