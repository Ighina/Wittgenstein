# Extending the Pipeline

This guide covers all extension points in the Paperena Verification pipeline. Each section provides a step-by-step walkthrough with code examples.

---

## Extension Points Overview

| What You Want to Add | Files to Touch | Difficulty |
|----------------------|---------------|------------|
| New verifier | 1 new file + 2 lines in existing files | Easy |
| New Claude Code skill | 1 new SKILL.md + optional MCP tool | Easy |
| New snippet type | 1 file + routing config | Medium |
| New evaluation metric | 1 file + models | Medium |
| New content type in parser | 1 file | Medium |
| New LLM backend | 1 file | Easy |
| New CLI command | 1 function in main.py | Easy |
| New MCP tool | 1 function in server.py | Easy |

---

## Adding a New Verifier

This is the primary extension point. Verifiers are self-contained modules that analyze a specific type of content.

> **Note.** `StatisticalVerifier` and `CitationVerifier` now ship as **real**
> modules (`src/verifiers/statistical_verifier.py`,
> `src/verifiers/citation_verifier.py`) — the real statistical verifier is
> *deterministic* (LLM extracts closed arithmetic, Python recomputes via
> `safe_eval`), not the simple LLM-only version sketched below. The walkthrough
> here remains a useful generic template for any new LLM-backed verifier. The
> default registry already contains six verifiers: `math_equation`, `vision`,
> `text`, `statistical`, `citation`, `triage`. For triage-based routing see
> `docs/UNCERTAINTY_ORCHESTRATION.md`.

### Template Example: a simple LLM verifier

**Goal**: Verify statistical claims in papers (p-values, confidence intervals, sample sizes).

#### Step 1: Create the Verifier Class

```python
# src/verifiers/statistical_verifier.py
"""Verifies statistical claims in scientific papers."""

from __future__ import annotations

import time
from typing import Optional

from loguru import logger

from src.config import PipelineConfig, default_config
from src.models import (
    BaseVerificationResult,
    VerificationSnippet,
    VerificationStatus,
)
from src.verifiers.base import BaseVerifier


STATISTICAL_SYSTEM_PROMPT = """You are a statistical methodology expert.
Your task is to examine statistical claims in a scientific paper and identify errors.

## What to Check

1. **p-value consistency**: Do reported p-values match the described test?
2. **Sample size**: Is the sample size adequate for the claimed effect?
3. **Confidence intervals**: Are CIs correctly calculated?
4. **Multiple comparisons**: Are corrections applied where needed?
5. **Effect size**: Is the effect size reported and interpreted correctly?

## Output Format

Return a JSON object:
```json
{
  "error_detected": true,
  "confidence": 0.85,
  "reasoning": "Detailed explanation...",
  "predicted_error_category": "Statistical reporting"
}
```
"""


class StatisticalVerifier(BaseVerifier):
    """Verifies statistical methodology in text snippets."""

    name: str = "statistical"

    def verify(
        self,
        snippet: VerificationSnippet,
    ) -> BaseVerificationResult:
        start_time = time.monotonic()

        logger.debug(f"Verifying statistics in: {snippet.location}")

        if not self.can_verify(snippet):
            return self._make_result(
                snippet_id=snippet.snippet_id,
                status=VerificationStatus.SKIPPED,
                execution_time_ms=(time.monotonic() - start_time) * 1000,
            )

        prompt = self._build_prompt(snippet)

        try:
            response = self._call_llm_json(
                prompt=prompt,
                system_prompt=STATISTICAL_SYSTEM_PROMPT,
            )

            error_detected = response.get("error_detected", False)
            confidence = float(response.get("confidence", 0.0))

            return self._make_result(
                snippet_id=snippet.snippet_id,
                status=(
                    VerificationStatus.ERROR_DETECTED if error_detected
                    else VerificationStatus.NO_ERROR
                ),
                error_detected=error_detected,
                confidence=confidence,
                reasoning=response.get("reasoning", ""),
                predicted_error_category=response.get("predicted_error_category"),
                execution_time_ms=(time.monotonic() - start_time) * 1000,
            )

        except Exception as exc:
            logger.error(f"Statistical verification failed: {exc}")
            return self._make_result(
                snippet_id=snippet.snippet_id,
                status=VerificationStatus.UNVERIFIABLE,
                reasoning=f"Verification error: {exc}",
                execution_time_ms=(time.monotonic() - start_time) * 1000,
            )

    def can_verify(self, snippet: VerificationSnippet) -> bool:
        """Only verify snippets that contain statistical content."""
        if not self.verifier_config.enabled:
            return False

        statistical_keywords = [
            "p-value", "p value", "confidence interval", "sample size",
            "t-test", "anova", "chi-square", "regression", "effect size",
            "standard deviation", "standard error", "significance",
        ]
        content_lower = snippet.content.lower()
        return any(kw in content_lower for kw in statistical_keywords)

    def _build_prompt(self, snippet: VerificationSnippet) -> str:
        return (
            f"Analyze the following text for statistical errors.\n\n"
            f"Location: {snippet.location}\n\n"
            f"Content:\n{snippet.content[:3000]}\n\n"
            f"Return ONLY the JSON response."
        )
```

#### Step 2: Register the Verifier

Edit `src/orchestrator/router.py`:

```python
# Add this import at the top
from src.verifiers.statistical_verifier import StatisticalVerifier

# Add this line in create_default_registry()
def create_default_registry() -> VerifierRegistry:
    registry = VerifierRegistry()
    registry.register("math_equation", MathEquationVerifier)
    registry.register("vision", VisionVerifier)
    registry.register("text", TextVerifier)
    registry.register("statistical", StatisticalVerifier)
    registry.register("citation", CitationVerifier)
    registry.register("triage", TriageVerifier)
    registry.register("my_new_verifier", MyNewVerifier)  # ← your addition
    return registry
```

> To make a new verifier reachable in **uncertainty mode**, also add its route to
> `config.triage_route_map` (e.g. `"my_route": "my_new_verifier"`) and emit that
> route from the triage prompt.

#### Step 3: Add Routing

In `src/config.py`, add entries to the default routing table:

```python
verifier_routing: dict[str, str] = field(
    default_factory=lambda: {
        # … existing entries …
        "SECTION": "text",
        "PARAGRAPH": "text",
        # New: route sections with statistical content
        # Note: can_verify() does the actual filtering
    }
)
```

Alternatively, if you want to route by snippet type:

```python
# In src/models.py, add to SnippetType:
class SnippetType(str, Enum):
    # … existing …
    STATISTICAL = "STATISTICAL"   # ← NEW

# In src/config.py:
verifier_routing = {
    "STATISTICAL": "statistical",
}
```

#### Step 4: Configure Threshold

In `src/config.py`, add verifier config:

```python
def __post_init__(self) -> None:
    if not self.verifiers:
        self.verifiers = {
            "math_equation": VerifierConfig(confidence_threshold=0.7),
            "vision": VerifierConfig(confidence_threshold=0.6),
            "text": VerifierConfig(confidence_threshold=0.5),
            "statistical": VerifierConfig(confidence_threshold=0.65),  # ← NEW
        }
```

#### Step 5: Test

```python
from src.models import SnippetType, VerificationSnippet
from src.verifiers.statistical_verifier import StatisticalVerifier

snippet = VerificationSnippet(
    snippet_id="test",
    snippet_type=SnippetType.SECTION,
    paper_id="test",
    location="Section Results",
    content="The p-value was 0.03 with a sample size of n=12…",
)

verifier = StatisticalVerifier()
result = verifier.verify(snippet)
print(result.status, result.confidence)
```

**That's it.** No changes to the orchestrator, CLI, or evaluation code.

---

## Adding a New LLM Backend

To add support for a new LLM provider (e.g., a local model via Ollama):

```python
# src/utils/llm.py

def _ollama_call(
    prompt: str,
    system_prompt: str = "",
    model: str = "llama3",
    image_path: Optional[str] = None,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    **_kwargs: Any,
) -> str:
    """Call a local Ollama model."""
    import requests

    payload = {
        "model": model,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
        },
    }

    response = requests.post(
        "http://localhost:11434/api/generate",
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"]


# Add the branch in llm_call():
def llm_call(…):
    # …
    if provider == "ollama":
        return _ollama_call(…)
    # …
```

Then set `provider = "ollama"` in `LLMConfig`.

---

## Adding a New Snippet Type

To add a new snippet type (e.g., for citation analysis):

### Step 1: Add to SnippetType Enum

```python
# src/models.py
class SnippetType(str, Enum):
    # … existing …
    CITATION = "CITATION"   # ← NEW
```

### Step 2: Produce Snippets in Segmenter

```python
# src/segmentation/segmenter.py

def _segment_citations(paper: NormalizedPaper) -> list[VerificationSnippet]:
    """Extract citation/bibliography sections as snippets."""
    snippets = []
    for section in paper.sections:
        if any(kw in section.section_title.lower()
               for kw in ["reference", "bibliography", "citation"]):
            snippets.append(VerificationSnippet(
                snippet_id=f"{paper.paper_id}_cite",
                snippet_type=SnippetType.CITATION,
                paper_id=paper.paper_id,
                location=f"References: {section.section_title}",
                content=section.content,
            ))
    return snippets


def segment_paper(paper, config=None):
    snippets = []
    # … existing …
    snippets.extend(_segment_citations(paper))   # ← NEW
    return snippets
```

### Step 3: Add Routing

```python
# src/config.py
verifier_routing = {
    # … existing …
    "CITATION": "citation",   # ← NEW
}
```

---

## Adding a New Evaluation Metric

To add a custom metric (e.g., Mean Reciprocal Rank for error ranking):

### Step 1: Add to the Model

```python
# src/models.py
class EvaluationMetrics(BaseModel):
    # … existing fields …
    mean_reciprocal_rank: float = 0.0   # ← NEW
```

### Step 2: Implement Computation

```python
# src/evaluation/metrics.py

def _compute_mrr(aligned: list[AlignedPrediction]) -> float:
    """Compute Mean Reciprocal Rank for error ranking."""
    # Group by paper
    by_paper = {}
    for a in aligned:
        if a.paper_id not in by_paper:
            by_paper[a.paper_id] = []
        by_paper[a.paper_id].append(a)

    reciprocal_ranks = []
    for paper_id, preds in by_paper.items():
        # Sort by confidence descending
        sorted_preds = sorted(preds, key=lambda p: p.predicted.confidence, reverse=True)
        for rank, pred in enumerate(sorted_preds, start=1):
            if pred.is_true_positive:
                reciprocal_ranks.append(1.0 / rank)
                break

    if not reciprocal_ranks:
        return 0.0
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def evaluate_predictions(aligned, ground_truth_df):
    # … existing code …
    mrr = _compute_mrr(aligned)
    metrics = EvaluationMetrics(
        # … existing fields …
        mean_reciprocal_rank=mrr,   # ← NEW
    )
    return metrics
```

### Step 3: Add to Report

```python
# src/reporting/reporter.py

def _build_summary(…):
    # … existing …
    lines.append(f"| Mean Reciprocal Rank | {metrics.mean_reciprocal_rank:.4f} |")
```

---

## Adding a New Content Parser Pattern

To support a new content format in paper_content (e.g., code blocks):

```python
# src/parser/content_parser.py

CODE_BLOCK_PATTERN = re.compile(
    r"```(\w+)?\n(.*?)```",
    re.DOTALL,
)

class CodeBlock(BaseModel):
    id: str
    language: Optional[str] = None
    code: str
    context_before: Optional[str] = None
    context_after: Optional[str] = None


def _extract_code_blocks(
    raw_items: list[RawContentItem],
    paper_id: str,
) -> list[CodeBlock]:
    """Extract code blocks from text content."""
    blocks = []
    full_text = "\n".join(
        item.text or "" for item in raw_items if item.text
    )

    for idx, match in enumerate(CODE_BLOCK_PATTERN.finditer(full_text)):
        language = match.group(1)
        code = match.group(2).strip()
        blocks.append(CodeBlock(
            id=f"{paper_id}_code_{idx}",
            language=language,
            code=code,
            context_before=full_text[max(0, match.start()-200):match.start()],
            context_after=full_text[match.end():match.end()+200],
        ))

    return blocks
```

Then call `_extract_code_blocks()` from `parse_paper_content()`.

---

## Adding a New CLI Command

```python
# main.py

@app.command()
def verify_category(
    category: str = typer.Argument(..., help="Paper category to filter by."),
    parquet_path: str = typer.Option("data/train-00000-of-00001.parquet", "--parquet"),
) -> None:
    """Verify all papers in a specific category."""
    setup_logging("INFO")

    df = pd.read_parquet(parquet_path)
    df = df[df["paper_category"] == category]

    console.print(f"Verifying [bold]{len(df)}[/bold] papers in category: {category}")

    orchestrator = VerificationOrchestrator()
    for _, row in df.iterrows():
        paper = parse_paper_content(
            paper_id=str(row["doi/arxiv_id"]),
            title=row["title"],
            paper_category=row["paper_category"],
            paper_content=row["paper_content"],
        )
        prediction = orchestrator.run(paper)
        console.print(
            f"  {prediction.paper_id}: "
            f"{prediction.errors_detected} errors detected"
        )
```

---

## Extension Checklist

When adding a new verifier, verify all of these:

- [ ] Verifier class extends `BaseVerifier` and sets `name`
- [ ] `verify()` returns a `BaseVerificationResult` (or subclass)
- [ ] `can_verify()` checks `verifier_config.enabled`
- [ ] Registered in `create_default_registry()` in `router.py`
- [ ] Routing entry added in `config.py` (if a new snippet type)
- [ ] Verifier config added to `PipelineConfig.__post_init__()`
- [ ] Verifier imported in `src/verifiers/__init__.py` (optional, for convenience)
- [ ] Unit tests added in `tests/` covering `verify()` and `can_verify()`
- [ ] LLM prompt includes clear instructions and JSON output format

---

## Design Rules for Verifiers

1. **Statelessness** — Verifier instances may be reused across snippets. Don't store per-snippet state on `self`.
2. **Error resilience** — Catch all exceptions in `verify()` and return a result with `UNVERIFIABLE` status. Never let exceptions propagate to the orchestrator.
3. **Prompt quality** — System prompts should be specific about what to check and what JSON format to return. The pipeline relies on `parse_json_response()` to extract structured data.
4. **Confidence calibration** — Use `confidence_threshold` from `self.verifier_config` for filtering. Different verifiers naturally have different confidence profiles.
5. **Time tracking** — Record `execution_time_ms` for observability.

---

## Adding a New Claude Code Skill

In addition to Python verifiers, the pipeline supports Claude Code skills — self-contained verification agents defined as markdown files. Skills are the recommended way to add a new verifier when the logic is primarily prompt-based (LLM reasoning + MCP tools for deterministic steps).

### Skill Structure

Each skill lives in `.claude/skills/<skill-name>/SKILL.md` and follows this template:

```markdown
---
name: verify-example
description: Verify example-type content in scientific papers.
---

You are an expert at verifying [specific content type] in scientific papers.

## What to Check

1. [Check 1 description]
2. [Check 2 description]
3. [Check 3 description]

## Available Tools

Use these MCP tools for deterministic operations:
- `segment_paper` — Get paper snippets
- `run_sympy_sandbox_exec` — Execute SymPy code (for math)
- `safe_arithmetic_eval` — Evaluate arithmetic expressions

## Output Format

Return findings as structured JSON:
```json
{
  "error_detected": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "Explanation...",
  "predicted_error_category": "Category name"
}
```

## Verification Rules

- Only flag errors that are provable from the snippet alone
- Default to NO_ERROR if uncertain
- For math: convert LaTeX to SymPy, execute in sandbox, interpret verdict
```

### Step 1: Create the Skill File

```bash
mkdir -p .claude/skills/verify-example
# Write SKILL.md following the template above
```

### Step 2: Add MCP Tools (if needed)

If your skill needs a new deterministic tool, add it to `mcp-server/server.py`:

```python
@mcp.tool()
def my_new_tool(param: str) -> dict[str, Any]:
    """Description of what this tool does."""
    # Your implementation
    return {"result": "..."}
```

The tool becomes automatically available to all skills.

### Step 3: Wire into the Orchestrator

To make your skill reachable from `/verify-paper`, update the orchestrator skill's routing logic in `.claude/skills/verify-paper/SKILL.md` to route the appropriate snippet types to your new skill.

For **uncertainty mode**, also update the triage skill's route suggestions in `.claude/skills/verify-triage/SKILL.md` so it can suggest your verifier for relevant snippets.

### Step 4: Add MCP Permissions

If your skill uses new MCP tools, ensure `.claude/settings.json` grants permission:

```json
{
  "permissions": {
    "allow": [
      "mcp__paperena__my_new_tool"
    ]
  }
}
```

### Step 5: Test

```bash
# Test the skill directly
claude -p "/verify-example Check this content: ..."

# Or as part of the full pipeline
claude -p "/verify-paper Verify paper 2405.01133v3 from data/train-00000-of-00001.parquet"
```

### Skill Design Rules

1. **Skills use MCP for determinism** — Any operation that can be computed (parsing, math, arithmetic) should use an MCP tool, not LLM generation
2. **Skills are self-contained** — A SKILL.md should stand alone; the orchestrator skill handles coordination
3. **Structured output** — Skills should return structured JSON findings that the orchestrator can aggregate
4. **Fail open** — On error or uncertainty, default to no error rather than fabricating a finding
5. **MCP tools are lazy** — Tools in `mcp-server/server.py` use lazy imports so the server starts fast; follow the same pattern for new tools
