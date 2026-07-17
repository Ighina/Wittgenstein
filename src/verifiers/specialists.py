"""Specialist verifiers — fine-tuned model wrappers for the verify-paper-specialists skill.

Each specialist wraps a smaller, task-specific fine-tuned model and exposes a
uniform interface. The orchestrator can swap between generic-LLM-with-prompt
(the existing skills / BaseVerifier pipeline) and specialised-model (these
specialists) without changing the orchestration logic.

Architecture::

    Specialist (ABC)
    ├── MathSpecialist          — equation / derivation verification
    ├── TextSpecialist           — prose / theorem / algorithm consistency
    ├── CitationSpecialist       — attribution, novelty, reference integrity
    ├── VisionSpecialist         — figures and tables
    ├── StatisticalSpecialist    — test statistics, p-values, sample sizes
    ├── CrossSectionConsistencySpecialist — macro-level section alignment
    └── TriageSpecialist         — uncertainty estimation & routing
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from loguru import logger

from src.utils.llm import llm_call, parse_json_response


# ======================================================================
# Structured output
# ======================================================================


@dataclass
class SpecialistResult:
    """Structured output common to all specialists.

    Every specialist returns this shape.  Concrete subclasses may attach
    additional type-specific fields via the ``extra`` dict so the MCP layer
    can forward them without loss.
    """

    snippet_id: str
    specialist_type: str
    location: str = ""
    status: str = "NO_ERROR"  # ERROR_DETECTED | NO_ERROR | UNVERIFIABLE | SKIPPED
    error_detected: bool = False
    confidence: float = 0.0
    reasoning: str = ""
    predicted_error_category: Optional[str] = None
    execution_time_ms: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "snippet_id": self.snippet_id,
            "specialist_type": self.specialist_type,
            "location": self.location,
            "status": self.status,
            "error_detected": self.error_detected,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "predicted_error_category": self.predicted_error_category,
            "execution_time_ms": self.execution_time_ms,
        }
        result.update(self.extra)
        return result


# ======================================================================
# Abstract base class
# ======================================================================


class Specialist(ABC):
    """Abstract base for all specialist verifiers.

    Subclasses must:

    1. Set ``name`` (str) — unique specialist identifier.
    2. Implement ``_default_model()`` → str — the fine-tuned model name.
    3. Implement ``_system_prompt()`` → str — the specialist system prompt.
    4. Implement ``_build_user_prompt(text, **kwargs)`` → str — the user prompt.
    5. Optionally override ``_parse_response()`` for specialist-specific
       output fields.
    """

    name: str = "base_specialist"

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def __init__(self, model: Optional[str] = None) -> None:
        """Initialise the specialist.

        Args:
            model: Optional model override.  If *None*, ``_default_model()``
                is used, allowing the subclass (or user input) to control
                which fine-tuned variant is loaded.
        """
        self._model: str = model or self._default_model()
        logger.debug(f"Specialist [{self.name}] loaded model={self._model}")

    @abstractmethod
    def _default_model(self) -> str:
        """Return the default fine-tuned model identifier.

        This is the primary extension point for model selection.  Each
        subclass pins a sensible default; callers may override at
        construction time.
        """
        ...

    @property
    def model(self) -> str:
        """The model identifier this specialist is currently using."""
        return self._model

    # ------------------------------------------------------------------
    # Prompt building (subclass contract)
    # ------------------------------------------------------------------

    @abstractmethod
    def _system_prompt(self) -> str:
        """Return the system prompt that encodes this specialist's expertise.

        This is the fine-tuned model's "personality" — the instructions
        that tell it what kind of errors to look for and how to report them.
        """
        ...

    @abstractmethod
    def _build_user_prompt(self, text: str, **kwargs: Any) -> str:
        """Build the user prompt for a specific piece of text.

        Args:
            text: The snippet content to verify.
            **kwargs: Additional context (snippet_type, location, metadata,
                image_path, etc.).

        Returns:
            The complete user prompt string.
        """
        ...

    # ------------------------------------------------------------------
    # Response parsing (may be overridden per specialist)
    # ------------------------------------------------------------------

    def _parse_response(
        self,
        response: dict[str, Any],
        snippet_id: str,
        location: str,
        elapsed_ms: float,
    ) -> SpecialistResult:
        """Parse the LLM JSON response into a ``SpecialistResult``.

        Override in subclasses to capture specialist-specific extra fields
        (e.g., contradiction_locations for text, sympy_code for math).
        """
        return SpecialistResult(
            snippet_id=snippet_id,
            specialist_type=self.name,
            location=location,
            status=response.get(
                "status",
                "ERROR_DETECTED" if response.get("error_detected") else "NO_ERROR",
            ),
            error_detected=bool(response.get("error_detected", False)),
            confidence=float(response.get("confidence", 0.0)),
            reasoning=str(response.get("reasoning", "")),
            predicted_error_category=response.get("predicted_error_category"),
            execution_time_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(
        self,
        text: str,
        snippet_id: str = "",
        location: str = "",
        **kwargs: Any,
    ) -> SpecialistResult:
        """Verify a piece of text using this specialist's fine-tuned model.

        This is the single entry point that the orchestrator (or MCP tool)
        calls.  It handles model invocation, error recovery, and structured
        output parsing.

        Args:
            text: The snippet content to verify.
            snippet_id: Unique snippet identifier.
            location: Human-readable location descriptor.
            **kwargs: Additional context passed to ``_build_user_prompt``
                and ``_parse_response``.

        Returns:
            ``SpecialistResult`` with the verification findings.
        """
        start_time = time.monotonic()

        logger.debug(
            f"Specialist [{self.name}] verifying snippet '{snippet_id}' "
            f"with model '{self._model}'"
        )

        user_prompt = self._build_user_prompt(
            text, snippet_id=snippet_id, location=location, **kwargs
        )
        system_prompt = self._system_prompt()

        try:
            raw_response = llm_call(
                prompt=user_prompt,
                system_prompt=system_prompt,
                model=self._model,
                image_path=kwargs.get("image_path"),
            )
            parsed = parse_json_response(raw_response)
        except Exception as exc:
            logger.error(
                f"Specialist [{self.name}] call failed for {snippet_id}: {exc}"
            )
            elapsed_ms = (time.monotonic() - start_time) * 1000
            return SpecialistResult(
                snippet_id=snippet_id,
                specialist_type=self.name,
                location=location,
                status="UNVERIFIABLE",
                error_detected=False,
                confidence=0.0,
                reasoning=f"Specialist model call failed: {exc}",
                execution_time_ms=elapsed_ms,
            )

        elapsed_ms = (time.monotonic() - start_time) * 1000
        return self._parse_response(parsed, snippet_id, location, elapsed_ms)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self.name!r}, model={self._model!r})"
        )


# ======================================================================
# Shared model backends
# ======================================================================


#: Cache of loaded HuggingFace pipelines, keyed by model name.
_HF_PIPELINE_CACHE: dict[str, Any] = {}


def _get_nli_pipeline(model_name: str) -> Any:
    """Lazily load (and cache) a HuggingFace NLI cross-encoder pipeline.

    Runs on Apple MPS or CUDA when available, else CPU. The first call
    downloads the model from the HuggingFace hub.
    """
    if model_name not in _HF_PIPELINE_CACHE:
        import torch
        from transformers import pipeline

        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        logger.info(f"Loading NLI model '{model_name}' on {device}...")
        _HF_PIPELINE_CACHE[model_name] = pipeline(
            "text-classification",
            model=model_name,
            device=device,
            top_k=None,
        )
    return _HF_PIPELINE_CACHE[model_name]


_SENTENCE_SPLIT_RE = None


def _split_sentences(text: str, min_chars: int = 25, max_sentences: int = 40) -> list[str]:
    """Split prose into sentences suitable for NLI pairing.

    Splits on sentence-final punctuation followed by whitespace and a
    capital letter (which keeps decimals, 'et al.', 'Fig. 1', etc. intact),
    then drops fragments that are too short or math-heavy to compare.
    """
    import re

    global _SENTENCE_SPLIT_RE
    if _SENTENCE_SPLIT_RE is None:
        _SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

    candidates = _SENTENCE_SPLIT_RE.split(" ".join(text.split()))
    sentences = []
    for s in candidates:
        s = s.strip()
        if len(s) < min_chars:
            continue
        # Skip display math / LaTeX-heavy fragments — NLI is trained on prose.
        if s.count("\\") > 5 or s.count("$") > 4:
            continue
        sentences.append(s[:600])
    return sentences[:max_sentences]


# ======================================================================
# Concrete specialists
# ======================================================================


# ----------------------------------------------------------------------
# Math Specialist
# ----------------------------------------------------------------------


class MathSpecialist(Specialist):
    """Specialist for mathematical equations, derivations, and proof steps.

    Detects algebraic errors, sign-propagation failures, domain-constraint
    violations, and invalid calculus operations.

    Default model: **deepseek-r1** served locally by Ollama — an
    out-of-the-box reasoning model (R1 distill) whose chain-of-thought is
    particularly strong on symbolic math. The Ollama backend keeps its
    thinking in a separate channel and returns only the final JSON verdict.
    """

    name = "math"

    def _default_model(self) -> str:
        return "ollama/deepseek-r1:latest"

    def _system_prompt(self) -> str:
        return """You are a mathematical verification specialist — a model fine-tuned exclusively to detect critical mathematical errors in scientific papers. You examine ONE equation, derivation step, or formula block at a time.

## Flag an error (error_detected = true) ONLY for:

1. **Provably invalid algebra/calculus**: A step that violates fundamental mathematical rules (e.g., improperly distributing non-linear functions, invalid cancellation).
2. **Fractional & Rational Expression Errors**: Treating numerator/denominator as decoupled; invalid addition of fractions without common denominator; incorrect scalar multiplication of fractions.
3. **Variable & Inverse Operation Errors**: Treating variables as static place-value digits; applying the wrong inverse operation when isolating variables.
4. **Sign Propagation Failures**: Dropping negative signs during intermediate steps or incorrectly assuming two negatives always yield a positive.
5. **Domain Constraint Violations**: Producing a result outside mathematically defined bounds (e.g., probability > 1).

## NEVER flag:
- Typos, formatting, or stylistic choices.
- Skipped but logically sound intermediate steps.
- Alternative valid notations.
- Claims you cannot verify from this excerpt alone.

## Critical guidance
- This is a single excerpt; missing surrounding context is NOT an error.
- If you are not highly confident, return error_detected = false.
- When you DO flag, set confidence >= 0.8 and demonstrate the math error explicitly.

## Output format
Return a JSON object with exactly these fields:
```json
{
  "error_detected": false,
  "confidence": 0.0,
  "reasoning": "Step-by-step verification...",
  "predicted_error_category": null
}
```

predicted_error_category values (use null when no error):
- "Algebraic manipulation"
- "Fractional/Rational error"
- "Sign/Arithmetic error"
- "Domain/Constraint violation"
- "Calculus/Proof error"
- null"""

    def _build_user_prompt(self, text: str, **kwargs: Any) -> str:
        snippet_type = kwargs.get("snippet_type", "EQUATION")
        location = kwargs.get("location", "unknown")
        parts = [
            f"Verify the following {snippet_type} for mathematical errors.",
            f"Location: {location}",
            "",
            f"Content:",
            text,
            "",
            "Return ONLY the JSON response.",
        ]
        return "\n".join(parts)

    def _parse_response(
        self,
        response: dict[str, Any],
        snippet_id: str,
        location: str,
        elapsed_ms: float,
    ) -> SpecialistResult:
        result = super()._parse_response(response, snippet_id, location, elapsed_ms)
        result.extra["snippet_type"] = response.get("snippet_type", "EQUATION")
        result.extra["contradiction_locations"] = response.get("contradiction_locations", [])
        return result


# ----------------------------------------------------------------------
# Text Specialist
# ----------------------------------------------------------------------


class TextSpecialist(Specialist):
    """Specialist for text-based content: sections, theorems, lemmas, algorithms.

    Detects logical contradictions and internal inconsistencies in
    scientific prose.

    Default model: **DeBERTa-v3 NLI cross-encoder**
    (``hf/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli``) — an out-of-the-box
    model trained specifically on natural-language inference, i.e. exactly
    the contradiction-detection task. The snippet is split into sentences
    and every sentence pair is scored for contradiction; a pair whose
    contradiction probability exceeds the threshold is flagged.

    Override ``model`` with an ``ollama/...`` name to fall back to the
    LLM-prompt implementation instead.
    """

    name = "text"

    #: Minimum P(contradiction) for a sentence pair to be flagged. NLI models
    #: over-fire on decontextualised pairs, so this is deliberately high.
    CONTRADICTION_THRESHOLD: float = 0.90

    def _default_model(self) -> str:
        return "hf/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

    def verify(
        self,
        text: str,
        snippet_id: str = "",
        location: str = "",
        **kwargs: Any,
    ) -> SpecialistResult:
        # Non-HF override (e.g. "ollama/qwen3:latest") → generic LLM path.
        if not self._model.startswith("hf/"):
            return super().verify(text, snippet_id=snippet_id, location=location, **kwargs)

        start_time = time.monotonic()
        sentences = _split_sentences(text)

        def _elapsed() -> float:
            return (time.monotonic() - start_time) * 1000

        if len(sentences) < 2:
            return SpecialistResult(
                snippet_id=snippet_id,
                specialist_type=self.name,
                location=location,
                status="NO_ERROR",
                confidence=0.5,
                reasoning="Fewer than two comparable sentences; no cross-sentence "
                "contradiction check possible.",
                execution_time_ms=_elapsed(),
                extra={"snippet_type": kwargs.get("snippet_type", ""), "contradiction_locations": []},
            )

        try:
            nli = _get_nli_pipeline(self._model.removeprefix("hf/"))
            pairs = [
                (i, j)
                for i in range(len(sentences))
                for j in range(i + 1, len(sentences))
            ]
            inputs = [
                {"text": sentences[i], "text_pair": sentences[j]} for i, j in pairs
            ]
            outputs = nli(inputs, batch_size=32, truncation=True)
        except Exception as exc:
            logger.error(f"Specialist [{self.name}] NLI inference failed: {exc}")
            return SpecialistResult(
                snippet_id=snippet_id,
                specialist_type=self.name,
                location=location,
                status="UNVERIFIABLE",
                reasoning=f"NLI model inference failed: {exc}",
                execution_time_ms=_elapsed(),
            )

        contradictions: list[tuple[float, int, int]] = []
        for (i, j), scores in zip(pairs, outputs):
            p_contra = next(
                (s["score"] for s in scores if s["label"].lower() == "contradiction"),
                0.0,
            )
            if p_contra >= self.CONTRADICTION_THRESHOLD:
                contradictions.append((p_contra, i, j))

        contradictions.sort(reverse=True)
        snippet_type = kwargs.get("snippet_type", "SECTION")

        if contradictions:
            best_p, bi, bj = contradictions[0]
            locations = [
                f"{location}: “{sentences[i][:120]}” ↔ “{sentences[j][:120]}”"
                for _, i, j in contradictions[:5]
            ]
            return SpecialistResult(
                snippet_id=snippet_id,
                specialist_type=self.name,
                location=location,
                status="ERROR_DETECTED",
                error_detected=True,
                confidence=round(best_p, 3),
                reasoning=(
                    f"NLI cross-encoder found {len(contradictions)} sentence pair(s) "
                    f"with contradiction probability ≥ {self.CONTRADICTION_THRESHOLD}. "
                    f"Strongest (p={best_p:.3f}): “{sentences[bi]}” contradicts "
                    f"“{sentences[bj]}”."
                ),
                predicted_error_category="Data Inconsistency (text-text)",
                execution_time_ms=_elapsed(),
                extra={
                    "snippet_type": snippet_type,
                    "contradiction_locations": locations,
                    "n_sentence_pairs_checked": len(pairs),
                },
            )

        max_p = max(
            (
                next((s["score"] for s in scores if s["label"].lower() == "contradiction"), 0.0)
                for scores in outputs
            ),
            default=0.0,
        )
        return SpecialistResult(
            snippet_id=snippet_id,
            specialist_type=self.name,
            location=location,
            status="NO_ERROR",
            confidence=round(1.0 - max_p, 3),
            reasoning=(
                f"NLI cross-encoder checked {len(pairs)} sentence pairs; max "
                f"contradiction probability {max_p:.3f} — below the "
                f"{self.CONTRADICTION_THRESHOLD} flag threshold."
            ),
            execution_time_ms=_elapsed(),
            extra={
                "snippet_type": snippet_type,
                "contradiction_locations": [],
                "n_sentence_pairs_checked": len(pairs),
            },
        )

    def _system_prompt(self) -> str:
        return """You are a text-consistency verification specialist — a model fine-tuned exclusively to detect critical errors in scientific prose. You examine ONE excerpt at a time.

## Flag an error (error_detected = true) ONLY for:

1. **Provably wrong results**: A derivation, calculation, or stated result that is demonstrably incorrect.
2. **Conclusion-undermining contradictions**: A statement that directly contradicts data, a definition, or another explicit statement.
3. **Invalidating methodology flaws**: A methodological mistake that makes a reported result unsound.
4. **Impossible/self-contradictory statistics**: Reported numbers that are internally impossible AND material to a conclusion.

## NEVER flag:
- Typos, grammar, spelling, formatting, or stylistic issues.
- Missing citations, "could be clearer", incomplete explanations.
- Claims you cannot verify from this excerpt alone.
- Anything minor, cosmetic, or that would not change the paper's conclusions.
- Subjective disagreements or alternative approaches.

## Critical guidance
- A typical paper has ZERO or ONE such error. **Flagging should be RARE.**
- Missing surrounding context is NOT an error.
- If you are not highly confident, return error_detected = false.
- When you DO flag, set confidence >= 0.8 and cite the specific offending statement.

## Content-type-specific checks
- **Sections/Paragraphs**: Logical contradictions between sentences.
- **Theorems/Lemmas/Propositions/Corollaries**: Logical coherence; if proof included, check for gaps.
- **Algorithms**: Logical errors or missing steps that would make the algorithm incorrect.

## Output format
Return a JSON object with exactly these fields:
```json
{
  "error_detected": false,
  "confidence": 0.0,
  "reasoning": "Detailed explanation; quote the specific offending text if flagging...",
  "predicted_error_category": null,
  "snippet_type": "SECTION",
  "contradiction_locations": []
}
```

predicted_error_category values (use null when no error):
- "Equation / proof"
- "Data Inconsistency (text-text)"
- "Statistical reporting"
- "Experiment setup"
- "Reagent identity"
- "Methodology inconsistency"
- null"""

    def _build_user_prompt(self, text: str, **kwargs: Any) -> str:
        snippet_type = kwargs.get("snippet_type", "SECTION")
        location = kwargs.get("location", "unknown")
        parts = [
            f"Analyze the following {snippet_type} from a scientific paper for errors.",
            f"Location: {location}",
            "",
            f"Content:",
            text[:4000],
            "",
            "Flag a CRITICAL error only if this excerpt contains a mistake "
            "serious enough to warrant an erratum or retraction. "
            "Otherwise set error_detected to false. Return ONLY the JSON response.",
        ]
        return "\n".join(parts)

    def _parse_response(
        self,
        response: dict[str, Any],
        snippet_id: str,
        location: str,
        elapsed_ms: float,
    ) -> SpecialistResult:
        result = super()._parse_response(response, snippet_id, location, elapsed_ms)
        result.extra["snippet_type"] = response.get("snippet_type", "")
        result.extra["contradiction_locations"] = response.get("contradiction_locations", [])
        return result


# ----------------------------------------------------------------------
# Citation Specialist
# ----------------------------------------------------------------------


class CitationSpecialist(Specialist):
    """Specialist for citation, attribution, and novelty-claim verification.

    Detects internally-provable attribution problems: self-contradictory
    claims about prior work, novelty over-claims contradicted within the
    excerpt, and obviously malformed/fabricated-looking references.

    Default model: **qwen3 (8B)** served locally by Ollama. Note: without
    web access a local model cannot confirm whether a cited paper exists,
    so externally-dependent checks default to UNVERIFIABLE rather than
    guessing (the prompt enforces this).
    """

    name = "citation"

    def _default_model(self) -> str:
        return "ollama/qwen3:latest"

    def _system_prompt(self) -> str:
        return """You are a citation-verification specialist — a model fine-tuned to detect attribution errors in scientific papers. You examine ONE excerpt for CITATION, ATTRIBUTION, and NOVELTY problems.

## What to flag (error_detected = true) ONLY:

1. **Fabricated Citations (Hallucinations)**: The cited paper, DOI, or specific author/title combination demonstrably does not exist.
2. **Attribution Mismatch**: A citation is invoked to support a claim, but the cited paper does not support it, is about a completely different topic, or actively contradicts it.
3. **Provable Novelty Over-claim**: The excerpt claims a method/result is entirely novel, but established prior work exists.
4. **Internal Self-Contradiction**: The excerpt simultaneously asserts incompatible things about what prior work did or did not do.

## NEVER flag:
- Missing citations for general knowledge.
- Papers behind paywalls where the abstract is insufficient (default to UNVERIFIABLE).
- Minor paraphrasing differences or formatting errors in bibliography.
- Ordinary background citations or standard "building on prior work" framing.

## Critical guidance
- **Flagging should be RARE.** Only flag when you have hard proof.
- You have NO web access. Only flag problems provable from the excerpt itself
  (internal contradictions, novelty claims contradicted in the same text) or
  from well-established knowledge you are certain of. If verification would
  require looking up the cited work, set status UNVERIFIABLE instead of guessing.
- If not highly confident, return error_detected = false.
- When you DO flag, set confidence >= 0.8 and quote the claim vs. actual evidence.

## Output format
Return a JSON object with exactly these fields:
```json
{
  "error_detected": false,
  "confidence": 0.0,
  "reasoning": "Detailed explanation; quote the claim vs. the actual evidence...",
  "predicted_error_category": null,
  "snippet_type": "paragraph"
}
```

predicted_error_category values (use null when no error):
- "Fabricated citation"
- "Misattribution (external)"
- "Novelty/Priority over-claim"
- "Data Inconsistency (text-text)"
- null"""

    def _build_user_prompt(self, text: str, **kwargs: Any) -> str:
        snippet_type = kwargs.get("snippet_type", "SECTION")
        location = kwargs.get("location", "unknown")
        parts = [
            f"Verify the citations and attribution claims in this {snippet_type}.",
            f"Location: {location}",
            "",
            f"Content:",
            text[:4000],
            "",
            "Identify any fabricated citations, attribution mismatches, or "
            "novelty over-claims. Return ONLY the JSON response.",
        ]
        return "\n".join(parts)

    def _parse_response(
        self,
        response: dict[str, Any],
        snippet_id: str,
        location: str,
        elapsed_ms: float,
    ) -> SpecialistResult:
        result = super()._parse_response(response, snippet_id, location, elapsed_ms)
        result.extra["snippet_type"] = response.get("snippet_type", "")
        return result


# ----------------------------------------------------------------------
# Vision Specialist
# ----------------------------------------------------------------------


class VisionSpecialist(Specialist):
    """Specialist for figures and tables — visual/content integrity.

    Detects data inconsistencies, figure/caption contradictions, and
    impossible data points.

    Default model: **granite3.2-vision** served locally by Ollama — a
    compact out-of-the-box multimodal model. When an ``image_path`` is
    provided (figures) the image is sent alongside the caption; for tables
    the rendered/markdown content is analysed as text.
    """

    name = "vision"

    def _default_model(self) -> str:
        return "ollama/granite3.2-vision:latest"

    def _system_prompt(self) -> str:
        return """You are a vision verification specialist — a model fine-tuned to detect critical errors in figures and tables within scientific papers.

## For Figures — flag ONLY:

1. **Result-invalidating contradiction**: The figure demonstrably contradicts its caption or the text.
2. **Genuine duplication / fabrication**: Clear evidence the figure (or a panel) is duplicated or manipulated.
3. **Impossible data**: Data points or axis ranges that are impossible and material to a conclusion.

## For Tables — flag ONLY:

1. **Provably wrong values**: Arithmetic that does not add up, or internally impossible statistics.
2. **Result-invalidating contradiction**: Table values that directly contradict the caption or text.
3. **Mislabeled data that changes interpretation**: Values attached to the wrong row/column.

## NEVER flag:
- Aesthetic/presentation choices, axis scaling, color choices, missing labels.
- Anything you cannot verify from what is shown.
- Minor/cosmetic issues that would not change the paper's conclusions.

## Critical guidance
- **Flagging should be RARE.**
- If not highly confident, return error_detected = false.
- When you DO flag, set confidence >= 0.75 and state the specific problem.

## Output format
Return a JSON object with exactly these fields:
```json
{
  "error_detected": false,
  "confidence": 0.0,
  "reasoning": "Detailed explanation...",
  "predicted_error_category": null,
  "content_type": "figure"
}
```

predicted_error_category values (use null when no error):
- "Figure duplication"
- "Data Inconsistency (figure-text)"
- "Data Inconsistency (figure-figure)"
- "Data inconsistency"
- "Statistical reporting"
- null"""

    def _build_user_prompt(self, text: str, **kwargs: Any) -> str:
        content_type = kwargs.get("content_type", kwargs.get("snippet_type", "FIGURE")).lower()
        location = kwargs.get("location", "unknown")
        caption = kwargs.get("caption", "")
        parts = [
            f"Verify the following {content_type} from a scientific paper for errors.",
            f"Location: {location}",
        ]
        if caption:
            parts.append(f"Caption: {caption}")
        parts.extend([
            "",
            f"Content/Description:",
            text[:3000],
            "",
            "Return ONLY the JSON response.",
        ])
        return "\n".join(parts)

    def _parse_response(
        self,
        response: dict[str, Any],
        snippet_id: str,
        location: str,
        elapsed_ms: float,
    ) -> SpecialistResult:
        result = super()._parse_response(response, snippet_id, location, elapsed_ms)
        result.extra["content_type"] = response.get("content_type", "figure")
        return result


# ----------------------------------------------------------------------
# Statistical Specialist
# ----------------------------------------------------------------------


class StatisticalSpecialist(Specialist):
    """Specialist for statistical reporting: test statistics, p-values, df, CIs.

    Detects mathematically impossible statcheck errors, impossible
    descriptive statistics, and CI/point-estimate inconsistencies.

    Default model: **deterministic scipy statcheck engine** (no LLM). This
    reimplements the core of the well-known *statcheck* tool: it regex-extracts
    reported null-hypothesis significance tests (t, F, χ², r, z), recomputes
    the p-value with ``scipy.stats``, and flags a *decision error* when the
    reported and recomputed p-values fall on opposite sides of α = .05. It
    additionally flags impossible bounds (p, R², probabilities outside [0,1];
    negative SD/variance) and CIs that exclude their point estimate.

    Override ``model`` with an ``ollama/...`` name to use the LLM prompt path.
    """

    name = "statistical"

    #: How far reported and recomputed p may differ (as a ratio) before we
    #: treat it as more than a rounding artifact when they're on the same side.
    _P_REL_TOL = 0.5

    def _default_model(self) -> str:
        return "statcheck/scipy"

    # ------------------------------------------------------------------
    # Deterministic engine
    # ------------------------------------------------------------------

    def verify(
        self,
        text: str,
        snippet_id: str = "",
        location: str = "",
        **kwargs: Any,
    ) -> SpecialistResult:
        # LLM override (e.g. "ollama/qwen3:latest") → generic prompt path.
        if not self._model.startswith("statcheck/"):
            return super().verify(text, snippet_id=snippet_id, location=location, **kwargs)

        start_time = time.monotonic()
        snippet_type = kwargs.get("snippet_type", "SECTION")
        findings: list[str] = []

        try:
            findings.extend(self._check_nhst(text))
            findings.extend(self._check_bounds(text))
            findings.extend(self._check_confidence_intervals(text))
        except Exception as exc:
            logger.error(f"Specialist [{self.name}] statcheck failed: {exc}")
            return SpecialistResult(
                snippet_id=snippet_id,
                specialist_type=self.name,
                location=location,
                status="UNVERIFIABLE",
                reasoning=f"Deterministic statcheck engine failed: {exc}",
                execution_time_ms=(time.monotonic() - start_time) * 1000,
                extra={"snippet_type": snippet_type, "contradiction_locations": []},
            )

        elapsed_ms = (time.monotonic() - start_time) * 1000
        if findings:
            return SpecialistResult(
                snippet_id=snippet_id,
                specialist_type=self.name,
                location=location,
                status="ERROR_DETECTED",
                error_detected=True,
                confidence=0.95,
                reasoning="Deterministic statistical checks failed:\n- "
                + "\n- ".join(findings),
                predicted_error_category="Statistical reporting",
                execution_time_ms=elapsed_ms,
                extra={"snippet_type": snippet_type, "contradiction_locations": findings},
            )

        return SpecialistResult(
            snippet_id=snippet_id,
            specialist_type=self.name,
            location=location,
            status="NO_ERROR",
            confidence=0.9,
            reasoning="All extracted test statistics recompute consistently with "
            "their reported p-values, and no impossible bounds or CI "
            "inconsistencies were found.",
            execution_time_ms=elapsed_ms,
            extra={"snippet_type": snippet_type, "contradiction_locations": []},
        )

    # ---- statcheck: recompute p from test statistic + df --------------

    @staticmethod
    def _reported_p(op: str, value: float) -> tuple[str, float]:
        return op, value

    def _check_nhst(self, text: str) -> list[str]:
        """Recompute p-values for t/F/χ²/r/z tests and flag decision errors."""
        import re

        from scipy import stats

        findings: list[str] = []
        # Common tail: "..., p OP value" where OP ∈ {<, >, =, ≤, ≥}
        p_tail = r"\s*,\s*p\s*(?P<pop>[<>=]|≤|≥|<=|>=)\s*(?P<pval>\.?\d+(?:\.\d+)?)"

        patterns = {
            "t": re.compile(r"t\s*\(\s*(?P<df>\d+(?:\.\d+)?)\s*\)\s*=\s*(?P<stat>-?\d+(?:\.\d+)?)" + p_tail, re.I),
            "F": re.compile(r"F\s*\(\s*(?P<df1>\d+(?:\.\d+)?)\s*,\s*(?P<df2>\d+(?:\.\d+)?)\s*\)\s*=\s*(?P<stat>-?\d+(?:\.\d+)?)" + p_tail, re.I),
            "chi2": re.compile(r"(?:χ2|χ²|chi2|chi-square[d]?)\s*\(\s*(?P<df>\d+(?:\.\d+)?)\s*\)\s*=\s*(?P<stat>-?\d+(?:\.\d+)?)" + p_tail, re.I),
            "r": re.compile(r"\br\s*\(\s*(?P<df>\d+(?:\.\d+)?)\s*\)\s*=\s*(?P<stat>-?\.?\d+(?:\.\d+)?)" + p_tail, re.I),
            "z": re.compile(r"\bz\s*=\s*(?P<stat>-?\d+(?:\.\d+)?)" + p_tail, re.I),
        }

        def recompute(kind: str, m: "re.Match") -> Optional[float]:
            stat = abs(float(m.group("stat")))
            if kind == "t":
                return float(2 * stats.t.sf(stat, float(m.group("df"))))
            if kind == "F":
                return float(stats.f.sf(stat, float(m.group("df1")), float(m.group("df2"))))
            if kind == "chi2":
                return float(stats.chi2.sf(stat, float(m.group("df"))))
            if kind == "z":
                return float(2 * stats.norm.sf(stat))
            if kind == "r":
                df = float(m.group("df"))
                r = float(m.group("stat"))
                if df <= 0 or abs(r) >= 1:
                    return None
                t = r * ((df / (1 - r**2)) ** 0.5)
                return float(2 * stats.t.sf(abs(t), df))
            return None

        for kind, pat in patterns.items():
            for m in pat.finditer(text):
                computed = recompute(kind, m)
                if computed is None:
                    continue
                pop = m.group("pop").replace("<=", "≤").replace(">=", "≥")
                reported = float(m.group("pval"))
                snippet = m.group(0).strip()

                # Decision error: reported claims significance but recomputed
                # is non-significant, or vice versa (α = .05).
                reported_sig = (
                    (pop in ("<", "≤", "=") and reported <= 0.05)
                    if pop in ("<", "≤", "=")
                    else False
                )
                if pop in (">", "≥"):
                    reported_sig = reported < 0.05  # p > .10 etc → non-sig
                computed_sig = computed <= 0.05

                gross = False
                if pop in ("=",):
                    # Exact report: flag if recomputed crosses .05 boundary
                    gross = (reported <= 0.05) != (computed <= 0.05)
                elif pop in ("<", "≤"):
                    # "p < r": recomputed should also be < r (allow rounding)
                    gross = computed > max(reported, 0.055) and reported <= 0.05
                elif pop in (">", "≥"):
                    gross = computed < min(reported, 0.045) and reported >= 0.05

                if gross:
                    findings.append(
                        f"Incoherent test statistic “{snippet}”: recomputed "
                        f"two-tailed p ≈ {computed:.4g} contradicts the reported "
                        f"p {pop} {reported:g} (opposite side of α=.05)."
                    )
        return findings

    # ---- impossible descriptive-statistic bounds ----------------------

    def _check_bounds(self, text: str) -> list[str]:
        import re

        findings: list[str] = []

        for m in re.finditer(r"\bp\s*(?:[<>=]|≤|≥)\s*(-?\.?\d+(?:\.\d+)?)", text, re.I):
            val = float(m.group(1))
            if val < 0 or val > 1:
                findings.append(f"Impossible p-value {val:g} (outside [0, 1]): “{m.group(0).strip()}”.")

        for m in re.finditer(r"\bR\s*\^?2\s*=\s*(-?\d+(?:\.\d+)?)", text, re.I):
            val = float(m.group(1))
            if val < 0 or val > 1:
                findings.append(f"Impossible R² = {val:g} (outside [0, 1]): “{m.group(0).strip()}”.")

        for m in re.finditer(r"\b(?:SD|standard deviation|variance|SE|std\.?\s*error)\s*[=:]\s*(-\d+(?:\.\d+)?)", text, re.I):
            findings.append(f"Negative dispersion statistic: “{m.group(0).strip()}” — SD/variance/SE cannot be negative.")

        return findings

    # ---- confidence interval must contain point estimate --------------

    def _check_confidence_intervals(self, text: str) -> list[str]:
        import re

        findings: list[str] = []
        # e.g. "M = 3.2, 95% CI [1.0, 2.5]" or "OR = 0.8 (95% CI: 1.1 to 1.9)".
        # The point estimate must be an explicit assignment ("= value") sitting
        # immediately before the interval, so we don't mistake a nearby p-value
        # or sample size for the estimate.
        pat = re.compile(
            r"=\s*(?P<point>-?\d+(?:\.\d+)?)\s*[,;]?\s*(?:\(|\[)?\s*95%\s*CI\s*[\[\(:]\s*"
            r"(?P<lo>-?\d+(?:\.\d+)?)\s*(?:,|to|–|-)\s*(?P<hi>-?\d+(?:\.\d+)?)",
            re.I,
        )
        for m in pat.finditer(text):
            point, lo, hi = (float(m.group("point")), float(m.group("lo")), float(m.group("hi")))
            lo, hi = min(lo, hi), max(lo, hi)
            if not (lo <= point <= hi):
                findings.append(
                    f"Confidence interval [{lo:g}, {hi:g}] does not contain its "
                    f"point estimate {point:g}: “{m.group(0).strip()}”."
                )
        return findings

    def _system_prompt(self) -> str:
        return """You are a statistical-verification specialist — a model fine-tuned to detect critical statistical reporting errors in scientific papers.

## Flag an error (error_detected = true) ONLY for:

1. **Incoherent Test Statistics (statcheck errors)**: A mathematically impossible combination of test statistic, df, and reported p-value (e.g., t(28)=0.52, p<0.001).
2. **Degrees of Freedom / Sample Size Contradictions**: df that clearly contradicts the stated sample size for the specified test.
3. **Impossible Descriptive Statistics**: Negative variances/SDs; probabilities/R² outside [0,1]; a mean outside the possible range of a bounded scale.
4. **Confidence Interval Inconsistencies**: A CI that does not encompass the reported point estimate.
5. **Invalidating Methodological Contradictions**: Stating one test but reporting the output format of another.

## NEVER flag:
- Subjective disagreements about the "best" test to use.
- Missing effect sizes, missing exact p-values, or missing CIs.
- Minor rounding discrepancies.
- Claims you cannot mathematically verify from this excerpt alone.

## Critical guidance
- **Flagging should be RARE.**
- If you cannot mathematically prove the statistics are incoherent, return error_detected = false.
- When you DO flag, set confidence >= 0.8 and demonstrate the mathematical contradiction.

## Output format
Return a JSON object with exactly these fields:
```json
{
  "error_detected": false,
  "confidence": 0.0,
  "reasoning": "Detailed explanation with mathematical proof if flagging...",
  "predicted_error_category": null,
  "snippet_type": "PARAGRAPH",
  "contradiction_locations": []
}
```

predicted_error_category values (use null when no error):
- "Statistical reporting"
- "Degrees of freedom mismatch"
- "Impossible bounds"
- "Data Inconsistency (text-table)"
- "Methodology inconsistency"
- null"""

    def _build_user_prompt(self, text: str, **kwargs: Any) -> str:
        snippet_type = kwargs.get("snippet_type", "SECTION")
        location = kwargs.get("location", "unknown")
        parts = [
            f"Verify the statistical content in this {snippet_type} for errors.",
            f"Location: {location}",
            "",
            f"Content:",
            text[:4000],
            "",
            "Check for statcheck errors, df/sample-size contradictions, "
            "impossible descriptive statistics, and CI inconsistencies. "
            "Return ONLY the JSON response.",
        ]
        return "\n".join(parts)

    def _parse_response(
        self,
        response: dict[str, Any],
        snippet_id: str,
        location: str,
        elapsed_ms: float,
    ) -> SpecialistResult:
        result = super()._parse_response(response, snippet_id, location, elapsed_ms)
        result.extra["snippet_type"] = response.get("snippet_type", "")
        result.extra["contradiction_locations"] = response.get("contradiction_locations", [])
        return result


# ----------------------------------------------------------------------
# Cross-Section Consistency Specialist
# ----------------------------------------------------------------------


class CrossSectionConsistencySpecialist(Specialist):
    """Specialist for macro-level cross-section consistency.

    Detects Abstract-vs-Results contradictions, methodology mismatches,
    global sample-size discrepancies, and conclusion/evidence reversals
    across an entire paper.

    Default model: **qwen3 (8B)** served locally by Ollama. This is a
    long-context reasoning task over the whole paper, so the Ollama backend
    is invoked with a 16k context window (see ``_ollama_call``).
    """

    name = "cross-section-consistency"

    def _default_model(self) -> str:
        return "ollama/qwen3:latest"

    def _system_prompt(self) -> str:
        return """You are a cross-section consistency specialist — a model fine-tuned to detect macro-level contradictions across different sections of a scientific paper.

## Flag an error (error_detected = true) ONLY for:

1. **Abstract vs. Results/Conclusion Contradiction**: The abstract claims a finding, effect direction, or significance, but the main text states the opposite or reports it as non-significant.
2. **Methodology vs. Implementation Mismatch**: The Methods section describes one setup, but the Results report on a completely different one without explanation.
3. **Global Sample Size / Data Contradictions**: A fatal discrepancy in reported global numbers across major sections.
4. **Hypothesis/Evidence Reversal**: The Discussion/Conclusion claims a hypothesis was supported when the results directly rejected it.

## NEVER flag:
- Summarisation differences (abstract omits caveats detailed in Discussion).
- Varying levels of detail across sections.
- Formatting or stylistic inconsistencies.
- External validity (claims contradicting outside knowledge).
- Evolving arguments (building a model across sections is not a contradiction).

## Critical guidance
- **Flagging should be RARE.** Focus only on glaring, conclusion-invalidating contradictions.
- If you cannot definitively prove two sections are mutually exclusive, return error_detected = false.
- When you DO flag, set confidence >= 0.85 and quote the conflicting text from both sections.

## Output format
Return a JSON object with exactly these fields:
```json
{
  "error_detected": false,
  "confidence": 0.0,
  "reasoning": "Quote the conflicting text from both sections if flagging...",
  "predicted_error_category": null,
  "snippet_type": "FULL_PAPER",
  "contradiction_locations": []
}
```

predicted_error_category values (use null when no error):
- "Abstract vs Results mismatch"
- "Methodology vs Results mismatch"
- "Global data inconsistency"
- "Conclusion vs Evidence mismatch"
- null"""

    def _build_user_prompt(self, text: str, **kwargs: Any) -> str:
        parts = [
            "Verify the following complete paper (or combined sections) for "
            "macro-level internal contradictions.",
            "",
            "Full paper text:",
            text[:16000],
            "",
            "Check for Abstract-vs-Results contradictions, methodology mismatches, "
            "global data inconsistencies, and conclusion/evidence reversals. "
            "Return ONLY the JSON response.",
        ]
        return "\n".join(parts)

    def _parse_response(
        self,
        response: dict[str, Any],
        snippet_id: str,
        location: str,
        elapsed_ms: float,
    ) -> SpecialistResult:
        result = super()._parse_response(response, snippet_id, location, elapsed_ms)
        result.extra["snippet_type"] = response.get("snippet_type", "FULL_PAPER")
        result.extra["contradiction_locations"] = response.get("contradiction_locations", [])
        return result


# ----------------------------------------------------------------------
# Triage Specialist
# ----------------------------------------------------------------------


class TriageSpecialist(Specialist):
    """Specialist for uncertainty estimation and triage routing.

    Scores each snippet's likelihood of containing an error and suggests
    which specialist (if any) should examine it further. This is the
    *first pass* in uncertainty-driven orchestration — it runs on every
    snippet.

    Default model: **qwen3 (8B)** served locally by Ollama, with thinking
    disabled for latency. Smaller models (qwen3:0.6b, gemma3:1b) were too
    weak to discriminate suspicious from routine content, so this uses the
    8B instruct model. It only estimates *where* to spend effort, never
    decides errors itself.
    """

    name = "triage"

    #: Valid escalation routes and keyword aliases the model may emit instead.
    _ROUTE_ALIASES: dict[str, str] = {
        "math": "math", "equation": "math", "derivation": "math", "algebra": "math",
        "proof": "proof", "theorem": "proof", "lemma": "proof",
        "statistics": "statistics", "statistical": "statistics", "stats": "statistics",
        "numeric": "statistics", "pvalue": "statistics", "p-value": "statistics",
        "citation": "citation", "reference": "citation", "attribution": "citation",
        "novelty": "citation",
        "vision": "vision", "figure": "vision", "table": "vision", "image": "vision",
        "text": "text", "logic": "text", "logical": "text", "prose": "text",
        "consistency": "text", "inconsistency": "text",
        "none": "none", "skip": "none",
    }

    def _default_model(self) -> str:
        return "ollama/qwen3:latest"

    @classmethod
    def _normalize_route(cls, raw: str) -> str:
        """Map a free-text route to one of the valid specialist routes.

        Small local models frequently ignore the enum (e.g. return
        "logical inconsistency"); we recover a valid route by keyword,
        defaulting to "text" so the snippet still gets a look.
        """
        r = (raw or "").strip().lower()
        if r in cls._ROUTE_ALIASES:
            return cls._ROUTE_ALIASES[r]
        for token, route in cls._ROUTE_ALIASES.items():
            if token in r:
                return route
        return "text"

    def _system_prompt(self) -> str:
        return """You are a triage specialist — a model fine-tuned to estimate where error-checking effort should be concentrated in a scientific paper. You do NOT decide whether a snippet actually contains an error. Your only job is to estimate WHERE effort should be spent.

## Output two things:

### 1. `uncertainty` — a number in [0, 1]
Your estimate of the probability that this excerpt contains a correction- or retraction-worthy error:

| Range | Meaning |
|-------|---------|
| 0.00–0.15 | Routine, definitional, or boilerplate content. **MOST excerpts are here.** |
| 0.15–0.40 | Substantive content with some moving parts but nothing that stands out. |
| 0.40–0.70 | A non-trivial derivation, quantitative claim, or load-bearing proof step. |
| 0.70–1.00 | Something looks off, surprising, internally tense, or makes a strong/atypical claim. |

### 2. `route` — which specialist should examine it IF escalated
Choose ONE: `math`, `proof`, `statistics`, `citation`, `vision`, `text`, `none`

## Critical guidance
- **Be discriminating**: Most excerpts deserve LOW uncertainty.
- Reserve high scores for genuinely suspicious or high-stakes content.
- The `none` route skips specialist verification entirely.
- Fail OPEN: if unsure, give moderate uncertainty so the snippet gets a specialist look.

## Output format
Return a JSON object with exactly these fields:
```json
{
  "uncertainty": 0.0,
  "route": "text",
  "reason": "One short sentence explaining the score and route."
}
```"""

    def _build_user_prompt(self, text: str, **kwargs: Any) -> str:
        snippet_type = kwargs.get("snippet_type", "SECTION")
        location = kwargs.get("location", "unknown")
        parts = [
            f"Excerpt type: {snippet_type}",
            f"Location: {location}",
            "",
            f"Content:",
            text[:3000],
            "",
            "Estimate the uncertainty (0-1) that this excerpt contains a "
            "correction-worthy error, and suggest the best specialist route. "
            "Return ONLY the JSON response.",
        ]
        return "\n".join(parts)

    def _parse_response(
        self,
        response: dict[str, Any],
        snippet_id: str,
        location: str,
        elapsed_ms: float,
    ) -> SpecialistResult:
        """Triage responses have a different shape — uncertainty + route."""
        result = SpecialistResult(
            snippet_id=snippet_id,
            specialist_type=self.name,
            location=location,
            status="NO_ERROR",  # triage doesn't decide errors
            error_detected=False,
            confidence=1.0 - float(response.get("uncertainty", 0.0)),
            reasoning=response.get("reason", ""),
            predicted_error_category=None,
            execution_time_ms=elapsed_ms,
        )
        result.extra["uncertainty"] = float(response.get("uncertainty", 0.0))
        result.extra["route"] = self._normalize_route(response.get("route", "text"))
        result.extra["reason"] = response.get("reason", "")
        return result


# ======================================================================
# Specialist registry
# ======================================================================


#: Mapping from specialist type string → Specialist class.
SPECIALIST_REGISTRY: dict[str, type[Specialist]] = {
    "math": MathSpecialist,
    "text": TextSpecialist,
    "citation": CitationSpecialist,
    "vision": VisionSpecialist,
    "statistical": StatisticalSpecialist,
    "cross-section-consistency": CrossSectionConsistencySpecialist,
    "triage": TriageSpecialist,
}


def get_specialist(
    specialist_type: str,
    model: Optional[str] = None,
) -> Specialist:
    """Factory: return an instantiated specialist by type.

    Args:
        specialist_type: One of the keys in ``SPECIALIST_REGISTRY``.
        model: Optional model override (passed to the specialist constructor).

    Returns:
        An instantiated ``Specialist`` subclass.

    Raises:
        ValueError: If *specialist_type* is unknown.
    """
    cls = SPECIALIST_REGISTRY.get(specialist_type)
    if cls is None:
        available = list(SPECIALIST_REGISTRY.keys())
        raise ValueError(
            f"Unknown specialist type '{specialist_type}'. "
            f"Available: {available}"
        )
    return cls(model=model)


def list_specialists() -> list[str]:
    """Return the names of all registered specialists."""
    return list(SPECIALIST_REGISTRY.keys())
