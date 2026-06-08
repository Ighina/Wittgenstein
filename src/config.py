"""Central configuration for the Paperena Verification pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class PathsConfig:
    """File system paths used throughout the pipeline."""

    data_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("PAPERENA_DATA_DIR", "data"))
    )
    output_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("PAPERENA_OUTPUT_DIR", "outputs"))
    )
    parquet_file: str = "train-00000-of-00001.parquet"

    @property
    def parquet_path(self) -> Path:
        return self.data_dir / self.parquet_file

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)


@dataclass
class LLMConfig:
    """Configuration for LLM backend."""

    provider: str = "deepseek"  # "mock", "anthropic", "openai", "deepseek"
    model: str = "deepseek-v4-pro"
    api_key_env: str = "DEEPSEEK_API_KEY"
    # Output-token budget. Set high enough that reasoning-style models can emit
    # the final answer after their chain-of-thought; a too-small value yields
    # empty `message.content`. Verifiers/baseline inherit this via llm_call.
    max_tokens: int = 8192
    temperature: float = 0.0
    timeout_seconds: int = 120

    # Concurrency: number of snippets verified in parallel within a paper.
    # LLM calls are network-bound, so threads (not processes) are used.
    # Set to 1 for fully sequential / deterministic debugging.
    num_workers: int = 8

    # Transient-failure handling for concurrent API calls (rate limits, blips).
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0


@dataclass
class VerifierConfig:
    """Configuration for individual verifiers."""

    enabled: bool = True
    confidence_threshold: float = 0.6


@dataclass
class SandboxConfig:
    """Configuration for the SymPy sandbox execution environment."""

    timeout_seconds: int = 10
    max_output_bytes: int = 65536
    python_executable: str = "python3"


@dataclass
class SegmentationConfig:
    """Configuration for paper segmentation."""

    max_snippet_chars: int = 4000
    max_section_chars: int = 8000
    overlap_chars: int = 200


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    verifiers: dict[str, VerifierConfig] = field(default_factory=dict)

    # Error-detection strictness. "strict" (default) only flags critical,
    # erratum/retraction-worthy errors; "lenient" reproduces the original,
    # broader prompts and lower confidence thresholds.
    strictness: str = "strict"  # "strict" | "lenient"

    # ------------------------------------------------------------------
    # Orchestration mode
    # ------------------------------------------------------------------
    # "exhaustive" (default): every snippet is routed to a verifier by type.
    # "uncertainty": a cheap general triage pass first scores each snippet's
    # error likelihood (an "uncertainty map"); specialized verifiers run ONLY
    # on snippets above `uncertainty_threshold` (optionally capped by
    # `uncertainty_budget`). This routes effort by expected error density
    # rather than document structure. See UncertaintyOrchestrator.
    orchestration_mode: str = "exhaustive"  # "exhaustive" | "uncertainty"

    # Snippets with triage uncertainty >= this are escalated to a specialist.
    uncertainty_threshold: float = 0.30

    # ------------------------------------------------------------------
    # LLM-only verification (bypasses specialist verifiers)
    # ------------------------------------------------------------------
    # When set, ALL snippets are verified by a single general-purpose LLM
    # call instead of being routed to type-specific specialist verifiers
    # (no SymPy sandbox, no deterministic checks, no multimodal vision).
    #   "same-prompt"     → one unified system prompt for every snippet type
    #   "separate-prompts"→ different prompts per snippet type (text, math,
    #                        figure/table, citation), but still pure LLM
    #   None              → use the normal specialist verifier pipeline
    llm_only_mode: Optional[str] = None  # None | "same-prompt" | "separate-prompts"

    # ------------------------------------------------------------------
    # Chunking (text-based verifiers)
    # ------------------------------------------------------------------
    # Text/citation verifiers split content longer than `verify_chunk_chars`
    # into overlapping chunks and verify each independently, then aggregate.
    # This keeps each LLM call short (dense proofs otherwise return empty/garbled
    # completions), tolerates a single empty chunk, and localizes findings.
    verify_chunk_chars: int = 2000
    verify_chunk_overlap: int = 200

    # Use the progressive math verifier (context-accumulating, multi-layer)
    # instead of the single-equation MathEquationVerifier for EQUATION snippets.
    # Requires num_workers=1 for fully deterministic results.
    use_progressive_math: bool = False

    # When set, filter evaluation to only consider ground-truth errors and
    # predictions in this category.  Useful for isolating, e.g.,
    # "Equation / proof" performance.  None = evaluate all categories.
    eval_category_filter: Optional[str] = None

    # Optional hard cap on the number of specialist verifications per paper
    # (the highest-uncertainty snippets are kept). None = no cap.
    uncertainty_budget: Optional[int] = None

    # Semantic triage routes → registered verifier names. The triage model
    # suggests a route per snippet; this maps it to an actual verifier. Unknown
    # routes (and "none") fall back to type-based routing / are skipped.
    triage_route_map: dict[str, str] = field(
        default_factory=lambda: {
            "math": "math_equation",
            "equation": "math_equation",
            "proof": "text",
            "statistics": "statistical",
            "statistical": "statistical",
            "numeric": "statistical",
            "citation": "citation",
            "reference": "citation",
            "vision": "vision",
            "figure": "vision",
            "table": "vision",
            "text": "text",
            "logic": "text",
            "none": "",  # empty → no specialist (accept as low-risk)
        }
    )

    # Evaluation: how predictions are matched to ground-truth annotations.
    # The original benchmark (see src/run_eval.py) used an LLM judge to decide
    # whether a predicted error semantically corresponds to an annotated one,
    # which is far more forgiving than location-string matching. When True we
    # reproduce that behavior; the fuzzy location matcher is kept as a fallback
    # for offline/mock runs and whenever a judge call fails.
    use_llm_judge: bool = True
    # Model used by the judge. None → reuse the reviewer model (self.llm.model).
    # The original benchmark used a separate, stronger judge model.
    judge_model: Optional[str] = None

    # Routing table: snippet_type → verifier_name
    verifier_routing: dict[str, str] = field(
        default_factory=lambda: {
            "EQUATION": "math_equation",
            "FIGURE": "vision",
            "TABLE": "vision",
            "SECTION": "text",
            "SUBSECTION": "text",
            "THEOREM": "text",
            "LEMMA": "text",
            "PROPOSITION": "text",
            "ALGORITHM": "text",
            "APPENDIX": "text",
            "PARAGRAPH": "text",
        }
    )

    def __post_init__(self) -> None:
        # Default verifier configs. Thresholds depend on strictness: under
        # "strict" we demand higher confidence as a second line of defense
        # beyond the stricter prompts; "lenient" keeps the original values.
        if not self.verifiers:
            if self.strictness == "lenient":
                self.verifiers = {
                    "math_equation": VerifierConfig(confidence_threshold=0.7),
                    "vision": VerifierConfig(confidence_threshold=0.6),
                    "text": VerifierConfig(confidence_threshold=0.5),
                    "statistical": VerifierConfig(confidence_threshold=0.7),
                    "citation": VerifierConfig(confidence_threshold=0.6),
                    "triage": VerifierConfig(confidence_threshold=0.0),
                    "llm_only": VerifierConfig(confidence_threshold=0.5),
                    "progressive_math": VerifierConfig(confidence_threshold=0.7),
                }
            else:
                self.verifiers = {
                    "math_equation": VerifierConfig(confidence_threshold=0.7),
                    "vision": VerifierConfig(confidence_threshold=0.75),
                    "text": VerifierConfig(confidence_threshold=0.8),
                    "statistical": VerifierConfig(confidence_threshold=0.8),
                    "citation": VerifierConfig(confidence_threshold=0.8),
                    "triage": VerifierConfig(confidence_threshold=0.0),
                    "llm_only": VerifierConfig(confidence_threshold=0.8),
                    "progressive_math": VerifierConfig(confidence_threshold=0.7),
                }

        # When progressive math is enabled, route EQUATION snippets to it
        if self.use_progressive_math:
            self.verifier_routing["EQUATION"] = "progressive_math"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineConfig":
        """Create config from a dictionary (e.g., loaded from JSON/YAML)."""
        paths = PathsConfig(**data.get("paths", {}))
        llm = LLMConfig(**data.get("llm", {}))
        sandbox = SandboxConfig(**data.get("sandbox", {}))
        segmentation = SegmentationConfig(**data.get("segmentation", {}))
        verifiers = {
            k: VerifierConfig(**v) for k, v in data.get("verifiers", {}).items()
        }
        routing = data.get("verifier_routing", {})
        kwargs: dict[str, Any] = dict(
            paths=paths,
            llm=llm,
            sandbox=sandbox,
            segmentation=segmentation,
            verifiers=verifiers,
            verifier_routing=routing,
            strictness=data.get("strictness", "strict"),
            use_llm_judge=data.get("use_llm_judge", True),
            judge_model=data.get("judge_model"),
            orchestration_mode=data.get("orchestration_mode", "exhaustive"),
            uncertainty_threshold=data.get("uncertainty_threshold", 0.30),
            uncertainty_budget=data.get("uncertainty_budget"),
            llm_only_mode=data.get("llm_only_mode"),
            verify_chunk_chars=data.get("verify_chunk_chars", 2000),
            verify_chunk_overlap=data.get("verify_chunk_overlap", 200),
            use_progressive_math=data.get("use_progressive_math", False),
            eval_category_filter=data.get("eval_category_filter"),
        )
        if "triage_route_map" in data:
            kwargs["triage_route_map"] = data["triage_route_map"]
        return cls(**kwargs)


# Global default configuration instance
default_config = PipelineConfig()
