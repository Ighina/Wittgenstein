"""Pydantic models for the Paperena Verification pipeline.

All shared data structures are defined here to ensure consistency across
the parser, segmentation, verifier, orchestrator, and evaluation layers.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Phase 1 – Dataset exploration
# ---------------------------------------------------------------------------


class ContentItemSchema(BaseModel):
    """Describes the structure of a single element in paper_content."""

    keys_found: list[str] = Field(default_factory=list)
    content_type_value: Optional[str] = None
    has_text: bool = False
    has_image: bool = False
    image_format: Optional[str] = None
    sample_text_preview: Optional[str] = None


class PaperContentSchemaReport(BaseModel):
    """Report produced by analyze_dataset_schema()."""

    total_rows: int
    total_columns: int
    column_names: list[str]
    column_dtypes: dict[str, str]

    content_types: list[str] = Field(default_factory=list)
    keys_found: list[str] = Field(default_factory=list)
    text_item_count: int = 0
    image_item_count: int = 0
    rows_with_images: int = 0
    rows_with_local_content: int = 0
    sample_content_items: list[dict[str, Any]] = Field(default_factory=list)

    error_categories: list[dict[str, Any]] = Field(default_factory=list)
    error_locations_sample: list[str] = Field(default_factory=list)
    error_severities: list[dict[str, Any]] = Field(default_factory=list)
    paper_categories: list[dict[str, Any]] = Field(default_factory=list)

    generated_at: str = Field(
        default_factory=lambda: datetime.now().isoformat()
    )


# ---------------------------------------------------------------------------
# Phase 2 – Paper parsing
# ---------------------------------------------------------------------------


class ContentType(str, Enum):
    """Known content types in paper_content items."""

    TEXT = "text"
    IMAGE_URL = "image_url"
    UNKNOWN = "unknown"


class RawContentItem(BaseModel):
    """A single raw item from the paper_content list."""

    content_type: ContentType
    text: Optional[str] = None
    image_url: Optional[dict[str, Any]] = None
    raw: dict[str, Any] = Field(default_factory=dict)


class EquationBlock(BaseModel):
    """A parsed LaTeX equation block."""

    id: str
    equation_label: Optional[str] = None
    latex: str
    display_mode: bool = True  # True = \[...\], False = \(...\)
    context_before: Optional[str] = None
    context_after: Optional[str] = None


class ImageBlock(BaseModel):
    """A parsed image/figure block."""

    id: str
    caption: Optional[str] = None
    image_path: Optional[str] = None  # Path to decoded temp file
    base64_data: Optional[str] = None  # Raw base64 string
    context_before: Optional[str] = None
    context_after: Optional[str] = None


class TableBlock(BaseModel):
    """A parsed table block (detected from markdown or text)."""

    id: str
    caption: Optional[str] = None
    raw_content: str = ""
    rows: Optional[list[list[str]]] = None
    context_before: Optional[str] = None
    context_after: Optional[str] = None


class PaperSection(BaseModel):
    """A parsed section of the paper."""

    id: str
    section_title: str
    section_level: int = 1
    content: str
    start_index: int = 0
    end_index: int = 0


class TheoremBlock(BaseModel):
    """A detected theorem/lemma/proposition/corollary environment."""

    id: str
    theorem_type: str  # "theorem", "lemma", "proposition", "corollary"
    label: Optional[str] = None  # e.g., "Theorem 1.1"
    statement: str
    proof: Optional[str] = None
    context_before: Optional[str] = None
    context_after: Optional[str] = None


class NormalizedPaper(BaseModel):
    """Fully parsed and normalized paper representation."""

    paper_id: str
    title: str
    paper_category: str

    sections: list[PaperSection] = Field(default_factory=list)
    equations: list[EquationBlock] = Field(default_factory=list)
    images: list[ImageBlock] = Field(default_factory=list)
    tables: list[TableBlock] = Field(default_factory=list)
    theorems: list[TheoremBlock] = Field(default_factory=list)

    # Full text with inline tags for images/tables/equations
    tagged_full_text: str = ""

    # Raw content items (preserved for reference)
    raw_items: list[RawContentItem] = Field(default_factory=list)

    # Metadata
    parse_timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat()
    )


# ---------------------------------------------------------------------------
# Phase 2b – LLM-based content parsing (EnrichedPaper pathway)
# ---------------------------------------------------------------------------


class VerifiableUnit(BaseModel):
    """A single verifiable claim/equation/theorem/table/figure identified by the LLM parser.

    Each unit carries its core content, a list of dependencies on other units
    or symbol definitions, and a verifier route assigned by the LLM.
    """

    unit_id: str
    unit_type: str  # equation, theorem, lemma, proposition, definition, claim, etc.
    content: str  # Core content to verify (latex for equations, statement for theorems, etc.)
    location: str = ""  # Human-readable location descriptor
    dependencies: list[str] = Field(default_factory=list)  # IDs of prerequisite units
    required_context: str = ""  # Assembled context from dependencies (populated at segment time)
    verifier_route: str = ""  # Which verifier should handle this (assigned by LLM)
    is_verifiable: bool = True  # False for boilerplate / standard definitions
    confidence: float = 1.0  # LLM's confidence in its classification [0, 1]
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_chunk_index: int = 0  # Which LLM parse chunk this came from


class SymbolDefinition(BaseModel):
    """A tracked symbol/term definition extracted by the LLM parser."""

    symbol_name: str  # e.g., "X", "f", "\\mathcal{H}"
    domain: str = ""  # e.g., "real", "integer", "Banach space"
    latex: str = ""  # Original LaTeX representation
    natural_language: str = ""  # "Let X be a real Banach space"
    defining_unit_id: str = ""  # The VerifiableUnit that introduces this symbol
    metadata: dict[str, Any] = Field(default_factory=dict)


class EnrichedPaper(BaseModel):
    """Paper representation produced by the LLM-based parser.

    Extends the concept of NormalizedPaper with a graph of verifiable units,
    symbol definitions, and context tracking.
    """

    paper_id: str
    title: str
    paper_category: str

    # Verifiable units identified by the LLM
    verifiable_units: list[VerifiableUnit] = Field(default_factory=list)

    # Symbol/term registry across the paper
    symbol_registry: list[SymbolDefinition] = Field(default_factory=list)

    # Dependency graph: unit_id → list of prerequisite unit_ids
    context_graph: dict[str, list[str]] = Field(default_factory=dict)

    # Full text of unverifiable sections (preserved for reference context)
    unverifiable_context: str = ""

    # The original NormalizedPaper is still built for backward compatibility
    sections: list[PaperSection] = Field(default_factory=list)
    equations: list[EquationBlock] = Field(default_factory=list)
    images: list[ImageBlock] = Field(default_factory=list)
    tables: list[TableBlock] = Field(default_factory=list)
    theorems: list[TheoremBlock] = Field(default_factory=list)
    tagged_full_text: str = ""
    raw_items: list[RawContentItem] = Field(default_factory=list)
    parse_timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class LLMParseChunkResult(BaseModel):
    """Structured output from a single LLM parser call over one text chunk."""

    chunk_index: int
    units: list[VerifiableUnit] = Field(default_factory=list)
    symbols: list[SymbolDefinition] = Field(default_factory=list)
    unverifiable_text: str = ""
    section_headers: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Phase 3 – Error location parsing
# ---------------------------------------------------------------------------


class LocationType(str, Enum):
    """Types of locations that can be referenced in error annotations."""

    EQUATION = "equation"
    FIGURE = "figure"
    TABLE = "table"
    SECTION = "section"
    THEOREM = "theorem"
    LEMMA = "lemma"
    PROPOSITION = "proposition"
    COROLLARY = "corollary"
    CLAIM = "claim"
    ALGORITHM = "algorithm"
    APPENDIX = "appendix"
    PAGE = "page"
    OVERALL = "overall"
    UNKNOWN = "unknown"


class LocationReference(BaseModel):
    """Structured representation of an error location."""

    raw: str
    location_type: LocationType
    identifier: str  # e.g., "4", "3.1", "2.3", "3,4"
    identifiers: list[str] = Field(default_factory=list)  # Split multiple refs
    is_range: bool = False
    normalized: str = ""  # Canonical form for fuzzy matching

    def model_post_init(self, __context: Any) -> None:
        if not self.normalized:
            self.normalized = f"{self.location_type.value} {self.identifier}"


# ---------------------------------------------------------------------------
# Phase 4 – Segmentation
# ---------------------------------------------------------------------------


class SnippetType(str, Enum):
    """Types of verification snippets."""

    SECTION = "SECTION"
    SUBSECTION = "SUBSECTION"
    EQUATION = "EQUATION"
    FIGURE = "FIGURE"
    TABLE = "TABLE"
    THEOREM = "THEOREM"
    LEMMA = "LEMMA"
    PROPOSITION = "PROPOSITION"
    COROLLARY = "COROLLARY"
    ALGORITHM = "ALGORITHM"
    APPENDIX = "APPENDIX"
    PARAGRAPH = "PARAGRAPH"


class VerificationSnippet(BaseModel):
    """A single unit of content to be verified."""

    snippet_id: str
    snippet_type: SnippetType
    paper_id: str

    # Human-readable location descriptor e.g., "Equation 7", "Section 3.1"
    location: str

    # The actual content for verification
    content: str

    # For image/table snippets, path to the decoded image file
    image_path: Optional[str] = None

    # Structured reference for alignment
    location_ref: Optional[LocationReference] = None

    # Explicit verifier assignment from LLM parser (overrides type-based routing)
    verifier_route: Optional[str] = None

    # Context assembled from dependency graph (LLM parser mode only)
    dependency_context: str = ""

    # Additional context
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Character / token estimate
    content_length: int = 0
    estimated_tokens: int = 0

    def model_post_init(self, __context: Any) -> None:
        if not self.content_length:
            self.content_length = len(self.content)
        if not self.estimated_tokens:
            # Rough estimate: ~4 chars per token
            self.estimated_tokens = max(1, self.content_length // 4)


# ---------------------------------------------------------------------------
# Phase 5-9 – Verification results
# ---------------------------------------------------------------------------


class VerificationStatus(str, Enum):
    """Status of a verification attempt."""

    VALID = "VALID"
    INVALID = "INVALID"
    MALFORMED = "MALFORMED"
    UNVERIFIABLE = "UNVERIFIABLE"
    ERROR_DETECTED = "ERROR_DETECTED"
    NO_ERROR = "NO_ERROR"
    SKIPPED = "SKIPPED"


class BaseVerificationResult(BaseModel):
    """Base result from any verifier."""

    snippet_id: str
    verifier_name: str
    status: VerificationStatus
    error_detected: bool = False
    confidence: float = 0.0
    reasoning: str = ""
    predicted_error_category: Optional[str] = None
    execution_time_ms: float = 0.0


class TriageResult(BaseModel):
    """Output of the general triage pass over a single snippet.

    The triage verifier does NOT decide whether an error exists — it estimates
    *where uncertainty is concentrated* so specialized verifiers can be routed
    only to the regions that warrant deeper checking (see
    UncertaintyOrchestrator). ``uncertainty`` is the model's estimated
    probability that this snippet contains a correction/retraction-worthy error.
    """

    snippet_id: str
    snippet_type: str = ""
    location: str = ""
    uncertainty: float = 0.0          # [0, 1] — expected error density
    suggested_route: str = "text"     # semantic specialist label (see config.triage_route_map)
    reason: str = ""
    selected: bool = False            # escalated to a specialized verifier?
    routed_to: Optional[str] = None   # registered verifier the snippet was sent to
    execution_time_ms: float = 0.0


class EquationVerificationResult(BaseVerificationResult):
    """Result from MathEquationVerifier or ProgressiveMathVerifier."""

    sympy_code: Optional[str] = None
    execution_output: Optional[str] = None
    execution_error: Optional[str] = None
    return_code: Optional[int] = None

    # -- Progressive verifier fields (optional, default-empty) --
    statement_class: Optional[str] = None
    proof_obligations: list[dict[str, Any]] = Field(default_factory=list)
    verification_layer: Optional[str] = None
    context_snapshot: Optional[dict[str, Any]] = None


class VisionVerificationResult(BaseVerificationResult):
    """Result from VisionVerifier (figure or table)."""

    content_type: str = "figure"  # "figure" or "table"
    image_path: Optional[str] = None
    caption_text: Optional[str] = None


class TextVerificationResult(BaseVerificationResult):
    """Result from TextVerifier."""

    snippet_type: str = "paragraph"
    contradiction_locations: list[str] = Field(default_factory=list)


class StatisticalVerificationResult(BaseVerificationResult):
    """Result from StatisticalVerifier (deterministic numeric recomputation)."""

    # Each check: {description, expr, expected, computed, tolerance, passed, error}
    checks: list[dict[str, Any]] = Field(default_factory=list)


class CitationVerificationResult(BaseVerificationResult):
    """Result from CitationVerifier (attribution / novelty / reference use)."""

    snippet_type: str = "paragraph"


# Union type for verifier results
VerificationResult = (
    EquationVerificationResult
    | VisionVerificationResult
    | TextVerificationResult
    | StatisticalVerificationResult
    | CitationVerificationResult
    | BaseVerificationResult
)


# ---------------------------------------------------------------------------
# Phase 10 – Aggregation & predictions
# ---------------------------------------------------------------------------


class PredictedError(BaseModel):
    """A single predicted error in a paper."""

    error_category: str
    error_location: str
    confidence: float
    supporting_evidence: str
    verifier_name: str = ""
    snippet_id: str = ""


class PaperPrediction(BaseModel):
    """Paper-level prediction aggregating all verifier findings."""

    paper_id: str
    title: str = ""
    paper_category: str = ""

    predicted_errors: list[PredictedError] = Field(default_factory=list)

    # Statistics
    total_snippets: int = 0
    snippets_verified: int = 0
    errors_detected: int = 0
    verifier_usage: dict[str, int] = Field(default_factory=dict)

    # Raw verifier results for debugging
    raw_results: list[dict[str, Any]] = Field(default_factory=list)

    # Uncertainty map produced by the triage pass (empty in exhaustive mode).
    # Each entry is a serialized TriageResult.
    uncertainty_map: list[dict[str, Any]] = Field(default_factory=list)

    generation_timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat()
    )


# ---------------------------------------------------------------------------
# Phase 11 – Ground truth alignment
# ---------------------------------------------------------------------------


class AlignedPrediction(BaseModel):
    """A prediction matched against a ground-truth annotation."""

    paper_id: str
    predicted: PredictedError
    matched_ground_truth: bool = False
    ground_truth_category: Optional[str] = None
    ground_truth_location: Optional[str] = None
    ground_truth_severity: Optional[str] = None
    ground_truth_annotation: Optional[str] = None
    match_quality: float = 0.0  # 0.0 to 1.0, quality of fuzzy match
    is_true_positive: bool = False
    is_false_positive: bool = False


# ---------------------------------------------------------------------------
# Phase 12 – Evaluation metrics
# ---------------------------------------------------------------------------


class CategoryMetrics(BaseModel):
    """Metrics for a single category (error type, severity, or paper field)."""

    category_name: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    accuracy: float = 0.0
    support: int = 0


class EvaluationMetrics(BaseModel):
    """Complete evaluation metrics for a pipeline run."""

    # Binary counts
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    # Overall metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0

    # Per-category breakdowns
    by_paper_category: list[CategoryMetrics] = Field(default_factory=list)
    by_error_category: list[CategoryMetrics] = Field(default_factory=list)
    by_error_severity: list[CategoryMetrics] = Field(default_factory=list)

    # Metadata
    total_papers: int = 0
    total_ground_truth_errors: int = 0
    total_predictions: int = 0
    matched_predictions: int = 0

    # Timestamp
    computed_at: str = Field(
        default_factory=lambda: datetime.now().isoformat()
    )
