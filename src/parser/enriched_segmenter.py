"""Phase 4 (LLM path): Enriched paper segmentation.

Takes an EnrichedPaper (produced by the LLM parser) and produces a list of
VerificationSnippets. Each snippet carries:
  - The unit's core content
  - Full dependency context baked in (definitions, prior lemmas)
  - An explicit verifier_route assigned by the LLM
  - A location descriptor for ground-truth alignment

Unverifiable units are skipped (they contribute context but are not separately
verified). Figure and table units that reference images in the base paper are
assigned image_paths from the NormalizedPaper.
"""

from __future__ import annotations

from typing import Optional

from loguru import logger

from src.config import PipelineConfig, SegmentationConfig, default_config
from src.models import (
    EnrichedPaper,
    LocationReference,
    SnippetType,
    VerificationSnippet,
)
from src.parser.location_parser import parse_error_location


# Map from the LLM parser's unit_type strings to SnippetType enum values.
UNIT_TYPE_TO_SNIPPET: dict[str, SnippetType] = {
    "equation": SnippetType.EQUATION,
    "theorem": SnippetType.THEOREM,
    "lemma": SnippetType.LEMMA,
    "proposition": SnippetType.PROPOSITION,
    "corollary": SnippetType.COROLLARY,
    "definition": SnippetType.PARAGRAPH,
    "claim": SnippetType.PARAGRAPH,
    "numeric_claim": SnippetType.PARAGRAPH,
    "proof_step": SnippetType.PARAGRAPH,
    "table_data": SnippetType.TABLE,
    "figure_reference": SnippetType.FIGURE,
    "boilerplate": SnippetType.PARAGRAPH,
}


def segment_enriched_paper(
    paper: EnrichedPaper,
    config: Optional[PipelineConfig] = None,
) -> list[VerificationSnippet]:
    """Produce VerificationSnippets from an EnrichedPaper.

    Only verifiable units (is_verifiable=True) are turned into snippets.
    Each snippet's content is the unit's core content prefixed by its
    dependency context. The LLM-assigned verifier_route is preserved.

    Args:
        paper: The EnrichedPaper from the LLM parser.
        config: Pipeline configuration.

    Returns:
        List of VerificationSnippets ready for the orchestrator.
    """
    if config is None:
        config = default_config

    seg_config = config.segmentation

    snippets: list[VerificationSnippet] = []
    snippet_counts: dict[str, int] = {}

    for unit in paper.verifiable_units:
        if not unit.is_verifiable:
            continue

        snippet_type = _map_unit_type(unit.unit_type)

        # Build full content: dependency context + core content
        content_parts: list[str] = []
        if unit.required_context:
            content_parts.append("## Prerequisite Context\n")
            content_parts.append(unit.required_context)
            content_parts.append("\n## Verifiable Unit\n")
        content_parts.append(unit.content)

        full_content = "\n".join(content_parts)

        # Enforce max size by truncating context if needed
        if len(full_content) > seg_config.max_snippet_chars:
            # Keep the unit content intact; trim context
            overhead = len("\n## Verifiable Unit\n") + len(unit.content)
            max_ctx = seg_config.max_snippet_chars - overhead
            if unit.required_context and max_ctx > 200:
                truncated_ctx = unit.required_context[:max_ctx]
                truncated_ctx += "\n[...context truncated...]"
                full_content = (
                    "## Prerequisite Context\n"
                    + truncated_ctx
                    + "\n## Verifiable Unit\n"
                    + unit.content
                )
            else:
                full_content = unit.content[:seg_config.max_snippet_chars]

        # Route: LLM-assigned verifier_route takes priority
        verifier_route = unit.verifier_route if unit.verifier_route else None

        # Location reference
        location_str = unit.location or f"{unit.unit_type} {unit.unit_id}"
        location_ref = parse_error_location(location_str)

        # For visual content, try to find the matching image
        image_path: Optional[str] = None
        if unit.unit_type in ("figure_reference", "table_data"):
            image_path = _find_image_for_unit(unit, paper)

        snippet = VerificationSnippet(
            snippet_id=unit.unit_id,
            snippet_type=snippet_type,
            paper_id=paper.paper_id,
            location=location_str,
            content=full_content,
            image_path=image_path,
            location_ref=location_ref,
            verifier_route=verifier_route,
            dependency_context=unit.required_context,
            metadata={
                "unit_type": unit.unit_type,
                "verifier_route": unit.verifier_route,
                "llm_confidence": unit.confidence,
                "dependency_count": len(unit.dependencies),
                "source_chunk_index": unit.source_chunk_index,
            },
        )
        snippets.append(snippet)

        snippet_counts[unit.unit_type] = (
            snippet_counts.get(unit.unit_type, 0) + 1
        )

    # Log summary
    unverifiable_count = sum(
        1 for u in paper.verifiable_units if not u.is_verifiable
    )
    logger.info(
        f"Segmented enriched paper {paper.paper_id} into {len(snippets)} snippets "
        f"(skipped {unverifiable_count} unverifiable units). "
        f"Breakdown: {dict(snippet_counts)}"
    )

    return snippets


def _map_unit_type(unit_type: str) -> SnippetType:
    """Map an LLM parser unit_type string to a SnippetType enum value.

    Args:
        unit_type: The unit_type assigned by the LLM parser.

    Returns:
        The corresponding SnippetType.
    """
    return UNIT_TYPE_TO_SNIPPET.get(unit_type, SnippetType.PARAGRAPH)


def _find_image_for_unit(unit, paper: EnrichedPaper) -> Optional[str]:
    """Try to find an image file path for a figure_reference or table_data unit.

    Matches by looking at unit content for figure/table numbers and finding
    the corresponding ImageBlock in the base NormalizedPaper.

    Args:
        unit: The VerifiableUnit referencing a figure or table.
        paper: The EnrichedPaper with base paper data.

    Returns:
        Image file path if found, else None.
    """
    # Simple heuristic: match figure/table number in unit location
    import re

    numbers = re.findall(r"(\d+)", unit.location or "")
    if not numbers:
        numbers = re.findall(r"(\d+)", unit.content or "")

    target_num = numbers[0] if numbers else None
    if target_num is None:
        return None

    # Search images for matching caption
    for img in paper.images:
        if img.caption and target_num in img.caption:
            return img.image_path
        if img.id and target_num in img.id:
            return img.image_path

    return None
