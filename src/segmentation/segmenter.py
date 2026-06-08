"""Phase 4: Verification-oriented paper segmentation.

Splits a NormalizedPaper into compact VerificationSnippets suitable for
LLM-based verification. Each snippet is small enough to fit in a typical
context window while preserving enough context for meaningful analysis.
"""

from __future__ import annotations

from typing import Optional

from loguru import logger

from src.config import SegmentationConfig, default_config
from src.models import (
    EquationBlock,
    ImageBlock,
    LocationType,
    NormalizedPaper,
    PaperSection,
    SnippetType,
    TableBlock,
    TheoremBlock,
    VerificationSnippet,
)
from src.parser.location_parser import parse_error_location


def segment_paper(
    paper: NormalizedPaper,
    config: Optional[SegmentationConfig] = None,
) -> list[VerificationSnippet]:
    """Split a normalized paper into verification snippets.

    Follows the natural scientific structure of the paper, producing
    snippets typed as SECTION, EQUATION, FIGURE, TABLE, THEOREM, LEMMA, etc.

    Args:
        paper: The parsed NormalizedPaper.
        config: Segmentation configuration (max sizes, overlap).

    Returns:
        List of VerificationSnippets ready for verification.
    """
    if config is None:
        config = default_config.segmentation

    snippets: list[VerificationSnippet] = []

    # 1. Section-based snippets
    snippets.extend(_segment_sections(paper, config))

    # 2. Equation snippets
    snippets.extend(_segment_equations(paper))

    # 3. Figure snippets
    snippets.extend(_segment_images(paper))

    # 4. Table snippets
    snippets.extend(_segment_tables(paper))

    # 5. Theorem/lemma/proposition snippets
    snippets.extend(_segment_theorems(paper))

    logger.info(
        f"Segmented paper {paper.paper_id} into {len(snippets)} snippets: "
        f"sections={sum(1 for s in snippets if s.snippet_type == SnippetType.SECTION)}, "
        f"equations={sum(1 for s in snippets if s.snippet_type == SnippetType.EQUATION)}, "
        f"figures={sum(1 for s in snippets if s.snippet_type == SnippetType.FIGURE)}, "
        f"tables={sum(1 for s in snippets if s.snippet_type == SnippetType.TABLE)}, "
        f"theorems={sum(1 for s in snippets if s.snippet_type in (
            SnippetType.THEOREM, SnippetType.LEMMA,
            SnippetType.PROPOSITION, SnippetType.COROLLARY
        ))}"
    )

    return snippets


def _segment_sections(
    paper: NormalizedPaper,
    config: SegmentationConfig,
) -> list[VerificationSnippet]:
    """Create SECTION snippets from paper sections.

    Sections may be split into smaller chunks if they exceed max_section_chars.
    """
    snippets: list[VerificationSnippet] = []

    for i, section in enumerate(paper.sections):
        content = section.content
        section_level = section.section_level
        snippet_type = (
            SnippetType.SUBSECTION if section_level >= 3 else SnippetType.SECTION
        )

        # Split long sections into manageable chunks with overlap
        if len(content) > config.max_section_chars:
            chunks = _chunk_text(
                content,
                chunk_size=config.max_snippet_chars,
                overlap=config.overlap_chars,
            )
            for j, chunk in enumerate(chunks):
                loc = (
                    f"Section {section.section_title}"
                    if j == 0
                    else f"Section {section.section_title} (part {j + 1})"
                )
                snippets.append(VerificationSnippet(
                    snippet_id=f"{paper.paper_id}_sec_{i}_part_{j}",
                    snippet_type=snippet_type,
                    paper_id=paper.paper_id,
                    location=loc,
                    content=chunk,
                    location_ref=parse_error_location(f"Section {section.section_title}"),
                    metadata={
                        "section_title": section.section_title,
                        "section_level": section_level,
                        "part": j,
                        "total_parts": len(chunks),
                    },
                ))
        else:
            snippets.append(VerificationSnippet(
                snippet_id=f"{paper.paper_id}_sec_{i}",
                snippet_type=snippet_type,
                paper_id=paper.paper_id,
                location=f"Section {section.section_title}",
                content=content,
                location_ref=parse_error_location(f"Section {section.section_title}"),
                metadata={
                    "section_title": section.section_title,
                    "section_level": section_level,
                },
            ))

    return snippets


def _segment_equations(paper: NormalizedPaper) -> list[VerificationSnippet]:
    """Create EQUATION snippets from extracted equation blocks."""
    snippets: list[VerificationSnippet] = []

    for eq in paper.equations:
        # Build context-rich content
        parts: list[str] = []
        if eq.context_before:
            parts.append(f"Context before:\n{eq.context_before[-500:]}")
        parts.append(f"Equation:\n{eq.latex}")
        if eq.context_after:
            parts.append(f"Context after:\n{eq.context_after[:500]}")

        content = "\n\n".join(parts)
        label = eq.equation_label or f"Equation {eq.id}"

        snippets.append(VerificationSnippet(
            snippet_id=eq.id,
            snippet_type=SnippetType.EQUATION,
            paper_id=paper.paper_id,
            location=label,
            content=content,
            location_ref=parse_error_location(label),
            metadata={
                "equation_label": label,
                "display_mode": eq.display_mode,
                "latex": eq.latex,
            },
        ))

    return snippets


def _segment_images(paper: NormalizedPaper) -> list[VerificationSnippet]:
    """Create FIGURE snippets from extracted image blocks."""
    snippets: list[VerificationSnippet] = []

    for img in paper.images:
        # Build content with caption and context
        parts: list[str] = []
        if img.caption:
            parts.append(f"Caption: {img.caption}")
        if img.context_before:
            parts.append(f"Context before:\n{img.context_before[-300:]}")
        if img.context_after:
            parts.append(f"Context after:\n{img.context_after[:300]}")

        content = "\n\n".join(parts) if parts else f"Figure: {img.caption or img.id}"

        snippets.append(VerificationSnippet(
            snippet_id=img.id,
            snippet_type=SnippetType.FIGURE,
            paper_id=paper.paper_id,
            location=img.caption or f"Figure {img.id}",
            content=content,
            image_path=img.image_path,
            location_ref=parse_error_location(img.caption or f"Figure {img.id}"),
            metadata={
                "caption": img.caption,
                "has_image_file": img.image_path is not None,
            },
        ))

    return snippets


def _segment_tables(paper: NormalizedPaper) -> list[VerificationSnippet]:
    """Create TABLE snippets from extracted table blocks."""
    snippets: list[VerificationSnippet] = []

    for tbl in paper.tables:
        parts: list[str] = []
        if tbl.caption:
            parts.append(f"Caption: {tbl.caption}")
        parts.append(f"Table content:\n{tbl.raw_content}")

        content = "\n\n".join(parts)

        snippets.append(VerificationSnippet(
            snippet_id=tbl.id,
            snippet_type=SnippetType.TABLE,
            paper_id=paper.paper_id,
            location=tbl.caption or f"Table {tbl.id}",
            content=content,
            location_ref=parse_error_location(tbl.caption or f"Table {tbl.id}"),
            metadata={
                "caption": tbl.caption,
                "row_count": len(tbl.rows) if tbl.rows else 0,
            },
        ))

    return snippets


def _segment_theorems(paper: NormalizedPaper) -> list[VerificationSnippet]:
    """Create THEOREM/LEMMA/PROPOSITION snippets from theorem blocks."""
    snippets: list[VerificationSnippet] = []

    type_to_snippet: dict[str, SnippetType] = {
        "theorem": SnippetType.THEOREM,
        "lemma": SnippetType.LEMMA,
        "proposition": SnippetType.PROPOSITION,
        "corollary": SnippetType.COROLLARY,
        "claim": SnippetType.THEOREM,
    }

    for thm in paper.theorems:
        snippet_type = type_to_snippet.get(
            thm.theorem_type, SnippetType.THEOREM
        )

        parts: list[str] = []
        if thm.label:
            parts.append(f"**{thm.label}**")
        parts.append(f"Statement: {thm.statement}")
        if thm.proof:
            parts.append(f"Proof: {thm.proof}")
        if thm.context_before:
            parts.append(f"Context before:\n{thm.context_before[-300:]}")
        if thm.context_after:
            parts.append(f"Context after:\n{thm.context_after[:300]}")

        content = "\n\n".join(parts)
        loc = thm.label or f"{thm.theorem_type} {thm.id}"

        snippets.append(VerificationSnippet(
            snippet_id=thm.id,
            snippet_type=snippet_type,
            paper_id=paper.paper_id,
            location=loc,
            content=content,
            location_ref=parse_error_location(loc),
            metadata={
                "theorem_type": thm.theorem_type,
                "label": thm.label,
                "has_proof": thm.proof is not None,
            },
        ))

    return snippets


def _chunk_text(
    text: str,
    chunk_size: int = 4000,
    overlap: int = 200,
) -> list[str]:
    """Split long text into overlapping chunks, respecting paragraph boundaries.

    Args:
        text: The text to split.
        chunk_size: Maximum characters per chunk.
        overlap: Number of overlapping characters between chunks.

    Returns:
        List of text chunks.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            chunks.append(text[start:])
            break

        # Try to break at a paragraph boundary
        chunk = text[start:end]
        para_break = chunk.rfind("\n\n")
        if para_break > chunk_size // 2:
            end = start + para_break

        chunks.append(text[start:end])
        start = end - overlap

    return chunks
