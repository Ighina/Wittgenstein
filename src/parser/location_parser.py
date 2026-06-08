"""Phase 3: Error location parsing.

Parses human-written error location strings into structured LocationReference
objects that can be used by both verifiers and the evaluation stage.
"""

from __future__ import annotations

import re
from typing import Optional

from loguru import logger

from src.models import LocationReference, LocationType


# ---------------------------------------------------------------------------
# Location parsing patterns — ordered from most specific to most general
# ---------------------------------------------------------------------------

LOCATION_PATTERNS: list[tuple[re.Pattern[str], LocationType]] = [
    # Equation references
    (
        re.compile(
            r"(?:Equation|Eq\.?)\s*\(?\s*(\d+(?:[a-zA-Z])?(?:\s*[,&]\s*\d+(?:[a-zA-Z])?)*)\s*\)?",
            re.IGNORECASE,
        ),
        LocationType.EQUATION,
    ),
    # Multi-figure references: Fig1, Fig2, Fig 6A
    (
        re.compile(
            r"(?:Figure|Fig\.?)\s*(\d+(?:[a-zA-Z])?)"
            r"(?:\s*[,&]\s*(?:Figure|Fig\.?)\s*(\d+(?:[a-zA-Z])?))+",
            re.IGNORECASE,
        ),
        LocationType.FIGURE,
    ),
    # Figure references: Fig 5, Fig. 4, Figure 2d, Fig1, Fig 2B, Fig 6A
    (
        re.compile(
            r"(?:Figure|Fig\.?)\s*(\d+(?:[a-zA-Z])?(?:\s*[,&]\s*\d+(?:[a-zA-Z])?)*)",
            re.IGNORECASE,
        ),
        LocationType.FIGURE,
    ),
    # Table references: Table 2, Table. 1
    (
        re.compile(
            r"(?:Table)\.?\s*(\d+(?:[a-zA-Z])?(?:\s*[,&]\s*\d+(?:[a-zA-Z])?)*)",
            re.IGNORECASE,
        ),
        LocationType.TABLE,
    ),
    # Lemma references: Lemma 3,4, Lemma 1, Lemma 4.2
    (
        re.compile(
            r"(?:Lemma)\s*(\d+(?:\.\d+)?(?:\s*[,&]\s*\d+(?:\.\d+)?)*)",
            re.IGNORECASE,
        ),
        LocationType.LEMMA,
    ),
    # Theorem references: Theorem 1.1, Theorem 2.3, Theorems 1.2, 1.3, Theorem 7
    (
        re.compile(
            r"(?:Theorem)s?\s*(\d+(?:\.\d+)?(?:\s*[,&]\s*\d+(?:\.\d+)?)*)",
            re.IGNORECASE,
        ),
        LocationType.THEOREM,
    ),
    # Proposition references: Proposition 2, Proposition 3.9, Proposition 4.6
    (
        re.compile(
            r"(?:Proposition)\s*(\d+(?:\.\d+)?(?:\s*[,&]\s*\d+(?:\.\d+)?)*)",
            re.IGNORECASE,
        ),
        LocationType.PROPOSITION,
    ),
    # Corollary references: 1.10. Corollary
    (
        re.compile(
            r"(\d+(?:\.\d+)*)\.?\s*(?:Corollary)",
            re.IGNORECASE,
        ),
        LocationType.COROLLARY,
    ),
    (
        re.compile(
            r"(?:Corollary)\s*(\d+(?:\.\d+)?)",
            re.IGNORECASE,
        ),
        LocationType.COROLLARY,
    ),
    # Claim references: Claim 3, Claim 7
    (
        re.compile(
            r"(?:Claim)\s*(\d+(?:\.\d+)?)",
            re.IGNORECASE,
        ),
        LocationType.CLAIM,
    ),
    # Section references: Section 4.2.3, Sec 3.1, §3.1, Sec 3, Sec4, Section. 3.1.2
    (
        re.compile(
            r"(?:Section|Sec|§)\.?\s*(\d+(?:\.\d+)*(?:\s*[,&]\s*\d+(?:\.\d+)*)*)",
            re.IGNORECASE,
        ),
        LocationType.SECTION,
    ),
    # Appendix references: Appendix B
    (
        re.compile(
            r"(?:Appendix)\s+([A-Za-z0-9]+(?:\s*[,&]\s*[A-Za-z0-9]+)*)",
            re.IGNORECASE,
        ),
        LocationType.APPENDIX,
    ),
    # Page references: Page 4, Page 5
    (
        re.compile(
            r"(?:Page)\s+(\d+(?:\s*[,&]\s*\d+)*)",
            re.IGNORECASE,
        ),
        LocationType.PAGE,
    ),
    # Algorithm references
    (
        re.compile(
            r"(?:Algorithm)\s+(\d+(?:[a-zA-Z])?)",
            re.IGNORECASE,
        ),
        LocationType.ALGORITHM,
    ),
]


def parse_error_location(raw: str) -> LocationReference:
    """Parse a raw error location string into a LocationReference.

    Handles the wide variety of location formats found in the dataset:
    - "Equation 6", "Eq. 1", "Eq. (12)"
    - "Fig 5", "Fig. 4", "Figure 5", "Fig 2d", "Fig1, Fig2"
    - "Lemma 3,4", "Lemma 1", "Lemma 4.2"
    - "Theorem 1.1", "Theorems 1.2, 1.3", "Theorem 7"
    - "Section 4.2.3", "Sec 3", "§3.1"
    - "Table 2", "Table. 1"
    - "Page 4", "Page 5"

    Args:
        raw: The raw error_location string from the dataset.

    Returns:
        A structured LocationReference.
    """
    raw = raw.strip()

    # Try each pattern in order
    for pattern, loc_type in LOCATION_PATTERNS:
        match = pattern.search(raw)
        if match:
            # Collect all numeric groups (handle multi-reference patterns)
            groups = [g for g in match.groups() if g is not None]
            if groups:
                id_str = ",".join(groups)
            else:
                id_str = match.group(0)

            identifiers = _split_identifiers(id_str)
            is_range = len(identifiers) > 1

            normalized = _normalize_location(loc_type, id_str)

            return LocationReference(
                raw=raw,
                location_type=loc_type,
                identifier=id_str,
                identifiers=identifiers,
                is_range=is_range,
                normalized=normalized,
            )

    # Fallback: unknown or non-standard locations
    # Check for "Overall", "Introduction", "Unknown", "Computations"
    raw_lower = raw.lower()
    if any(w in raw_lower for w in ["overall", "overview", "unknown"]):
        loc_type = LocationType.OVERALL if "over" in raw_lower else LocationType.UNKNOWN
        return LocationReference(
            raw=raw,
            location_type=loc_type,
            identifier=raw,
            identifiers=[raw],
            is_range=False,
            normalized=raw_lower.replace(" ", "_"),
        )

    if "introduction" in raw_lower:
        return LocationReference(
            raw=raw,
            location_type=LocationType.SECTION,
            identifier="introduction",
            identifiers=["introduction"],
            is_range=False,
            normalized="section introduction",
        )

    if "appendix" in raw_lower:
        return LocationReference(
            raw=raw,
            location_type=LocationType.APPENDIX,
            identifier=raw,
            identifiers=[raw],
            is_range=False,
            normalized=raw_lower,
        )

    # Names like "Kahler Twists"
    if re.match(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$", raw):
        return LocationReference(
            raw=raw,
            location_type=LocationType.SECTION,
            identifier=raw.lower().replace(" ", "_"),
            identifiers=[raw],
            is_range=False,
            normalized=f"section {raw.lower().replace(' ', '_')}",
        )

    # Default: unknown
    logger.debug(f"Unrecognized location format: '{raw}'")
    return LocationReference(
        raw=raw,
        location_type=LocationType.UNKNOWN,
        identifier=raw,
        identifiers=[raw],
        is_range=False,
        normalized=raw_lower,
    )


def _split_identifiers(id_str: str) -> list[str]:
    """Split a compound identifier string into individual identifiers.

    Handles: "3,4", "1.2, 1.3", "3.1.2", "2d"
    """
    # Split on comma or "and"/"&"
    parts = re.split(r"\s*[,&]\s*|\s+and\s+", id_str)
    return [p.strip() for p in parts if p.strip()]


def _normalize_location(loc_type: LocationType, identifier: str) -> str:
    """Create a canonical normalized form for fuzzy matching.

    Examples:
        equation 6, eq 1
        figure 5, fig 4
        lemma 3,4
        theorem 1.1
        section 4.2.3
        table 2
    """
    type_name = loc_type.value
    # Remove spaces and standardize
    clean_id = identifier.replace(" ", "")
    return f"{type_name} {clean_id}"


def fuzzy_match_locations(
    loc_a: str | LocationReference,
    loc_b: str | LocationReference,
) -> float:
    """Compute a fuzzy match score (0.0 to 1.0) between two locations.

    Handles equivalence classes like:
        "Equation 7" ↔ "Eq. (7)" ↔ "equation 7"
        "Fig 5" ↔ "Figure 5" ↔ "FIGURE 5"
        "Section 3.1" ↔ "Sec 3.1" ↔ "§3.1"

    Args:
        loc_a: First location (string or LocationReference).
        loc_b: Second location (string or LocationReference).

    Returns:
        Match score between 0.0 (no match) and 1.0 (exact match).
    """
    # Parse strings into LocationReferences
    ref_a = loc_a if isinstance(loc_a, LocationReference) else parse_error_location(loc_a)
    ref_b = loc_b if isinstance(loc_b, LocationReference) else parse_error_location(loc_b)

    # If types differ, no match
    if ref_a.location_type != ref_b.location_type:
        # Special case: both are "unknown" → partial match possible
        if ref_a.location_type == LocationType.UNKNOWN or ref_b.location_type == LocationType.UNKNOWN:
            return _string_similarity(ref_a.normalized, ref_b.normalized) * 0.5
        return 0.0

    # Compare identifiers
    a_ids = set(ref_a.identifiers)
    b_ids = set(ref_b.identifiers)

    if not a_ids or not b_ids:
        return 0.0

    # Exact identifier match
    if a_ids == b_ids:
        return 1.0

    # Partial overlap
    overlap = a_ids & b_ids
    if overlap:
        jaccard = len(overlap) / len(a_ids | b_ids)
        return 0.5 + 0.5 * jaccard

    # String similarity fallback
    return _string_similarity(ref_a.normalized, ref_b.normalized)


def _string_similarity(a: str, b: str) -> float:
    """Compute a simple string similarity score."""
    a_lower = a.lower().replace(" ", "").replace(".", "").replace("(", "").replace(")", "")
    b_lower = b.lower().replace(" ", "").replace(".", "").replace("(", "").replace(")", "")

    if a_lower == b_lower:
        return 1.0

    # Check one is a substring of the other
    if a_lower in b_lower or b_lower in a_lower:
        return 0.85

    # Simple character overlap
    a_chars = set(a_lower)
    b_chars = set(b_lower)
    if not a_chars or not b_chars:
        return 0.0

    overlap = len(a_chars & b_chars)
    total = len(a_chars | b_chars)
    return overlap / total if total > 0 else 0.0
