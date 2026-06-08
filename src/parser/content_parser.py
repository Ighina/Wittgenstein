"""Phase 2: Paper content parsing.

Transforms the raw paper_content list of dicts into a structured
NormalizedPaper with sections, equations, images, tables, and theorems.
"""

from __future__ import annotations

import base64
import re
import tempfile
from pathlib import Path
from typing import Any, Optional

from loguru import logger
from PIL import Image

from src.models import (
    ContentType,
    EquationBlock,
    ImageBlock,
    NormalizedPaper,
    PaperSection,
    RawContentItem,
    TableBlock,
    TheoremBlock,
)


# ---------------------------------------------------------------------------
# Regex patterns for structure detection
# ---------------------------------------------------------------------------

# Markdown-style section headers: ## 2. Title, ### 3.1 Subtitle
SECTION_PATTERN = re.compile(
    r"^(#{1,4})\s*"  # 1-4 hash marks
    r"(\d+(?:\.\d+)*)?"  # Optional number like 2, 3.1, 4.2.3
    r"\.?\s*"  # Optional period
    r"(.+?)"  # Title text
    r"\s*$",
    re.MULTILINE,
)

# Bold-style headers: **Theorem 1.1.**, **Lemma 2.**
BOLD_HEADER_PATTERN = re.compile(
    r"\*\*"  # Opening bold
    r"(Theorem|Lemma|Proposition|Corollary|Definition|Remark|Example|Algorithm|Proof)"
    r"(?:\s+(\d+(?:\.\d+)*))?"  # Optional number
    r"\.?"  # Optional period
    r"\*\*"  # Closing bold
    r"\s*",
)

# LaTeX equation patterns
DISPLAY_EQUATION_PATTERN = re.compile(
    r"\\\[(.*?)\\\]"  # \[ ... \]
    r"|"  # or
    r"\$\$(.*?)\$\$"  # $$ ... $$
    r"|"  # or
    r"\\begin\{equation\*?\}(.*?)\\end\{equation\*?\}"  # \begin{equation}...\end{equation}
    r"|"  # or
    r"\\begin\{align\*?\}(.*?)\\end\{align\*?\}"  # \begin{align}...\end{align}
    r"|"  # or
    r"\\begin\{eqnarray\*?\}(.*?)\\end\{eqnarray\*?\}",  # \begin{eqnarray}...\end{eqnarray}
    re.DOTALL,
)

INLINE_EQUATION_PATTERN = re.compile(
    r"(?<!\$)"  # Not preceded by $
    r"\\\((.+?)\\\)"  # \( ... \)
    r"(?!\$)",  # Not followed by $
)

# Table detection (markdown tables)
TABLE_PATTERN = re.compile(
    r"\|[^\n]+\|\s*\n"  # Header row
    r"\|[\s\-:]+\|\s*\n"  # Separator row
    r"(?:\|[^\n]+\|\s*\n)*",  # Data rows
)

# Theorem/lemma/proposition environment detection
THEOREM_ENV_PATTERN = re.compile(
    r"\*\*(Theorem|Lemma|Proposition|Corollary|Claim)\s*"
    r"(\d+(?:\.\d+)*(?:,\d+(?:\.\d+)*)*)?"  # Numbers like 1.1 or 3,4
    r"\.?\*\*"  # Closing bold
    r"\s*",  # Whitespace after
)

# Image/figure reference in text
FIGURE_REF_PATTERN = re.compile(
    r"(?:Figure|Fig\.?|FIGURE)\s+(\d+[a-zA-Z]?)",
    re.IGNORECASE,
)
TABLE_REF_PATTERN = re.compile(
    r"(?:Table)\s+(\d+[a-zA-Z]?)",
    re.IGNORECASE,
)
EQUATION_REF_PATTERN = re.compile(
    r"(?:Equation|Eq\.?)\s*\(?\s*(\d+[a-zA-Z]?)\s*\)?",
    re.IGNORECASE,
)


def parse_paper_content(
    paper_id: str,
    title: str,
    paper_category: str,
    paper_content: list[dict[str, Any]],
    decode_images: bool = True,
    image_output_dir: Optional[Path] = None,
) -> NormalizedPaper:
    """Transform raw paper_content into a NormalizedPaper.

    Args:
        paper_id: Unique paper identifier (DOI/arXiv ID).
        title: Paper title.
        paper_category: Paper's scientific category.
        paper_content: Raw content list from the dataset.
        decode_images: If True, decode base64 images to temp files.
        image_output_dir: Directory for decoded images; uses temp dir if None.

    Returns:
        A fully parsed NormalizedPaper.
    """
    logger.info(f"Parsing paper: {paper_id} ({len(paper_content)} content items)")

    # Step 1: Convert raw items
    raw_items = _extract_raw_items(paper_content)

    # Step 2: Identify structural elements
    sections = _extract_sections(raw_items, paper_id)
    equations = _extract_equations(raw_items, paper_id)
    images = _extract_images(raw_items, paper_id, decode_images, image_output_dir)
    tables = _extract_tables(raw_items, paper_id)
    theorems = _extract_theorems(raw_items, paper_id)

    # Step 3: Build tagged full text
    tagged_text = _build_tagged_text(raw_items, images, tables, equations)

    paper = NormalizedPaper(
        paper_id=paper_id,
        title=title,
        paper_category=paper_category,
        sections=sections,
        equations=equations,
        images=images,
        tables=tables,
        theorems=theorems,
        tagged_full_text=tagged_text,
        raw_items=raw_items,
    )

    logger.info(
        f"Parsed paper {paper_id}: "
        f"{len(sections)} sections, {len(equations)} equations, "
        f"{len(images)} images, {len(tables)} tables, "
        f"{len(theorems)} theorems"
    )

    return paper


def _extract_raw_items(
    paper_content: list[dict[str, Any]],
) -> list[RawContentItem]:
    """Extract and normalize raw content items."""
    items: list[RawContentItem] = []
    for entry in paper_content:
        content_type_str = entry.get("type", "unknown")
        try:
            content_type = ContentType(content_type_str)
        except ValueError:
            content_type = ContentType.UNKNOWN

        items.append(RawContentItem(
            content_type=content_type,
            text=entry.get("text"),
            image_url=entry.get("image_url"),
            raw=entry,
        ))
    return items


def _extract_sections(
    raw_items: list[RawContentItem],
    paper_id: str,
) -> list[PaperSection]:
    """Detect paper sections from text content.

    Sections are identified by markdown headers (##, ###) or bold section titles.
    """
    sections: list[PaperSection] = []
    full_text = "\n".join(
        item.text or "" for item in raw_items if item.text
    )

    # Find all section header matches
    matches = list(SECTION_PATTERN.finditer(full_text))

    if not matches:
        # Fallback: treat entire text as one section
        sections.append(PaperSection(
            id=f"{paper_id}_sec_0",
            section_title="Full Paper",
            section_level=1,
            content=full_text[:10000],
            start_index=0,
            end_index=len(full_text),
        ))
        return sections

    for i, match in enumerate(matches):
        hashes = match.group(1)
        number = match.group(2)
        title = match.group(3).strip() if match.group(3) else ""

        level = len(hashes)
        section_title = f"{number}. {title}" if number else title

        # Extract content from this header to the next (or end)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        content = full_text[start:end].strip()

        sections.append(PaperSection(
            id=f"{paper_id}_sec_{i}",
            section_title=section_title,
            section_level=level,
            content=content,
            start_index=start,
            end_index=end,
        ))

    return sections


def _extract_equations(
    raw_items: list[RawContentItem],
    paper_id: str,
) -> list[EquationBlock]:
    """Extract LaTeX equations from text content."""
    equations: list[EquationBlock] = []
    full_text = "\n".join(
        item.text or "" for item in raw_items if item.text
    )

    eq_idx = 0

    # Display equations
    for match in DISPLAY_EQUATION_PATTERN.finditer(full_text):
        latex = ""
        for group in match.groups():
            if group:
                latex = group.strip()
                break

        if not latex:
            continue

        # Try to find equation label
        label_match = re.search(r"\\label\{([^}]+)\}", latex)
        label = label_match.group(1) if label_match else None

        # Context around equation
        start = max(0, match.start() - 200)
        end = min(len(full_text), match.end() + 200)
        context_before = full_text[start:match.start()].strip()
        context_after = full_text[match.end():end].strip()

        eq_id = f"{paper_id}_eq_{eq_idx}"
        equations.append(EquationBlock(
            id=eq_id,
            equation_label=label or f"Equation {eq_idx + 1}",
            latex=latex,
            display_mode=True,
            context_before=context_before,
            context_after=context_after,
        ))
        eq_idx += 1

    # Inline equations
    for match in INLINE_EQUATION_PATTERN.finditer(full_text):
        latex = match.group(1).strip()
        if not latex or len(latex) < 2:
            continue

        start = max(0, match.start() - 100)
        end = min(len(full_text), match.end() + 100)

        eq_id = f"{paper_id}_eq_{eq_idx}"
        equations.append(EquationBlock(
            id=eq_id,
            equation_label=f"Inline {eq_idx + 1}",
            latex=latex,
            display_mode=False,
            context_before=full_text[start:match.start()].strip(),
            context_after=full_text[match.end():end].strip(),
        ))
        eq_idx += 1

    return equations


def _extract_images(
    raw_items: list[RawContentItem],
    paper_id: str,
    decode_images: bool,
    output_dir: Optional[Path],
) -> list[ImageBlock]:
    """Extract images from raw content items.

    Base64-encoded JPEG images are decoded and saved to temporary files
    for use by the vision verifier.
    """
    images: list[ImageBlock] = []
    img_idx = 0
    text_items = [item for item in raw_items if item.content_type == ContentType.TEXT]

    for i, item in enumerate(raw_items):
        if item.content_type != ContentType.IMAGE_URL:
            continue

        image_url_data = item.image_url or {}
        raw_url = image_url_data.get("url", "")
        base64_data = None
        image_path = None

        # Extract base64 from data URI
        if raw_url and "base64," in raw_url:
            base64_data = raw_url.split("base64,", 1)[1]
        elif raw_url:
            base64_data = raw_url

        # Try to find caption from nearby text items
        caption = None
        context_before = None
        context_after = None

        if i > 0 and raw_items[i - 1].text:
            context_before = raw_items[i - 1].text[-500:]
        if i + 1 < len(raw_items) and raw_items[i + 1].text:
            context_after = raw_items[i + 1].text[:500]

        # Look for "Figure N" caption pattern in context
        for ctx in [context_before, context_after]:
            if ctx:
                fig_match = re.search(
                    r"(?:Figure|Fig\.?)\s*\d+[a-zA-Z]?"
                    r"(?:[\.:]?\s*(.+?))?(?:\n|$)",
                    ctx,
                    re.IGNORECASE,
                )
                if fig_match and fig_match.group(1):
                    caption = fig_match.group(1).strip()
                    break

        # Decode image
        if decode_images and base64_data:
            try:
                img_data = base64.b64decode(base64_data)
                if output_dir:
                    output_dir = Path(output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    image_path = str(output_dir / f"{paper_id}_img_{img_idx}.jpg")
                    with open(image_path, "wb") as f:
                        f.write(img_data)
                else:
                    # Save to temp file
                    with tempfile.NamedTemporaryFile(
                        suffix=".jpg",
                        prefix=f"paperena_img_{img_idx}_",
                        delete=False,
                    ) as tmp:
                        tmp.write(img_data)
                        image_path = tmp.name
            except Exception as exc:
                logger.warning(f"Failed to decode image {img_idx} in {paper_id}: {exc}")

        img_id = f"{paper_id}_img_{img_idx}"
        images.append(ImageBlock(
            id=img_id,
            caption=caption or f"Figure {img_idx + 1}",
            image_path=image_path,
            base64_data=base64_data,
            context_before=context_before,
            context_after=context_after,
        ))
        img_idx += 1

    return images


def _extract_tables(
    raw_items: list[RawContentItem],
    paper_id: str,
) -> list[TableBlock]:
    """Detect markdown-style tables in text content."""
    tables: list[TableBlock] = []
    full_text = "\n".join(
        item.text or "" for item in raw_items if item.text
    )

    tbl_idx = 0
    for match in TABLE_PATTERN.finditer(full_text):
        raw = match.group(0).strip()

        # Try to parse rows
        lines = raw.split("\n")
        rows: list[list[str]] = []
        for line in lines:
            if re.match(r"^\|[\s\-:]+\|$", line):
                continue  # Skip separator
            cells = [c.strip() for c in line.split("|")[1:-1]]
            if cells:
                rows.append(cells)

        if len(rows) < 2:
            continue  # Need at least header + one data row

        # Find caption from preceding text
        caption = None
        start = max(0, match.start() - 300)
        before = full_text[start:match.start()]
        cap_match = re.search(
            r"(?:Table)\s*\d+[a-zA-Z]?"
            r"(?:[\.:]?\s*(.+?))?(?:\n|$)",
            before,
            re.IGNORECASE,
        )
        if cap_match:
            caption = cap_match.group(0).strip()

        tbl_id = f"{paper_id}_tbl_{tbl_idx}"
        tables.append(TableBlock(
            id=tbl_id,
            caption=caption or f"Table {tbl_idx + 1}",
            raw_content=raw,
            rows=rows,
            context_before=before[-200:] if before else None,
            context_after=full_text[match.end():match.end() + 200],
        ))
        tbl_idx += 1

    return tables


def _extract_theorems(
    raw_items: list[RawContentItem],
    paper_id: str,
) -> list[TheoremBlock]:
    """Detect theorem/lemma/proposition environments in text."""
    theorems: list[TheoremBlock] = []
    full_text = "\n".join(
        item.text or "" for item in raw_items if item.text
    )

    thm_idx = 0
    for match in THEOREM_ENV_PATTERN.finditer(full_text):
        thm_type = match.group(1).lower()
        label = match.group(0).strip().rstrip(".")

        # Extract statement: from end of match to next theorem env or section
        stmt_start = match.end()
        remaining = full_text[stmt_start:]

        # Try to find the end of the statement
        # Look for next theorem env, section header, or Proof marker
        end_match = re.search(
            r"\n\*\*(?:Theorem|Lemma|Proposition|Corollary|Claim|Proof)"
            r"|"
            r"\n#{1,4}\s",
            remaining,
        )
        stmt_end = stmt_start + end_match.start() if end_match else stmt_start + len(remaining)
        statement = full_text[stmt_start:stmt_end].strip()

        # Look for proof in the following text
        proof = None
        proof_match = re.search(
            r"\*\*Proof\.?\*\*\s*(.+?)(?=\n\*\*(?:Theorem|Lemma|Proposition|Corollary)|\n#{1,4}\s|$)",
            full_text[stmt_end:],
            re.DOTALL,
        )
        if proof_match:
            proof = proof_match.group(1).strip()

        thm_id = f"{paper_id}_thm_{thm_idx}"
        theorems.append(TheoremBlock(
            id=thm_id,
            theorem_type=thm_type,
            label=label,
            statement=statement,
            proof=proof,
            context_before=full_text[max(0, match.start() - 300):match.start()],
            context_after=full_text[stmt_end:stmt_end + 300],
        ))
        thm_idx += 1

    return theorems


def _build_tagged_text(
    raw_items: list[RawContentItem],
    images: list[ImageBlock],
    tables: list[TableBlock],
    equations: list[EquationBlock],
) -> str:
    """Build a version of the full text with image/table/equation tags.

    Tags follow the form: [IMAGE:FIGURE_N], [TABLE:TABLE_N], [EQUATION:EQ_N].
    This preserves element locations within the text flow.
    """
    parts: list[str] = []
    img_counter = 0
    tbl_counter = 0
    eq_counter = 0

    for item in raw_items:
        if item.content_type == ContentType.TEXT and item.text:
            text = item.text

            # Tag figure references
            text = FIGURE_REF_PATTERN.sub(
                lambda m: f"[IMAGE:FIGURE_{m.group(1)}]", text
            )
            # Tag table references
            text = TABLE_REF_PATTERN.sub(
                lambda m: f"[TABLE:TABLE_{m.group(1)}]", text
            )
            # Tag equation references
            text = EQUATION_REF_PATTERN.sub(
                lambda m: f"[EQUATION:EQ_{m.group(1)}]", text
            )

            parts.append(text)

        elif item.content_type == ContentType.IMAGE_URL:
            if img_counter < len(images):
                tag = f"[IMAGE:FIGURE_{img_counter + 1}]"
            else:
                tag = f"[IMAGE:UNKNOWN_{img_counter + 1}]"
            parts.append(f"\n{tag}\n")
            img_counter += 1

    return "\n".join(parts)
