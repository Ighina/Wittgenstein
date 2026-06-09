"""Phase 2 (LLM path): LLM-based paper content parsing.

Replaces the regex-based content_parser.py when config.parser_mode == "llm".
Uses an LLM to:
  - Identify verifiable claims / equations / theorems / tables / figures
  - Track symbol/term definitions
  - Flag unverifiable boilerplate
  - Assign each unit to a specific verifier
  - List dependencies between units

Produces an EnrichedPaper with a full ContextGraph.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from src.config import PipelineConfig, default_config
from src.models import (
    EnrichedPaper,
    ImageBlock,
    LLMParseChunkResult,
    RawContentItem,
    SymbolDefinition,
    VerifiableUnit,
)
from src.parser.content_parser import _extract_raw_items, _extract_images
from src.parser.context_graph import ContextGraph
from src.utils.llm import llm_call, parse_json_response


# ---------------------------------------------------------------------------
# System prompt for the LLM parser
# ---------------------------------------------------------------------------

PARSER_SYSTEM_PROMPT = """You are a scientific-paper parser. Read a chunk of text and output a structured JSON breakdown for downstream verifiers. Return ONLY a JSON object.

## Output fields

1. **units** — verifiable content units. Each unit:
   - unit_id: stable descriptive slug (e.g., "sec3_eq1", "thm2.1")
   - unit_type: "equation" | "theorem" | "lemma" | "proposition" | "corollary" | "definition" | "claim" | "numeric_claim" | "proof_step" | "table_data" | "figure_reference" | "boilerplate"
   - content: EXACT verbatim text/LaTeX from source (do NOT paraphrase)
   - location: human-readable descriptor (e.g., "Equation 7", "Theorem 2.1")
   - dependencies: list of unit_id strings this unit depends on (empty list if none)
   - verifier_route: "math" (symbolic/SymPy) | "text" (logical consistency) | "statistical" (numeric claims) | "citation" (attribution/novelty) | "vision" (figures/tables) | "none" (unverifiable)
   - is_verifiable: true if independently checkable; false for definitions, boilerplate, notation setup
   - confidence: your confidence in this classification [0.0, 1.0]

2. **symbols** — symbol/term definitions introduced in this chunk. Each:
   - symbol_name: e.g., "X", "f", "\\mathcal{H}", "radiative forcing"
   - domain: mathematical domain (e.g., "real", "integer", "Banach space", or "")
   - latex: LaTeX representation if applicable
   - natural_language: full sentence defining this symbol
   - defining_unit_id: unit_id of the unit introducing this symbol (must match a unit in this chunk)

3. **unverifiable_text** — boilerplate/acknowledgements/standard notation setup (preserved as context, not routed to verifiers)

4. **section_headers** — list of section/subsection headers found in this chunk

## Rules

- Be EXHAUSTIVE: capture every equation (display and inline), every theorem/lemma/proposition/corollary, every quantitative claim
- Copy content VERBATIM from source — never paraphrase
- Track dependencies: if Theorem 2.1 uses symbols from Definition 1.1, list Definition 1.1's unit_id as a dependency
- Definitions ("Let X be a Banach space") → is_verifiable=false; theorems claiming properties → is_verifiable=true
- Boilerplate (acknowledgements, grants, affiliations, "we thank...") → unit_type:"boilerplate", is_verifiable:false, verifier_route:"none"
- LaTeX: display = \\[..\\] or $$..$$ or \\begin{equation}..\\end{equation}; inline = \\(..\\)

```json
{
  "units": [{"unit_id": "...", "unit_type": "...", "content": "...", "location": "...", "dependencies": [], "verifier_route": "...", "is_verifiable": true, "confidence": 0.95}],
  "symbols": [{"symbol_name": "...", "domain": "...", "latex": "...", "natural_language": "...", "defining_unit_id": "..."}],
  "unverifiable_text": "...",
  "section_headers": ["..."]
}
```
"""


# ---------------------------------------------------------------------------
# Mock responses for testing
# ---------------------------------------------------------------------------

def _mock_llm_parse_chunk(
    chunk_text: str,
    chunk_index: int,
) -> dict[str, Any]:
    """Deterministic mock responses for the LLM parser.

    Returns plausible parse results based on keyword detection in the chunk text.
    """
    chunk_lower = chunk_text.lower()
    units: list[dict[str, Any]] = []
    symbols: list[dict[str, Any]] = []
    unverifiable = ""
    headers: list[str] = []

    unit_counter = [0]  # mutable counter

    def _next_id(prefix: str) -> str:
        unit_counter[0] += 1
        return f"{prefix}_{chunk_index}_{unit_counter[0]}"

    # -- Section headers --
    import re
    header_matches = re.findall(r"^#{1,4}\s*.+", chunk_text, re.MULTILINE)
    for h in header_matches:
        headers.append(h.strip())

    # -- Equations (display) --
    display_eqs = re.findall(
        r"\\\[(.*?)\\\]|\$\$(.*?)\$\$|\\begin\{equation\*?\}(.*?)\\end\{equation\*?\}",
        chunk_text, re.DOTALL,
    )
    for groups in display_eqs:
        latex = next((g.strip() for g in groups if g), "")
        if latex and len(latex) > 2:
            uid = _next_id("eq")
            units.append({
                "unit_id": uid,
                "unit_type": "equation",
                "content": latex,
                "location": f"Equation {unit_counter[0]}",
                "dependencies": [],
                "verifier_route": "math",
                "is_verifiable": True,
                "confidence": 0.90,
            })

    # -- Inline equations --
    inline_eqs = re.findall(r"\\\((.+?)\\\)", chunk_text)
    for latex in inline_eqs:
        latex = latex.strip()
        if latex and len(latex) > 2:
            uid = _next_id("eq")
            units.append({
                "unit_id": uid,
                "unit_type": "equation",
                "content": latex,
                "location": f"Inline Equation {unit_counter[0]}",
                "dependencies": [],
                "verifier_route": "math",
                "is_verifiable": True,
                "confidence": 0.90,
            })

    # -- Theorem-like environments --
    thm_patterns = [
        (r"\*\*Theorem\s*(\d+(?:\.\d+)*)?\.?\*\*", "theorem"),
        (r"\*\*Lemma\s*(\d+(?:\.\d+)*)?\.?\*\*", "lemma"),
        (r"\*\*Proposition\s*(\d+(?:\.\d+)*)?\.?\*\*", "proposition"),
        (r"\*\*Corollary\s*(\d+(?:\.\d+)*)?\.?\*\*", "corollary"),
    ]
    for pattern, thm_type in thm_patterns:
        for match in re.finditer(pattern, chunk_text):
            uid = _next_id("thm")
            # Extract statement: text after the match until next section/theorem or 1000 chars
            stmt_start = match.end()
            stmt_end = stmt_start + min(1000, len(chunk_text) - stmt_start)
            statement = chunk_text[stmt_start:stmt_end].strip()
            # Truncate at next section or theorem marker
            for cutoff in re.finditer(
                r"\n\*\*(?:Theorem|Lemma|Proposition|Corollary|Proof)"
                r"|\n#{1,4}\s",
                statement,
            ):
                statement = statement[:cutoff.start()].strip()
                break
            units.append({
                "unit_id": uid,
                "unit_type": thm_type,
                "content": statement[:2000],
                "location": match.group(0).strip().rstrip("*").strip(),
                "dependencies": [],
                "verifier_route": "math" if thm_type == "theorem" else "text",
                "is_verifiable": True,
                "confidence": 0.85,
            })

    # -- Numeric claims --
    numeric_pattern = re.compile(
        r"(\d+(?:\.\d+)?)\s*%|"
        r"p\s*[<>=]\s*0\.\d+|"
        r"(\d+(?:\.\d+)?)\s*(?:times|fold|percent)",
        re.IGNORECASE,
    )
    for match in numeric_pattern.finditer(chunk_text):
        if match.group(0).strip():
            context_start = max(0, match.start() - 100)
            context_end = min(len(chunk_text), match.end() + 100)
            uid = _next_id("num")
            units.append({
                "unit_id": uid,
                "unit_type": "numeric_claim",
                "content": chunk_text[context_start:context_end].strip(),
                "location": f"Numeric claim near char {match.start()}",
                "dependencies": [],
                "verifier_route": "statistical",
                "is_verifiable": True,
                "confidence": 0.75,
            })
            break  # One numeric claim per chunk in mock mode

    # -- Table detection --
    if "|" in chunk_text and re.search(r"\|[^\n]+\|\s*\n\|[\s\-:]+\|", chunk_text):
        uid = _next_id("tbl")
        tbl_match = re.search(
            r"\|[^\n]+\|\s*\n\|[\s\-:]+\|\s*\n(?:\|[^\n]+\|\s*\n)*", chunk_text
        )
        tbl_text = tbl_match.group(0) if tbl_match else "Table content"
        units.append({
            "unit_id": uid,
            "unit_type": "table_data",
            "content": tbl_text[:2000],
            "location": f"Table {unit_counter[0]}",
            "dependencies": [],
            "verifier_route": "vision",
            "is_verifiable": True,
            "confidence": 0.85,
        })

    # -- Symbol definitions --
    symbol_defs = re.findall(
        r"(?:Let|Define|Set)\s+"
        r"(?:\$|\\\()?\s*([a-zA-Z\\][a-zA-Z0-9\\]*(?:\{[^}]*\})?)"
        r"(?:\s*\$|\\\))?\s*(?:be\s+(?:a|an)\s+)?(.+?)(?:\.|,|$)",
        chunk_text,
    )
    for sym_match in symbol_defs[:5]:  # limit
        sym_name = sym_match[0].strip()
        domain = sym_match[1].strip()[:100]
        sym_uid = _next_id("sym")
        symbols.append({
            "symbol_name": sym_name,
            "domain": domain,
            "latex": sym_name,
            "natural_language": f"Let {sym_name} be {domain}",
            "defining_unit_id": sym_uid,
        })
        # Also add a definition unit
        units.append({
            "unit_id": sym_uid,
            "unit_type": "definition",
            "content": f"Let {sym_name} be {domain}",
            "location": f"Definition of {sym_name}",
            "dependencies": [],
            "verifier_route": "none",
            "is_verifiable": False,
            "confidence": 0.90,
        })

    # -- Boilerplate detection --
    boilerplate_keywords = [
        "acknowledg", "thank", "grant", "funding",
        "corresponding author", "email:", "affiliation",
        "we would like to", "the authors declare",
    ]
    for kw in boilerplate_keywords:
        if kw in chunk_lower:
            unverifiable = (
                "Boilerplate section detected (acknowledgements/funding/etc.)"
            )
            break

    return {
        "units": units,
        "symbols": symbols,
        "unverifiable_text": unverifiable,
        "section_headers": headers,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def llm_parse_paper(
    paper_id: str,
    title: str,
    paper_category: str,
    paper_content: list[dict[str, Any]],
    config: Optional[PipelineConfig] = None,
    decode_images: bool = True,
    image_output_dir: Optional[str] = None,
) -> EnrichedPaper:
    """Parse a paper using the LLM-based parser.

    Chunks the raw paper content into windows, calls the LLM on each chunk
    **concurrently**, and assembles the results into an EnrichedPaper with
    ContextGraph.

    Args:
        paper_id: Unique paper identifier.
        title: Paper title.
        paper_category: Scientific category.
        paper_content: Raw content list from the dataset.
        config: Pipeline configuration.
        decode_images: If True, decode base64 images.
        image_output_dir: Directory for decoded images.

    Returns:
        An EnrichedPaper with verifiable units, symbol registry, and context graph.
    """
    if config is None:
        config = default_config

    num_workers = config.llm_parser_num_workers or config.llm.num_workers

    logger.info(
        f"LLM parsing paper: {paper_id} "
        f"(mode={config.parser_mode}, chunk_size={config.llm_parser_chunk_size}, "
        f"workers={num_workers})"
    )

    # Lightweight extraction: only raw_items + images (the only NormalizedPaper
    # fields actually consumed downstream by the enriched segmenter).  Skipping
    # the full regex parser (sections, equations, tables, theorems) saves a
    # significant amount of CPU work on large papers.
    image_dir = Path(image_output_dir) if image_output_dir else None
    raw_items = _extract_raw_items(paper_content)
    images = _extract_images(raw_items, paper_id, decode_images, image_dir)

    # Extract full text from raw items
    full_text = "\n".join(
        item.get("text", "") or ""
        for item in paper_content
        if item.get("type") == "text"
    )

    # Chunk the text
    chunks = _chunk_text(full_text, config.llm_parser_chunk_size)
    logger.info(f"Split paper into {len(chunks)} chunks for LLM parsing")

    # Parse chunks concurrently (same ThreadPoolExecutor pattern used by the
    # orchestrator for parallel verification).  Each chunk is a self-contained
    # LLM call — no shared mutable state.
    chunk_results: list[LLMParseChunkResult] = []
    non_empty = [(i, c) for i, c in enumerate(chunks) if c.strip()]

    if num_workers == 1 or len(non_empty) <= 1:
        # Sequential path for deterministic debugging / single-chunk papers.
        for i, chunk in non_empty:
            chunk_results.append(_parse_chunk(chunk, i, len(chunks), config))
    else:
        with ThreadPoolExecutor(max_workers=min(num_workers, len(non_empty))) as executor:
            futures = {
                executor.submit(_parse_chunk, chunk, i, len(chunks), config): i
                for i, chunk in non_empty
            }
            for future in as_completed(futures):
                chunk_results.append(future.result())

    # Sort by chunk_index to preserve document order during assembly.
    chunk_results.sort(key=lambda r: r.chunk_index)

    # Assemble into EnrichedPaper
    enriched = _assemble_enriched_paper(
        paper_id=paper_id,
        title=title,
        paper_category=paper_category,
        chunk_results=chunk_results,
        raw_items=raw_items,
        images=images,
        config=config,
    )

    logger.info(
        f"LLM parsing complete for {paper_id}: "
        f"{len(enriched.verifiable_units)} units "
        f"({sum(1 for u in enriched.verifiable_units if u.is_verifiable)} verifiable), "
        f"{len(enriched.symbol_registry)} symbols"
    )

    return enriched


def _chunk_text(text: str, chunk_size: int = 8000) -> list[str]:
    """Split text into overlapping chunks at paragraph boundaries.

    Args:
        text: The full paper text.
        chunk_size: Target characters per chunk.

    Returns:
        List of text chunks.
    """
    if len(text) <= chunk_size:
        return [text]

    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)
        if current_len + para_len > chunk_size and current:
            chunks.append("\n\n".join(current))
            # Overlap: keep the last paragraph for continuity
            if len(current) >= 1:
                current = [current[-1]]
                current_len = len(current[0])
            else:
                current = []
                current_len = 0

        current.append(para)
        current_len += para_len

    if current:
        chunks.append("\n\n".join(current))

    return chunks if chunks else [text]


def _parse_chunk(
    chunk_text: str,
    chunk_index: int,
    total_chunks: int,
    config: PipelineConfig,
) -> LLMParseChunkResult:
    """Parse a single text chunk with the LLM.

    Args:
        chunk_text: The text content of this chunk.
        chunk_index: 0-based index of this chunk.
        total_chunks: Total number of chunks in the paper.
        config: Pipeline configuration.

    Returns:
        An LLMParseChunkResult with units, symbols, and metadata.
    """
    logger.debug(
        f"Parsing chunk {chunk_index + 1}/{total_chunks} "
        f"({len(chunk_text)} chars)"
    )

    t0 = time.monotonic()

    try:
        if config.llm.provider == "mock":
            parsed = _mock_llm_parse_chunk(chunk_text, chunk_index)
        else:
            prompt = _build_parse_prompt(chunk_text, chunk_index, total_chunks)
            model = config.llm_parser_model or config.llm.model
            response = llm_call(
                prompt=prompt,
                system_prompt=PARSER_SYSTEM_PROMPT,
                model=model,
                config=config.llm,
            )
            parsed = parse_json_response(response)

        # Build units
        units: list[VerifiableUnit] = []
        for u_dict in parsed.get("units", []):
            unit = VerifiableUnit(
                unit_id=str(u_dict.get("unit_id", f"u_{chunk_index}_{len(units)}")),
                unit_type=str(u_dict.get("unit_type", "claim")),
                content=str(u_dict.get("content", "")),
                location=str(u_dict.get("location", "")),
                dependencies=[
                    str(d) for d in u_dict.get("dependencies", [])
                ],
                verifier_route=str(u_dict.get("verifier_route", "text")),
                is_verifiable=bool(u_dict.get("is_verifiable", True)),
                confidence=float(u_dict.get("confidence", 0.8)),
                source_chunk_index=chunk_index,
            )
            units.append(unit)

        # Build symbols
        symbols: list[SymbolDefinition] = []
        for s_dict in parsed.get("symbols", []):
            symbol = SymbolDefinition(
                symbol_name=str(s_dict.get("symbol_name", "")),
                domain=str(s_dict.get("domain", "")),
                latex=str(s_dict.get("latex", "")),
                natural_language=str(s_dict.get("natural_language", "")),
                defining_unit_id=str(s_dict.get("defining_unit_id", "")),
            )
            symbols.append(symbol)

        result = LLMParseChunkResult(
            chunk_index=chunk_index,
            units=units,
            symbols=symbols,
            unverifiable_text=str(parsed.get("unverifiable_text", "")),
            section_headers=[
                str(h) for h in parsed.get("section_headers", [])
            ],
        )

    except Exception as exc:
        logger.warning(
            f"LLM parse failed for chunk {chunk_index + 1}/{total_chunks}: {exc}. "
            f"Falling back to empty result."
        )
        result = LLMParseChunkResult(
            chunk_index=chunk_index,
            unverifiable_text=chunk_text,  # Preserve as context
        )

    elapsed = (time.monotonic() - t0) * 1000
    logger.debug(
        f"Chunk {chunk_index + 1}/{total_chunks} parsed in {elapsed:.0f}ms: "
        f"{len(result.units)} units, {len(result.symbols)} symbols"
    )

    return result


def _build_parse_prompt(
    chunk_text: str,
    chunk_index: int,
    total_chunks: int,
) -> str:
    """Build the user prompt for a single parse chunk.

    Args:
        chunk_text: The text content to parse.
        chunk_index: 0-based index of this chunk.
        total_chunks: Total number of chunks.

    Returns:
        The user prompt string.
    """
    lines = [
        f"Parse chunk {chunk_index + 1} of {total_chunks} from this scientific paper.",
        "",
        "## Text to parse",
        "",
        chunk_text[:12000],  # Safety cap
        "",
        "Return the JSON parse result now.",
    ]
    return "\n".join(lines)


def _assemble_enriched_paper(
    paper_id: str,
    title: str,
    paper_category: str,
    chunk_results: list[LLMParseChunkResult],
    raw_items: list[RawContentItem],
    images: list[ImageBlock],
    config: PipelineConfig,
) -> EnrichedPaper:
    """Assemble chunk results into a complete EnrichedPaper.

    Merges units across chunks, builds the ContextGraph, resolves cross-chunk
    dependencies, and collects unverifiable text.

    Args:
        paper_id: Paper identifier.
        title: Paper title.
        paper_category: Scientific category.
        chunk_results: Parse results for each chunk.
        raw_items: Lightweight-extracted raw content items.
        images: Lightweight-extracted image blocks.
        config: Pipeline configuration.

    Returns:
        An assembled EnrichedPaper.
    """
    graph = ContextGraph()
    all_symbols: dict[str, SymbolDefinition] = {}

    # First pass: collect all units and symbols
    for cr in chunk_results:
        for unit in cr.units:
            graph.add_unit(unit)
            for dep in unit.dependencies:
                # Dependencies may reference units from other chunks
                graph.add_dependency(unit.unit_id, dep)

        for sym in cr.symbols:
            if sym.symbol_name not in all_symbols:
                all_symbols[sym.symbol_name] = sym
                graph.add_symbol(sym)

        if cr.unverifiable_text:
            graph.add_unverifiable_text(cr.unverifiable_text)

    # Second pass: resolve symbol-based dependencies for each verifiable unit
    all_verifiable = [
        u for cr in chunk_results for u in cr.units if u.is_verifiable
    ]
    all_definitional = [
        u for cr in chunk_results for u in cr.units if not u.is_verifiable
    ]

    # Build context for each verifiable unit
    enriched_units: list[VerifiableUnit] = []
    for unit in all_verifiable:
        # Resolve context from the dependency graph
        context = graph.resolve_context(unit, config.llm_parser_max_context_chars)

        # Create enriched copy
        enriched = VerifiableUnit(
            unit_id=unit.unit_id,
            unit_type=unit.unit_type,
            content=unit.content,
            location=unit.location,
            dependencies=unit.dependencies,
            required_context=context,
            verifier_route=unit.verifier_route,
            is_verifiable=True,
            confidence=unit.confidence,
            metadata=unit.metadata,
            source_chunk_index=unit.source_chunk_index,
        )
        enriched_units.append(enriched)

    # Collect unverifiable context
    unverifiable_context = graph.unverifiable_text
    # Also include definitional units as reference context
    for unit in all_definitional:
        unverifiable_context += f"\n\n[{unit.unit_type}] {unit.location}\n{unit.content}"

    # Build dependency dict for serialization
    deps_dict: dict[str, list[str]] = {}
    for unit in enriched_units:
        if unit.dependencies:
            deps_dict[unit.unit_id] = unit.dependencies
        # Also include inferred deps from the graph
        graph_deps = graph.get_dependencies(unit.unit_id)
        if graph_deps:
            merged = list(set(unit.dependencies + graph_deps))
            deps_dict[unit.unit_id] = merged

    enriched = EnrichedPaper(
        paper_id=paper_id,
        title=title,
        paper_category=paper_category,
        verifiable_units=enriched_units,
        symbol_registry=list(graph.symbols_as_list()),
        context_graph=deps_dict,
        unverifiable_context=unverifiable_context,
        # Backward-compatible fields: only images are populated (needed by
        # enriched_segmenter._find_image_for_unit); the rest are left empty
        # since we skipped the regex parser.
        sections=[],
        equations=[],
        images=images,
        tables=[],
        theorems=[],
        tagged_full_text="",
        raw_items=raw_items,
    )

    return enriched
