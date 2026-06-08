"""Boundary-aware text chunking for verification.

Long, dense snippets (multi-page proofs, whole sections) are where the LLM most
often returns an empty or unparseable completion — and even when it answers, a
2-page prompt dilutes attention so a single load-bearing wrong step is easy to
miss. Splitting such a snippet into smaller overlapping chunks and verifying each
independently mitigates both: each call is short enough to answer reliably, a
single empty chunk no longer sinks the whole snippet, and a flagged error is
localized to a chunk rather than the entire section.

`chunk_text` packs whole paragraphs (then sentences, then a hard char split as a
last resort) up to ``max_chars``, and carries ``overlap`` characters of trailing
context from one chunk into the next so a claim split across a boundary is still
seen whole by at least one chunk.
"""

from __future__ import annotations

import re

_PARA_RE = re.compile(r"\n\s*\n")
# Sentence-ish boundary: end punctuation followed by whitespace. Good enough for
# prose and proof text; we never rely on it for correctness, only for nicer cuts.
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text: str, max_chars: int = 2000, overlap: int = 200) -> list[str]:
    """Split ``text`` into chunks of at most ~``max_chars`` characters.

    Args:
        text: The text to split.
        max_chars: Soft upper bound on chunk size (a single unsplittable token
            run may exceed it slightly).
        overlap: Characters of trailing context prepended to each chunk after the
            first, for continuity across boundaries.

    Returns:
        A list of chunk strings (at least one). Returns ``[""]`` for empty input
        so callers always have something to iterate.
    """
    text = (text or "").strip()
    if not text:
        return [""]
    if len(text) <= max_chars:
        return [text]

    # 1) Pack paragraphs; split oversized paragraphs into sentences, and an
    #    oversized sentence by hard character slicing.
    units: list[str] = []
    for para in _PARA_RE.split(text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            units.append(para)
            continue
        for sent in _SENT_RE.split(para):
            sent = sent.strip()
            if not sent:
                continue
            if len(sent) <= max_chars:
                units.append(sent)
            else:
                units.extend(_hard_split(sent, max_chars))

    # 2) Greedily pack units into chunks.
    chunks: list[str] = []
    cur = ""
    for unit in units:
        if not cur:
            cur = unit
        elif len(cur) + 1 + len(unit) <= max_chars:
            cur = f"{cur}\n{unit}"
        else:
            chunks.append(cur)
            cur = unit
    if cur:
        chunks.append(cur)

    # 3) Add trailing-context overlap between consecutive chunks.
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for prev, nxt in zip(chunks, chunks[1:]):
            tail = prev[-overlap:]
            overlapped.append(f"{tail}\n{nxt}" if tail else nxt)
        chunks = overlapped

    return chunks or [text[:max_chars]]


def _hard_split(s: str, max_chars: int) -> list[str]:
    """Last-resort fixed-width split for an unsplittable run."""
    return [s[i : i + max_chars] for i in range(0, len(s), max_chars)]
