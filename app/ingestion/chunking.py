"""
Chunking: splitting a parsed document into retrieval-sized pieces.

STRATEGY COMPARISON (why recursive chunking is the Phase-2 default):

  Fixed-size chunking
    Split every N characters, full stop. Trivial to implement, but
    routinely slices a sentence or a code block in half, which both
    hurts embedding quality (a half-sentence embeds to a blurry
    average) and hurts the LLM's ability to use the chunk faithfully.

  Recursive character chunking (what we implement here)
    Try to split on "big" boundaries first (paragraph breaks), and
    only fall back to smaller boundaries (sentences, then words) for
    pieces that are still too large. This keeps semantically whole
    units together whenever the text allows it, and only cuts
    mid-sentence as a last resort. This is the standard production
    default (it's what LangChain's RecursiveCharacterTextSplitter
    does) because it's a large quality improvement over fixed-size
    for near-zero extra complexity.

  Semantic chunking
    Embed individual sentences, then cut where consecutive sentence
    embeddings diverge past a threshold (a topic-shift detector).
    Higher quality boundaries, but requires an embedding call per
    sentence at ingestion time — meaningfully slower and costs more
    compute. Worth adding as an alternative strategy in Phase 3 once
    we have the embedding model wired up; not needed to get a working
    system today.

  Parent-child chunking
    Retrieve on small "child" chunks (precise matching) but feed the
    LLM the larger "parent" chunk they belong to (more context). We
    revisit this in Phase 5 once reranking is in place — it's a
    retrieval-time technique, not a chunking-time one.

IMPLEMENTATION NOTE — offsets are tracked natively, not re-searched.
An earlier version of this module joined pieces back together with a
synthetic " " separator and then called full_text.find(chunk) to
recover position. That's wrong whenever the original separator wasn't
a single space (e.g. "\\n\\n" between paragraphs) — the reconstructed
string is never an exact substring of the source, so find() silently
returns -1 for every chunk. This version carries (start, end) offsets
through the recursion and merge steps, then slices the *original*
text for the final chunk content, so chunk text is always guaranteed
to be a real substring at a real offset.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.ingestion.parsers import ParsedDocument

# Ordered from "biggest, most meaningful" boundary to smallest.
_DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

Span = tuple[str, int, int]  # (piece_text, start_offset, end_offset) — offsets into the ORIGINAL text


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    file_type: str
    chunk_index: int
    char_start: int
    char_end: int
    page_number: int | None = None


def _split_with_offsets(text: str, base_offset: int, separator: str) -> list[Span]:
    """Split text on separator, returning each piece with its true offset in the original document."""
    if separator == "":
        return [(ch, base_offset + i, base_offset + i + 1) for i, ch in enumerate(text)]

    spans: list[Span] = []
    pos = 0
    for part in text.split(separator):
        start = base_offset + pos
        end = start + len(part)
        spans.append((part, start, end))
        pos = end - base_offset + len(separator)
    return spans


def _recursive_split(text: str, base_offset: int, separators: list[str], chunk_size: int) -> list[Span]:
    sep = separators[0]
    spans = _split_with_offsets(text, base_offset, sep)

    result: list[Span] = []
    for piece_text, start, end in spans:
        if len(piece_text) > chunk_size and len(separators) > 1:
            result.extend(_recursive_split(piece_text, start, separators[1:], chunk_size))
        elif piece_text:
            result.append((piece_text, start, end))
    return result


def _merge_with_overlap(spans: list[Span], chunk_size: int, overlap: int) -> list[tuple[int, int]]:
    """Greedily pack small spans into chunks up to chunk_size, sliding forward with overlap.

    Returns (start, end) offset pairs only — the caller slices the original
    text, so chunk content is always an exact, verifiable substring.
    """
    chunks: list[tuple[int, int]] = []
    current: list[Span] = []

    for span in spans:
        if current:
            tentative_len = span[2] - current[0][1]
            if tentative_len > chunk_size:
                chunks.append((current[0][1], current[-1][2]))
                while len(current) > 1 and (current[-1][2] - current[0][1]) > overlap:
                    current.pop(0)
        current.append(span)

    if current:
        chunks.append((current[0][1], current[-1][2]))
    return chunks


def chunk_document(doc: ParsedDocument, chunk_size: int, overlap: int) -> list[Chunk]:
    """Chunk a parsed document and attach source/page/offset metadata to every chunk."""
    spans = _recursive_split(doc.full_text, 0, _DEFAULT_SEPARATORS, chunk_size)
    offset_pairs = _merge_with_overlap(spans, chunk_size, overlap)

    # For PDFs, map each chunk's start offset back to a page number by
    # tracking where each page's text falls in the concatenated full_text.
    page_boundaries: list[tuple[int, int]] = []  # (start_offset, page_number)
    if doc.pages:
        offset = 0
        for i, page_text in enumerate(doc.pages):
            page_boundaries.append((offset, i + 1))
            offset += len(page_text) + 2  # "\n\n" join separator between pages

    chunks: list[Chunk] = []
    for idx, (start, end) in enumerate(offset_pairs):
        text = doc.full_text[start:end]

        page_number = None
        if page_boundaries:
            page_number = next(
                (pg for boundary, pg in reversed(page_boundaries) if start >= boundary),
                page_boundaries[0][1],
            )

        chunks.append(
            Chunk(
                chunk_id=f"{doc.source}::{idx}",
                text=text,
                source=doc.source,
                file_type=doc.file_type,
                chunk_index=idx,
                char_start=start,
                char_end=end,
                page_number=page_number,
            )
        )
    return chunks