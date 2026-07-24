"""
Orchestrates: file(s) on disk -> parse -> chunk -> list of Chunk objects,
ready to be embedded and indexed in Phase 3.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.config import settings
from app.ingestion.chunking import Chunk, chunk_document
from app.ingestion.parsers import parse_document

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".md", ".markdown", ".txt"}


def ingest_file(path: Path) -> list[Chunk]:
    doc = parse_document(path)
    return chunk_document(doc, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)


def ingest_directory(directory: Path) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() in SUPPORTED_SUFFIXES:
            all_chunks.extend(ingest_file(path))
    return all_chunks


def save_chunks(chunks: list[Chunk], output_path: Path) -> None:
    """Persist chunks as JSON so Phase 3 can load them without re-parsing."""
    payload = [
        {
            "chunk_id": c.chunk_id,
            "text": c.text,
            "source": c.source,
            "file_type": c.file_type,
            "chunk_index": c.chunk_index,
            "char_start": c.char_start,
            "char_end": c.char_end,
            "page_number": c.page_number,
        }
        for c in chunks
    ]
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raw_dir = Path(__file__).resolve().parents[2] / "data" / "raw"
    processed_dir = Path(__file__).resolve().parents[2] / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    chunks = ingest_directory(raw_dir)
    save_chunks(chunks, processed_dir / "chunks.json")
    print(f"Ingested {len(chunks)} chunks from {raw_dir} -> {processed_dir / 'chunks.json'}")