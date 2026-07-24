"""
Ties Phase 2 (chunks.json on disk) to Phase 3 (embeddings + Qdrant):
load chunks -> embed in batch -> upsert into the vector collection.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.ingestion.chunking import Chunk
from app.retrieval.embeddings import embed_texts
from app.retrieval.vector_store import ensure_collection, get_client, upsert_chunks


def load_chunks(path: Path) -> list[Chunk]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [Chunk(**item) for item in raw]


def index_chunks(chunks_path: Path) -> int:
    """Full rebuild from chunks.json -- the CLI entrypoint below, and Phase
    1-6's manual verification step. Deliberately recreates the collection,
    since it's meant to be a from-scratch reindex of everything on disk."""
    chunks = load_chunks(chunks_path)
    vectors = embed_texts([c.text for c in chunks])

    client = get_client()
    ensure_collection(client, recreate=True)
    upsert_chunks(client, chunks, vectors)
    return len(chunks)


def index_new_chunks(chunks: list[Chunk]) -> int:
    """Incremental indexing for the Phase 7 upload endpoint: creates the
    collection only if it doesn't exist yet (recreate=False) and upserts
    just the new chunks -- never wipes previously indexed documents.
    Safe to call repeatedly for the same chunks because point IDs are
    deterministic (see vector_store._point_id)."""
    if not chunks:
        return 0
    vectors = embed_texts([c.text for c in chunks])

    client = get_client()
    ensure_collection(client, recreate=False)
    upsert_chunks(client, chunks, vectors)
    return len(chunks)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    chunks_path = project_root / "data" / "processed" / "chunks.json"

    count = index_chunks(chunks_path)
    print(f"Indexed {count} chunks into Qdrant collection.")