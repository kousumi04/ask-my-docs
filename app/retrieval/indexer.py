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
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [Chunk(**item) for item in raw]


def index_chunks(chunks_path: Path) -> int:
    chunks = load_chunks(chunks_path)
    client = get_client()
    ensure_collection(client, recreate=True)

    if not chunks:
        return 0

    vectors = embed_texts([c.text for c in chunks])
    upsert_chunks(client, chunks, vectors)
    return len(chunks)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    chunks_path = project_root / "data" / "processed" / "chunks.json"

    count = index_chunks(chunks_path)
    print(f"Indexed {count} chunks into Qdrant collection.")