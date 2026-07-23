"""
Qdrant vector store integration.

WHY QDRANT OVER FAISS: FAISS is an in-process library with no
persistence, no metadata filtering, and no server API -- every process
that wants to search has to rebuild or reload the index itself. Qdrant
is a real service: it persists to disk, supports filtering search
results by payload (e.g. "only chunks from source=X"), and is what
teams actually run in production, not just prototype with.

DEV VS PRODUCTION MODE: Qdrant supports an embedded local-file mode
(QdrantClient(path=...)) with no server process required, which is
what we use here for a from-scratch build without requiring Docker to
already be running. It uses the identical client API as the real
server mode -- the only change to go to production is removing
qdrant_local_path and pointing host/port at the Dockerized Qdrant
container from Phase 9. Nothing else in this module changes.

HNSW: Qdrant indexes vectors with HNSW (Hierarchical Navigable Small
World), an approximate-nearest-neighbor graph structure. Exact nearest-
neighbor search is O(n) per query -- fine for a few thousand vectors,
too slow once a corpus reaches millions. HNSW trades a small amount of
recall for logarithmic-time search, which is the standard trade-off
every production vector database makes.

Every function takes an explicit collection_name (defaulting to the
configured production collection) rather than hardcoding it, so tests
can point at an isolated test collection without touching real data --
and so a future multi-tenant use case (one collection per customer)
is a parameter, not a rewrite.
"""

from __future__ import annotations

from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config import settings
from app.ingestion.chunking import Chunk


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    if settings.qdrant_local_path:
        return QdrantClient(path=settings.qdrant_local_path)
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


def ensure_collection(client: QdrantClient, collection_name: str | None = None, recreate: bool = False) -> None:
    collection_name = collection_name or settings.qdrant_collection
    exists = client.collection_exists(collection_name)
    if exists and not recreate:
        return
    if exists and recreate:
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=qmodels.VectorParams(
            size=settings.embedding_dim,
            distance=qmodels.Distance.COSINE,
        ),
    )


def upsert_chunks(client: QdrantClient, chunks: list[Chunk], vectors: list[list[float]], collection_name: str | None = None) -> None:
    collection_name = collection_name or settings.qdrant_collection
    points = [
        qmodels.PointStruct(
            id=idx,
            vector=vector,
            payload={
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "source": chunk.source,
                "file_type": chunk.file_type,
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
            },
        )
        for idx, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]
    client.upsert(collection_name=collection_name, points=points)


def search(client: QdrantClient, query_vector: list[float], top_k: int, collection_name: str | None = None) -> list[dict]:
    collection_name = collection_name or settings.qdrant_collection
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
    ).points
    return [{"score": r.score, **r.payload} for r in results]