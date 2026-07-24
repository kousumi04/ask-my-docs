"""
Ties every phase's module together into the two operations the API
actually needs: "answer a query" and "add a new document." Nothing
retrieval- or generation-specific lives here -- this module is purely
orchestration and the BM25 index caching policy.

WHY BM25 NEEDS ITS OWN CACHING POLICY, UNLIKE QDRANT: Qdrant is a real
store -- the client just queries whatever's on disk, so a new upload's
vectors are visible to the very next request automatically. rank-bm25
has no such persistence: build_bm25_index() constructs an in-memory
index from a list of chunks, so if we rebuilt it from disk on every
single query, we'd pay a full-corpus tokenization cost per request.
Instead we cache the built index in memory and only rebuild it when
the underlying chunks.json file's modification time changes -- cheap
to check, correct after every upload, no unnecessary rebuilds.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.config import settings
from app.generation.generate import answer_query
from app.ingestion.chunking import Chunk
from app.ingestion.pipeline import ingest_file, save_chunks
from app.retrieval.embeddings import embed_query
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.indexer import index_new_chunks
from app.retrieval.reranker import rerank
from app.retrieval.sparse import BM25Index, build_bm25_index
from app.retrieval.sparse import search as bm25_search
from app.retrieval.vector_store import get_client
from app.retrieval.vector_store import search as vector_search

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.json"

_bm25_cache: dict[str, object] = {"index": None, "mtime": None}


def _load_chunks_from_disk() -> list[Chunk]:
    if not CHUNKS_PATH.exists():
        return []
    raw = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    return [Chunk(**item) for item in raw]


def get_bm25_index() -> BM25Index | None:
    """Returns the cached BM25 index, rebuilding it if chunks.json has
    changed (or doesn't exist yet) since the last build."""
    if not CHUNKS_PATH.exists():
        return None

    current_mtime = CHUNKS_PATH.stat().st_mtime
    if _bm25_cache["index"] is None or _bm25_cache["mtime"] != current_mtime:
        chunks = _load_chunks_from_disk()
        _bm25_cache["index"] = build_bm25_index(chunks) if chunks else None
        _bm25_cache["mtime"] = current_mtime

    return _bm25_cache["index"]


def add_document(file_path: Path) -> dict:
    """Ingest one uploaded file end-to-end: parse+chunk, append to
    chunks.json (so BM25 picks it up on next query), and incrementally
    index its embeddings into Qdrant (never wiping existing data)."""
    new_chunks = ingest_file(file_path)

    existing_chunks = _load_chunks_from_disk()
    save_chunks(existing_chunks + new_chunks, CHUNKS_PATH)  # bumps mtime -> invalidates BM25 cache

    indexed_count = index_new_chunks(new_chunks)

    return {"filename": file_path.name, "chunks_added": len(new_chunks), "chunks_indexed": indexed_count}


def query(question: str, top_k_rerank: int | None = None) -> dict:
    """The full retrieve -> rerank -> generate pipeline for one question."""
    top_k_rerank = top_k_rerank or settings.top_k_rerank

    bm25_index = get_bm25_index()
    if bm25_index is None:
        return {
            "answer": "No documents have been indexed yet -- upload a document first.",
            "sources": [],
            "is_fully_grounded": False,
            "warning": "no_documents_indexed",
        }

    sparse_results = bm25_search(bm25_index, question, top_k=settings.top_k_sparse)

    query_vector = embed_query(question)
    dense_results = vector_search(get_client(), query_vector, top_k=settings.top_k_dense)

    fused = reciprocal_rank_fusion([dense_results, sparse_results], top_k=settings.top_k_dense)
    reranked = rerank(question, fused, top_k=top_k_rerank)

    return answer_query(question, reranked)