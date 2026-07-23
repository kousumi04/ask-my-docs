"""
Hybrid search: run dense (vector) and sparse (BM25) retrieval in
parallel, then fuse with RRF.

Dense retrieval is injected as a callable rather than called directly
here, deliberately: dense_search requires the embedding model (network
access to Hugging Face, unavailable in this sandbox) while sparse
search does not. Depending on an injected function means this module's
fusion logic is fully testable without the model, and Phase 7 wires in
the real vector_store.search + embeddings.embed_query without this
file changing at all.
"""

from __future__ import annotations

from typing import Callable

from app.config import settings
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.sparse import BM25Index
from app.retrieval.sparse import search as bm25_search

DenseSearchFn = Callable[[str, int], list[dict]]  # (query, top_k) -> ranked results


def hybrid_search(
    query: str,
    bm25_index: BM25Index,
    dense_search_fn: DenseSearchFn,
    top_k: int | None = None,
) -> list[dict]:
    top_k = top_k or settings.top_k_rerank

    dense_results = dense_search_fn(query, settings.top_k_dense)
    sparse_results = bm25_search(bm25_index, query, settings.top_k_sparse)

    return reciprocal_rank_fusion([dense_results, sparse_results], top_k=top_k)