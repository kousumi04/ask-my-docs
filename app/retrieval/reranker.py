"""
Cross-encoder re-ranking: the final precision pass over the hybrid
fusion shortlist.

BI-ENCODER (Phase 3) VS CROSS-ENCODER (here) -- why both exist:
A bi-encoder embeds the query and each document independently, so a
document's vector is computed once and reused for every future query --
that's what makes it fast enough to run over an entire corpus. But
because the model never sees the query and document together, it can
only compare two independently-computed summaries, which caps how
precisely it can judge relevance.

A cross-encoder feeds the query and ONE document into the model
together as a single input, and the model attends across both texts
jointly to output one relevance score. This is meaningfully more
accurate -- the model can directly notice "these share a specific
phrase" or "this document contradicts the question's premise" -- but
it must run a full forward pass per query-document PAIR, so it cannot
be run over an entire corpus per query the way a bi-encoder can.

WHY BOTH, IN THIS ORDER (retrieve-then-rerank): hybrid fusion (Phase 4)
narrows an entire corpus down to a small shortlist (top_k_dense +
top_k_sparse candidates, deduplicated) cheaply. The cross-encoder then
only has to score that small shortlist, not the whole corpus -- getting
cross-encoder-level precision at bi-encoder-level cost, by paying the
expensive model only where it matters.

TESTABILITY: rerank() takes an optional score_fn, following the same
dependency-injection pattern as hybrid_search() in Phase 4 -- the real
cross-encoder needs BAAI/bge-reranker-base from Hugging Face (same
network constraint as Phase 3's embedding model), but the sorting and
truncation logic itself needs no model at all and is fully testable
with a stub score function.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable

from app.config import settings

ScoreFn = Callable[[str, list[str]], list[float]]  # (query, texts) -> relevance scores


@lru_cache(maxsize=1)
def _get_cross_encoder():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(settings.reranker_model)


def _default_score_fn(query: str, texts: list[str]) -> list[float]:
    model = _get_cross_encoder()
    pairs = [[query, text] for text in texts]
    return model.predict(pairs).tolist()


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int | None = None,
    score_fn: ScoreFn | None = None,
) -> list[dict]:
    """Re-score and re-sort a shortlist of candidates (each a dict with a
    'text' key) by cross-encoder relevance to the query."""
    top_k = top_k or settings.top_k_rerank
    score_fn = score_fn or _default_score_fn

    if not candidates:
        return []

    texts = [c["text"] for c in candidates]
    scores = score_fn(query, texts)

    reranked = [{**c, "rerank_score": float(s)} for c, s in zip(candidates, scores)]
    reranked.sort(key=lambda c: c["rerank_score"], reverse=True)
    return reranked[:top_k]