"""
Embedding generation.

WHAT AN EMBEDDING IS: a bi-encoder model (BAAI/bge-small-en-v1.5) maps a
piece of text to a fixed-length vector (384 numbers here) such that
texts with similar *meaning* end up as vectors that are close together
in that 384-dimensional space, measured by cosine similarity. "Bi-encoder"
means the query and the document are each embedded independently (query
never sees the document during encoding) — that's what makes it fast
enough to run over an entire corpus. This is the opposite trade-off from
the cross-encoder we add in Phase 5, which is more accurate but must see
query+document together and is too slow to run at corpus scale.

Why bge-small-en-v1.5 specifically: 384 dimensions keeps the index small
and search fast, while still landing near the top of the MTEB retrieval
leaderboard for its size class. e5-base-v2 (768-dim) is a valid upgrade
if retrieval quality matters more than latency/storage for your corpus.

The model is loaded once as a module-level singleton — reloading a
transformer model per request would add seconds of latency to every
single API call, which is an easy performance mistake to make by
accident if embedding logic gets inlined into a request handler.
"""

from __future__ import annotations

from functools import lru_cache

from app.config import settings


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer  # lazy: avoids pulling in torch at

    # app startup / import time -- reranker.py and llm_client.py both follow this same
    # lazy-import-inside-the-function pattern for their heavy/network-dependent deps, for
    # the same reason: importing app.retrieval.embeddings (transitively, importing app.main)
    # should not itself cost a multi-second torch import before the model is ever used.
    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch-embed a list of document chunks. Batching is meaningfully faster
    than one-at-a-time calls because the model can vectorize across the batch."""
    model = _get_model()
    vectors = model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    """Embed a single search query. bge models recommend a query instruction
    prefix for retrieval tasks — it measurably improves ranking quality
    versus embedding the raw query, because it nudges the model toward
    'this text is a question, match it against passages that answer it'
    rather than treating it as just another passage to compare."""
    model = _get_model()
    prefixed = f"Represent this sentence for searching relevant passages: {text}"
    vector = model.encode(prefixed, normalize_embeddings=True)
    return vector.tolist()