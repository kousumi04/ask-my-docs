"""
BM25 sparse retrieval.

THE MATH, BRIEFLY: BM25 scores a document for a query as a sum, over
each query term, of:

    IDF(term) * ( f(term, doc) * (k1 + 1) ) / ( f(term, doc) + k1 * (1 - b + b * |doc| / avgdl) )

- f(term, doc): how many times the term appears in this document.
- IDF(term): rarer terms across the corpus score higher -- a term that
  appears in every document carries no discriminating information.
- The denominator applies *term-frequency saturation*: the 5th
  occurrence of a word adds much less score than the 1st. Without this,
  a document could rank #1 purely by repeating a word many times.
- |doc| / avgdl with parameter b: length normalization. Without it,
  long documents would win searches just by containing more words,
  not by being more relevant. b=0.75 and k1=1.5 (rank-bm25's defaults)
  are the standard starting point used in production search systems.

WHY THIS MATTERS ALONGSIDE DENSE RETRIEVAL: bge-small-en-v1.5 embeds
"E1101" as just another token blended into a 384-dim average -- it has
no special mechanism for exact identifier matching. BM25's IDF term
means a rare, specific token like "E1101" or "queuely_prefetch_count"
dominates the score for a query containing it. This is precisely the
class of query dense retrieval alone tends to miss.

WHY rank-bm25 SPECIFICALLY: it's a small, dependency-light, pure-Python
implementation of the exact BM25Okapi variant described above -- no
server process needed, unlike Elasticsearch/OpenSearch, which would be
considerable operational overhead for what is fundamentally an
in-memory scoring computation over a corpus this size.
"""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from app.ingestion.chunking import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric-only tokenization. Deliberately simple:
    BM25's value comes from term-frequency/IDF statistics, not from
    sophisticated tokenization -- stemming or lemmatization could be
    added later, but would apply equally to indexing and querying, so
    the fusion/reranking logic downstream is unaffected either way."""
    return _TOKEN_RE.findall(text.lower())


@dataclass
class BM25Index:
    bm25: BM25Okapi
    chunks: list[Chunk]  # positional alignment with the corpus passed to BM25Okapi


def build_bm25_index(chunks: list[Chunk]) -> BM25Index:
    tokenized_corpus = [tokenize(c.text) for c in chunks]
    return BM25Index(bm25=BM25Okapi(tokenized_corpus), chunks=chunks)


def search(index: BM25Index, query: str, top_k: int) -> list[dict]:
    query_tokens = tokenize(query)
    scores = index.bm25.get_scores(query_tokens)

    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    results = []
    for i in ranked[:top_k]:
        if scores[i] <= 0:
            continue  # a zero score means no query term matched at all -- not a real result
        chunk = index.chunks[i]
        results.append(
            {
                "score": float(scores[i]),
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "source": chunk.source,
                "file_type": chunk.file_type,
                "page_number": chunk.page_number,
            }
        )
    return results


def save_index(index: BM25Index, path: Path) -> None:
    path.write_bytes(pickle.dumps(index))


def load_index(path: Path) -> BM25Index:
    return pickle.loads(path.read_bytes())