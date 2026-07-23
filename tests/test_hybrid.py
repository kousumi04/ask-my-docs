"""
Demonstrates hybrid retrieval's fusion behavior using the real BM25
index over the real 12 sample chunks -- including a real limitation
this test uncovered rather than one that was designed in from the
start.

WHAT WENT WRONG THE FIRST TIME THIS TEST WAS WRITTEN: the original
version asserted that BM25's #1 result for "E1101 broker connection
timeout" would reach #1 in the fused ranking even against an
unrelated dense list. It failed. Investigation showed two things:

  1. The "unrelated" dense chunks weren't actually disjoint from BM25's
     own top-5 for this query -- shared vocabulary ("broker") caused
     overlap, contaminating the scenario.
  2. More fundamentally: plain RRF is rank-only and has no concept of
     match *magnitude*. A chunk that is BM25's #1 result with an
     overwhelming keyword match, and a chunk that is dense's #1 result
     with only a mediocre semantic score, get IDENTICAL fused scores if
     each appears only in its own list (weight/(k+1) each). Plain RRF
     cannot express "sparse is far more confident here."

The tests below reflect that corrected understanding: one test pins
down the tie precisely (rather than assuming a winner), and the other
shows the actual production fix -- weighting sparse higher when you
know it should be trusted more for a given query pattern.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.ingestion.chunking import Chunk
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.sparse import build_bm25_index
from app.retrieval.sparse import search as bm25_search

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_real_chunks() -> list[Chunk]:
    raw = json.loads((PROJECT_ROOT / "data" / "processed" / "chunks.json").read_text())
    return [Chunk(**c) for c in raw]


def test_plain_rrf_ties_two_independent_rank1_results():
    """The honest baseline: with no overlap between lists and equal weights,
    a rank-1-only result from EITHER list scores identically. This is a
    real property of plain RRF, not something to paper over."""
    dense = [{"chunk_id": "dense_only_top"}]
    sparse = [{"chunk_id": "sparse_only_top"}]

    fused = reciprocal_rank_fusion([dense, sparse], k=60, top_k=10)
    scores = {r["chunk_id"]: r["rrf_score"] for r in fused}

    assert scores["dense_only_top"] == pytest.approx(scores["sparse_only_top"])


def test_weighting_sparse_higher_lets_it_win_a_genuine_disagreement():
    """The production fix for the limitation above: when you know sparse
    should be trusted more (e.g. queries containing error-code-shaped
    tokens), a weight on the sparse list lets its rank-1 result win a
    genuine disagreement with dense -- using the REAL BM25 index and
    REAL sample chunks, not synthetic data."""
    chunks = _load_real_chunks()
    index = build_bm25_index(chunks)

    query = "E1101 broker connection timeout"
    sparse_results = bm25_search(index, query, top_k=5)
    sparse_top_id = sparse_results[0]["chunk_id"]

    # A dense list whose top result is a genuinely different, non-overlapping
    # chunk -- simulating a real disagreement between the two retrievers.
    sparse_ids = {r["chunk_id"] for r in sparse_results}
    disagreeing_chunk = next(c for c in chunks if c.chunk_id not in sparse_ids)
    dense_results = [{"chunk_id": disagreeing_chunk.chunk_id, "text": disagreeing_chunk.text}]

    unweighted = reciprocal_rank_fusion([dense_results, sparse_results], top_k=1)
    weighted = reciprocal_rank_fusion([dense_results, sparse_results], top_k=1, weights=[1.0, 2.0])

    assert unweighted[0]["chunk_id"] == disagreeing_chunk.chunk_id, (
        "Unweighted: dense's rank-1 ties or wins since it's rank 1 in its own list too"
    )
    assert weighted[0]["chunk_id"] == sparse_top_id, (
        "Weighted 2x toward sparse: BM25's strong match should now win the disagreement"
    )