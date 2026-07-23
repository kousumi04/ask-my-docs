"""
Tests the reranker's sort/truncate logic using a stub score function --
isolating it from the real cross-encoder model, which needs Hugging
Face access this sandbox doesn't have (same constraint as Phase 3).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from app.ingestion.chunking import Chunk
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.reranker import rerank
from app.retrieval.sparse import build_bm25_index
from app.retrieval.sparse import search as bm25_search

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_rerank_sorts_by_score_and_truncates_to_top_k():
    candidates = [
        {"chunk_id": "a", "text": "low relevance"},
        {"chunk_id": "b", "text": "high relevance"},
        {"chunk_id": "c", "text": "medium relevance"},
    ]
    # Stub score_fn: deliberately returns scores in an order that DISAGREES
    # with the candidates' input order, so passing this test proves rerank()
    # is actually re-sorting, not just preserving input order by accident.
    stub_scores = {"a": 0.1, "b": 0.9, "c": 0.5}

    def score_fn(query: str, texts: list[str]) -> list[float]:
        return [stub_scores[c["chunk_id"]] for c in candidates]

    result = rerank("irrelevant query", candidates, top_k=2, score_fn=score_fn)

    assert [r["chunk_id"] for r in result] == ["b", "c"], "Should sort desc by score and keep only top 2"
    assert result[0]["rerank_score"] == 0.9


def test_rerank_handles_empty_candidates():
    assert rerank("query", [], top_k=5) == []


def _word_overlap_score_fn(query: str, texts: list[str]) -> list[float]:
    """A crude but real relevance heuristic (shared-token count) -- stands
    in for the real cross-encoder to prove the PIPELINE'S reordering logic
    works, not to claim this heuristic IS a cross-encoder."""
    query_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
    scores = []
    for text in texts:
        text_tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
        scores.append(float(len(query_tokens & text_tokens)))
    return scores


def test_rerank_can_correct_the_rrf_tie_from_phase4():
    """Phase 4 documented a real RRF limitation: a chunk that's rank-1 in
    ONLY the dense list can tie with (or beat, by insertion order) a chunk
    that's rank-1 in ONLY the sparse list, even when the sparse match is
    obviously stronger. This test shows the reranker, as the next stage in
    the pipeline, can still recover the right answer -- using the real
    BM25 index and real chunks, not synthetic data."""
    chunks_raw = json.loads((PROJECT_ROOT / "data" / "processed" / "chunks.json").read_text())
    chunks = [Chunk(**c) for c in chunks_raw]
    index = build_bm25_index(chunks)

    query = "E1101 broker connection timeout"
    sparse_results = bm25_search(index, query, top_k=5)
    sparse_top_id = sparse_results[0]["chunk_id"]

    sparse_ids = {r["chunk_id"] for r in sparse_results}
    disagreeing_chunk = next(c for c in chunks if c.chunk_id not in sparse_ids)
    dense_results = [{"chunk_id": disagreeing_chunk.chunk_id, "text": disagreeing_chunk.text}]

    # Unweighted RRF ties/loses the disagreement (documented in Phase 4) --
    # the dense-only result lands at #1 despite being unrelated to the query.
    fused = reciprocal_rank_fusion([dense_results, sparse_results], top_k=5)
    assert fused[0]["chunk_id"] == disagreeing_chunk.chunk_id, "Confirms the Phase 4 tie still exists pre-rerank"

    # The reranker scores the full fused shortlist against the actual query
    # text and corrects the ordering.
    reranked = rerank(query, fused, top_k=5, score_fn=_word_overlap_score_fn)
    assert reranked[0]["chunk_id"] == sparse_top_id, "Reranker should promote the true E1101 match to #1"