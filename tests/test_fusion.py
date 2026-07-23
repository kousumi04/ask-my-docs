"""
RRF is pure arithmetic over ranks, so it's fully testable with
hand-computed expected values -- no embedding model or Qdrant needed.
"""

from __future__ import annotations

import pytest

from app.retrieval.fusion import reciprocal_rank_fusion


def test_rrf_hand_computed_ordering():
    # k=1 chosen only to keep the arithmetic easy to verify by hand;
    # production uses k=60 (settings.rrf_k) per the original RRF paper.
    dense = [{"chunk_id": "A"}, {"chunk_id": "B"}, {"chunk_id": "C"}]
    sparse = [{"chunk_id": "C"}, {"chunk_id": "A"}, {"chunk_id": "D"}]

    # RRF(A) = 1/(1+1) [dense rank1] + 1/(1+2) [sparse rank2] = 0.8333
    # RRF(B) = 1/(1+2) [dense rank2 only]                      = 0.3333
    # RRF(C) = 1/(1+3) [dense rank3] + 1/(1+1) [sparse rank1]  = 0.75
    # RRF(D) = 1/(1+3) [sparse rank3 only]                     = 0.25
    fused = reciprocal_rank_fusion([dense, sparse], k=1, top_k=10)

    scores = {r["chunk_id"]: r["rrf_score"] for r in fused}
    assert scores["A"] == pytest.approx(0.8333, abs=1e-3)
    assert scores["B"] == pytest.approx(0.3333, abs=1e-3)
    assert scores["C"] == pytest.approx(0.75, abs=1e-3)
    assert scores["D"] == pytest.approx(0.25, abs=1e-3)

    order = [r["chunk_id"] for r in fused]
    assert order == ["A", "C", "B", "D"], f"Expected A > C > B > D, got {order}"


def test_rrf_respects_top_k():
    dense = [{"chunk_id": f"d{i}"} for i in range(20)]
    fused = reciprocal_rank_fusion([dense], top_k=5)
    assert len(fused) == 5


def test_document_in_both_lists_outranks_single_list_top_result():
    """A doc appearing near the top of BOTH lists should beat a doc that's
    #1 in only one list -- this is the entire point of hybrid retrieval."""
    dense = [{"chunk_id": "only_in_dense"}, {"chunk_id": "shared"}]
    sparse = [{"chunk_id": "shared"}, {"chunk_id": "only_in_sparse"}]

    fused = reciprocal_rank_fusion([dense, sparse], k=60, top_k=10)
    assert fused[0]["chunk_id"] == "shared"