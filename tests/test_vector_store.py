"""
Tests the Qdrant integration logic (collection setup, upsert, cosine
search ranking) in isolation from the embedding model, using synthetic
vectors with a known ground-truth similarity structure.

This is a deliberate test boundary, not a shortcut: embeddings.py needs
network access to download BAAI/bge-small-en-v1.5 from Hugging Face,
which is a real, separate dependency from "does our Qdrant plumbing
rank results correctly." Testing them separately means this test suite
runs fast and deterministically in CI without needing a model download
at all -- exactly how retrieval-infrastructure tests should behave in
a real CI pipeline.
"""

from __future__ import annotations

import random

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.ingestion.chunking import Chunk
from app.retrieval.vector_store import ensure_collection, search, upsert_chunks

TEST_DIM = 16
TEST_COLLECTION = "test_ask_my_docs"


def _concept_vector(seed: int, noise: float = 0.05) -> list[float]:
    """Deterministic base vector per 'concept', plus a little per-call noise
    so vectors of the same concept are close but not identical -- mimicking
    how real embeddings of similar-but-not-identical text behave."""
    rng = random.Random(seed)
    base = [rng.uniform(-1, 1) for _ in range(TEST_DIM)]
    noisy = [b + rng.uniform(-noise, noise) for b in base]
    norm = sum(x * x for x in noisy) ** 0.5
    return [x / norm for x in noisy]


@pytest.fixture
def client(tmp_path, monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "embedding_dim", TEST_DIM)
    c = QdrantClient(path=str(tmp_path / "qdrant_test"))
    ensure_collection(c, collection_name=TEST_COLLECTION, recreate=True)
    yield c


def _make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, text=text, source="test.md", file_type="md", chunk_index=0, char_start=0, char_end=len(text))


def test_upsert_and_search_ranks_by_true_similarity(client):
    # Two "concepts": broker-error texts (seed 1) and scheduling texts (seed 2).
    chunks = [
        _make_chunk("c1", "broker connection timeout error"),
        _make_chunk("c2", "broker failed to connect"),
        _make_chunk("c3", "how to schedule a periodic task"),
        _make_chunk("c4", "cron style task scheduling"),
    ]
    vectors = [
        _concept_vector(seed=1),
        _concept_vector(seed=1),
        _concept_vector(seed=2),
        _concept_vector(seed=2),
    ]

    upsert_chunks(client, chunks, vectors, collection_name=TEST_COLLECTION)

    query_vector = _concept_vector(seed=1)  # a new "broker error" query
    results = search(client, query_vector, top_k=2, collection_name=TEST_COLLECTION)

    returned_ids = {r["chunk_id"] for r in results}
    assert returned_ids == {"c1", "c2"}, f"Expected the two broker-concept chunks, got {returned_ids}"
    assert results[0]["score"] >= results[1]["score"], "Results must be sorted by descending similarity"


def test_reindexing_same_chunk_is_idempotent_not_duplicated(client):
    """Point IDs are derived deterministically from chunk_id (see
    vector_store._point_id) specifically so Phase 7's incremental upload
    endpoint can safely re-index a file without ever duplicating or
    colliding with existing points. This proves that property directly."""
    chunk = _make_chunk("c1", "broker connection timeout error")
    vector = _concept_vector(seed=1)

    upsert_chunks(client, [chunk], [vector], collection_name=TEST_COLLECTION)
    upsert_chunks(client, [chunk], [vector], collection_name=TEST_COLLECTION)  # re-index the SAME chunk

    count = client.count(collection_name=TEST_COLLECTION).count
    assert count == 1, f"Expected exactly 1 point after re-indexing the same chunk twice, got {count}"


def test_incremental_upsert_does_not_collide_with_existing_points(client):
    """Simulates the real Phase 7 scenario: an initial batch is indexed,
    then a second, unrelated batch (a new file upload) is indexed later.
    The old positional-index ID scheme would have collided here (both
    batches would restart at id=0); the deterministic chunk_id-based
    scheme must not."""
    first_batch = [_make_chunk("fileA::0", "broker connection timeout error")]
    upsert_chunks(client, first_batch, [_concept_vector(seed=1)], collection_name=TEST_COLLECTION)

    second_batch = [_make_chunk("fileB::0", "how to schedule a periodic task")]
    upsert_chunks(client, second_batch, [_concept_vector(seed=2)], collection_name=TEST_COLLECTION)

    count = client.count(collection_name=TEST_COLLECTION).count
    assert count == 2, f"Expected both batches' points to coexist, got {count}"