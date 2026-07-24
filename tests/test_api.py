"""
Tests the full /upload -> /query flow through the actual FastAPI app,
using TestClient. Real ingestion (parsing + chunking) and real BM25
run unmodified -- neither needs network access. The three genuinely
network-dependent calls (embedding, reranking, generation) are stubbed
at their exact call sites, the same dependency-injection boundary used
in every previous phase's tests.

Isolation: this test points chunks.json, the Qdrant collection, and the
raw-upload directory at tmp_path, so it never touches the real
data/processed/chunks.json or data/qdrant_local/ built up over Phases
1-6 during manual verification.
"""

from __future__ import annotations

import random
import re

import pytest
from fastapi.testclient import TestClient

import app.api.routes as routes_module
import app.core.pipeline as pipeline_module
import app.generation.generate as generate_module
import app.retrieval.indexer as indexer_module
import app.retrieval.reranker as reranker_module
from app.config import settings
from app.main import app
from app.retrieval.vector_store import get_client

EMBED_DIM = 384


def _fake_vector(text: str) -> list[float]:
    """Deterministic pseudo-embedding -- dense retrieval's exact ranking
    doesn't matter for this test (BM25 and the word-overlap reranker
    stub below carry the real correctness signal); this only needs to
    produce a valid-shaped vector so Qdrant's upsert/search don't error."""
    rng = random.Random(hash(text) % (2**32))
    vec = [rng.uniform(-1, 1) for _ in range(EMBED_DIM)]
    norm = sum(x * x for x in vec) ** 0.5
    return [x / norm for x in vec]


def _word_overlap_score_fn(query: str, texts: list[str]) -> list[float]:
    query_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
    return [float(len(query_tokens & set(re.findall(r"[a-z0-9]+", t.lower())))) for t in texts]


def _stub_complete_fn(messages: list[dict]) -> str:
    # Real generation logic isn't under test here (Phase 6 covers that) --
    # just needs a plausible, correctly-cited response referencing the
    # top-ranked (and therefore presumed most relevant) context passage.
    return "Based on the documentation, this is the answer to your question [1]."


@pytest.fixture
def isolated_client(tmp_path, monkeypatch):
    # Isolate chunks.json + BM25 cache from the real data built in Phases 1-6.
    monkeypatch.setattr(pipeline_module, "CHUNKS_PATH", tmp_path / "chunks.json")
    pipeline_module._bm25_cache["index"] = None
    pipeline_module._bm25_cache["mtime"] = None

    # Isolate Qdrant: temp local-file path, dedicated collection, fresh client.
    monkeypatch.setattr(settings, "qdrant_local_path", str(tmp_path / "qdrant_test"))
    monkeypatch.setattr(settings, "qdrant_collection", "test_api_flow")
    get_client.cache_clear()

    # Isolate the raw-upload destination directory.
    monkeypatch.setattr(routes_module, "RAW_DATA_DIR", tmp_path / "raw")

    # Stub the three network-dependent calls at their real call sites.
    monkeypatch.setattr(indexer_module, "embed_texts", lambda texts: [_fake_vector(t) for t in texts])
    monkeypatch.setattr(pipeline_module, "embed_query", _fake_vector)
    monkeypatch.setattr(reranker_module, "_default_score_fn", _word_overlap_score_fn)
    monkeypatch.setattr(generate_module, "_default_complete_fn", _stub_complete_fn)

    return TestClient(app)


def test_upload_then_query_end_to_end(isolated_client, tmp_path):
    sample_path = tmp_path / "sample.md"
    sample_path.write_text(
        "# Widget API\n\nThe widget_timeout_error occurs when a widget request "
        "exceeds 30 seconds. Increase the timeout_seconds config value to fix it."
    )

    with sample_path.open("rb") as f:
        upload_response = isolated_client.post("/upload", files={"file": ("sample.md", f, "text/markdown")})

    assert upload_response.status_code == 200
    upload_data = upload_response.json()
    assert upload_data["filename"] == "sample.md"
    assert upload_data["chunks_added"] >= 1
    assert upload_data["chunks_indexed"] == upload_data["chunks_added"]

    query_response = isolated_client.post("/query", json={"question": "What causes widget_timeout_error?"})
    assert query_response.status_code == 200
    query_data = query_response.json()

    assert query_data["is_fully_grounded"] is True
    assert len(query_data["sources"]) >= 1
    assert query_data["sources"][0]["source"] == "sample.md"


def test_upload_rejects_unsupported_file_type(isolated_client, tmp_path):
    bad_path = tmp_path / "sample.exe"
    bad_path.write_bytes(b"not a real document")

    with bad_path.open("rb") as f:
        response = isolated_client.post("/upload", files={"file": ("sample.exe", f, "application/octet-stream")})

    assert response.status_code == 400


def test_query_with_no_documents_indexed_short_circuits(isolated_client):
    response = isolated_client.post("/query", json={"question": "anything"})
    assert response.status_code == 200
    data = response.json()
    assert data["warning"] == "no_documents_indexed"
    assert data["is_fully_grounded"] is False