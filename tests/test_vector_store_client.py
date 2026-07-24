"""
get_client()'s priority order (Cloud url -> local path -> self-hosted
host/port) is a real branch a config typo could silently get wrong --
e.g. leaving both qdrant_url AND qdrant_local_path set would only ever
use Cloud, which might surprise someone expecting local mode. This
locks the priority order down with a test rather than leaving it as an
unverified docstring claim.
"""

from __future__ import annotations

from app.config import settings
from app.retrieval.vector_store import get_client


def test_get_client_prefers_cloud_url_when_set(monkeypatch):
    monkeypatch.setattr(settings, "qdrant_url", "https://fake-cluster.aws.cloud.qdrant.io:6333")
    monkeypatch.setattr(settings, "qdrant_api_key", "fake-key")
    monkeypatch.setattr(settings, "qdrant_local_path", "data/qdrant_local")  # set too, should be ignored
    get_client.cache_clear()

    client = get_client()
    # qdrant-client stores the configured REST host internally; confirming it
    # points at our fake cloud URL (not local-file mode) proves the branch taken.
    assert "fake-cluster" in str(client._client.rest_uri)


def test_get_client_falls_back_to_local_path_when_no_cloud_url(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "qdrant_url", "")
    monkeypatch.setattr(settings, "qdrant_local_path", str(tmp_path / "qtest"))
    get_client.cache_clear()

    client = get_client()
    # Local-file mode clients don't have a rest_uri in the same sense;
    # constructing without error and being usable is the meaningful check.
    client.get_collections()  # would raise if this weren't a working local client