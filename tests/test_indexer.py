from __future__ import annotations

import app.retrieval.indexer as indexer_module
from app.config import settings
from app.ingestion.chunking import Chunk


def _make_chunk(idx: int) -> Chunk:
    text = f"chunk text {idx}"
    return Chunk(
        chunk_id=f"file.md::{idx}",
        text=text,
        source="file.md",
        file_type="md",
        chunk_index=idx,
        char_start=0,
        char_end=len(text),
    )


def test_index_new_chunks_embeds_and_upserts_in_batches(monkeypatch):
    chunks = [_make_chunk(i) for i in range(5)]
    embedded_batches: list[list[str]] = []
    upserted_batches: list[list[str]] = []

    monkeypatch.setattr(settings, "indexing_batch_size", 2)
    monkeypatch.setattr(indexer_module, "get_client", lambda: object())
    monkeypatch.setattr(indexer_module, "ensure_collection", lambda client, recreate=False: None)

    def fake_embed_texts(texts: list[str]) -> list[list[float]]:
        embedded_batches.append(texts)
        return [[1.0] for _ in texts]

    def fake_upsert_chunks(client, batch: list[Chunk], vectors: list[list[float]]) -> None:
        upserted_batches.append([chunk.chunk_id for chunk in batch])
        assert len(batch) == len(vectors)

    monkeypatch.setattr(indexer_module, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(indexer_module, "upsert_chunks", fake_upsert_chunks)

    assert indexer_module.index_new_chunks(chunks) == 5
    assert embedded_batches == [
        ["chunk text 0", "chunk text 1"],
        ["chunk text 2", "chunk text 3"],
        ["chunk text 4"],
    ]
    assert upserted_batches == [
        ["file.md::0", "file.md::1"],
        ["file.md::2", "file.md::3"],
        ["file.md::4"],
    ]
