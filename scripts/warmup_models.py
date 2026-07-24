"""
Run this once, from the project root, before your first upload or
query: python scripts/warmup_models.py

Forces both one-time model downloads (bge-small-en-v1.5 for embeddings,
~130MB, and bge-reranker-base for reranking, ~1.1GB) to happen here,
with visible progress, rather than silently inside the first real
/upload or /query request -- where a slow connection can easily exceed
any reasonable HTTP timeout and surface as a confusing error instead of
an expected one-time wait.

Safe to re-run any time: both models are cached locally after the
first download (in ~/.cache/huggingface), so subsequent runs of this
script complete in a second or two.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Running `python scripts/warmup_models.py` only puts scripts/ on sys.path,
# not the project root -- so `import app.*` fails with ModuleNotFoundError
# unless we add the root explicitly. (Same underlying class of issue as the
# pytest ModuleNotFoundError fixed earlier via pytest.ini's pythonpath
# setting, which only applies to pytest runs, not plain `python script.py`.)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    print("Warming up models used by Ask My Docs. This downloads them once")
    print("and caches them locally -- safe to re-run, later runs are instant.\n")

    print("[1/2] Embedding model (BAAI/bge-small-en-v1.5, ~130MB)...")
    start = time.time()
    from app.retrieval.embeddings import embed_query

    embed_query("warmup")
    print(f"      Done in {time.time() - start:.1f}s.\n")

    print("[2/2] Reranker model (BAAI/bge-reranker-base, ~1.1GB -- this is the big one)...")
    start = time.time()
    from app.retrieval.reranker import rerank

    rerank("warmup", [{"chunk_id": "warmup", "text": "warmup"}], top_k=1)
    print(f"      Done in {time.time() - start:.1f}s.\n")

    print("Both models are cached. Uploads and queries will be fast from here on")
    print("(the LLM call to Groq is a fast API call, not a local download, so")
    print("nothing to warm up there).")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 -- this is a diagnostic CLI script, not library code
        print(f"\nWarmup failed: {exc}")
        print("Check your internet connection can reach huggingface.co, then try again.")
        sys.exit(1)