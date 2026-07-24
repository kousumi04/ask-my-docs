# Ask My Docs — Local Setup & Verification (Phases 1–6)

This project has no long-running server yet (that's Phase 7 — the full
FastAPI service). Right now, "is it running" means: can each piece be
executed and does it produce the expected output. This doc walks
through exactly that, phase by phase.

## 0. One-time environment setup

```bash
cd ask-my-docs
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Open `.env` and fill in `GROQ_API_KEY` (free, no card required, from
console.groq.com) — needed for Phase 6 generation. Everything through
Phase 5 works without it.

**How do I know this step worked?**
```bash
python -c "from app.config import settings; print(settings.app_name)"
# -> Ask My Docs
```
If that prints without an ImportError, your venv and dependencies are correctly installed.

---

## 1. Phase 1 — Is the API skeleton running?

```bash
uvicorn app.main:app --reload --port 8000
```

In a second terminal:
```bash
curl http://localhost:8000/health
# -> {"status":"ok","app":"Ask My Docs","environment":"development"}
```

You can also open `http://localhost:8000/docs` in a browser for the
auto-generated Swagger UI — if that page loads, the server is up.
Stop it with `Ctrl+C` when done (or leave it running in that terminal
while you work in another).

---

## 2. Phase 2 — Does ingestion produce chunks?

```bash
python -m app.ingestion.pipeline
```

**Expected output:**
```
Ingested 12 chunks from .../data/raw -> .../data/processed/chunks.json
```

Check the result directly:
```bash
python -c "import json; c = json.load(open('data/processed/chunks.json')); print(len(c), 'chunks'); print(c[0]['text'][:80])"
```
If `data/processed/chunks.json` exists and has content, ingestion worked.

---

## 3. Phase 3 — Connecting to Qdrant Cloud

This project is configured to use **Qdrant Cloud** (managed hosting)
instead of local-file mode. Here's the exact setup:

### 3.1 Create a free Qdrant Cloud cluster

1. Go to **cloud.qdrant.io** and sign up (free tier available, no
   card required for the free cluster).
2. Create a new cluster — the free tier is enough for this project.
3. Once it's provisioned, open the cluster and copy its **Cluster URL**
   (looks like `https://xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx.us-east-1-0.aws.cloud.qdrant.io:6333`).
4. Go to the cluster's **API Keys** tab, create a key, and copy it
   immediately — Qdrant Cloud only shows it once.

### 3.2 Add both to `.env`

```
QDRANT_URL=https://your-actual-cluster-url.aws.cloud.qdrant.io:6333
QDRANT_API_KEY=your_actual_api_key_here
QDRANT_LOCAL_PATH=
```

Leaving `QDRANT_LOCAL_PATH` blank matters less than it looks — `get_client()`
checks `QDRANT_URL` first regardless — but keeping it blank avoids
confusion about which mode is actually active.

### 3.3 Index into Cloud

```bash
python -m app.retrieval.indexer
```

Same command as local mode — the only thing that changed is *where*
the data goes, not how you call it. First run still downloads
`bge-small-en-v1.5` from Hugging Face (one-time, ~130MB) for the
embedding model itself, which is separate from where vectors are stored.

**Expected output:**
```
Indexed 12 chunks into Qdrant collection.
```

**How do I know Qdrant Cloud actually has the data?**

Easiest: log into the Qdrant Cloud dashboard, open your cluster, and
check the **Collections** tab — you should see `ask_my_docs` with
12 points.

Or from the command line:
```bash
python3 -c "
from app.retrieval.vector_store import get_client
from app.config import settings
client = get_client()
info = client.get_collection(settings.qdrant_collection)
print('Points in collection:', info.points_count)
"
# -> Points in collection: 12
```
If that prints 12, embeddings were generated and stored in your Cloud cluster correctly.

**Troubleshooting:**
- `Unauthorized` / 403 errors → `QDRANT_API_KEY` is wrong or missing in `.env`
- Connection timeout → double check `QDRANT_URL` — it must include `:6333` and the `https://` prefix exactly as shown on your cluster's dashboard
- Free tier clusters can auto-pause after inactivity — if requests suddenly start timing out after it's worked before, check your cluster's status on the dashboard

---

## 4. Phase 4 — BM25 + fusion (no external service needed)

BM25 is pure Python — nothing to "connect" to, it's rebuilt in memory
from `chunks.json` each time it's used. Sanity-check it directly:

```bash
python3 -c "
import json
from app.ingestion.chunking import Chunk
from app.retrieval.sparse import build_bm25_index, search

chunks = [Chunk(**c) for c in json.load(open('data/processed/chunks.json'))]
index = build_bm25_index(chunks)
for r in search(index, 'E1101 broker timeout', top_k=3):
    print(f'{r[\"score\"]:.2f}  {r[\"source\"]}')
"
```
If the top result is from `queuely_docs.md` with the highest score, it's working.

---

## 5. Phase 5 & 6 — Reranking and generation

These need their own models/APIs (`bge-reranker-base` from Hugging
Face, and Groq for generation) — same one-time-download pattern as
Phase 3's embedding model. There's no persistent "indexer" step for
these; they run per-query. A full manual smoke test once you have a
Groq key:

```bash
python3 << 'EOF'
import json
from app.ingestion.chunking import Chunk
from app.retrieval.sparse import build_bm25_index, search as bm25_search
from app.retrieval.embeddings import embed_query
from app.retrieval.vector_store import get_client, search as vector_search
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.reranker import rerank
from app.generation.generate import answer_query

chunks = [Chunk(**c) for c in json.load(open('data/processed/chunks.json'))]
bm25_index = build_bm25_index(chunks)

query = "What does error E1101 mean?"
sparse = bm25_search(bm25_index, query, top_k=5)
dense = vector_search(get_client(), embed_query(query), top_k=5)
fused = reciprocal_rank_fusion([dense, sparse], top_k=5)
reranked = rerank(query, fused, top_k=3)

result = answer_query(query, reranked)
print(result["answer"])
print("Sources:", result["sources"])
print("Fully grounded:", result["is_fully_grounded"])
EOF
```

If this prints a real answer citing `queuely_docs.md` with
`Fully grounded: True`, the entire Phase 1–6 pipeline is working
end-to-end for real.

---

## 6. Fastest overall health check: run the test suite

Every phase's *logic* (as opposed to the live models/APIs) is unit
tested and needs **no internet access at all**:

```bash
pytest tests/ -v
```

**Expected:** `21 passed`. This won't catch a bad Groq API key or a
Hugging Face download failure, but it will immediately catch any
environment/dependency/import problem — run this first whenever
something feels broken, before chasing model or network issues.

---

## 7. Phase 7 — Running the full app (API + Streamlit frontend)

**Recommended first step — warm up the models:**
```bash
python scripts/warmup_models.py
```
This forces the one-time downloads of the embedding model (~130MB) and
reranker model (~1.1GB, the slow one) to happen here, with visible
progress, instead of silently inside your first upload or query — where
a slow connection can exceed the UI's timeout and look like a failure
when it's actually just a slow download in progress. Safe to skip if
you don't mind waiting through it inside the UI instead; either way
it's a one-time cost, cached after the first run.

Two processes, two terminals:

**Terminal 1 — the API:**
```bash
uvicorn app.main:app --reload --port 8000
```

**Terminal 2 — the frontend:**
```bash
streamlit run streamlit_app.py
```

Streamlit opens in your browser automatically (usually
`http://localhost:8501`). Upload a document from the sidebar, then ask
a question about it in the chat box.

**How do I know it's actually working end-to-end?**
- The sidebar should show "Indexed N chunks from `<filename>`" after upload — if you instead see a red error box, check that Terminal 1 (the API) is actually running and reachable at `http://localhost:8000`.
- A question about the uploaded document should come back with an answer, a "Sources: [1] ..." caption, and no warning banner. A `no_documents_indexed` or `no_citations_in_answer` warning means something upstream didn't work as expected — check Terminal 1's logs for errors first.

If you'd rather test the API directly without the UI:
```bash
curl -X POST http://localhost:8000/upload -F "file=@data/raw/queuely_docs.md"
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" \
  -d '{"question": "What does error E1101 mean?"}'
```

---

## Troubleshooting quick reference
|---|---|
| `ModuleNotFoundError` on any `app.*` import | venv not activated, or you're running from the wrong directory (must be repo root) |
| `pytest` fails on collection/import | dependency missing — re-run `pip install -r requirements.txt` |
| Indexer hangs or errors reaching huggingface.co | check real internet access; corporate VPN/proxy can also block it |
| `Points in collection: 0` after indexing | check the indexer actually printed "Indexed 12 chunks" with no error above it |
| Groq call fails with auth error | `GROQ_API_KEY` missing or wrong in `.env` |
| Ollama call fails to connect | Ollama isn't running locally — only relevant if you set `LLM_PROVIDER=ollama` |