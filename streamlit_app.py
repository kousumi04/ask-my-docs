"""
Streamlit frontend for Ask My Docs.

Deliberately a thin HTTP client, nothing more: every piece of real
logic (ingestion, retrieval, generation, citation validation) already
lives in the FastAPI service from Phase 7. This file's only job is
rendering a chat UI and a file uploader against that API -- it holds
no business logic of its own, so the same backend could equally serve
a different frontend (a CLI, a Slack bot) without any duplication.

TIMEOUT NOTE: the first /upload and the first /query each trigger a
one-time model download (bge-small-en-v1.5 for embeddings, then
bge-reranker-base for reranking -- ~1.1GB). On a slow connection this
can take several minutes, well past a typical HTTP timeout. Run
`python scripts/warmup_models.py` once before using this UI to force
both downloads up front, outside of any request's time budget --
otherwise the timeout below is set generously (15 minutes) specifically
to survive a slow first-time download rather than fail with a raw
exception mid-conversation.
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API_BASE_URL = os.environ.get("ASK_MY_DOCS_API_URL", "http://localhost:8000")
REQUEST_TIMEOUT_SECONDS = 900  # generous: covers a slow one-time model download, not just normal query latency

st.set_page_config(page_title="Ask My Docs", page_icon="\U0001F4DA", layout="centered")
st.title("\U0001F4DA Ask My Docs")
st.caption("Hybrid-retrieval RAG over your documents, with enforced citations.")
st.caption(
    "First upload/query after a fresh install may take several minutes (one-time model "
    "downloads). Run `python scripts/warmup_models.py` beforehand to avoid this wait."
)

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"|"assistant", "content": ..., "sources": [...]}


def _post_with_friendly_errors(url: str, **kwargs) -> requests.Response | None:
    """Wraps requests.post with the two failure modes users actually hit:
    the API not running at all, and a slow first-time model download
    exceeding even our generous timeout. Returns None on failure after
    showing an st.error -- callers just check for None."""
    try:
        return requests.post(url, timeout=REQUEST_TIMEOUT_SECONDS, **kwargs)
    except requests.exceptions.ConnectionError:
        st.error(f"Can't reach the API at {API_BASE_URL}. Is `uvicorn app.main:app` running?")
    except requests.exceptions.Timeout:
        st.error(
            f"Request timed out after {REQUEST_TIMEOUT_SECONDS}s. If this is your first "
            "upload or query, a model may still be downloading in the API's terminal -- "
            "check that window for progress, then try again once it finishes."
        )
    return None


with st.sidebar:
    st.header("Upload a document")
    uploaded_file = st.file_uploader("PDF, DOCX, Markdown, or TXT", type=["pdf", "docx", "md", "markdown", "txt"])
    if uploaded_file is not None and st.button("Index this document"):
        with st.spinner(f"Ingesting and indexing {uploaded_file.name}... (can take minutes on first run)"):
            response = _post_with_friendly_errors(
                f"{API_BASE_URL}/upload",
                files={"file": (uploaded_file.name, uploaded_file.getvalue())},
            )
            if response is not None:
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"Indexed {data['chunks_indexed']} chunks from {data['filename']}.")
                else:
                    st.error(f"Upload failed ({response.status_code}): {response.text}")

    st.divider()
    st.caption(f"API: {API_BASE_URL}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("sources"):
            source_lines = []
            for s in message["sources"]:
                loc = s["source"]
                if s.get("page_number"):
                    loc += f", page {s['page_number']}"
                source_lines.append(f"[{s['marker']}] {loc}")
            st.caption("Sources: " + "; ".join(source_lines))
        if message.get("warning"):
            st.warning(f"\u26a0\ufe0f {message['warning']}")

if question := st.chat_input("Ask a question about your indexed documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching and generating an answer... (can take minutes on first run)"):
            response = _post_with_friendly_errors(f"{API_BASE_URL}/query", json={"question": question})
            if response is not None:
                if response.status_code == 200:
                    data = response.json()
                    st.write(data["answer"])
                    if data.get("sources"):
                        source_lines = []
                        for s in data["sources"]:
                            loc = s["source"]
                            if s.get("page_number"):
                                loc += f", page {s['page_number']}"
                            source_lines.append(f"[{s['marker']}] {loc}")
                        st.caption("Sources: " + "; ".join(source_lines))
                    if data.get("warning"):
                        st.warning(f"\u26a0\ufe0f {data['warning']}")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": data["answer"], "sources": data.get("sources"), "warning": data.get("warning")}
                    )
                else:
                    error_text = f"Query failed ({response.status_code}): {response.text}"
                    st.error(error_text)
                    st.session_state.messages.append({"role": "assistant", "content": error_text})