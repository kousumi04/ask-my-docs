# Multi-stage build: the `builder` stage installs dependencies into a
# venv, and the final stage copies only that venv plus the app code --
# build tools and pip's cache never end up in the shipped image, which
# is most of what keeps this smaller than a single-stage `pip install`.

FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN python -m venv /venv \
    && /venv/bin/pip install --no-cache-dir --upgrade pip

# CRITICAL: install CPU-only torch BEFORE requirements.txt. Without this,
# pip's default Linux wheel for torch (pulled in transitively by
# sentence-transformers) includes full CUDA/GPU support -- several GB of
# nvidia_cublas/cudnn/cusparselt/etc downloads for a container that never
# touches a GPU (embeddings/reranking run on CPU; the LLM call goes to
# Groq's API, not a local GPU). Installing the CPU build first satisfies
# sentence-transformers' torch requirement, so the requirements.txt install
# below does not re-resolve and re-download the CUDA version afterward.
RUN /venv/bin/pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu

RUN /venv/bin/pip install --no-cache-dir -r requirements.txt


FROM python:3.11-slim

# Run as a non-root user -- a container running as root is a real
# privilege-escalation risk if the app is ever compromised, and it
# costs nothing to avoid here.
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH"

COPY app/ ./app/
COPY data/raw/ ./data/raw/
COPY data/eval/ ./data/eval/

# Model downloads (bge-small-en-v1.5, bge-reranker-base) happen at
# runtime on first use, same as local dev -- see QUICKSTART.md section
# on scripts/warmup_models.py. Baking them into the image is a valid
# Phase 10 optimization (faster cold start, no first-request latency
# spike) but adds real build time and image size, so it's deliberately
# left out of this baseline Dockerfile.

RUN mkdir -p /app/data/processed /app/data/qdrant_local && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]