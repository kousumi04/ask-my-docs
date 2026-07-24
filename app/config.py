"""
Centralized configuration using pydantic-settings.

Why this exists: hardcoding model names, ports, or API keys throughout
the codebase makes a project unmaintainable and is an instant red flag
in a code review. Every configurable value lives here, sourced from
environment variables (with sane defaults for local dev).
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- App metadata ---
    app_name: str = "Ask My Docs"
    app_version: str = "0.1.0"
    environment: str = "development"  # development | production

    # --- Vector store (Qdrant) ---
    # Three ways to connect, checked in this priority order by get_client():
    #   1. qdrant_url set       -> Qdrant Cloud (managed, needs api_key too)
    #   2. qdrant_local_path set -> embedded local-file mode (no server, dev/test default)
    #   3. neither set           -> self-hosted server at qdrant_host:qdrant_port (e.g. Docker)
    qdrant_url: str = ""       # e.g. "https://xxxx-xxxx.aws.cloud.qdrant.io:6333"
    qdrant_api_key: str = ""   # from your Qdrant Cloud cluster's API Keys tab
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "ask_my_docs"
    qdrant_local_path: str = "data/qdrant_local"

    # --- Embedding model ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # --- Reranker ---
    reranker_model: str = "BAAI/bge-reranker-base"

    # --- LLM ---
    llm_provider: str = "groq"  # groq | ollama
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"
    ollama_model: str = "llama3.1:8b"
    ollama_host: str = "http://localhost:11434"

    # --- Retrieval tuning ---
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k_dense: int = 20
    top_k_sparse: int = 20
    top_k_rerank: int = 5
    rrf_k: int = 60  # Reciprocal Rank Fusion constant


settings = Settings()