from pydantic_settings import BaseSettings, SettingsConfigDict
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    app_name: str = "Ask My Docs"
    environment: str = "development"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "ask_my_docs"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    reranker_model: str = "BAAI/bge-reranker-base"
    llm_provider: str = "groq"
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k_dense: int = 20
    top_k_sparse: int = 20
    top_k_rerank: int = 5
    rrf_k: int = 60
settings = Settings()
