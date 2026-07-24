"""
RAGAS's evaluate() defaults to OpenAI for its LLM judge and embeddings
if you don't pass your own -- which would silently require an OpenAI
API key and cost money, directly contradicting this project's
free-tools-only constraint. This module exists so evaluate() is NEVER
called without explicit llm/embeddings arguments (see harness.py) --
wired to the same Groq LLM and bge embedding model already used
elsewhere in the project, wrapped in RAGAS's langchain adapters.
"""

from __future__ import annotations

from functools import lru_cache

from app.config import settings


@lru_cache(maxsize=1)
def get_ragas_llm():
    from langchain_groq import ChatGroq
    from ragas.llms import LangchainLLMWrapper

    chat_model = ChatGroq(api_key=settings.groq_api_key, model=settings.groq_model, temperature=0.0)
    return LangchainLLMWrapper(chat_model)


@lru_cache(maxsize=1)
def get_ragas_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    hf_embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    return LangchainEmbeddingsWrapper(hf_embeddings)