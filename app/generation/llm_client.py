"""
LLM client abstraction: one complete(messages) -> str interface, two
implementations. This is what lets generate.py stay provider-agnostic
-- swapping Groq for Ollama, or adding a third provider later, is a
new class here, not a change anywhere generation logic lives.

WHY GROQ AS PRIMARY: Groq's free tier runs open models (Llama 3.1 8B)
on their own fast inference hardware, so responses come back quickly
without needing a local GPU -- meaningfully better for a live demo
than CPU-bound local inference. Ollama is kept as a same-interface
fallback for fully offline use, or if a Groq API key isn't available.

NEITHER CLIENT CAN BE EXERCISED IN THIS SANDBOX: Groq's API and a
local Ollama server both require network/service access this sandboxed
environment doesn't have (see Phase 3's note on the same constraint
for Hugging Face). The classes below are the real implementation you
run locally; generate.py's tests use a stub complete_fn instead, the
same dependency-injection pattern used for the embedding model, the
reranker, and hybrid search in every phase so far.
"""

from __future__ import annotations

from typing import Protocol

from app.config import settings


class LLMClient(Protocol):
    def complete(self, messages: list[dict]) -> str: ...


class GroqClient:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or settings.groq_api_key
        self.model = model or settings.groq_model

    def complete(self, messages: list[dict]) -> str:
        from groq import Groq

        client = Groq(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,  # low temperature: favor grounded, consistent answers over creative ones
        )
        return response.choices[0].message.content


class OllamaClient:
    def __init__(self, host: str | None = None, model: str | None = None):
        self.host = host or settings.ollama_host
        self.model = model or settings.ollama_model

    def complete(self, messages: list[dict]) -> str:
        import httpx

        response = httpx.post(
            f"{self.host}/api/chat",
            json={"model": self.model, "messages": messages, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]


def get_llm_client() -> LLMClient:
    if settings.llm_provider == "ollama":
        return OllamaClient()
    return GroqClient()