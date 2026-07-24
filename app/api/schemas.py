from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to answer using indexed documents.")
    top_k: int | None = Field(None, description="How many reranked chunks to feed the LLM. Defaults to settings.top_k_rerank.")


class SourceCitation(BaseModel):
    marker: int
    source: str
    page_number: int | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]
    is_fully_grounded: bool
    warning: str | None = None


class UploadResponse(BaseModel):
    filename: str
    chunks_added: int
    chunks_indexed: int