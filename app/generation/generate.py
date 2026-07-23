"""
Ties together prompts.py (build the citation-enforcing prompt),
llm_client.py (get an answer), and citation.py (verify the answer
actually cited correctly) into one call.

complete_fn is injected (defaulting to the real LLM client) for the
same reason as every other phase: the orchestration and validation
logic is fully testable without a live model, and swapping in the real
client for production use touches zero logic here.
"""

from __future__ import annotations

from typing import Callable

from app.generation.citation import cited_sources, validate_citations
from app.generation.llm_client import get_llm_client
from app.generation.prompts import build_prompt

CompleteFn = Callable[[list[dict]], str]


def _default_complete_fn(messages: list[dict]) -> str:
    return get_llm_client().complete(messages)


def answer_query(query: str, chunks: list[dict], complete_fn: CompleteFn | None = None) -> dict:
    """chunks: the reranked shortlist from Phase 5, in order (chunk 1 = most relevant)."""
    complete_fn = complete_fn or _default_complete_fn

    if not chunks:
        return {
            "answer": "I don't have any relevant documents to answer this question.",
            "sources": [],
            "is_fully_grounded": False,
            "warning": "no_context_retrieved",
        }

    messages = build_prompt(query, chunks)
    answer = complete_fn(messages)

    check = validate_citations(answer, num_context_chunks=len(chunks))
    sources = cited_sources(answer, chunks)

    result = {
        "answer": answer,
        "sources": sources,
        "is_fully_grounded": check.is_fully_grounded,
    }
    if not check.has_any_citation:
        result["warning"] = "no_citations_in_answer"
    elif check.invalid_numbers:
        result["warning"] = f"hallucinated_citation_numbers: {sorted(check.invalid_numbers)}"

    return result