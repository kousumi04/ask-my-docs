from __future__ import annotations

from app.generation.generate import answer_query

CHUNKS = [
    {"source": "queuely_docs.md", "page_number": None, "text": "E1101 is a broker connection timeout, raised after 5 seconds by default."},
    {"source": "architecture_guide.docx", "page_number": None, "text": "New brokers implement BrokerProtocol with enqueue, dequeue, and ack."},
]


def test_answer_query_with_well_grounded_stub_response():
    def stub_complete(messages: list[dict]) -> str:
        return "E1101 means the broker connection timed out after the default 5 second window [1]."

    result = answer_query("What does E1101 mean?", CHUNKS, complete_fn=stub_complete)

    assert result["is_fully_grounded"] is True
    assert "warning" not in result
    assert result["sources"] == [{"marker": 1, "source": "queuely_docs.md", "page_number": None}]


def test_answer_query_flags_hallucinated_citation():
    def stub_complete(messages: list[dict]) -> str:
        # Only 2 chunks were provided, so [4] cannot be a real citation.
        return "This is answered according to passage [4]."

    result = answer_query("some question", CHUNKS, complete_fn=stub_complete)

    assert result["is_fully_grounded"] is False
    assert "warning" in result and "hallucinated_citation_numbers" in result["warning"]


def test_answer_query_flags_missing_citations():
    def stub_complete(messages: list[dict]) -> str:
        return "Broker timeouts happen sometimes."  # no [n] markers at all

    result = answer_query("some question", CHUNKS, complete_fn=stub_complete)

    assert result["warning"] == "no_citations_in_answer"
    assert result["is_fully_grounded"] is False


def test_answer_query_with_no_retrieved_context_short_circuits_without_calling_llm():
    calls = []

    def stub_complete(messages: list[dict]) -> str:
        calls.append(messages)
        return "should never be called"

    result = answer_query("unanswerable question", [], complete_fn=stub_complete)

    assert calls == [], "LLM should not be called at all when there's no retrieved context"
    assert result["warning"] == "no_context_retrieved"
    assert result["is_fully_grounded"] is False