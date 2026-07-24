from __future__ import annotations

from app.evaluation.dataset import GoldenExample
from app.evaluation.harness import build_evaluation_samples


def test_build_evaluation_samples_maps_fields_correctly():
    examples = [
        GoldenExample(question="What does E1101 mean?", reference_answer="A broker timeout.", reference_source="queuely_docs.md"),
        GoldenExample(question="How do I add a backend?", reference_answer="Implement BrokerProtocol.", reference_source="architecture_guide.docx"),
    ]

    def stub_query_fn(question: str) -> dict:
        return {
            "answer": f"Stub answer for: {question}",
            "contexts": [f"Some retrieved context for {question}"],
            "sources": [],
            "is_fully_grounded": True,
        }

    samples = build_evaluation_samples(examples, stub_query_fn)

    assert len(samples) == 2
    assert samples[0]["user_input"] == "What does E1101 mean?"
    assert samples[0]["response"] == "Stub answer for: What does E1101 mean?"
    assert samples[0]["retrieved_contexts"] == ["Some retrieved context for What does E1101 mean?"]
    assert samples[0]["reference"] == "A broker timeout."


def test_build_evaluation_samples_handles_missing_contexts_key_gracefully():
    """query_fn's no-documents-indexed short-circuit path might omit
    'contexts' in older callers -- .get() with a default prevents a KeyError
    from turning into a confusing evaluation-run crash."""
    examples = [GoldenExample(question="q", reference_answer="r", reference_source="s.md")]

    def stub_query_fn(question: str) -> dict:
        return {"answer": "no docs", "sources": [], "is_fully_grounded": False}  # no "contexts" key

    samples = build_evaluation_samples(examples, stub_query_fn)
    assert samples[0]["retrieved_contexts"] == []