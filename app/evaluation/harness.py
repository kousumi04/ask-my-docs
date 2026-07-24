"""
Runs the golden dataset through the live pipeline and scores the
results with RAGAS. query_fn is injected (defaulting to the real
app.core.pipeline.query) for the same reason as every phase since 3:
the dataset-building logic is fully testable with a stub, while the
real evaluation run needs the real pipeline, models, and network.

THE FOUR METRICS AND WHAT EACH ONE ACTUALLY CATCHES:
  faithfulness       -- does the answer's content follow from the retrieved
                         contexts, or does it contain claims the contexts
                         don't support? Catches hallucination directly.
  answer_relevancy    -- does the answer actually address the question asked,
                         rather than being faithful-but-off-topic?
  context_precision   -- of the contexts retrieved, how many were actually
                         relevant to the question? Low precision means the
                         retriever is pulling in noise alongside signal.
  context_recall      -- of what the reference answer needed, how much did
                         the retrieved contexts actually cover? Low recall
                         means relevant material exists in the corpus but
                         retrieval didn't surface it.

Faithfulness and context_precision/recall are retrieval-and-generation
diagnostics respectively -- a low score on one but not the other tells
you WHICH stage of the pipeline to investigate, not just that something
is wrong somewhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from app.evaluation.dataset import GoldenExample, load_golden_dataset

QueryFn = Callable[[str], dict]  # question -> {"answer": ..., "contexts": [...], ...}


def build_evaluation_samples(examples: list[GoldenExample], query_fn: QueryFn) -> list[dict]:
    """Runs each golden question through query_fn and assembles the fields
    RAGAS needs: user_input, response, retrieved_contexts, reference.

    Returned as plain dicts rather than ragas.SingleTurnSample objects so
    this function has no import-time dependency on ragas at all -- makes
    it trivially testable without ragas even installed, and keeps ragas's
    (fairly heavy, langchain-coupled) dependency chain out of anything
    that doesn't actually run a real evaluation.
    """
    samples = []
    for example in examples:
        result = query_fn(example.question)
        samples.append(
            {
                "user_input": example.question,
                "response": result["answer"],
                "retrieved_contexts": result.get("contexts", []),
                "reference": example.reference_answer,
            }
        )
    return samples


def run_evaluation(samples: list[dict], llm=None, embeddings=None) -> dict[str, float]:
    """Scores the assembled samples with RAGAS's four core metrics.

    llm/embeddings default to None here deliberately -- the CALLER (see
    scripts/run_eval.py) is responsible for passing the real Groq/bge
    adapters from llm_adapter.py. This function never silently falls
    back to RAGAS's own OpenAI default; if you forget to pass them,
    RAGAS will raise rather than quietly bill an OpenAI account.
    """
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    dataset = EvaluationDataset(samples=[SingleTurnSample(**s) for s in samples])
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )
    return dict(result)


def load_and_build_samples(golden_dataset_path: Path, query_fn: QueryFn) -> list[dict]:
    examples = load_golden_dataset(golden_dataset_path)
    return build_evaluation_samples(examples, query_fn)