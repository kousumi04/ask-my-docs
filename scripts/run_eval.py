"""
The actual CI entrypoint: run the golden dataset through the real,
live pipeline, score it with RAGAS, and exit non-zero if any metric
falls below its threshold -- this is what a GitHub Actions step calls
to fail a build on quality regression, not just on broken code.

Usage:
    python scripts/run_eval.py

Requires GROQ_API_KEY (for both generation and RAGAS's LLM judge) and
either QDRANT_URL+QDRANT_API_KEY or a populated local Qdrant, plus an
indexed corpus (run scripts/warmup_models.py and the indexer first if
you haven't already -- see QUICKSTART.md).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Same reasoning as scripts/warmup_models.py: a directly-run script only
# gets its own directory on sys.path, not the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.pipeline import query as pipeline_query  # noqa: E402
from app.evaluation.gating import check_thresholds  # noqa: E402
from app.evaluation.harness import load_and_build_samples, run_evaluation  # noqa: E402
from app.evaluation.llm_adapter import get_ragas_embeddings, get_ragas_llm  # noqa: E402
from app.evaluation.report import format_report  # noqa: E402

GOLDEN_DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "eval" / "golden_dataset.json"


def main() -> int:
    print(f"Loading golden dataset from {GOLDEN_DATASET_PATH}...")
    samples = load_and_build_samples(GOLDEN_DATASET_PATH, query_fn=pipeline_query)
    print(f"Ran {len(samples)} questions through the live pipeline.\n")

    print("Scoring with RAGAS (faithfulness, answer_relevancy, context_precision, context_recall)...")
    print("This calls the LLM judge once per metric per sample -- expect this to take a few minutes.\n")
    scores = run_evaluation(samples, llm=get_ragas_llm(), embeddings=get_ragas_embeddings())

    gate_result = check_thresholds(scores)
    print(format_report(scores, gate_result))

    return 0 if gate_result.passed else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 -- diagnostic CLI script, not library code
        print(f"\nEvaluation run failed to complete: {exc}")
        print("Common causes: GROQ_API_KEY missing/invalid, no documents indexed yet, or network issues.")
        sys.exit(1)