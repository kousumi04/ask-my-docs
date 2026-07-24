from __future__ import annotations

from pathlib import Path

from app.evaluation.dataset import GoldenExample, load_golden_dataset

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "data" / "eval" / "golden_dataset.json"


def test_load_golden_dataset_from_real_file():
    examples = load_golden_dataset(GOLDEN_PATH)

    assert len(examples) == 8
    assert all(isinstance(e, GoldenExample) for e in examples)
    assert all(e.question and e.reference_answer and e.reference_source for e in examples)


def test_golden_dataset_sources_match_real_sample_corpus():
    """Every reference_source should be one of the actual files in data/raw/
    -- catches a typo'd filename in the golden dataset itself."""
    examples = load_golden_dataset(GOLDEN_PATH)
    real_sources = {"queuely_docs.md", "deployment_notes.txt", "architecture_guide.docx", "api_reference.pdf"}

    for example in examples:
        assert example.reference_source in real_sources, f"Unknown source: {example.reference_source}"