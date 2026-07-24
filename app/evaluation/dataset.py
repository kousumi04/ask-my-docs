"""
Loads the golden evaluation dataset -- hand-written question/reference-answer
pairs grounded in the actual sample corpus (see data/eval/golden_dataset.json),
used to measure the live pipeline's answers against known-correct references.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GoldenExample:
    question: str
    reference_answer: str
    reference_source: str


def load_golden_dataset(path: Path) -> list[GoldenExample]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [GoldenExample(**item) for item in raw]