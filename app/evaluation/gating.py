"""
Decides pass/fail for CI given a dict of metric scores and thresholds.

Deliberately has zero dependency on ragas or any model -- this is the
part of "CI-gated evaluation" that has to be simple, fast, and 100%
deterministic, since it's the thing standing between a merge and a
blocked build. The scores it receives can come from a real RAGAS run
or a stub in a test; this function can't tell the difference and
doesn't need to.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

DEFAULT_THRESHOLDS: dict[str, float] = {
    "faithfulness": 0.80,
    "answer_relevancy": 0.70,
    "context_precision": 0.70,
    "context_recall": 0.70,
}


@dataclass
class GateResult:
    passed: bool
    failures: list[str] = field(default_factory=list)  # e.g. ["faithfulness: 0.62 < 0.80"]


def check_thresholds(scores: dict[str, float], thresholds: dict[str, float] | None = None) -> GateResult:
    thresholds = thresholds if thresholds is not None else DEFAULT_THRESHOLDS
    failures = []

    for metric, minimum in thresholds.items():
        if metric not in scores:
            failures.append(f"{metric}: missing from evaluation results")
            continue
        if not math.isfinite(scores[metric]):
            failures.append(f"{metric}: did not produce a finite score")
            continue
        if scores[metric] < minimum:
            failures.append(f"{metric}: {scores[metric]:.3f} < {minimum:.3f}")

    return GateResult(passed=len(failures) == 0, failures=failures)
