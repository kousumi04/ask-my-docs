from __future__ import annotations

from app.evaluation.gating import GateResult


def format_report(scores: dict[str, float], gate_result: GateResult) -> str:
    lines = ["RAGAS Evaluation Report", "=" * 24, ""]
    for metric, value in scores.items():
        lines.append(f"  {metric:<20} {value:.3f}")
    lines.append("")
    if gate_result.passed:
        lines.append("PASSED -- all metrics meet their thresholds.")
    else:
        lines.append("FAILED -- the following metrics are below threshold:")
        for failure in gate_result.failures:
            lines.append(f"  - {failure}")
    return "\n".join(lines)