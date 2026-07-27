from __future__ import annotations

import math

from app.evaluation.gating import DEFAULT_THRESHOLDS, check_thresholds


def test_check_thresholds_passes_when_all_scores_meet_minimums():
    scores = {"faithfulness": 0.85, "answer_relevancy": 0.75, "context_precision": 0.80, "context_recall": 0.72}
    result = check_thresholds(scores)
    assert result.passed is True
    assert result.failures == []


def test_check_thresholds_fails_and_reports_which_metric():
    scores = {"faithfulness": 0.62, "answer_relevancy": 0.75, "context_precision": 0.80, "context_recall": 0.72}
    result = check_thresholds(scores)
    assert result.passed is False
    assert len(result.failures) == 1
    assert "faithfulness" in result.failures[0]
    assert "0.620" in result.failures[0]


def test_check_thresholds_reports_multiple_failures():
    scores = {"faithfulness": 0.50, "answer_relevancy": 0.40, "context_precision": 0.90, "context_recall": 0.90}
    result = check_thresholds(scores)
    assert result.passed is False
    assert len(result.failures) == 2


def test_check_thresholds_flags_a_missing_metric_rather_than_ignoring_it():
    """If RAGAS silently fails to compute one metric (e.g. an API error for
    just that metric), a missing key must fail the gate, not pass by default."""
    scores = {"faithfulness": 0.90, "answer_relevancy": 0.90, "context_precision": 0.90}  # context_recall missing
    result = check_thresholds(scores)
    assert result.passed is False
    assert any("context_recall" in f and "missing" in f for f in result.failures)


def test_check_thresholds_flags_nan_metric_rather_than_passing_it():
    scores = {"faithfulness": 0.90, "answer_relevancy": math.nan, "context_precision": 0.90, "context_recall": 0.90}
    result = check_thresholds(scores)
    assert result.passed is False
    assert any("answer_relevancy" in f and "finite" in f for f in result.failures)


def test_default_thresholds_cover_all_four_ragas_metrics_used():
    assert set(DEFAULT_THRESHOLDS.keys()) == {"faithfulness", "answer_relevancy", "context_precision", "context_recall"}
