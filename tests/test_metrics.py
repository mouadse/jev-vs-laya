from __future__ import annotations

import pytest

from darija_eval.metrics import compute_metrics, expected_calibration_error, wilson_interval, error_analysis


def record(gold, predicted, probabilities, style="Arabic", topic="it", latency=10.0):
    return {
        "gold": gold,
        "predicted": predicted,
        "correct": gold == predicted,
        "probabilities": probabilities,
        "confidence": probabilities[predicted] if probabilities else None,
        "writing_style": style,
        "topic": topic,
        "latency_ms": latency,
    }


def test_quality_confusion_and_calibration() -> None:
    rows = [
        record("positive", "positive", {"positive": 0.8, "neutral": 0.1, "negative": 0.1}),
        record("neutral", "positive", {"positive": 0.6, "neutral": 0.3, "negative": 0.1}, "Arabizi"),
        record("negative", "negative", {"positive": 0.1, "neutral": 0.2, "negative": 0.7}, "Arabizi"),
    ]
    metrics = compute_metrics(rows, requested=3, api_failures=0, wall_clock_seconds=1.0)
    assert metrics["quality"]["accuracy"] == pytest.approx(2 / 3)
    assert metrics["quality"]["confusion_matrix"] == [[1, 0, 0], [1, 0, 0], [0, 0, 1]]
    assert metrics["calibration"]["average_confidence"] == pytest.approx(0.7)
    assert metrics["calibration"]["brier_score"] == pytest.approx((0.06 + 0.86 + 0.14) / 3)
    assert metrics["calibration"]["thresholds"][0]["coverage"] == 1.0


def test_ece_known_value() -> None:
    assert expected_calibration_error([True, False], [0.8, 0.6], bins=2) == pytest.approx(0.2)


def test_probability_metrics_skipped_when_missing() -> None:
    metrics = compute_metrics(
        [record("positive", "positive", None)],
        requested=1,
        api_failures=0,
        wall_clock_seconds=0.1,
    )
    assert metrics["calibration"] is None
    assert "skipped" in metrics["warning"]


def test_wilson_interval_and_majority_reference():
    assert wilson_interval(0, 0) is None
    assert wilson_interval(0, 10)["high"] == pytest.approx(0.2775328)
    assert wilson_interval(10, 10)["low"] == pytest.approx(0.7224672)
    rows = [record("positive", "positive", None), record("positive", "negative", None)]
    metrics = compute_metrics(rows, requested=2, api_failures=0, wall_clock_seconds=1)
    assert metrics["quality"]["observed_majority_baseline"]["accuracy"] == 1
    assert metrics["quality"]["accuracy_ci95"]["low"] < 0.5


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1, 0.5])
def test_reject_invalid_probability_distributions(bad):
    row = record("positive", "positive", {"positive": bad, "neutral": 0, "negative": 0})
    with pytest.raises(ValueError, match="probabilities"):
        compute_metrics([row], requested=1, api_failures=0, wall_clock_seconds=1)


@pytest.mark.parametrize("correct,confidence,bins", [([True], [], 10), ([True], [1.1], 10), ([True], [0.8], 0)])
def test_ece_rejects_invalid_inputs(correct, confidence, bins):
    with pytest.raises(ValueError):
        expected_calibration_error(correct, confidence, bins)


def test_calibration_uses_labels_and_reports_requested_coverage():
    row = record("negative", "positive", {"positive": 0.9, "neutral": 0.05, "negative": 0.05})
    row["correct"] = True  # A stale stored field cannot inflate calibration accuracy.
    metrics = compute_metrics([row], requested=2, api_failures=1, wall_clock_seconds=1)
    threshold = metrics["calibration"]["thresholds"][0]
    assert threshold["coverage"] == 1
    assert threshold["requested_coverage"] == 0.5
    assert threshold["accuracy"] == 0
    assert sum(b["n"] for b in metrics["calibration"]["reliability_bins"]) == 1
    assert metrics["calibration"]["ece"] == 0.9


def test_cached_latency_not_counted_as_current_run():
    rows = [record("positive", "positive", None, latency=100), record("neutral", "neutral", None, latency=20)]
    rows[0]["cached"] = True
    rows[1]["cached"] = False
    metrics = compute_metrics(rows, requested=2, api_failures=0, wall_clock_seconds=1)
    assert metrics["latency_ms"]["mean"] == 60
    assert metrics["current_run_latency_ms"]["mean"] == 20
    assert metrics["cache"] == {"reused": 1, "fresh": 1, "unknown": 0}


def test_unknown_labels_rejected_and_missing_confidence_not_ranked():
    row = record("positive", "positive", None)
    assert error_analysis([row])["lowest_confidence_correct"] == []
    row["gold"] = "unknown"
    with pytest.raises(ValueError, match="sentiment"):
        compute_metrics([row], requested=1, api_failures=0, wall_clock_seconds=1)


def probs_for(predicted, confidence):
    rest = (1.0 - confidence) / 2
    return {label: (confidence if label == predicted else rest) for label in ("positive", "neutral", "negative")}


def test_risk_coverage_perfect_predictions():
    rows = [
        record("positive", "positive", probs_for("positive", 0.9)),
        record("neutral", "neutral", probs_for("neutral", 0.7)),
        record("negative", "negative", probs_for("negative", 0.6)),
    ]
    calibration = compute_metrics(rows, requested=3, api_failures=0, wall_clock_seconds=1)["calibration"]
    risk_coverage = calibration["risk_coverage"]
    assert risk_coverage["aurc"] == pytest.approx(0.0)
    assert [point["threshold"] for point in risk_coverage["curve"]] == [0.9, 0.7, 0.6]
    assert all(point["risk"] == pytest.approx(0.0) for point in risk_coverage["curve"])
    assert all(point["accuracy"] == pytest.approx(1.0) for point in risk_coverage["curve"])
    assert [point["n"] for point in risk_coverage["curve"]] == [1, 2, 3]
    assert risk_coverage["curve"][-1]["coverage"] == pytest.approx(1.0)
    for entry in risk_coverage["accuracy_at_coverage"]:
        assert entry["accuracy"] == pytest.approx(1.0)
        assert entry["risk"] == pytest.approx(0.0)


def test_risk_coverage_all_wrong_predictions():
    rows = [
        record("positive", "negative", probs_for("negative", 0.9)),
        record("neutral", "positive", probs_for("positive", 0.7)),
        record("negative", "neutral", probs_for("neutral", 0.6)),
    ]
    calibration = compute_metrics(rows, requested=3, api_failures=0, wall_clock_seconds=1)["calibration"]
    risk_coverage = calibration["risk_coverage"]
    assert risk_coverage["aurc"] == pytest.approx(1.0)
    assert all(point["risk"] == pytest.approx(1.0) for point in risk_coverage["curve"])
    assert all(point["accuracy"] == pytest.approx(0.0) for point in risk_coverage["curve"])
    for entry in risk_coverage["accuracy_at_coverage"]:
        assert entry["accuracy"] == pytest.approx(0.0)
        assert entry["risk"] == pytest.approx(1.0)


def test_risk_coverage_hand_calculable_mixed_case():
    rows = [
        record("positive", "positive", probs_for("positive", 0.9)),
        record("positive", "negative", probs_for("negative", 0.8)),
        record("neutral", "neutral", probs_for("neutral", 0.7)),
        record("neutral", "positive", probs_for("positive", 0.6)),
    ]
    calibration = compute_metrics(rows, requested=4, api_failures=0, wall_clock_seconds=1)["calibration"]
    risk_coverage = calibration["risk_coverage"]
    assert [point["threshold"] for point in risk_coverage["curve"]] == [0.9, 0.8, 0.7, 0.6]
    assert [point["risk"] for point in risk_coverage["curve"]] == pytest.approx([0.0, 0.5, 1 / 3, 0.5])
    # Grouped right-step AURC: 0.25*0 + 0.25*0.5 + 0.25*(1/3) + 0.25*0.5.
    assert risk_coverage["aurc"] == pytest.approx(1 / 3)
    at_half, at_high = risk_coverage["accuracy_at_coverage"]
    assert (at_half["target_coverage"], at_half["n"], at_half["threshold"]) == (0.5, 2, 0.8)
    assert at_half["coverage"] == pytest.approx(0.5)
    assert at_half["accuracy"] == pytest.approx(0.5)
    # Target 0.8 is first reached at full coverage; actual coverage disclosed.
    assert at_high["threshold"] == pytest.approx(0.6)
    assert at_high["n"] == 4
    assert at_high["coverage"] == pytest.approx(1.0)
    assert at_high["accuracy"] == pytest.approx(0.5)


def test_risk_coverage_ties_accepted_jointly_and_permutation_invariant():
    rows = [
        record("positive", "positive", probs_for("positive", 0.9)),
        record("positive", "negative", probs_for("negative", 0.9)),
        record("neutral", "neutral", probs_for("neutral", 0.5)),
        record("neutral", "positive", probs_for("positive", 0.5)),
    ]
    first = compute_metrics(rows, requested=4, api_failures=0, wall_clock_seconds=1)["calibration"]["risk_coverage"]
    # Whole tie groups accepted jointly: one endpoint per distinct confidence.
    assert [point["threshold"] for point in first["curve"]] == [0.9, 0.5]
    assert [point["n"] for point in first["curve"]] == [2, 4]
    assert [point["accuracy"] for point in first["curve"]] == pytest.approx([0.5, 0.5])
    assert first["aurc"] == pytest.approx(0.5)
    # Target 0.5 lands exactly on the first tie-group boundary.
    assert first["accuracy_at_coverage"][0]["n"] == 2
    assert first["accuracy_at_coverage"][0]["coverage"] == pytest.approx(0.5)
    # Input order cannot influence results: reversed and rotated orders agree.
    for permuted in (rows[::-1], rows[2:] + rows[:2], [rows[1], rows[3], rows[0], rows[2]]):
        other = compute_metrics(permuted, requested=4, api_failures=0, wall_clock_seconds=1)["calibration"]["risk_coverage"]
        assert other == first


def test_risk_coverage_tie_group_straddling_target_discloses_overshoot():
    rows = [record("positive", "positive", probs_for("positive", 0.9)) for _ in range(3)]
    risk_coverage = compute_metrics(rows, requested=3, api_failures=0, wall_clock_seconds=1)["calibration"]["risk_coverage"]
    assert len(risk_coverage["curve"]) == 1
    assert risk_coverage["aurc"] == pytest.approx(0.0)
    for entry in risk_coverage["accuracy_at_coverage"]:
        assert entry["coverage"] == pytest.approx(1.0)
        assert entry["n"] == 3
        assert entry["threshold"] == pytest.approx(0.9)


def test_risk_coverage_requested_denominator_counts_api_failures():
    rows = [
        record("positive", "positive", probs_for("positive", 0.9)),
        record("positive", "negative", probs_for("negative", 0.8)),
        record("neutral", "neutral", probs_for("neutral", 0.7)),
        record("neutral", "positive", probs_for("positive", 0.6)),
    ]
    risk_coverage = compute_metrics(rows, requested=8, api_failures=4, wall_clock_seconds=1)["calibration"]["risk_coverage"]
    # Successful-coverage denominators ignore failures; requested denominators include them.
    assert [point["coverage"] for point in risk_coverage["curve"]] == pytest.approx([0.25, 0.5, 0.75, 1.0])
    assert [point["requested_coverage"] for point in risk_coverage["curve"]] == pytest.approx([1 / 8, 2 / 8, 3 / 8, 4 / 8])
    # AURC integrates over successful coverage, so failures leave it unchanged.
    assert risk_coverage["aurc"] == pytest.approx(1 / 3)
    at_half, at_high = risk_coverage["accuracy_at_coverage"]
    assert at_half["coverage"] == pytest.approx(0.5)
    assert at_half["requested_coverage"] == pytest.approx(0.25)
    assert at_high["coverage"] == pytest.approx(1.0)
    assert at_high["requested_coverage"] == pytest.approx(0.5)


def test_risk_coverage_missing_probabilities_preserved():
    metrics = compute_metrics(
        [record("positive", "positive", None)],
        requested=1,
        api_failures=0,
        wall_clock_seconds=0.1,
    )
    assert metrics["calibration"] is None
    assert "skipped" in metrics["warning"]
