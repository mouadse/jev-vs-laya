from __future__ import annotations

from collections import Counter, defaultdict
from math import sqrt
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

LABELS = ("positive", "neutral", "negative")
THRESHOLDS = (0.50, 0.70, 0.80, 0.90, 0.95, 0.99)


def wilson_interval(correct: int, n: int) -> dict[str, Any] | None:
    """Two-sided 95% Wilson interval; assumes independent sampled examples."""
    if not 0 <= correct <= n:
        raise ValueError("Wilson counts must satisfy 0 <= correct <= n")
    if n == 0:
        return None
    z = 1.959963984540054
    p = correct / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return {"low": max(0.0, center - half), "high": min(1.0, center + half),
            "level": 0.95, "method": "wilson"}


def expected_calibration_error(
    correct: Sequence[bool], confidence: Sequence[float], bins: int = 10
) -> float:
    if len(correct) == 0:
        raise ValueError("ECE requires at least one prediction")
    correct_array = np.asarray(correct, dtype=float)
    confidence_array = np.asarray(confidence, dtype=float)
    if (not isinstance(bins, int) or bins < 1 or correct_array.ndim != 1
            or confidence_array.shape != correct_array.shape
            or not np.isin(correct_array, [0, 1]).all()
            or not np.isfinite(confidence_array).all()
            or ((confidence_array < 0) | (confidence_array > 1)).any()):
        raise ValueError("ECE requires aligned binary correctness and finite [0, 1] confidence, and positive bins")
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidence_array >= lower) & (
            confidence_array <= upper if index == bins - 1 else confidence_array < upper
        )
        if mask.any():
            result += float(mask.mean()) * abs(
                float(correct_array[mask].mean()) - float(confidence_array[mask].mean())
            )
    return result


def _quality(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    gold = [row["gold"] for row in records]
    predicted = [row["predicted"] for row in records]
    precision, recall, f1, support = precision_recall_fscore_support(
        gold, predicted, labels=LABELS, zero_division=0
    )
    _, _, macro_f1, _ = precision_recall_fscore_support(
        gold, predicted, labels=LABELS, average="macro", zero_division=0
    )
    macro_precision, macro_recall, _, _ = precision_recall_fscore_support(
        gold, predicted, labels=LABELS, average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        gold, predicted, labels=LABELS, average="weighted", zero_division=0
    )
    return {
        "n": len(records),
        "correct": sum(a == b for a, b in zip(gold, predicted)),
        "incorrect": sum(a != b for a, b in zip(gold, predicted)),
        "accuracy": float(accuracy_score(gold, predicted)),
        "accuracy_ci95": wilson_interval(sum(a == b for a, b in zip(gold, predicted)), len(records)),
        "observed_majority_baseline": {
            "label": max(LABELS, key=gold.count),
            "accuracy": max(gold.count(label) for label in LABELS) / len(gold),
            "description": "Descriptive majority of this scored sample; selected post hoc, not a dev-selected baseline.",
        },
        "macro_average_labels": list(LABELS),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_sentiment": {
            label: {
                "support": int(support[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
            }
            for index, label in enumerate(LABELS)
        },
        "confusion_matrix": confusion_matrix(gold, predicted, labels=LABELS).tolist(),
    }


def _group_quality(records: Sequence[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[row[field]].append(row)
    result = {}
    for name, rows in sorted(groups.items()):
        quality = _quality(rows)
        result[name] = {
            "n": len(rows),
            "accuracy": quality["accuracy"],
            "accuracy_ci95": quality["accuracy_ci95"],
            "macro_f1": quality["macro_f1"],
            "sentiment_support": dict(Counter(row["gold"] for row in rows)),
        }
    return result


def _risk_coverage(
    correct: Sequence[bool], confidence: Sequence[float], requested: int
) -> dict[str, Any]:
    """Selective risk-coverage curve with whole tie groups.

    Endpoints are taken at each distinct confidence value, descending, after
    accepting every prediction tied at that value, so input order cannot
    influence the curve. Coverage divides by successful predictions; requested
    coverage divides by all requested examples (including API failures). AURC
    uses grouped right-step integration (sum of delta successful coverage
    times endpoint risk), not conventional per-rank integration with arbitrary
    within-tie ordering.
    """
    total = len(correct)
    groups: dict[float, list[int]] = defaultdict(list)
    for index, value in enumerate(confidence):
        groups[float(value)].append(index)
    curve: list[dict[str, Any]] = []
    accepted = 0
    accepted_correct = 0
    previous_coverage = 0.0
    aurc = 0.0
    for threshold in sorted(groups, reverse=True):
        members = groups[threshold]
        accepted += len(members)
        accepted_correct += sum(1 for index in members if correct[index])
        coverage = accepted / total
        accuracy = accepted_correct / accepted
        risk = 1.0 - accuracy
        aurc += (coverage - previous_coverage) * risk
        previous_coverage = coverage
        curve.append(
            {
                "threshold": threshold,
                "n": accepted,
                "coverage": coverage,
                "requested_coverage": accepted / requested if requested else 0.0,
                "accuracy": accuracy,
                "risk": risk,
            }
        )
    accuracy_at_coverage = []
    for target in (0.5, 0.8):
        endpoint = next(item for item in curve if item["coverage"] >= target)
        accuracy_at_coverage.append(
            {
                "target_coverage": target,
                "n": endpoint["n"],
                "coverage": endpoint["coverage"],
                "requested_coverage": endpoint["requested_coverage"],
                "accuracy": endpoint["accuracy"],
                "risk": endpoint["risk"],
                "threshold": endpoint["threshold"],
            }
        )
    return {"aurc": aurc, "curve": curve, "accuracy_at_coverage": accuracy_at_coverage}


def _calibration(records: Sequence[dict[str, Any]], requested: int) -> tuple[dict[str, Any] | None, str | None]:
    for row in records:
        probabilities = row.get("probabilities")
        if probabilities is not None:
            if not isinstance(probabilities, dict) or set(probabilities) != set(LABELS):
                raise ValueError("Supplied probabilities must contain exactly the three sentiment classes")
            values = np.asarray(list(probabilities.values()), dtype=float)
            if (not np.isfinite(values).all() or ((values < 0) | (values > 1)).any()
                    or not np.isclose(values.sum(), 1.0, atol=1e-3, rtol=0)):
                raise ValueError("Class probabilities must be finite, between 0 and 1, and sum to 1")
    if not records or any(
        not isinstance(row.get("probabilities"), dict)
        or set(row["probabilities"]) != set(LABELS)
        for row in records
    ):
        return None, "Probability-dependent metrics skipped: complete class probabilities unavailable."
    matrix = np.asarray(
        [[float(row["probabilities"][label]) for label in LABELS] for row in records]
    )
    gold = [row["gold"] for row in records]
    confidence = [float(row["probabilities"][row["predicted"]]) for row in records]
    correct = [row["gold"] == row["predicted"] for row in records]
    one_hot = np.asarray([[float(label == target) for label in LABELS] for target in gold])
    clipped = np.clip(matrix, 1e-15, 1.0)
    thresholds = []
    for threshold in THRESHOLDS:
        chosen = [index for index, value in enumerate(confidence) if value >= threshold]
        thresholds.append(
            {
                "threshold": threshold,
                "n": len(chosen),
                "coverage": len(chosen) / len(records),
                "requested_coverage": len(chosen) / requested,
                "accuracy": (
                    sum(correct[index] for index in chosen) / len(chosen) if chosen else None
                ),
                "errors": sum(not correct[index] for index in chosen),
                "accuracy_ci95": wilson_interval(sum(correct[index] for index in chosen), len(chosen)),
            }
        )
    correct_confidence = [value for value, ok in zip(confidence, correct) if ok]
    wrong_confidence = [value for value, ok in zip(confidence, correct) if not ok]
    return {
        "n": len(records),
        "average_confidence": float(np.mean(confidence)),
        "confidence_correct": float(np.mean(correct_confidence)) if correct_confidence else None,
        "confidence_incorrect": float(np.mean(wrong_confidence)) if wrong_confidence else None,
        "log_loss": float(-np.mean(np.sum(one_hot * np.log(clipped), axis=1))),
        "brier_score": float(np.mean(np.sum((matrix - one_hot) ** 2, axis=1))),
        "ece": expected_calibration_error(correct, confidence),
        "ece_bins": 10,
        "reliability_bins": [
            {"lower": i / 10, "upper": (i + 1) / 10,
             "n": len(selected),
             "accuracy": float(np.mean([correct[j] for j in selected])) if selected else None,
             "confidence": float(np.mean([confidence[j] for j in selected])) if selected else None}
            for i in range(10)
            for selected in [[j for j, value in enumerate(confidence)
                              if i / 10 <= value and (value < (i + 1) / 10 or i == 9)]]
        ],
        "definitions": {
            "brier_score": "Mean sum of squared class-probability errors; multiclass range [0, 2].",
            "ece": "Predicted-class confidence ECE with 10 equal-width bins; descriptive and sample-size sensitive.",
            "coverage": "Accepted predictions / successful predictions.",
            "requested_coverage": "Accepted predictions / all requested examples, including API failures.",
            "log_loss": "Mean negative natural log probability of gold class, clipped at 1e-15.",
            "risk": "Selective error rate among accepted predictions (1 - accuracy) at a risk-coverage endpoint.",
            "aurc": "Grouped right-step area under the risk-coverage curve: endpoints at each distinct confidence with whole tie groups accepted jointly (input order cannot change the curve); AURC = sum(delta successful coverage * endpoint risk), not conventional per-rank integration with arbitrary within-tie ordering.",
            "accuracy_at_coverage": "First risk-coverage endpoint with successful coverage >= target; actual coverage and threshold disclosed and may exceed the target when a tie group straddles it.",
            "risk_coverage_scope": "Risk-coverage endpoints condition on successful predictions only; coverage divides by successful predictions while requested_coverage divides by all requested examples, including API failures.",
        },
        "thresholds": thresholds,
        "risk_coverage": _risk_coverage(correct, confidence, requested),
    }, None


def compute_metrics(
    records: Sequence[dict[str, Any]],
    *,
    requested: int,
    api_failures: int,
    wall_clock_seconds: float,
    topic_min_n: int = 10,
) -> dict[str, Any]:
    if requested < len(records) + api_failures or api_failures < 0 or requested < 0:
        raise ValueError("Requested count must cover successful predictions and API failures")
    if not np.isfinite(wall_clock_seconds) or wall_clock_seconds < 0:
        raise ValueError("Wall-clock time must be finite and nonnegative")
    for row in records:
        if row["gold"] not in LABELS or row["predicted"] not in LABELS:
            raise ValueError("Unknown gold or predicted sentiment label")
        if not np.isfinite(row["latency_ms"]) or row["latency_ms"] < 0:
            raise ValueError("Prediction latency must be finite and nonnegative")
    if not records:
        return {
            "requested": requested,
            "successful": 0,
            "api_failures": api_failures,
            "wall_clock_seconds": wall_clock_seconds,
            "quality": None,
            "warning": "No successful predictions; quality metrics unavailable.",
        }
    quality = _quality(records)
    by_style = _group_quality(records, "writing_style")
    arabic_accuracy = by_style.get("Arabic", {}).get("accuracy")
    arabizi_accuracy = by_style.get("Arabizi", {}).get("accuracy")
    style_delta = (
        arabizi_accuracy - arabic_accuracy
        if arabic_accuracy is not None and arabizi_accuracy is not None
        else None
    )
    topic_counts = Counter(row["topic"] for row in records)
    by_topic = {
        topic: values
        for topic, values in _group_quality(records, "topic").items()
        if topic_counts[topic] >= topic_min_n
    }
    latencies = np.asarray([row["latency_ms"] for row in records], dtype=float)
    current_latencies = np.asarray([row["latency_ms"] for row in records if row.get("cached") is False], dtype=float)
    calibration, warning = _calibration(records, requested)
    return {
        "requested": requested,
        "successful": len(records),
        "api_failures": api_failures,
        "completion_rate": len(records) / requested,
        "unresolved": requested - len(records) - api_failures,
        "statistical_notes": [
            "95% Wilson accuracy intervals assume independent sampled examples; they do not capture label uncertainty or dataset shift.",
            "Macro scores average all three fixed sentiment classes, assigning zero to undefined class scores.",
            "Quality metrics condition on successful predictions; API failures may introduce selection bias.",
        ],
        "quality": quality,
        "writing_style": by_style,
        "arabizi_minus_arabic_accuracy": style_delta,
        "topic_min_n": topic_min_n,
        "topics": by_topic,
        "latency_ms": {
            "mean": float(np.mean(latencies)),
            "p50": float(np.percentile(latencies, 50)),
            "p90": float(np.percentile(latencies, 90)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99)),
        },
        "latency_scope": "original_prediction_observations_including_cache",
        "current_run_latency_ms": ({
            "n": len(current_latencies),
            "mean": float(np.mean(current_latencies)),
            **{f"p{p}": float(np.percentile(current_latencies, p)) for p in (50, 90, 95, 99)},
        } if len(current_latencies) else None),
        "cache": {
            "reused": sum(row.get("cached") is True for row in records),
            "fresh": sum(row.get("cached") is False for row in records),
            "unknown": sum("cached" not in row for row in records),
        },
        "wall_clock_seconds": wall_clock_seconds,
        "calibration": calibration,
        "warning": warning,
    }


def error_analysis(records: Iterable[dict[str, Any]], limit: int = 5) -> dict[str, Any]:
    rows = list(records)
    errors = [row for row in rows if row["gold"] != row["predicted"]]
    correct = [row for row in rows if row["gold"] == row["predicted"]]
    confidence = lambda row: row.get("confidence") if row.get("confidence") is not None else -1.0
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in errors:
        grouped[f'{row["gold"]} -> {row["predicted"]}'].append(row)
    return {
        "highest_confidence_errors": sorted((row for row in errors if row.get("confidence") is not None), key=confidence, reverse=True)[:limit],
        "lowest_confidence_correct": sorted((row for row in correct if row.get("confidence") is not None), key=confidence)[:limit],
        "errors_by_transition": {key: value for key, value in sorted(grouped.items())},
        "arabizi_errors": [row for row in errors if row["writing_style"] == "Arabizi"],
        "arabic_errors": [row for row in errors if row["writing_style"] == "Arabic"],
    }
