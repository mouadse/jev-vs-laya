from __future__ import annotations

import json

import pytest

from darija_eval.compare import compare_runs
from darija_eval.artifacts import reanalyse_run
from darija_eval.dataset import Example, dataset_fingerprint


def _write_run(path, backend, predictions):
    path.mkdir()
    accuracy = sum(row["correct"] for row in predictions) / len(predictions)
    metrics = {
        "backend": backend,
        "split": "eval",
        "requested_model": backend + "-requested",
        "resolved_models": [backend + "-resolved"],
        "successful": len(predictions),
        "requested": len(predictions),
        "api_failures": 0,
        "quality": {"accuracy": accuracy, "macro_f1": accuracy},
        "writing_style": {
            "Arabic": {"accuracy": accuracy},
            "Arabizi": {"accuracy": accuracy},
        },
        "arabizi_minus_arabic_accuracy": 0.0,
        "calibration": {
            "ece": 0.1,
            "brier_score": 0.2,
            "log_loss": 0.3,
            "thresholds": [{"threshold": 0.9, "coverage": 0.5, "accuracy": 1.0, "errors": 0}],
        },
        "latency_ms": {"mean": 10, "p50": 9, "p90": 12, "p95": 13, "p99": 15},
    }
    (path / "metrics.json").write_text(json.dumps(metrics))
    (path / "predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in predictions)
    )


def _row(identifier, predicted, backend):
    return {
        "id": identifier,
        "split": "eval",
        "text": f"review {identifier}",
        "gold": "positive",
        "predicted": predicted,
        "correct": predicted == "positive",
        "confidence": 0.8,
        "probabilities": {label: 0.8 if label == predicted else 0.1 for label in ("positive", "neutral", "negative")},
        "latency_ms": 10.0,
        "writing_style": "Arabic" if identifier == 1 else "Arabizi",
        "topic": "it",
        "schema_version": "v1",
        "raw": {"inference_ms": 5} if backend == "laya" else {},
    }


def test_comparison_counts_paired_outcomes(tmp_path) -> None:
    jev = tmp_path / "jev"
    laya = tmp_path / "laya"
    _write_run(jev, "jev", [_row(1, "positive", "jev"), _row(2, "negative", "jev")])
    _write_run(laya, "laya", [_row(1, "negative", "laya"), _row(2, "positive", "laya")])
    output = compare_runs(jev, laya, tmp_path / "results", split_path=None)
    comparison = json.loads((output / "comparison.json").read_text())
    assert comparison["outcomes"]["jev_only"] == 1
    assert comparison["outcomes"]["laya_only"] == 1
    assert (output / "report.html").exists()


def test_comparison_rejects_mismatched_examples(tmp_path) -> None:
    jev = tmp_path / "jev"
    laya = tmp_path / "laya"
    _write_run(jev, "jev", [_row(1, "positive", "jev")])
    _write_run(laya, "laya", [_row(2, "positive", "laya")])
    with pytest.raises(ValueError, match="same successful IDs"):
        compare_runs(jev, laya, tmp_path / "results", split_path=None)


def test_comparison_rejects_api_failures(tmp_path) -> None:
    jev = tmp_path / "jev"
    laya = tmp_path / "laya"
    _write_run(jev, "jev", [_row(1, "positive", "jev")])
    _write_run(laya, "laya", [_row(1, "positive", "laya")])
    metrics_path = laya / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["api_failures"] = 1
    metrics_path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="missing predictions"):
        compare_runs(jev, laya, tmp_path / "results", split_path=None)


def test_comparison_recomputes_metrics_and_includes_identical_errors(tmp_path):
    jev, laya = tmp_path / "jev", tmp_path / "laya"
    for name, path in (("jev", jev), ("laya", laya)):
        _write_run(path, name, [_row(1, "negative", name)])
        metrics = json.loads((path / "metrics.json").read_text())
        metrics["quality"]["accuracy"] = 1.0
        (path / "metrics.json").write_text(json.dumps(metrics))
    output = compare_runs(jev, laya, tmp_path / "results", split_path=None)
    data = json.loads((output / "comparison.json").read_text())
    assert data["models"]["jev"]["accuracy"] == 0
    assert data["both_wrong_same_label"] == 1
    assert 'data-outcome="neither"' in (output / "report.html").read_text()


def test_comparison_handles_missing_probabilities_and_style(tmp_path):
    jev, laya = tmp_path / "jev", tmp_path / "laya"
    for name, path in (("jev", jev), ("laya", laya)):
        row = _row(1, "negative", name)
        row.update(probabilities=None, confidence=None)
        _write_run(path, name, [row])
    output = compare_runs(jev, laya, tmp_path / "results", split_path=None)
    data = json.loads((output / "comparison.json").read_text())
    assert data["delta_laya_minus_jev"]["ece"] is None
    assert data["models"]["jev"]["arabizi_accuracy"] is None
    assert "calibration comparison skipped" in (output / "report.html").read_text()


def test_comparison_labels_subset_and_rejects_non_eval_ids(tmp_path):
    jev, laya = tmp_path / "jev", tmp_path / "laya"
    for name, path in (("jev", jev), ("laya", laya)):
        _write_run(path, name, [_row(1, "positive", name)])
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"eval": [1, 2], "dataset_fingerprint": "test"}))
    output = compare_runs(jev, laya, tmp_path / "results", split_path=split)
    assert json.loads((output / "comparison.json").read_text())["scope"]["status"] == "subset"
    split.write_text(json.dumps({"eval": [2], "dataset_fingerprint": "test"}))
    with pytest.raises(ValueError, match="outside the frozen"):
        compare_runs(jev, laya, tmp_path / "results", split_path=split)


def test_reanalysis_preserves_source_evidence(tmp_path):
    source = tmp_path / "jev"
    _write_run(source, "jev", [_row(1, "negative", "jev")])
    original_predictions = (source / "predictions.jsonl").read_bytes()
    original_metrics = (source / "metrics.json").read_bytes()
    output = reanalyse_run(source, tmp_path / "results")
    assert (source / "metrics.json").read_bytes() == original_metrics
    assert (output / "predictions.jsonl").read_bytes() == original_predictions
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["analysis_provenance"]["inference_rerun"] is False
    assert metrics["quality"]["accuracy"] == 0
    assert (output / "report.html").exists()


def test_reanalysis_rejects_corrupt_correctness(tmp_path):
    source = tmp_path / "jev"
    row = _row(1, "negative", "jev")
    row["correct"] = True
    _write_run(source, "jev", [row])
    with pytest.raises(ValueError, match="correctness flag"):
        reanalyse_run(source, tmp_path / "results")


def _add_provenance(path, rows):
    metrics_path = path / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["provenance"] = {
        "selected_ids": [row["id"] for row in rows],
        "selected_size": len(rows),
        "selection_fingerprint": dataset_fingerprint([
            Example(row["id"], row["text"], row["gold"], row["writing_style"], row["topic"])
            for row in rows
        ]),
        "schema_fingerprint": "question-content-hash",
    }
    metrics_path.write_text(json.dumps(metrics))
    (path / "predictions.jsonl").write_text("".join(
        json.dumps({**row, "schema_fingerprint": "question-content-hash"}) + "\n"
        for row in rows
    ))
    return metrics


@pytest.mark.parametrize(("field", "value", "error"), [
    ("text", "a different review", "selection fingerprint"),
    ("id", 99, "selected IDs"),
    ("schema_fingerprint", "different-question", "schema fingerprint"),
])
def test_reanalysis_rejects_evidence_inconsistent_with_provenance(tmp_path, field, value, error):
    source = tmp_path / "jev"
    rows = [_row(1, "positive", "jev")]
    _write_run(source, "jev", rows)
    _add_provenance(source, rows)
    path = source / "predictions.jsonl"
    row = json.loads(path.read_text())
    row[field] = value
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match=error):
        reanalyse_run(source, tmp_path / "results")


def test_reanalysis_checks_manifest_and_preserves_it(tmp_path):
    source = tmp_path / "jev"
    rows = [_row(2, "positive", "jev"), _row(1, "negative", "jev")]
    _write_run(source, "jev", rows)
    metrics = _add_provenance(source, rows)
    manifest = {key: metrics[key] for key in ("backend", "requested_model", "split", "provenance")}
    manifest_path = source / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    # Completion order is irrelevant; fingerprints use the original selected-ID order.
    path = source / "predictions.jsonl"
    path.write_text("\n".join(reversed(path.read_text().splitlines())) + "\n")
    output = reanalyse_run(source, tmp_path / "results")
    assert (output / "manifest.json").read_bytes() == manifest_path.read_bytes()
    assert json.loads((output / "metrics.json").read_text())["quality"]["accuracy"] == 0.5
    manifest["split"] = "dev"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest split"):
        reanalyse_run(source, tmp_path / "results")


def test_reanalysis_validates_failed_examples_in_selection(tmp_path):
    source = tmp_path / "jev"
    rows = [_row(1, "positive", "jev"), _row(2, "negative", "jev")]
    _write_run(source, "jev", rows)
    metrics = _add_provenance(source, rows)
    metrics.update(successful=1, api_failures=1)
    (source / "metrics.json").write_text(json.dumps(metrics))
    path = source / "predictions.jsonl"
    path.write_text(path.read_text().splitlines()[0] + "\n")
    failure = {key: rows[1][key] for key in ("id", "split", "text", "gold", "writing_style", "topic")}
    failures_path = source / "api_failures.jsonl"
    failures_path.write_text(json.dumps(failure) + "\n")
    output = reanalyse_run(source, tmp_path / "results")
    assert json.loads((output / "metrics.json").read_text())["completion_rate"] == 0.5
    failure["text"] = "not the selected review"
    failures_path.write_text(json.dumps(failure) + "\n")
    with pytest.raises(ValueError, match="selection fingerprint"):
        reanalyse_run(source, tmp_path / "results")


def test_comparison_rejects_different_dataset_provenance(tmp_path):
    jev, laya = tmp_path / "jev", tmp_path / "laya"
    for name, path in (("jev", jev), ("laya", laya)):
        rows = [_row(1, "positive", name)]
        _write_run(path, name, rows)
        metrics = _add_provenance(path, rows)
        metrics["provenance"]["dataset_fingerprint"] = name
        (path / "metrics.json").write_text(json.dumps(metrics))
    with pytest.raises(ValueError, match="provenance mismatch for dataset_fingerprint"):
        compare_runs(jev, laya, tmp_path / "results", split_path=None)


def test_comparison_rejects_wrong_frozen_dataset_with_matching_ids(tmp_path):
    jev, laya = tmp_path / "jev", tmp_path / "laya"
    for name, path in (("jev", jev), ("laya", laya)):
        rows = [_row(1, "positive", name)]
        _write_run(path, name, rows)
        metrics = _add_provenance(path, rows)
        metrics["provenance"]["dataset_fingerprint"] = "recorded-dataset"
        (path / "metrics.json").write_text(json.dumps(metrics))
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"eval": [1], "dataset_fingerprint": "different-dataset"}))
    with pytest.raises(ValueError, match="dataset_fingerprint disagrees"):
        compare_runs(jev, laya, tmp_path / "results", split_path=split)
