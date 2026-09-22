from __future__ import annotations

from darija_eval.metrics import compute_metrics
from darija_eval.report import html_report


def test_report_contains_readable_sections_and_escapes_reviews() -> None:
    rows = [
        {
            "id": 7,
            "gold": "neutral",
            "predicted": "positive",
            "correct": False,
            "probabilities": {"positive": 0.8, "neutral": 0.1, "negative": 0.1},
            "confidence": 0.8,
            "writing_style": "Arabizi",
            "topic": "it",
            "latency_ms": 31.0,
            "text": '<script>alert("no")</script>',
        },
        {
            "id": 8,
            "gold": "positive",
            "predicted": "positive",
            "correct": True,
            "probabilities": {"positive": 0.9, "neutral": 0.05, "negative": 0.05},
            "confidence": 0.9,
            "writing_style": "Arabic",
            "topic": "it",
            "latency_ms": 20.0,
            "text": "زوين",
        },
    ]
    metrics = compute_metrics(
        rows, requested=2, api_failures=0, wall_clock_seconds=0.5, topic_min_n=1
    )
    metrics.update(
        {
            "backend": "jev",
            "requested_model": "jev-latest",
            "resolved_models": ["jev-1.13.0"],
            "schema_version": "v1",
            "split": "eval",
        }
    )
    output = html_report(metrics, rows)
    assert "Quality at a glance" in output
    assert "Accuracy by writing style" in output
    assert "not a causal effect" in output
    assert "Confusion matrix" in output
    assert "Read the misses" in output
    assert "&lt;script&gt;alert(&quot;no&quot;)&lt;/script&gt;" in output
    assert '<script>alert("no")</script>' not in output


def test_evaluation_writes_html_report(tmp_path) -> None:
    from darija_eval.backends.base import Prediction
    from darija_eval.cache import PredictionCache
    from darija_eval.dataset import Example
    from darija_eval.evaluate import evaluate_examples

    class Backend:
        name = "jev"
        model_identifier = "jev-latest"
        schema_version = "v1"

        def predict(self, text):
            return Prediction(
                "positive",
                {"positive": 0.9, "neutral": 0.05, "negative": 0.05},
                12.0,
                "jev-1.13.0",
            )

    run_dir, *_ = evaluate_examples(
        [Example(1, "mzyan", "positive", "Arabizi", "it")],
        split_name="eval",
        backend=Backend(),
        results_root=tmp_path / "results",
        cache=PredictionCache(tmp_path / "cache"),
    )
    report = run_dir / "report.html"
    assert report.exists()
    assert "Can a decision model" in report.read_text(encoding="utf-8")


def test_empty_dev_report_does_not_claim_held_out_results() -> None:
    output = html_report({"quality": None, "split": "dev", "api_failures": 3, "requested": 3}, [])
    assert "No successful predictions" in output
    assert "not held-out evaluation evidence" in output
    assert "api_failures.jsonl" in output


def test_report_explains_uncertainty_reference_and_cached_latency() -> None:
    rows = [{
        "id": 1, "gold": "positive", "predicted": "positive", "correct": True,
        "probabilities": {"positive": 0.9, "neutral": 0.05, "negative": 0.05},
        "confidence": 0.9, "writing_style": "Arabic", "topic": "it",
        "latency_ms": 20.0, "text": "زوين", "cached": True,
    }]
    metrics = compute_metrics(rows, requested=2, api_failures=1, wall_clock_seconds=0.1)
    metrics.update(backend="laya", split="dev", requested_model="laya")
    metrics["quality"]["accuracy_ci95"] = {"low": 0.2, "high": 1.0}
    metrics["quality"]["observed_majority_baseline"] = {"label": "positive", "accuracy": 1.0}
    output = html_report(metrics, rows)
    assert "20.0%–100.0%" in output
    assert "post-hoc reference" in output
    assert "not held-out evaluation evidence" in output
    assert "1 reused predictions" in output
    assert "Historical request latency" in output
    assert "have not been independently adjudicated" in output
    assert "Accepted N" in output
    assert "Phase 01" not in output
    assert "frozen 20% sample" not in output
    assert "API failures</a>" in output


def _calibration_rows() -> list[dict]:
    return [
        {
            "id": 1, "gold": "positive", "predicted": "positive", "correct": True,
            "probabilities": {"positive": 0.9, "neutral": 0.05, "negative": 0.05},
            "confidence": 0.9, "writing_style": "Arabic", "topic": "it",
            "latency_ms": 20.0, "text": "زوين",
        },
        {
            "id": 2, "gold": "negative", "predicted": "positive", "correct": False,
            "probabilities": {"positive": 0.6, "neutral": 0.2, "negative": 0.2},
            "confidence": 0.6, "writing_style": "Arabizi", "topic": "it",
            "latency_ms": 25.0, "text": "ماشي زوين",
        },
    ]


def test_report_renders_without_new_probability_metrics() -> None:
    import copy

    rows = _calibration_rows()
    metrics = compute_metrics(rows, requested=2, api_failures=0, wall_clock_seconds=0.5, topic_min_n=1)
    metrics.update(backend="jev", split="dev", requested_model="jev")
    old = copy.deepcopy(metrics)
    old["calibration"].pop("risk_coverage", None)
    old["calibration"].pop("reliability_bins", None)
    output = html_report(old, rows)
    assert "<svg" not in output
    assert "Quality at a glance" in output


def test_report_renders_without_any_probabilities() -> None:
    rows = _calibration_rows()
    for row in rows:
        row.pop("probabilities")
        row["confidence"] = None
    metrics = compute_metrics(rows, requested=2, api_failures=0, wall_clock_seconds=0.5, topic_min_n=1)
    metrics.update(backend="jev", split="dev", requested_model="jev")
    assert metrics["calibration"] is None
    output = html_report(metrics, rows)
    assert "<svg" not in output
    assert "Quality at a glance" in output


def test_report_escapes_provenance_and_handles_empty_bins() -> None:
    rows = _calibration_rows()
    metrics = compute_metrics(rows, requested=2, api_failures=0, wall_clock_seconds=0.5, topic_min_n=1)
    metrics.update(backend="jev", split="dev", requested_model="jev")
    metrics["provenance"] = {
        "dataset_fingerprint": '"><script>alert(1)</script>',
        "reference_audit": {
            "reference_labels": {"source": '<script>alert("x")</script>'},
            "overall": {"n": 851},
            "duplicates": {
                "matching_rule": '<img src=x onerror=alert(1)>',
                "group_count": 5, "rows_in_groups": 10,
                "conflicting_label_group_count": 0, "cross_split_group_count": 0,
            },
        },
    }
    for entry in metrics["calibration"]["reliability_bins"][:-1]:
        entry["accuracy"] = None
        entry["confidence"] = None
        entry["n"] = 0
    output = html_report(metrics, rows)
    assert "<svg" in output
    assert "—" in output
    assert '<script>alert' not in output
    assert "<img src=x" not in output
    assert "&lt;script&gt;" in output


def test_report_charts_are_standalone() -> None:
    rows = _calibration_rows()
    metrics = compute_metrics(rows, requested=2, api_failures=0, wall_clock_seconds=0.5, topic_min_n=1)
    metrics.update(backend="jev", split="dev", requested_model="jev")
    output = html_report(metrics, rows)
    assert output.count("<svg") >= 2
    assert 'role="img"' in output
    assert 'src="http' not in output
    assert 'href="http' not in output
