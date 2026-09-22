from __future__ import annotations

from darija_eval.backends.base import Prediction
from darija_eval.cache import PredictionCache
from darija_eval.dataset import Example
from darija_eval.evaluate import evaluate_examples
from darija_eval.schema import configuration_fingerprint


class CountingBackend:
    name = "counting"
    model_identifier = "model-a"
    schema_version = "v1"

    def __init__(self):
        self.calls = 0

    def predict(self, text: str) -> Prediction:
        self.calls += 1
        return Prediction(
            "positive",
            {"positive": 0.8, "neutral": 0.1, "negative": 0.1},
            5.0,
            "model-a-1",
        )


def test_cache_key_invalidates_on_schema_model_or_text() -> None:
    key = PredictionCache.key("jev", "model", "v1", "hello")
    assert key != PredictionCache.key("jev", "model", "v2", "hello")
    assert key != PredictionCache.key("jev", "other", "v1", "hello")
    assert key != PredictionCache.key("jev", "model", "v1", "different")


def test_second_run_resumes_from_cache_without_backend_calls(tmp_path) -> None:
    rows = [Example(1, "mzyan", "positive", "Arabizi", "it")]
    cache = PredictionCache(tmp_path / "cache")
    backend = CountingBackend()
    first = evaluate_examples(
        rows, split_name="eval", backend=backend, results_root=tmp_path / "results", cache=cache
    )
    second = evaluate_examples(
        rows, split_name="eval", backend=backend, results_root=tmp_path / "results", cache=cache
    )
    assert backend.calls == 1
    assert first[1][0]["cached"] is False
    assert second[1][0]["cached"] is True


def test_cache_invalidates_when_schema_content_changes(tmp_path):
    rows = [Example(1, "mzyan", "positive", "Arabizi", "it")]
    backend = CountingBackend()
    backend.schema_fingerprint = "first"
    options = dict(split_name="dev", backend=backend, results_root=tmp_path / "results", cache=PredictionCache(tmp_path / "cache"))
    evaluate_examples(rows, **options)
    backend.schema_fingerprint = "second"
    evaluate_examples(rows, **options)
    assert backend.calls == 2


def test_duplicate_ids_rejected_before_predictions(tmp_path):
    import pytest
    backend = CountingBackend()
    row = Example(1, "mzyan", "positive", "Arabizi", "it")
    with pytest.raises(ValueError, match="unique"):
        evaluate_examples([row, row], split_name="dev", backend=backend, results_root=tmp_path)
    assert backend.calls == 0


def test_duplicate_review_uses_one_api_call(tmp_path):
    import time
    class SlowBackend(CountingBackend):
        def predict(self, text):
            time.sleep(0.02)
            return super().predict(text)
    backend = SlowBackend()
    rows = [Example(i, "same review", "positive", "Arabic", "it") for i in range(8)]
    result = evaluate_examples(rows, split_name="dev", backend=backend, concurrency=5, results_root=tmp_path / "results", cache=PredictionCache(tmp_path / "cache"))
    assert backend.calls == 1
    assert len(result[1]) == 8
    assert sum(row["cached"] for row in result[1]) == 7


def _cache_key_for(backend, text: str) -> str:
    return PredictionCache.key(
        backend.name,
        backend.model_identifier,
        backend.schema_version,
        text,
        getattr(backend, "schema_fingerprint", None),
        configuration_fingerprint({
            "identity_version": 2,
            "backend": backend.name,
            "requested_model": backend.model_identifier,
            "schema_version": backend.schema_version,
            "schema_fingerprint": getattr(backend, "schema_fingerprint", None),
            "option_order": getattr(backend, "option_order", None),
            "experiment": {},
        }),
    )


def test_corrupt_cache_file_is_treated_as_miss(tmp_path) -> None:
    rows = [Example(1, "mzyan", "positive", "Arabizi", "it")]
    cache = PredictionCache(tmp_path / "cache")
    backend = CountingBackend()
    path = cache._path(_cache_key_for(backend, "mzyan"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{truncated json", encoding="utf-8")
    assert cache.get(_cache_key_for(backend, "mzyan")) is None
    _, predictions, failures, _ = evaluate_examples(
        rows, split_name="eval", backend=backend, results_root=tmp_path / "results", cache=cache
    )
    assert backend.calls == 1
    assert failures == []
    assert len(predictions) == 1 and predictions[0]["cached"] is False
    assert predictions[0]["predicted"] == "positive"


def test_invalid_cached_prediction_is_refetched_not_scored(tmp_path) -> None:
    rows = [Example(1, "mzyan", "positive", "Arabizi", "it")]
    cache = PredictionCache(tmp_path / "cache")
    backend = CountingBackend()
    cache.put(
        _cache_key_for(backend, "mzyan"),
        Prediction("bogus", {"positive": 1.0}, 5.0, "model-a-1"),
    )
    _, predictions, failures, _ = evaluate_examples(
        rows, split_name="eval", backend=backend, results_root=tmp_path / "results", cache=cache
    )
    assert backend.calls == 1
    assert failures == []
    assert len(predictions) == 1
    assert predictions[0]["cached"] is False
    assert predictions[0]["predicted"] == "positive"


def test_invalid_fresh_prediction_goes_to_failures_and_is_not_cached(tmp_path) -> None:
    class BadBackend(CountingBackend):
        def predict(self, text: str) -> Prediction:
            self.calls += 1
            return Prediction("bogus", {"positive": 1.0}, 5.0, "model-a-1")

    rows = [Example(1, "mzyan", "positive", "Arabizi", "it")]
    cache = PredictionCache(tmp_path / "cache")
    backend = BadBackend()
    _, predictions, failures, _ = evaluate_examples(
        rows, split_name="eval", backend=backend, results_root=tmp_path / "results", cache=cache
    )
    assert predictions == []
    assert len(failures) == 1
    assert "invalid prediction" in failures[0]["error"]
    assert list(cache.root.rglob("*.json")) == []


def test_predictions_without_probabilities_are_scored_and_reused(tmp_path) -> None:
    class LabelOnlyBackend(CountingBackend):
        def predict(self, text):
            self.calls += 1
            return Prediction("positive", None, 5.0, "model-a-1")

    cache = PredictionCache(tmp_path / "cache")
    backend = LabelOnlyBackend()
    rows = [Example(1, "mzyan", "positive", "Arabizi", "it")]
    options = dict(split_name="dev", backend=backend, results_root=tmp_path / "results", cache=cache)
    first = evaluate_examples(rows, **options)
    second = evaluate_examples(rows, **options)
    assert backend.calls == 1
    assert first[3]["quality"]["accuracy"] == second[3]["quality"]["accuracy"] == 1.0
    assert first[3]["calibration"] is None
    assert second[1][0]["cached"] is True
