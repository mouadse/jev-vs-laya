from itertools import permutations
import hashlib
import json
from types import SimpleNamespace

import pytest

from darija_eval.backends.jev import JevBackend
from darija_eval.backends.laya import LayaBackend
from darija_eval.cache import PredictionCache
from darija_eval.artifacts import load_run
from darija_eval.dataset import Example
from darija_eval.evaluate import evaluate_examples
from darija_eval.schema import SENTIMENT_CRITERIA


class Client:
    calls = []

    def __init__(self, **kwargs):
        pass

    def system_one(self, *, state, questions):
        order = list(questions["sentiment"].criteria)
        self.calls.append(order)
        probabilities = dict(zip(order, (0.7, 0.2, 0.1)))
        return SimpleNamespace(
            model="jev-pinned",
            answers={"sentiment": SimpleNamespace(
                choice=order[0], probabilities=probabilities, confidence=0.5,
            )},
            model_dump=lambda **kwargs: {},
        )


def test_schema_hash_distinguishes_existing_criteria_permutations(monkeypatch):
    from darija_eval import schema

    backend = LayaBackend(endpoint_url="https://example.invalid")
    original = backend.schema_fingerprint
    monkeypatch.setattr(schema, "SENTIMENT_CRITERIA", dict(reversed(list(SENTIMENT_CRITERIA.items()))))
    assert LayaBackend(endpoint_url="https://example.invalid").schema_fingerprint != original


def test_permutations_isolate_cache_and_keep_semantic_probabilities(tmp_path):
    Client.calls = []
    rows = [Example(1, "mzyan", "positive", "Arabizi", "it")]
    cache = PredictionCache(tmp_path / "cache")
    identities = set()
    for order in permutations(SENTIMENT_CRITERIA):
        backend = JevBackend(api_key="test", client_factory=Client, option_order=order)
        for repeat in (False, True):
            _, records, failures, _ = evaluate_examples(
                rows, split_name="dev", backend=backend, cache=cache,
                results_root=tmp_path / "results",
            )
            assert not failures
            assert records[0]["cached"] is repeat
            assert records[0]["predicted"] == order[0]
            assert records[0]["probabilities"] == dict(zip(order, (0.7, 0.2, 0.1)))
            identities.add(records[0]["experiment_fingerprint"])
    assert len(identities) == 6
    assert Client.calls == [list(order) for order in permutations(SENTIMENT_CRITERIA)]


@pytest.mark.parametrize("field,value", [
    ("checkpoint_revision", "new-revision"),
    ("preprocessing", "normalizer-v2"),
    ("examples", ["example-2", "example-1"]),
    ("retrieval_corpus", "corpus-sha256"),
    ("calibration", {"temperature": 1.5}),
    ("decision_policy", "class-bias-v2"),
])
def test_configuration_changes_invalidate_cache(tmp_path, field, value):
    Client.calls = []
    backend = JevBackend(api_key="test", client_factory=Client)
    options = dict(
        split_name="dev", backend=backend, cache=PredictionCache(tmp_path / "cache"),
        results_root=tmp_path / "results",
    )
    rows = [Example(1, "mzyan", "positive", "Arabizi", "it")]
    baseline = evaluate_examples(rows, **options)
    changed = evaluate_examples(rows, experiment_config={field: value}, **options)
    repeated = evaluate_examples(rows, experiment_config={field: value}, **options)
    restored = evaluate_examples(rows, **options)
    assert len(Client.calls) == 2
    assert not changed[1][0]["cached"]
    assert repeated[1][0]["cached"] and restored[1][0]["cached"]
    assert baseline[1][0]["experiment_fingerprint"] != changed[1][0]["experiment_fingerprint"]


def test_backends_share_ordered_schema_identity():
    jev = JevBackend(api_key="test", client_factory=Client)
    laya = LayaBackend(endpoint_url="https://example.invalid")
    assert jev.schema_fingerprint == laya.schema_fingerprint
    reversed_jev = JevBackend(
        api_key="test", client_factory=Client, option_order=tuple(reversed(SENTIMENT_CRITERIA)),
    )
    assert reversed_jev.schema_fingerprint != laya.schema_fingerprint


def test_old_cache_is_preserved_but_not_reused(tmp_path):
    Client.calls = []
    backend = JevBackend(api_key="test", client_factory=Client)
    cache = PredictionCache(tmp_path / "cache")
    from darija_eval.schema import sentiment_question_dict

    old_schema = hashlib.sha256(json.dumps(sentiment_question_dict(), sort_keys=True).encode()).hexdigest()
    old_payload = json.dumps({
        "backend": "jev", "model": backend.model_identifier,
        "schema_version": backend.schema_version, "text": "mzyan",
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    old_key = hashlib.sha256((old_payload + "\n" + old_schema).encode()).hexdigest()
    cache.put(old_key, backend.predict("mzyan"))
    before = cache._path(old_key).read_bytes()
    _, records, _, _ = evaluate_examples(
        [Example(1, "mzyan", "positive", "Arabizi", "it")],
        split_name="dev", backend=backend, cache=cache, results_root=tmp_path / "results",
    )
    assert not records[0]["cached"] and len(Client.calls) == 2
    assert cache._path(old_key).read_bytes() == before


def test_config_order_is_canonical_and_provenance_is_validated(tmp_path):
    Client.calls = []
    backend = JevBackend(api_key="test", client_factory=Client)
    options = dict(split_name="dev", backend=backend,
                   cache=PredictionCache(tmp_path / "cache"), results_root=tmp_path / "results")
    rows = [Example(1, "mzyan", "positive", "Arabizi", "it")]
    first = evaluate_examples(rows, experiment_config={"preprocessing": "v1", "examples": ["a", "b"]}, **options)
    second = evaluate_examples(rows, experiment_config={"examples": ["a", "b"], "preprocessing": "v1"}, **options)
    third = evaluate_examples(rows, experiment_config={"examples": ["b", "a"], "preprocessing": "v1"}, **options)
    assert second[1][0]["cached"] and not third[1][0]["cached"]
    assert len(Client.calls) == 2
    assert load_run(first[0])["predictions"] == first[1]
    path = first[0] / "predictions.jsonl"
    path.write_text(json.dumps({**first[1][0], "experiment_fingerprint": "tampered"}) + "\n")
    with pytest.raises(ValueError, match="experiment fingerprint"):
        load_run(first[0])


@pytest.mark.parametrize("order", [
    (), ("positive", "negative"), ("positive", "positive", "negative"),
    ("positive", "neutral", "unknown"),
])
def test_invalid_option_sets_are_rejected(order):
    with pytest.raises(ValueError, match="option_order"):
        JevBackend(api_key="test", client_factory=Client, option_order=order)
