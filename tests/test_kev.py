from __future__ import annotations

import json
from urllib.error import HTTPError

import pytest

from darija_eval.backends.base import PermanentBackendError
from darija_eval.backends.kev import (
    BASE_REVISION,
    MODEL_IDENTIFIER,
    KevBackend,
    parse_response,
)
from darija_eval.modal_kev import checkpoint_base_matches
from darija_eval.schema import SCHEMA_VERSION, question_fingerprint, sentiment_question_dict


def _payload(**overrides):
    payload = {
        "label": "negative",
        "probabilities": {"positive": 0.05, "neutral": 0.15, "negative": 0.8},
        "model": MODEL_IDENTIFIER,
        "schema_version": SCHEMA_VERSION,
        "schema_fingerprint": question_fingerprint(sentiment_question_dict()),
        "inference_ms": 38.0,
    }
    payload.update(overrides)
    return payload


class Response:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return json.dumps(_payload()).encode()


def test_kev_backend_sends_review_only() -> None:
    calls = []

    def opener(request, timeout):
        calls.append((request, timeout))
        return Response()

    backend = KevBackend(endpoint_url="https://example.test/predict", opener=opener)
    prediction = backend.predict("hada khayb")
    assert json.loads(calls[0][0].data) == {"review": "hada khayb"}
    assert prediction.label == "negative"
    assert prediction.probabilities["negative"] == 0.8


def test_kev_requires_endpoint_url(monkeypatch) -> None:
    monkeypatch.delenv("KEV_ENDPOINT_URL", raising=False)
    with pytest.raises(PermanentBackendError, match="KEV_ENDPOINT_URL"):
        KevBackend(endpoint_url="")


@pytest.mark.parametrize(
    "probabilities",
    [
        {"negative": 1.0},
        {"positive": 0.8, "neutral": 0.4, "negative": 0.1},
        {"positive": -0.1, "neutral": 0.2, "negative": 0.9},
    ],
)
def test_kev_rejects_invalid_probabilities(probabilities) -> None:
    with pytest.raises(PermanentBackendError):
        parse_response(_payload(probabilities=probabilities), 1.0)


def test_kev_does_not_retry_permanent_http_error() -> None:
    calls = 0

    def opener(request, timeout):
        nonlocal calls
        calls += 1
        raise HTTPError(request.full_url, 401, "unauthorized", {}, None)

    backend = KevBackend(
        endpoint_url="https://example.test/predict", opener=opener, max_retries=3
    )
    with pytest.raises(PermanentBackendError):
        backend.predict("review")
    assert calls == 1


@pytest.mark.parametrize("field,value", [("model", "different"), ("schema_version", "obsolete")])
def test_endpoint_configuration_mismatch_is_rejected(field, value):
    class WrongResponse(Response):
        def read(self):
            return json.dumps(_payload(**{field: value})).encode()

    backend = KevBackend(endpoint_url="https://example.test", opener=lambda *a, **kw: WrongResponse())
    with pytest.raises(PermanentBackendError, match="model/schema"):
        backend.predict("review")


@pytest.mark.parametrize("fingerprint", ["different-question-content", None])
def test_missing_or_wrong_schema_fingerprint_is_rejected(fingerprint):
    class FingerprintResponse(Response):
        def read(self):
            payload = _payload()
            if fingerprint is None:
                del payload["schema_fingerprint"]
            else:
                payload["schema_fingerprint"] = fingerprint
            return json.dumps(payload).encode()

    backend = KevBackend(endpoint_url="https://example.test", opener=lambda *a, **kw: FingerprintResponse())
    with pytest.raises(PermanentBackendError, match="schema content"):
        backend.predict("review")




@pytest.mark.parametrize(
    ("meta_base", "meta_revision", "expected"),
    [
        ("Qwen/Qwen3.5-4B-Base", BASE_REVISION, True),
        ("Qwen/Qwen3.5-4B-Base", "1001bb4d", True),
        ("Qwen/Qwen3.5-4B-Base", "deadbeef", False),
        ("Qwen/Qwen3.5-4B-Base", "", False),
        ("Qwen/Qwen3.5-4B-Base", None, False),
        ("other/base", BASE_REVISION, False),
        (None, BASE_REVISION, False),
    ],
)
def test_checkpoint_base_gate_is_fail_closed(meta_base, meta_revision, expected) -> None:
    assert checkpoint_base_matches(meta_base, meta_revision) is expected


