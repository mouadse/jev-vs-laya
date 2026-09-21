from __future__ import annotations

import json
from urllib.error import HTTPError

import pytest

from darija_eval.backends.base import PermanentBackendError
from darija_eval.backends.laya import LayaBackend, MODEL_IDENTIFIER, parse_response


PAYLOAD = {
    "label": "negative",
    "probabilities": {"positive": 0.05, "neutral": 0.15, "negative": 0.8},
    "model": MODEL_IDENTIFIER,
    "schema_version": "v1",
    "inference_ms": 38.0,
}


class Response:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return json.dumps(PAYLOAD).encode()


def test_laya_backend_sends_review_only() -> None:
    calls = []

    def opener(request, timeout):
        calls.append((request, timeout))
        return Response()

    backend = LayaBackend(endpoint_url="https://example.test/predict", opener=opener)
    prediction = backend.predict("hada khayb")
    assert json.loads(calls[0][0].data) == {"review": "hada khayb"}
    assert prediction.label == "negative"
    assert prediction.probabilities["negative"] == 0.8


@pytest.mark.parametrize(
    "probabilities",
    [
        {"negative": 1.0},
        {"positive": 0.8, "neutral": 0.4, "negative": 0.1},
        {"positive": -0.1, "neutral": 0.2, "negative": 0.9},
    ],
)
def test_laya_rejects_invalid_probabilities(probabilities) -> None:
    payload = dict(PAYLOAD, probabilities=probabilities)
    with pytest.raises(PermanentBackendError):
        parse_response(payload, 1.0)


def test_laya_does_not_retry_permanent_http_error() -> None:
    calls = 0

    def opener(request, timeout):
        nonlocal calls
        calls += 1
        raise HTTPError(request.full_url, 401, "unauthorized", {}, None)

    backend = LayaBackend(
        endpoint_url="https://example.test/predict", opener=opener, max_retries=3
    )
    with pytest.raises(PermanentBackendError):
        backend.predict("review")
    assert calls == 1

@pytest.mark.parametrize("field,value", [("model", "different"), ("schema_version", "v2")])
def test_endpoint_configuration_mismatch_is_rejected(field, value):
    class WrongResponse(Response):
        def read(self):
            return json.dumps(dict(PAYLOAD, **{field: value})).encode()
    backend = LayaBackend(endpoint_url="https://example.test", opener=lambda *a, **kw: WrongResponse())
    with pytest.raises(PermanentBackendError, match="model/schema"):
        backend.predict("review")
