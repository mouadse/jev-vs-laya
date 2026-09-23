from __future__ import annotations

import json
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from darija_eval import cli
from darija_eval.backends.base import PermanentBackendError
from darija_eval.backends.djev import DjevBackend, MODEL_IDENTIFIER
from darija_eval.dataset import Example
from darija_eval.modal_djev import IMAGE, MODEL_REVISION
from darija_eval.schema import sentiment_question_dict


class Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return json.dumps(self.payload).encode()


HEALTH = {
    "status": "ok",
    "model": "dgemma",
    "model_revision": MODEL_REVISION,
    "image": IMAGE,
}
ANSWER = {
    "model": "dgemma",
    "answers": {
        "sentiment": {
            "type": "choice",
            "choice": "positive",
            "probabilities": {"positive": 0.8, "neutral": 0.1, "negative": 0.1},
        }
    },
}


def test_djev_backend_sends_only_review_and_shared_schema() -> None:
    calls = []

    def opener(request, timeout):
        calls.append(request)
        return Response(HEALTH if request.full_url.endswith("/health") else ANSWER)

    backend = DjevBackend(endpoint_url="https://example.test/", opener=opener)
    prediction = backend.predict("هاد المنتج زوين")
    assert [request.full_url for request in calls] == [
        "https://example.test/health",
        "https://example.test/v1/systemone",
    ]
    assert json.loads(calls[1].data) == {
        "model": "dgemma",
        "state": {"review": "هاد المنتج زوين"},
        "questions": {"sentiment": sentiment_question_dict()},
    }
    assert prediction.label == "positive"
    assert prediction.probabilities == ANSWER["answers"]["sentiment"]["probabilities"]
    assert prediction.model == MODEL_IDENTIFIER


def test_djev_requires_endpoint_url(monkeypatch) -> None:
    monkeypatch.delenv("DJEV_ENDPOINT_URL", raising=False)
    with pytest.raises(PermanentBackendError, match="DJEV_ENDPOINT_URL"):
        DjevBackend(endpoint_url="")


def test_djev_rejects_wrong_deployment_identity() -> None:
    calls = []

    def opener(request, timeout):
        calls.append(request)
        return Response({**HEALTH, "model_revision": "other"})

    backend = DjevBackend(endpoint_url="https://example.test", opener=opener)
    with pytest.raises(PermanentBackendError, match="identity"):
        backend.predict("mzyan")
    assert len(calls) == 1


def test_djev_waits_for_cold_start(monkeypatch) -> None:
    from darija_eval.backends import djev

    calls = []

    def opener(request, timeout):
        calls.append(request.full_url)
        if request.full_url.endswith("/health") and calls.count(request.full_url) == 1:
            raise HTTPError(request.full_url, 503, "starting", {}, None)
        return Response(HEALTH if request.full_url.endswith("/health") else ANSWER)

    monkeypatch.setattr(djev.time, "sleep", lambda _: None)
    backend = DjevBackend(endpoint_url="https://example.test", opener=opener)
    assert backend.predict("mzyan").label == "positive"
    assert calls == [
        "https://example.test/health",
        "https://example.test/health",
        "https://example.test/v1/systemone",
    ]


@pytest.mark.parametrize("probabilities", [
    {"positive": 1.0},
    {"positive": 0.8, "neutral": 0.4, "negative": 0.1},
    {"positive": -0.1, "neutral": 0.2, "negative": 0.9},
])
def test_djev_rejects_invalid_probabilities(probabilities) -> None:
    def opener(request, timeout):
        if request.full_url.endswith("/health"):
            return Response(HEALTH)
        return Response({
            **ANSWER,
            "answers": {"sentiment": {**ANSWER["answers"]["sentiment"], "probabilities": probabilities}},
        })

    backend = DjevBackend(endpoint_url="https://example.test", opener=opener)
    with pytest.raises(PermanentBackendError, match="probabilities"):
        backend.predict("mzyan")


def test_djev_rejects_unexpected_response_model() -> None:
    def opener(request, timeout):
        return Response(HEALTH if request.full_url.endswith("/health") else {**ANSWER, "model": "other"})

    backend = DjevBackend(endpoint_url="https://example.test", opener=opener)
    with pytest.raises(PermanentBackendError, match="unexpected model"):
        backend.predict("mzyan")


def test_djev_does_not_retry_permanent_http_error() -> None:
    calls = []

    def opener(request, timeout):
        calls.append(request)
        if request.full_url.endswith("/health"):
            return Response(HEALTH)
        raise HTTPError(request.full_url, 422, "invalid request", {}, None)

    backend = DjevBackend(endpoint_url="https://example.test", opener=opener)
    with pytest.raises(PermanentBackendError, match="422"):
        backend.predict("mzyan")
    assert len(calls) == 2


def test_cli_accepts_djev(monkeypatch) -> None:
    monkeypatch.setenv("DJEV_ENDPOINT_URL", "https://example.test")
    assert isinstance(cli._backend("djev", 0), DjevBackend)


def test_djev_eval_cli_writes_html_report_without_network(monkeypatch, tmp_path) -> None:
    from typer.testing import CliRunner

    example = Example(1, "synthetic review", "positive", "Arabizi", "test")
    split = SimpleNamespace(eval=(0,), dev=(), seed=42, strategy="test")

    def opener(request, timeout):
        return Response(HEALTH if request.full_url.endswith("/health") else ANSWER)

    monkeypatch.setattr(cli, "_load_validated", lambda: ([example], split))
    monkeypatch.setattr(cli, "_provenance", lambda *args: {})
    monkeypatch.setattr(
        cli,
        "DjevBackend",
        lambda max_retries: DjevBackend(
            endpoint_url="https://example.test", opener=opener, max_retries=max_retries
        ),
    )
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli.app, ["eval", "--backend", "djev"])
    assert result.exit_code == 0, result.output
    run = next((tmp_path / "results").glob("djev_eval_*"))
    assert (run / "report.html").exists()
    assert json.loads((run / "metrics.json").read_text())["successful"] == 1
