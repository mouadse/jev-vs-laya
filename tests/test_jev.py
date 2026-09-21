from __future__ import annotations

from types import SimpleNamespace

import pytest

from darija_eval.backends.base import PermanentBackendError
from darija_eval.backends.jev import JevBackend, parse_response


class FakeResponse:
    model = "jev-1.13.0"
    answers = {
        "sentiment": SimpleNamespace(
            choice="negative",
            probabilities={"positive": 0.05, "neutral": 0.15, "negative": 0.8},
            confidence=0.7,
        )
    }

    def model_dump(self, mode):
        assert mode == "json"
        return {"model": self.model, "answers": {}}


class FakeClient:
    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.calls = []
        self.closed = False

    def system_one(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse()

    def close(self):
        self.closed = True


def test_response_parsing_uses_actual_model_and_probabilities() -> None:
    prediction = parse_response(FakeResponse(), 12.3)
    assert prediction.label == "negative"
    assert prediction.model == "jev-1.13.0"
    assert prediction.probabilities["negative"] == 0.8
    assert prediction.raw["sdk_choice_confidence"] == 0.7


def test_backend_sends_only_review_state() -> None:
    created = []

    def factory(**kwargs):
        client = FakeClient(**kwargs)
        created.append(client)
        return client

    backend = JevBackend(api_key="test-key", client_factory=factory, max_retries=4)
    prediction = backend.predict("hada mzyan")
    assert prediction.label == "negative"
    assert created[0].calls[0]["state"] == {"review": "hada mzyan"}
    assert set(created[0].calls[0]["questions"]) == {"sentiment"}
    assert created[0].init_kwargs["retry"].max_retries == 4
    backend.close()
    assert created[0].closed


def test_malformed_response_is_not_fabricated() -> None:
    response = FakeResponse()
    response.answers = {
        "sentiment": SimpleNamespace(
            choice="negative", probabilities={"negative": 1.0}, confidence=1.0
        )
    }
    with pytest.raises(PermanentBackendError, match="unexpected labels"):
        parse_response(response, 1.0)


@pytest.mark.parametrize("values", [
    {"positive": float("nan"), "neutral": 0.2, "negative": 0.8},
    {"positive": -0.1, "neutral": 0.2, "negative": 0.9},
    {"positive": 0.3, "neutral": 0.3, "negative": 0.9},
])
def test_invalid_probability_values_are_rejected(values):
    response = FakeResponse()
    response.answers = {"sentiment": SimpleNamespace(choice="negative", probabilities=values, confidence=0.5)}
    with pytest.raises(PermanentBackendError):
        parse_response(response, 1.0)
