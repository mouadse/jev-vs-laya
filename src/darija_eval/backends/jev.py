from __future__ import annotations

import os
import math
import time
from typing import Any, Callable
from collections.abc import Sequence

from typesafe_sdk import (
    Choice,
    RetryPolicy,
    TypeSafeAPIConnectionError,
    TypeSafeAPIResponseValidationError,
    TypeSafeAPITimeoutError,
    TypeSafeAuthenticationError,
    TypeSafeBadRequestError,
    TypeSafeClient,
    TypeSafeInternalServerError,
    TypeSafeNotFoundError,
    TypeSafePermissionDeniedError,
    TypeSafeRateLimitError,
    TypeSafeUnprocessableEntityError,
)

from .base import PermanentBackendError, Prediction, TransientBackendError
from ..schema import SCHEMA_VERSION, SENTIMENT_CRITERIA, question_fingerprint, sentiment_question_dict

REQUESTED_MODEL = "jev-latest"
EXPECTED_LABELS = set(SENTIMENT_CRITERIA)

_TRANSIENT = (
    TypeSafeRateLimitError,
    TypeSafeInternalServerError,
    TypeSafeAPIConnectionError,
    TypeSafeAPITimeoutError,
)
_PERMANENT = (
    TypeSafeAuthenticationError,
    TypeSafePermissionDeniedError,
    TypeSafeBadRequestError,
    TypeSafeNotFoundError,
    TypeSafeUnprocessableEntityError,
    TypeSafeAPIResponseValidationError,
)


class JevBackend:
    name = "jev"
    model_identifier = REQUESTED_MODEL
    schema_version = SCHEMA_VERSION

    @property
    def schema_fingerprint(self) -> str:
        return question_fingerprint(self._question.model_dump(mode="json"))

    @property
    def option_order(self) -> tuple[str, ...]:
        return tuple(self._question.criteria)

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = REQUESTED_MODEL,
        max_retries: int = 3,
        timeout: float = 30.0,
        client_factory: Callable[..., Any] = TypeSafeClient,
        option_order: Sequence[str] | None = None,
    ) -> None:
        key = api_key or os.getenv("TYPESAFE_API_KEY")
        if not key:
            raise PermanentBackendError("TYPESAFE_API_KEY is not set")
        self.model_identifier = model
        self.max_retries = max_retries
        self._question = Choice(**sentiment_question_dict(option_order))
        self._client = client_factory(
            api_key=key,
            model=model,
            retry=RetryPolicy(max_retries=max_retries),
            timeout=timeout,
        )

    def predict(self, text: str) -> Prediction:
        started = time.perf_counter()
        try:
            response = self._client.system_one(
                state={"review": text},
                questions={"sentiment": self._question},
            )
        except _TRANSIENT as error:
            raise TransientBackendError(_safe_error(error)) from error
        except _PERMANENT as error:
            raise PermanentBackendError(_safe_error(error)) from error
        elapsed_ms = (time.perf_counter() - started) * 1000
        return parse_response(response, elapsed_ms)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "JevBackend":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def parse_response(response: Any, latency_ms: float) -> Prediction:
    try:
        answer = response.answers["sentiment"]
        label = answer.choice
        probabilities = {str(key): float(value) for key, value in answer.probabilities.items()}
        model = str(response.model)
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise PermanentBackendError(f"invalid Jev response: {error}") from error
    if label not in EXPECTED_LABELS:
        raise PermanentBackendError(f"Jev returned unknown label {label!r}")
    if set(probabilities) != EXPECTED_LABELS:
        raise PermanentBackendError(
            f"Jev probabilities have unexpected labels: {sorted(probabilities)}"
        )
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities.values()):
        raise PermanentBackendError("Jev returned invalid probability values")
    if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-3):
        raise PermanentBackendError("Jev probabilities do not sum to one")
    raw = response.model_dump(mode="json")
    raw["sdk_choice_confidence"] = float(answer.confidence)
    return Prediction(label, probabilities, latency_ms, model, raw)


def _safe_error(error: Exception) -> str:
    request_id = getattr(error, "request_id", None)
    suffix = f" (request_id={request_id})" if request_id else ""
    return f"{type(error).__name__}: {error}{suffix}"
