from __future__ import annotations

import json
import math
import os
import time
from threading import Lock
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..modal_djev import IMAGE, MODEL_IDENTIFIER, MODEL_REVISION
from ..schema import SCHEMA_VERSION, SENTIMENT_CRITERIA, question_fingerprint, sentiment_question_dict
from .base import PermanentBackendError, Prediction, TransientBackendError


class DjevBackend:
    name = "djev"
    model_identifier = MODEL_IDENTIFIER
    schema_version = SCHEMA_VERSION

    @property
    def schema_fingerprint(self) -> str:
        return question_fingerprint(sentiment_question_dict())

    @property
    def option_order(self) -> tuple[str, ...]:
        return tuple(sentiment_question_dict()["criteria"])

    def __init__(
        self,
        *,
        endpoint_url: str | None = None,
        max_retries: int = 3,
        timeout: float = 300.0,
        opener: Callable[..., Any] = urlopen,
    ) -> None:
        self.endpoint_url = (
            endpoint_url if endpoint_url is not None else os.getenv("DJEV_ENDPOINT_URL", "")
        ).strip().rstrip("/")
        if not self.endpoint_url:
            raise PermanentBackendError("DJEV_ENDPOINT_URL is not set")
        self.max_retries = max_retries
        self.timeout = timeout
        self._opener = opener
        self._identity_lock = Lock()
        self._identity_verified = False

    def _verify_identity(self) -> None:
        if self._identity_verified:
            return
        with self._identity_lock:
            if self._identity_verified:
                return
            deadline = time.monotonic() + self.timeout
            request = Request(f"{self.endpoint_url}/health")
            while True:
                try:
                    with self._opener(request, timeout=self.timeout) as response:
                        health = json.loads(response.read().decode("utf-8"))
                    if not isinstance(health, dict):
                        raise PermanentBackendError("invalid djev health response")
                    if health.get("status") == "ok":
                        expected = {
                            "model": "dgemma",
                            "model_revision": MODEL_REVISION,
                            "image": IMAGE,
                        }
                        if any(health.get(key) != value for key, value in expected.items()):
                            raise PermanentBackendError("djev endpoint identity does not match the pinned deployment")
                        self._identity_verified = True
                        return
                except HTTPError as error:
                    if error.code not in {408, 429} and error.code < 500:
                        raise PermanentBackendError(_http_error(error)) from error
                except (URLError, TimeoutError, ConnectionError):
                    pass
                except (json.JSONDecodeError, UnicodeDecodeError) as error:
                    raise PermanentBackendError(f"invalid djev health response: {error}") from error
                if time.monotonic() >= deadline:
                    raise TransientBackendError("djev endpoint did not become healthy before timeout")
                time.sleep(2)

    def predict(self, text: str) -> Prediction:
        started = time.perf_counter()
        self._verify_identity()
        body = json.dumps(
            {
                "model": "dgemma",
                "state": {"review": text},
                "questions": {"sentiment": sentiment_question_dict()},
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = Request(
            f"{self.endpoint_url}/v1/systemone",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        for attempt in range(self.max_retries + 1):
            try:
                with self._opener(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                return parse_response(payload, (time.perf_counter() - started) * 1000)
            except HTTPError as error:
                message = _http_error(error)
                if error.code not in {408, 429} and error.code < 500:
                    raise PermanentBackendError(message) from error
                if attempt == self.max_retries:
                    raise TransientBackendError(message) from error
            except (URLError, TimeoutError, ConnectionError) as error:
                if attempt == self.max_retries:
                    raise TransientBackendError(f"{type(error).__name__}: {error}") from error
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise PermanentBackendError(f"invalid djev response: {error}") from error
            time.sleep(min(0.5 * (2**attempt), 5.0))
        raise AssertionError("retry loop exhausted")

    def close(self) -> None:
        return None


def parse_response(payload: Any, latency_ms: float) -> Prediction:
    try:
        if not isinstance(payload, dict) or payload["model"] != "dgemma":
            raise ValueError("unexpected model")
        answer = payload["answers"]["sentiment"]
        if answer["type"] != "choice":
            raise ValueError("sentiment answer is not a choice")
        label = answer["choice"]
        raw_probabilities = answer["probabilities"]
        if not isinstance(raw_probabilities, dict):
            raise ValueError("probabilities must be an object")
        if any(type(value) not in (int, float) for value in raw_probabilities.values()):
            raise ValueError("probabilities must be numeric")
        probabilities = {str(key): float(value) for key, value in raw_probabilities.items()}
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise PermanentBackendError(f"invalid djev response: {error}") from error
    expected = set(SENTIMENT_CRITERIA)
    if not isinstance(label, str) or label not in expected:
        raise PermanentBackendError(f"djev returned unknown label {label!r}")
    if set(probabilities) != expected:
        raise PermanentBackendError("djev probabilities have unexpected labels")
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities.values()):
        raise PermanentBackendError("djev probabilities have invalid values")
    if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-3):
        raise PermanentBackendError("djev probabilities do not sum to one")
    return Prediction(label, probabilities, latency_ms, MODEL_IDENTIFIER, payload)


def _http_error(error: HTTPError) -> str:
    try:
        detail = error.read().decode("utf-8")[:500]
    except Exception:
        detail = ""
    return f"djev endpoint HTTP {error.code}: {detail or error.reason}"
