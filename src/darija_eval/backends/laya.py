from __future__ import annotations

import json
import hashlib
import math
import os
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..schema import SCHEMA_VERSION, SENTIMENT_CRITERIA, sentiment_question_dict
from .base import PermanentBackendError, Prediction, TransientBackendError

MODEL_IDENTIFIER = (
    "convaiinnovations/laya:multilingual@"
    "1c5edc17a7acd8701df6fc341c0d179f1c62c982+laya-0.3.4"
)


class LayaBackend:
    name = "laya"
    model_identifier = MODEL_IDENTIFIER
    schema_version = SCHEMA_VERSION

    @property
    def schema_fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(sentiment_question_dict(), sort_keys=True).encode()).hexdigest()

    def __init__(
        self,
        *,
        endpoint_url: str | None = None,
        max_retries: int = 3,
        timeout: float = 60.0,
        opener: Callable[..., Any] = urlopen,
    ) -> None:
        self.endpoint_url = (endpoint_url or os.getenv("LAYA_ENDPOINT_URL", "")).strip()
        if not self.endpoint_url:
            raise PermanentBackendError("LAYA_ENDPOINT_URL is not set")
        self.max_retries = max_retries
        self.timeout = timeout
        self._opener = opener

    def predict(self, text: str) -> Prediction:
        body = json.dumps({"review": text}, ensure_ascii=False).encode("utf-8")
        request = Request(
            self.endpoint_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        for attempt in range(self.max_retries + 1):
            try:
                with self._opener(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                latency_ms = (time.perf_counter() - started) * 1000
                if not isinstance(payload, dict):
                    raise PermanentBackendError("invalid Laya response: expected an object")
                if payload.get("model") != self.model_identifier or payload.get("schema_version") != self.schema_version:
                    raise PermanentBackendError("Laya endpoint model/schema does not match the requested configuration")
                if payload.get("schema_fingerprint", self.schema_fingerprint) != self.schema_fingerprint:
                    raise PermanentBackendError("Laya endpoint schema content does not match the requested configuration")
                return parse_response(payload, latency_ms)
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
                raise PermanentBackendError(f"invalid Laya response: {error}") from error
            time.sleep(min(0.5 * (2**attempt), 5.0))
        raise AssertionError("retry loop exhausted")

    def close(self) -> None:
        return None


def parse_response(payload: Any, latency_ms: float) -> Prediction:
    try:
        label = str(payload["label"])
        probabilities = {
            str(key): float(value) for key, value in payload["probabilities"].items()
        }
        model = str(payload["model"])
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise PermanentBackendError(f"invalid Laya response: {error}") from error
    expected = set(SENTIMENT_CRITERIA)
    if label not in expected:
        raise PermanentBackendError(f"Laya returned unknown label {label!r}")
    if set(probabilities) != expected:
        raise PermanentBackendError(
            f"Laya probabilities have unexpected labels: {sorted(probabilities)}"
        )
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities.values()):
        raise PermanentBackendError("Laya returned invalid probability values")
    if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-3):
        raise PermanentBackendError("Laya probabilities do not sum to one")
    return Prediction(label, probabilities, latency_ms, model, dict(payload))


def _http_error(error: HTTPError) -> str:
    try:
        detail = error.read().decode("utf-8")[:500]
    except Exception:
        detail = ""
    return f"Laya endpoint HTTP {error.code}: {detail or error.reason}"
