from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..schema import SCHEMA_VERSION, SENTIMENT_CRITERIA, question_fingerprint, sentiment_question_dict
from .base import PermanentBackendError, Prediction, TransientBackendError

ADAPTER_REPO = "jaredpalmer/kev-9b"
ADAPTER_REVISION = "2629c06a5aeb0feb3b9783bafed17ed8f39ecf5c"
BASE_REPO = "Qwen/Qwen3.5-9B-Base"
BASE_REVISION = "68c46c4b3498877f3ef123c856ecfde50c39f404"
KEV_CODE_REVISION = "90990a5fac2995b9faa3190f7d437e84f2067768"
MODEL_IDENTIFIER = (
    f"{ADAPTER_REPO}@{ADAPTER_REVISION}"
    f"+{BASE_REPO}@{BASE_REVISION}"
    f"+kev@{KEV_CODE_REVISION}"
)


class KevBackend:
    name = "kev"
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
        timeout: float = 60.0,
        opener: Callable[..., Any] = urlopen,
    ) -> None:
        self.endpoint_url = (endpoint_url or os.getenv("KEV_ENDPOINT_URL", "")).strip()
        if not self.endpoint_url:
            raise PermanentBackendError("KEV_ENDPOINT_URL is not set")
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
                    raise PermanentBackendError("invalid Kev response: expected an object")
                if payload.get("model") != self.model_identifier or payload.get("schema_version") != self.schema_version:
                    raise PermanentBackendError("Kev endpoint model/schema does not match the requested configuration")
                if payload.get("schema_fingerprint") != self.schema_fingerprint:
                    raise PermanentBackendError("Kev endpoint schema content does not match the requested configuration")
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
                raise PermanentBackendError(f"invalid Kev response: {error}") from error
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
        raise PermanentBackendError(f"invalid Kev response: {error}") from error
    expected = set(SENTIMENT_CRITERIA)
    if label not in expected:
        raise PermanentBackendError(f"Kev returned unknown label {label!r}")
    if set(probabilities) != expected:
        raise PermanentBackendError(
            f"Kev probabilities have unexpected labels: {sorted(probabilities)}"
        )
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities.values()):
        raise PermanentBackendError("Kev returned invalid probability values")
    if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-3):
        raise PermanentBackendError("Kev probabilities do not sum to one")
    return Prediction(label, probabilities, latency_ms, model, dict(payload))


def _http_error(error: HTTPError) -> str:
    try:
        detail = error.read().decode("utf-8")[:500]
    except Exception:
        detail = ""
    return f"Kev endpoint HTTP {error.code}: {detail or error.reason}"
