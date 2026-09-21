from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class Prediction:
    label: str
    probabilities: dict[str, float] | None
    latency_ms: float
    model: str
    raw: dict[str, Any] | None = None


class SentimentBackend(Protocol):
    name: str
    model_identifier: str
    schema_version: str

    def predict(self, text: str) -> Prediction: ...


class BackendError(RuntimeError):
    """Base error for backend calls."""


class TransientBackendError(BackendError):
    """A request may succeed when retried."""


class PermanentBackendError(BackendError):
    """A request should not be retried without changing configuration."""

