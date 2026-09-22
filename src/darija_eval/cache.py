from __future__ import annotations

import hashlib
import json
import math
import tempfile
from dataclasses import asdict
from pathlib import Path

from .backends.base import Prediction
from .schema import SENTIMENT_CRITERIA

EXPECTED_LABELS = frozenset(SENTIMENT_CRITERIA)


def is_valid_prediction(prediction: object) -> bool:
    """Reject malformed evidence while allowing unavailable probabilities."""
    if not isinstance(prediction, Prediction):
        return False
    if not isinstance(prediction.label, str) or prediction.label not in EXPECTED_LABELS:
        return False
    probabilities = prediction.probabilities
    if probabilities is not None:
        if not isinstance(probabilities, dict) or probabilities.keys() != EXPECTED_LABELS:
            return False
        total = 0.0
        for value in probabilities.values():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return False
            if not math.isfinite(value) or not 0 <= value <= 1:
                return False
            total += value
        if not math.isclose(total, 1.0, abs_tol=1e-3, rel_tol=0):
            return False
    if not isinstance(prediction.model, str) or not prediction.model:
        return False
    latency = prediction.latency_ms
    if isinstance(latency, bool) or not isinstance(latency, (int, float)):
        return False
    if not math.isfinite(latency) or latency < 0:
        return False
    return prediction.raw is None or isinstance(prediction.raw, dict)


class PredictionCache:
    def __init__(self, root: Path = Path("data/cache")) -> None:
        self.root = root

    @staticmethod
    def key(
        backend: str, model: str, schema_version: str, text: str,
        schema_fingerprint: str | None = None,
        experiment_fingerprint: str | None = None,
    ) -> str:
        payload = json.dumps(
            {
                "cache_identity_version": 2,
                "experiment_fingerprint": experiment_fingerprint,
                "backend": backend,
                "model": model,
                "schema_version": schema_version,
                "text": text,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if schema_fingerprint is not None:
            payload += "\n" + schema_fingerprint
        return hashlib.sha256(payload.encode()).hexdigest()

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> Prediction | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            # Corrupt entries are misses; filesystem permission/I/O errors propagate.
            return None
        if not isinstance(value, dict):
            return None
        try:
            prediction = Prediction(**value)
        except TypeError:
            # Missing, extra, or otherwise mismatched fields.
            return None
        if not is_valid_prediction(prediction):
            # Right shape but semantically invalid (unknown label, bad
            # probabilities, ...): never serve it as a reusable prediction.
            return None
        return prediction

    def put(self, key: str, prediction: Prediction) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, suffix=".tmp", delete=False, encoding="utf-8"
        ) as output:
            temporary = Path(output.name)
            output.write(json.dumps(asdict(prediction), ensure_ascii=False, sort_keys=True) + "\n")
        temporary.replace(path)
