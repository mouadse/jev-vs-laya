from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Sequence

from .backends.base import BackendError, PermanentBackendError, Prediction, SentimentBackend
from .cache import PredictionCache, is_valid_prediction
from .dataset import Example, dataset_fingerprint
from .metrics import compute_metrics, error_analysis
from .report import html_report


def evaluate_examples(
    examples: Sequence[Example],
    *,
    split_name: str,
    backend: SentimentBackend,
    concurrency: int = 5,
    results_root: Path = Path("results"),
    cache: PredictionCache | None = None,
    evaluation_provenance: dict[str, Any] | None = None,
) -> tuple[Path, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    if len({example.id for example in examples}) != len(examples):
        raise ValueError("evaluation example IDs must be unique")
    cache = cache or PredictionCache()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    run_dir = results_root / f"{backend.name}_{split_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    predictions: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started = time.perf_counter()
    schema_fingerprint = getattr(backend, "schema_fingerprint", None)
    provenance = {
        **(evaluation_provenance or {}),
        "selection_fingerprint": dataset_fingerprint(examples),
        "selected_ids": [example.id for example in examples],
        "schema_fingerprint": schema_fingerprint,
        "concurrency": concurrency,
    }
    (run_dir / "manifest.json").write_text(json.dumps({
        "backend": backend.name, "requested_model": backend.model_identifier,
        "schema_version": backend.schema_version, "split": split_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provenance": provenance,
    }, indent=2) + "\n", encoding="utf-8")
    _write_jsonl(run_dir / "predictions.jsonl", [])
    review_locks = {example.review: Lock() for example in examples}

    def run_one(example: Example) -> tuple[Example, Prediction, bool]:
        key = cache.key(
            backend.name, backend.model_identifier, backend.schema_version, example.review,
            schema_fingerprint,
        )
        with review_locks[example.review]:
            cached = cache.get(key)
            if cached is not None:
                return example, cached, True
            prediction = backend.predict(example.review)
            if not is_valid_prediction(prediction):
                raise PermanentBackendError(
                    f"backend {backend.name!r} returned an invalid prediction"
                )
            cache.put(key, prediction)
            return example, prediction, False

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(run_one, example): example for example in examples}
        for future in as_completed(futures):
            example = futures[future]
            try:
                row, prediction, cached = future.result()
                predictions.append(_record(row, split_name, backend, prediction, cached))
            except BackendError as error:
                failures.append(
                    {
                        "id": example.id,
                        "split": split_name,
                        "text": example.review,
                        "gold": example.label,
                        "writing_style": example.writing_style,
                        "topic": example.topic,
                        "backend": backend.name,
                        "model": backend.model_identifier,
                        "schema_version": backend.schema_version,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
            _write_jsonl(run_dir / "predictions.jsonl", sorted(predictions, key=lambda x: x["id"]))
            _write_jsonl(
                run_dir / "api_failures.jsonl", sorted(failures, key=lambda x: x["id"])
            )

    predictions.sort(key=lambda row: row["id"])
    failures.sort(key=lambda row: row["id"])
    wall_clock = time.perf_counter() - started
    metrics = compute_metrics(
        predictions,
        requested=len(examples),
        api_failures=len(failures),
        wall_clock_seconds=wall_clock,
    )
    metrics.update(
        {
            "format_version": 1,
            "backend": backend.name,
            "requested_model": backend.model_identifier,
            "resolved_models": sorted({row["model"] for row in predictions}),
            "schema_version": backend.schema_version,
            "split": split_name,
            "provenance": provenance,
        }
    )
    analysis = error_analysis(predictions)
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_jsonl(run_dir / "api_failures.jsonl", failures)
    _write_jsonl(run_dir / "failures.jsonl", [row for row in predictions if not row["correct"]])
    (run_dir / "summary.md").write_text(
        markdown_summary(metrics, analysis), encoding="utf-8"
    )
    (run_dir / "report.html").write_text(
        html_report(metrics, predictions), encoding="utf-8"
    )
    return run_dir, predictions, failures, metrics


def _record(
    example: Example,
    split_name: str,
    backend: SentimentBackend,
    prediction: Prediction,
    cached: bool,
) -> dict[str, Any]:
    confidence = (
        prediction.probabilities.get(prediction.label)
        if prediction.probabilities is not None
        else None
    )
    return {
        "id": example.id,
        "split": split_name,
        "text": example.review,
        "writing_style": example.writing_style,
        "topic": example.topic,
        "gold": example.label,
        "predicted": prediction.label,
        "correct": prediction.label == example.label,
        "probabilities": prediction.probabilities,
        "confidence": confidence,
        "latency_ms": prediction.latency_ms,
        "backend": backend.name,
        "model": prediction.model,
        "requested_model": backend.model_identifier,
        "schema_version": backend.schema_version,
        "schema_fingerprint": getattr(backend, "schema_fingerprint", None),
        "cached": cached,
        "raw": prediction.raw,
    }


def _write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def markdown_summary(metrics: dict[str, Any], analysis: dict[str, Any]) -> str:
    display_name = str(metrics["backend"]).title()
    lines = [
        f"# {display_name} — Moroccan Darija sentiment",
        "",
        f"- Split: {metrics['split']}",
        f"- Requested: {metrics['requested']}",
        f"- Successful: {metrics['successful']}",
        f"- API failures: {metrics['api_failures']}",
    ]
    quality = metrics.get("quality")
    if quality:
        lines.extend(
            [
                f"- Accuracy: {quality['accuracy']:.1%}",
                f"- Macro F1: {quality['macro_f1']:.1%}",
                "",
                "## Writing style",
                "",
            ]
        )
        for style, values in metrics["writing_style"].items():
            lines.append(
                f"- {style}: N={values['n']}, accuracy={values['accuracy']:.1%}, macro F1={values['macro_f1']:.1%}"
            )
        delta = metrics.get("arabizi_minus_arabic_accuracy")
        if delta is not None:
            lines.append(f"- Arabizi − Arabic accuracy: {delta * 100:+.1f} pp")
        lines.extend(["", "## Highest-confidence errors", ""])
        for row in analysis["highest_confidence_errors"]:
            lines.append(
                f"- `{row['id']}` {row['gold']} → {row['predicted']} "
                f"(confidence {row['confidence']:.3f}): {row['text']}"
            )
        if not analysis["highest_confidence_errors"]:
            lines.append("No errors with available confidence.")
    if metrics.get("warning"):
        lines.extend(["", f"> {metrics['warning']}"])
    return "\n".join(lines) + "\n"
