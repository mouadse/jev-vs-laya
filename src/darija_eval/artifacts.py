"""Offline, validated reanalysis of saved prediction evidence."""
from __future__ import annotations

import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .dataset import Example, dataset_fingerprint
from .metrics import LABELS, compute_metrics, error_analysis


def load_run(path: Path) -> dict:
    source = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    predictions = read_jsonl(path / "predictions.jsonl")
    failures = read_jsonl(path / "api_failures.jsonl") if (path / "api_failures.jsonl").exists() else []
    _validate_provenance(path, source, predictions, failures)
    ids = [row["id"] for row in predictions]
    failed_ids = [row["id"] for row in failures]
    if len(set(ids)) != len(ids) or len(set(failed_ids)) != len(failed_ids):
        raise ValueError("run contains duplicate example IDs")
    if set(ids) & set(failed_ids):
        raise ValueError("same example appears as both successful and failed")
    if source.get("successful") != len(predictions) or source.get("api_failures") != len(failures):
        raise ValueError("run has inconsistent counts or missing predictions/failures")
    requested = source.get("requested", len(predictions) + len(failures))
    if requested != len(predictions) + len(failures):
        raise ValueError("requested count does not match accounted examples")
    for row in predictions:
        if row["gold"] not in LABELS or row["predicted"] not in LABELS:
            raise ValueError(f"invalid sentiment for ID {row['id']}")
        if row.get("split") != source.get("split"):
            raise ValueError(f"row split disagrees with run metadata for ID {row['id']}")
        for field in ("backend", "schema_version"):
            if field in source and field in row and source[field] != row[field]:
                raise ValueError(f"row {field} disagrees with run metadata")
        if row["correct"] != (row["gold"] == row["predicted"]):
            raise ValueError(f"incorrect stored correctness flag for ID {row['id']}")
        probabilities = row.get("probabilities")
        if probabilities and row.get("confidence") is not None:
            if not math.isclose(row["confidence"], probabilities.get(row["predicted"], -1), abs_tol=1e-9):
                raise ValueError(f"stored confidence disagrees with probabilities for ID {row['id']}")
    if predictions and all("model" in row for row in predictions):
        actual_models = sorted({row["model"] for row in predictions})
        if source.get("resolved_models") != actual_models:
            raise ValueError("resolved model metadata disagrees with predictions")
    metrics = dict(source)
    metrics.update(compute_metrics(
        predictions, requested=requested, api_failures=len(failures),
        wall_clock_seconds=source.get("wall_clock_seconds", 0.0),
        topic_min_n=source.get("topic_min_n", 10),
    ))
    metrics["analysis_provenance"] = {
        "source_run": str(path.resolve()),
        "predictions_sha256": hashlib.sha256((path / "predictions.jsonl").read_bytes()).hexdigest(),
        "source_metrics_sha256": hashlib.sha256((path / "metrics.json").read_bytes()).hexdigest(),
        "inference_rerun": False,
        "analysis_version": 2,
        "note": "Metrics recomputed from saved predictions. Original timings retained; no new inference.",
    }
    return {"metrics": metrics, "predictions": predictions, "failures": failures}


def _validate_provenance(path: Path, source: dict, predictions: list[dict], failures: list[dict]) -> None:
    manifest_path = path / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for field in ("backend", "requested_model", "schema_version", "split", "provenance"):
            if manifest.get(field) != source.get(field):
                raise ValueError(f"manifest {field} disagrees with run metadata")
    provenance = source.get("provenance") or {}
    rows = predictions + failures
    for row in rows:
        if type(row.get("id")) is not int or row["id"] < 0:
            raise ValueError("example IDs must be nonnegative integers")
        if row.get("split") != source.get("split"):
            raise ValueError(f"row split disagrees with run metadata for ID {row['id']}")
    schema_fingerprint = provenance.get("schema_fingerprint")
    for row in predictions:
        if schema_fingerprint is not None and row.get("schema_fingerprint") != schema_fingerprint:
            raise ValueError(f"row schema fingerprint disagrees with provenance for ID {row['id']}")
        if "requested_model" in row and row["requested_model"] != source.get("requested_model"):
            raise ValueError("row requested model disagrees with run metadata")
    selected_ids = provenance.get("selected_ids")
    if selected_ids is None:
        if provenance.get("selection_fingerprint") is not None:
            raise ValueError("selection fingerprint cannot be checked without selected IDs")
        return
    if (not isinstance(selected_ids, list)
            or any(type(identifier) is not int or identifier < 0 for identifier in selected_ids)
            or len(set(selected_ids)) != len(selected_ids)):
        raise ValueError("selected IDs must be unique nonnegative integers")
    if len(rows) != len(selected_ids) or {row["id"] for row in rows} != set(selected_ids):
        raise ValueError("selected IDs disagree with accounted examples")
    if provenance.get("selected_size", len(selected_ids)) != len(selected_ids):
        raise ValueError("selected size disagrees with selected IDs")
    split_size = provenance.get("split_size")
    if split_size is not None:
        if type(split_size) is not int or split_size < len(selected_ids):
            raise ValueError("split size cannot be smaller than the selected sample")
        if provenance.get("full_split") is not (split_size == len(selected_ids)):
            raise ValueError("full-split flag disagrees with selection size")
    if provenance.get("selection_fingerprint") is not None:
        by_id = {row["id"]: row for row in rows}
        examples = [
            Example(identifier, by_id[identifier]["text"], by_id[identifier]["gold"],
                    by_id[identifier]["writing_style"], by_id[identifier]["topic"])
            for identifier in selected_ids
        ]
        if dataset_fingerprint(examples) != provenance["selection_fingerprint"]:
            raise ValueError("selection fingerprint disagrees with saved examples")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def reanalyse_run(path: Path, results_root: Path = Path("results")) -> Path:
    from .evaluate import markdown_summary
    from .report import html_report

    run = load_run(path)
    metrics, predictions = run["metrics"], run["predictions"]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = results_root / f"{metrics['backend']}_reanalysis_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(path / "predictions.jsonl", output / "predictions.jsonl")
    if (path / "manifest.json").exists():
        shutil.copyfile(path / "manifest.json", output / "manifest.json")
    for filename, rows in (("api_failures.jsonl", run["failures"]),
                           ("failures.jsonl", [row for row in predictions if not row["correct"]])):
        (output / filename).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "summary.md").write_text(markdown_summary(metrics, error_analysis(predictions)), encoding="utf-8")
    (output / "report.html").write_text(html_report(metrics, predictions), encoding="utf-8")
    return output
