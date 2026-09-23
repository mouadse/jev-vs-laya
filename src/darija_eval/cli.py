from __future__ import annotations

import random
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

from .backends.base import SentimentBackend
from .artifacts import reanalyse_run
from .backends.djev import DjevBackend
from .backends.jev import JevBackend
from .backends.kev import KevBackend
from .backends.laya import LayaBackend
from .dataset import (
    DEFAULT_SPLIT_PATH,
    Example,
    inspection,
    load_or_create_split,
    load_source_dataset,
    select_examples,
    validate_dataset,
    dataset_fingerprint,
    normalize_label,
    reference_audit,
)
from .evaluate import evaluate_examples
from .metrics import LABELS, error_analysis
from .compare import compare_runs

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)


@app.callback()
def initialize() -> None:
    """Load local credentials before any dataset or backend access."""
    load_dotenv()


@app.command()
def inspect(json_output: Annotated[Path | None, typer.Option("--json-output")] = None) -> None:
    """Inspect and validate the source dataset without calling a model."""
    dataset = load_source_dataset()
    stats = inspection(dataset)
    typer.echo(f"Dataset size: {stats['size']}")
    typer.echo(f"Columns: {', '.join(stats['columns'])}")
    _print_counts("Raw label counts", stats["label_counts_raw"])
    if stats["label_counts_normalized"] is not None:
        _print_counts("Normalized label counts", stats["label_counts_normalized"])
    _print_counts("Writing style counts", stats["writing_style_counts"])
    _print_counts("Topic counts", stats["topic_counts"])
    typer.echo("\nUnexpected/dirty labels")
    if not stats["dirty_labels"]:
        typer.echo("  None")
    for row in stats["dirty_labels"]:
        try:
            normalized = repr(normalize_label(row["label"]))
        except ValueError:
            normalized = "INVALID: no mapping"
        typer.echo(f"  row {row['id']}: {row['label']!r} -> {normalized}")
        typer.echo(f"    {row['review']}")
    if stats.get("validation_errors"):
        for error in stats["validation_errors"]:
            typer.echo(f"Validation error: {error}")
        raise typer.Exit(code=2)
    examples = validate_dataset(dataset)
    split = load_or_create_split(examples)
    typer.echo(
        f"\nFrozen split: dev={len(split.dev)}, eval={len(split.eval)}, "
        f"seed={split.seed}, stratification={split.strategy}"
    )
    typer.echo(f"Split file: {DEFAULT_SPLIT_PATH}")
    audit = reference_audit(examples, split)
    duplicates = audit["duplicates"]
    typer.echo(f"Duplicate review groups: {duplicates['group_count']}; cross-split: {duplicates['cross_split_group_count']}; conflicting labels: {duplicates['conflicting_label_group_count']}")
    if json_output is not None:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        typer.echo(f"Reference audit: {json_output}")


@app.command()
def demo(
    backend_name: Annotated[str, typer.Option("--backend", help="jev, djev, laya or kev")] = "jev",
    max_retries: Annotated[int, typer.Option(min=0)] = 3,
) -> None:
    """Run ten deterministic examples from the dev split."""
    examples, split = _load_validated()
    dev = select_examples(examples, split.dev)
    chosen = _demo_examples(dev, count=10)
    backend = _backend(backend_name, max_retries)
    display_name = backend.name.title()
    try:
        run_dir, predictions, failures, _ = evaluate_examples(
            chosen, split_name="demo", backend=backend, concurrency=1,
            evaluation_provenance=_provenance(examples, split, "dev", chosen),
        )
    finally:
        backend.close()
    by_id = {row["id"]: row for row in predictions}
    failed_by_id = {row["id"]: row for row in failures}
    for example in chosen:
        typer.echo("─" * 44)
        typer.echo(f'Review:\n"{example.review}"\n')
        typer.echo(f"Writing style: {example.writing_style}")
        typer.echo(f"Gold: {example.label}\n")
        if example.id in failed_by_id:
            typer.echo(f"{display_name} API failure: {failed_by_id[example.id]['error']}")
            continue
        row = by_id[example.id]
        typer.echo(f"{display_name}: {row['predicted']}")
        typer.echo(f"Confidence: {_number(row['confidence'])}")
        typer.echo(f"Latency: {row['latency_ms']:.1f} ms\n")
        typer.echo("✓ CORRECT" if row["correct"] else "✗ INCORRECT")
    typer.echo("─" * 44)
    typer.echo(f"Artifacts: {run_dir}/")


@app.command("eval")
def eval_command(
    backend_name: Annotated[str, typer.Option("--backend", help="jev, djev, laya or kev")] = "jev",
    limit: Annotated[int | None, typer.Option(min=1)] = None,
    concurrency: Annotated[int, typer.Option(min=1)] = 5,
    max_retries: Annotated[int, typer.Option(min=0)] = 3,
) -> None:
    """Evaluate only the frozen 20% holdout split."""
    _run("eval", backend_name, limit, concurrency, max_retries)


@app.command("dev-eval")
def dev_eval(
    backend_name: Annotated[str, typer.Option("--backend", help="jev, djev, laya or kev")] = "jev",
    limit: Annotated[int | None, typer.Option(min=1)] = None,
    concurrency: Annotated[int, typer.Option(min=1)] = 5,
    max_retries: Annotated[int, typer.Option(min=0)] = 3,
) -> None:
    """Evaluate the development split while iterating on prompts."""
    _run("dev", backend_name, limit, concurrency, max_retries)


@app.command()
def reanalyse(
    run: Path = typer.Argument(..., exists=True, file_okay=False),
) -> None:
    """Recompute metrics and HTML from saved predictions, with no API calls."""
    output = reanalyse_run(run)
    typer.echo(f"Reanalysis report: {output / 'report.html'}")


@app.command()
def compare(
    first_run: Path = typer.Argument(..., exists=True, file_okay=False),
    second_run: Path = typer.Argument(..., exists=True, file_okay=False),
    third_run: Path | None = typer.Argument(None, exists=True, file_okay=False),
) -> None:
    """Compare two or three eval runs from distinct backends or distinct kev model variants (e.g. kev-4b vs kev-9b); identical backend-and-model pairs are rejected."""
    try:
        output = compare_runs(first_run, second_run, third_run=third_run)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"Comparison report: {output / 'report.html'}")


def _backend(name: str, max_retries: int) -> SentimentBackend:
    normalized = name.strip().lower()
    if normalized == "jev":
        return JevBackend(max_retries=max_retries)
    if normalized == "djev":
        return DjevBackend(max_retries=max_retries)
    if normalized == "laya":
        return LayaBackend(max_retries=max_retries)
    if normalized == "kev":
        return KevBackend(max_retries=max_retries)
    raise typer.BadParameter("backend must be 'jev', 'djev', 'laya' or 'kev'", param_hint="--backend")


def _load_validated():
    dataset = load_source_dataset()
    examples = validate_dataset(dataset)
    return examples, load_or_create_split(examples)


def _provenance(examples, split, split_name, selected) -> dict:
    ids = split.eval if split_name == "eval" else split.dev
    return {
        "dataset_fingerprint": dataset_fingerprint(examples),
        "split_seed": split.seed, "stratification": split.strategy,
        "split_size": len(ids), "selected_size": len(selected),
        "full_split": {row.id for row in selected} == set(ids),
        "split_ids_sha256": hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest(),
        "reference_audit": reference_audit(examples, split),
    }


def _run(
    split_name: str,
    backend_name: str,
    limit: int | None,
    concurrency: int,
    max_retries: int,
) -> None:
    examples, split = _load_validated()
    indices = split.eval if split_name == "eval" else split.dev
    selected = select_examples(examples, indices)
    if limit is not None:
        selected = selected[:limit]
    backend = _backend(backend_name, max_retries)
    try:
        run_dir, predictions, failures, metrics = evaluate_examples(
            selected,
            split_name=split_name,
            backend=backend,
            concurrency=concurrency,
            evaluation_provenance=_provenance(examples, split, split_name, selected),
        )
    finally:
        backend.close()
    _print_report(metrics, predictions)
    if failures:
        typer.echo(f"\nAPI failure details: {run_dir / 'api_failures.jsonl'}")
    typer.echo(f"\nArtifacts: {run_dir}/")


def _print_report(metrics: dict, predictions: list[dict]) -> None:
    display_name = str(metrics["backend"]).upper()
    typer.echo(f"{display_name} — MOROCCAN DARIJA SENTIMENT")
    typer.echo("=" * 32)
    typer.echo("\nDataset")
    typer.echo(f"Requested:          {metrics['requested']:>5}")
    typer.echo(f"Successful:         {metrics['successful']:>5}")
    typer.echo(f"API failures:       {metrics['api_failures']:>5}")
    quality = metrics.get("quality")
    if not quality:
        typer.echo("\nNo successful predictions; quality metrics unavailable.")
        return
    typer.echo("\nQuality")
    typer.echo(f"Correct:            {quality['correct']:>5}")
    typer.echo(f"Incorrect:          {quality['incorrect']:>5}")
    typer.echo(f"Accuracy:           {quality['accuracy']:>7.1%}")
    interval = quality["accuracy_ci95"]
    typer.echo(f"95% accuracy CI:    {interval['low']:.1%}–{interval['high']:.1%}")
    baseline = quality["observed_majority_baseline"]
    typer.echo(f"Sample majority reference ({baseline['label']}): {baseline['accuracy']:.1%}")
    typer.echo(f"Macro precision:    {quality['macro_precision']:>7.1%}")
    typer.echo(f"Macro recall:       {quality['macro_recall']:>7.1%}")
    typer.echo(f"Macro F1:           {quality['macro_f1']:>7.1%}")
    typer.echo(f"Weighted F1:        {quality['weighted_f1']:>7.1%}")
    typer.echo("\nPer sentiment")
    typer.echo("Label       N  Precision  Recall     F1")
    for label in LABELS:
        row = quality["per_sentiment"][label]
        typer.echo(
            f"{label:<9} {row['support']:>3}  {row['precision']:>8.1%}  "
            f"{row['recall']:>6.1%}  {row['f1']:>6.1%}"
        )
    typer.echo("\nConfusion matrix")
    typer.echo("                 Predicted")
    typer.echo("              Pos   Neu   Neg")
    for name, values in zip(("Pos", "Neu", "Neg"), quality["confusion_matrix"]):
        typer.echo(f"Gold {name:<3}    {values[0]:>4}  {values[1]:>4}  {values[2]:>4}")
    typer.echo("\nBy writing style")
    for style, row in metrics["writing_style"].items():
        typer.echo(
            f"{style:<10} N={row['n']:<4} accuracy={row['accuracy']:.1%} "
            f"macro F1={row['macro_f1']:.1%}"
        )
    delta = metrics["arabizi_minus_arabic_accuracy"]
    if delta is not None:
        typer.echo(f"Arabizi - Arabic accuracy: {delta * 100:+.1f} pp")
    typer.echo(f"\nTopics (N >= {metrics['topic_min_n']})")
    for topic, row in metrics["topics"].items():
        typer.echo(f"{topic:<24} N={row['n']:<4} accuracy={row['accuracy']:.1%}")
    latency = metrics["latency_ms"]
    typer.echo("\nRecorded prediction latency (cached rows retain original measurements)")
    typer.echo(
        f"API mean/p50/p90/p95/p99: {latency['mean']:.1f} / {latency['p50']:.1f} / "
        f"{latency['p90']:.1f} / {latency['p95']:.1f} / {latency['p99']:.1f} ms"
    )
    typer.echo(f"Total wall clock: {metrics['wall_clock_seconds']:.2f} s")
    calibration = metrics.get("calibration")
    if calibration:
        typer.echo("\nCalibration")
        typer.echo(f"Average confidence:          {calibration['average_confidence']:.3f}")
        typer.echo(f"Confidence when correct:     {_number(calibration['confidence_correct'])}")
        typer.echo(f"Confidence when incorrect:   {_number(calibration['confidence_incorrect'])}")
        typer.echo(f"Log loss:                    {calibration['log_loss']:.4f}")
        typer.echo(f"Brier score:                 {calibration['brier_score']:.4f}")
        typer.echo(f"ECE:                         {calibration['ece']:.4f}")
        typer.echo("\nThreshold    Coverage    Accuracy    Errors")
        for row in calibration["thresholds"]:
            accuracy = "n/a" if row["accuracy"] is None else f"{row['accuracy']:.1%}"
            typer.echo(
                f">= {row['threshold']:.2f}      {row['coverage']:>7.1%}     "
                f"{accuracy:>7}    {row['errors']:>6}"
            )
    else:
        typer.echo(f"\nWARNING: {metrics['warning']}")
    _print_errors(predictions, str(metrics["backend"]).title())


def _print_errors(predictions: list[dict], display_name: str) -> None:
    analysis = error_analysis(predictions)
    typer.echo("\nHIGH-CONFIDENCE ERRORS")
    if not analysis["highest_confidence_errors"]:
        typer.echo("No errors with available confidence.")
    for index, row in enumerate(analysis["highest_confidence_errors"], 1):
        typer.echo(
            f"\n{index}.\nReview: {row['text']}\nStyle: {row['writing_style']}\n"
            f"Gold: {row['gold']}\n{display_name}: {row['predicted']}\n"
            f"Confidence: {_number(row['confidence'])}"
        )
    typer.echo("\nLOWEST-CONFIDENCE CORRECT")
    for index, row in enumerate(analysis["lowest_confidence_correct"], 1):
        typer.echo(
            f"{index}. [{row['writing_style']}] {_number(row['confidence'])} — {row['text']}"
        )
    typer.echo("\nERROR COUNTS BY GOLD → PREDICTED")
    for transition, rows in analysis["errors_by_transition"].items():
        typer.echo(f"{transition}: {len(rows)}")
        for row in sorted(rows, key=lambda item: item.get("confidence") or -1, reverse=True)[:3]:
            typer.echo(f"  [{row['id']}] {_number(row['confidence'])} — {row['text']}")
    _print_style_errors("ARABIZI ERRORS", analysis["arabizi_errors"])
    _print_style_errors("ARABIC-SCRIPT ERRORS", analysis["arabic_errors"])


def _print_style_errors(title: str, rows: list[dict]) -> None:
    typer.echo(f"\n{title} ({len(rows)})")
    for row in sorted(rows, key=lambda item: item.get("confidence") or -1, reverse=True)[:5]:
        typer.echo(
            f"[{row['id']}] {row['gold']} → {row['predicted']}, "
            f"{_number(row['confidence'])} — {row['text']}"
        )


def _demo_examples(examples: list[Example], count: int) -> list[Example]:
    groups: dict[tuple[str, str], list[Example]] = defaultdict(list)
    for example in examples:
        groups[(example.label, example.writing_style)].append(example)
    randomizer = random.Random(42)
    for rows in groups.values():
        randomizer.shuffle(rows)
    chosen: list[Example] = []
    keys = sorted(groups)
    while len(chosen) < count and any(groups.values()):
        for key in keys:
            if groups[key] and len(chosen) < count:
                chosen.append(groups[key].pop())
    return chosen


def _print_counts(title: str, counts: dict) -> None:
    typer.echo(f"\n{title}")
    for key, count in sorted(counts.items(), key=lambda item: (-item[1], str(item[0]))):
        typer.echo(f"  {key!r}: {count}")


def _number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    app()
