from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split

DATASET_NAME = "ohidaoui/darija-reviews"
DATASET_CONFIG = "default"
SOURCE_SPLIT = "test"
EXPECTED_COLUMNS = {"review", "label", "topic", "writing_style"}
LABELS = ("positive", "neutral", "negative")
LABEL_ALIASES = {"negative ": "negative"}
SPLIT_SEED = 42
DEFAULT_SPLIT_PATH = Path("data/splits/seed_42.json")


@dataclass(frozen=True)
class Example:
    id: int
    review: str
    label: str
    writing_style: str
    topic: str


@dataclass(frozen=True)
class FrozenSplit:
    dev: tuple[int, ...]
    eval: tuple[int, ...]
    strategy: str
    seed: int
    dataset_fingerprint: str


def load_source_dataset() -> Dataset:
    return load_dataset(DATASET_NAME, DATASET_CONFIG, split=SOURCE_SPLIT)


def normalize_label(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"invalid sentiment label: {value!r}")
    if value in LABELS:
        return value
    if value in LABEL_ALIASES:
        return LABEL_ALIASES[value]
    raise ValueError(f"unknown sentiment label: {value!r}")


def validate_dataset(dataset: Dataset) -> list[Example]:
    missing = EXPECTED_COLUMNS.difference(dataset.column_names)
    if missing:
        raise ValueError(f"dataset is missing columns: {sorted(missing)}")
    if len(dataset) == 0:
        raise ValueError("dataset is empty")
    examples: list[Example] = []
    for index, row in enumerate(dataset):
        review = row["review"]
        style = row["writing_style"]
        topic = row["topic"]
        if not isinstance(review, str) or not review.strip():
            raise ValueError(f"row {index} has an empty review")
        if not isinstance(style, str) or not style.strip():
            raise ValueError(f"row {index} has an invalid writing_style")
        if not isinstance(topic, str) or not topic.strip():
            raise ValueError(f"row {index} has an invalid topic")
        try:
            label = normalize_label(row["label"])
        except ValueError as error:
            raise ValueError(f"row {index}: {error}") from error
        examples.append(Example(index, review, label, style, topic))
    return examples


def dataset_fingerprint(examples: Sequence[Example]) -> str:
    digest = hashlib.sha256()
    for example in examples:
        digest.update(json.dumps(asdict(example), ensure_ascii=False, sort_keys=True).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _can_joint_stratify(strata: Sequence[str], eval_size: int) -> bool:
    counts = Counter(strata)
    return (
        min(counts.values(), default=0) >= 2
        and len(counts) <= eval_size
        and len(counts) <= len(strata) - eval_size
    )


def generate_split(examples: Sequence[Example], seed: int = SPLIT_SEED) -> FrozenSplit:
    indices = list(range(len(examples)))
    eval_size = math.ceil(len(indices) * 0.2)
    joint = [f"{row.label}\0{row.writing_style}" for row in examples]
    if _can_joint_stratify(joint, eval_size):
        strata = joint
        strategy = "sentiment+writing_style"
    else:
        strata = [row.label for row in examples]
        strategy = "sentiment"
    dev, evaluation = train_test_split(
        indices, test_size=0.2, random_state=seed, stratify=strata
    )
    return FrozenSplit(
        dev=tuple(sorted(dev)),
        eval=tuple(sorted(evaluation)),
        strategy=strategy,
        seed=seed,
        dataset_fingerprint=dataset_fingerprint(examples),
    )

def load_or_create_split(
    examples: Sequence[Example], path: Path = DEFAULT_SPLIT_PATH
) -> FrozenSplit:
    fingerprint = dataset_fingerprint(examples)
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"persisted split file is invalid: {error}") from error
        if not isinstance(payload, dict):
            raise ValueError("persisted split file is invalid: expected a JSON object")
        try:
            dev_raw, eval_raw = payload["dev"], payload["eval"]
            if not isinstance(dev_raw, (list, tuple)) or not isinstance(eval_raw, (list, tuple)):
                raise TypeError("split 'dev' and 'eval' must be JSON arrays")
            split = FrozenSplit(
                dev=tuple(dev_raw),
                eval=tuple(eval_raw),
                strategy=payload["strategy"],
                seed=payload["seed"],
                dataset_fingerprint=payload["dataset_fingerprint"],
            )
        except KeyError as error:
            raise ValueError(f"persisted split file is invalid: missing key {error}") from error
        except TypeError as error:
            raise ValueError(f"persisted split file is invalid: {error}") from error
        _validate_frozen_split(split, len(examples), fingerprint)
        return split
    split = generate_split(examples)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, json.dumps(asdict(split), indent=2) + "\n")
    return split


def _validate_frozen_split(split: FrozenSplit, size: int, fingerprint: str) -> None:
    if split.dataset_fingerprint != fingerprint:
        raise ValueError("persisted split does not match the current dataset")
    if split.strategy not in ("sentiment", "sentiment+writing_style"):
        raise ValueError(f"persisted split uses unexpected strategy {split.strategy!r}")
    if any(type(index) is not int for index in (*split.dev, *split.eval)):
        raise ValueError("persisted split indices are invalid")
    dev, evaluation = set(split.dev), set(split.eval)
    if (
        len(dev) != len(split.dev)
        or len(evaluation) != len(split.eval)
        or not dev or not evaluation
        or dev & evaluation or dev | evaluation != set(range(size))
    ):
        raise ValueError("persisted split indices are invalid")
    if split.seed != SPLIT_SEED:
        raise ValueError(f"persisted split uses unexpected seed {split.seed}")


def select_examples(examples: Sequence[Example], indices: Iterable[int]) -> list[Example]:
    selected = []
    for index in indices:
        if type(index) is not int:
            raise ValueError(f"selection index is invalid: {index!r}")
        if not 0 <= index < len(examples):
            raise ValueError(f"selection index out of range: {index!r}")
        selected.append(examples[index])
    return selected


def inspection(dataset: Dataset) -> dict[str, Any]:
    dirty = []
    errors = []
    normalized = Counter()
    missing = EXPECTED_COLUMNS.difference(dataset.column_names)
    if missing:
        errors.append(f"dataset is missing columns: {sorted(missing)}")
    labels = dataset["label"] if "label" in dataset.column_names else []
    for index, value in enumerate(labels):
        if value not in LABELS:
            dirty.append({"id": index, "label": value, "review": dataset[index].get("review")})
        try:
            normalized[normalize_label(value)] += 1
        except ValueError as error:
            errors.append(f"row {index}: {error}")
    return {
        "size": len(dataset),
        "columns": list(dataset.column_names),
        "label_counts_raw": dict(Counter(labels)),
        "label_counts_normalized": None if errors else dict(normalized),
        "writing_style_counts": dict(Counter(dataset["writing_style"])) if "writing_style" in dataset.column_names else {},
        "topic_counts": dict(Counter(dataset["topic"])) if "topic" in dataset.column_names else {},
        "dirty_labels": dirty,
        "validation_errors": errors,
    }


def reference_audit(examples: Sequence[Example], split: FrozenSplit) -> dict[str, Any]:
    """Audit dataset labels and split integrity without changing either."""
    fingerprint = dataset_fingerprint(examples)
    _validate_frozen_split(split, len(examples), fingerprint)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        groups[example.review.strip()].append(index)
    dev_indices, eval_indices = set(split.dev), set(split.eval)
    duplicate_groups = []
    for indices in groups.values():
        if len(indices) < 2:
            continue
        labels = sorted({examples[index].label for index in indices})
        duplicate_groups.append({
            "ids": [examples[index].id for index in indices],
            "labels": labels,
            "conflicting_labels": len(labels) > 1,
            "cross_split": bool(dev_indices.intersection(indices) and eval_indices.intersection(indices)),
        })

    def counts(rows: Sequence[Example]) -> dict[str, Any]:
        return {
            "n": len(rows),
            "sentiment": dict(sorted(Counter(row.label for row in rows).items())),
            "writing_style": dict(sorted(Counter(row.writing_style for row in rows).items())),
            "topic": dict(sorted(Counter(row.topic for row in rows).items())),
        }

    return {
        "format_version": 1,
        "dataset": DATASET_NAME,
        "source_split": SOURCE_SPLIT,
        "dataset_fingerprint": fingerprint,
        "reference_labels": {
            "source": "Dataset-provided sentiment labels; not independently adjudicated.",
            "classes": list(LABELS),
            "explicit_aliases": dict(LABEL_ALIASES),
            "unknown_label_policy": "Reject; never infer or silently map unknown labels.",
        },
        "overall": counts(examples),
        "split": {
            "seed": split.seed,
            "stratification": split.strategy,
            "dev": counts(select_examples(examples, split.dev)),
            "eval": counts(select_examples(examples, split.eval)),
        },
        "duplicates": {
            "matching_rule": "Exact review equality after stripping leading/trailing whitespace; no fuzzy matching.",
            "groups": duplicate_groups,
            "group_count": len(duplicate_groups),
            "rows_in_groups": sum(len(group["ids"]) for group in duplicate_groups),
            "conflicting_label_group_count": sum(group["conflicting_labels"] for group in duplicate_groups),
            "cross_split_group_count": sum(group["cross_split"] for group in duplicate_groups),
        },
    }


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
