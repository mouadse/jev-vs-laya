from __future__ import annotations

import json

import pytest
from datasets import Dataset

from darija_eval.dataset import (
    Example,
    generate_split,
    inspection,
    load_or_create_split,
    normalize_label,
    select_examples,
    validate_dataset,
)


def examples() -> list[Example]:
    rows = []
    index = 0
    for label in ("positive", "neutral", "negative"):
        for style in ("Arabic", "Arabizi"):
            for number in range(10):
                rows.append(Example(index, f"review-{index}", label, style, "it"))
                index += 1
    return rows


def test_split_is_deterministic_and_jointly_stratified() -> None:
    first = generate_split(examples())
    second = generate_split(examples())
    assert first == second
    assert first.strategy == "sentiment+writing_style"
    assert len(first.dev) == 48
    assert len(first.eval) == 12
    assert set(first.dev).isdisjoint(first.eval)


def test_split_falls_back_to_sentiment() -> None:
    rows = [Example(i, str(i), "positive" if i < 10 else "negative", "rare" if i == 0 else "Arabic", "it") for i in range(20)]
    split = generate_split(rows)
    assert split.strategy == "sentiment"


def test_split_persists_and_rejects_changed_dataset(tmp_path) -> None:
    path = tmp_path / "split.json"
    created = load_or_create_split(examples(), path)
    loaded = load_or_create_split(examples(), path)
    assert loaded == created
    changed = examples()
    changed[0] = Example(0, "changed", "positive", "Arabic", "it")
    with pytest.raises(ValueError, match="does not match"):
        load_or_create_split(changed, path)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("positive", "positive"), ("neutral", "neutral"), ("negative ", "negative")],
)
def test_label_normalization(raw, expected) -> None:
    assert normalize_label(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "Negative", "other", " neutral "])
def test_unknown_or_null_labels_are_rejected(raw) -> None:
    with pytest.raises(ValueError):
        normalize_label(raw)


def test_dataset_validation_reports_row() -> None:
    dataset = Dataset.from_list(
        [{"review": "mzyan", "label": "other", "topic": "it", "writing_style": "Arabizi"}]
    )
    with pytest.raises(ValueError, match="row 0"):
        validate_dataset(dataset)



def test_persisted_duplicate_indices_rejected(tmp_path):
    path = tmp_path / "split.json"
    load_or_create_split(examples(), path)
    payload = json.loads(path.read_text())
    payload["dev"].append(payload["dev"][0])
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="indices"):
        load_or_create_split(examples(), path)


def test_inspection_preserves_raw_stats_for_unknown_and_null_labels():
    dataset = Dataset.from_list([
        {"review": "a", "label": label, "writing_style": "Arabic", "topic": "it"}
        for label in ["positive", "negative ", "other", None]
    ])
    stats = inspection(dataset)
    assert stats["size"] == 4
    assert stats["label_counts_raw"] == {"positive": 1, "negative ": 1, "other": 1, None: 1}
    assert stats["label_counts_normalized"] is None
    assert len(stats["validation_errors"]) == 2
    assert stats["validation_errors"][0].startswith("row 2:")
    assert len(stats["dirty_labels"]) == 3


def test_inspection_explicit_known_alias_normalization():
    dataset = Dataset.from_list([
        {"review": "a", "label": "negative ", "writing_style": "Arabic", "topic": "it"}
    ])
    stats = inspection(dataset)
    assert stats["label_counts_normalized"] == {"negative": 1}
    assert stats["validation_errors"] == []


def test_joint_stratification_uses_ceil_for_eval_count():
    rows = [Example(i, str(i), ("positive", "neutral", "negative")[i % 3], "Arabic", "it") for i in range(11)]
    split = generate_split(rows)
    assert split.strategy == "sentiment+writing_style"
    assert len(split.eval) == 3


def test_stratification_requires_enough_dev_examples():
    from darija_eval.dataset import _can_joint_stratify
    assert not _can_joint_stratify(["a", "a", "b", "b"], 3)


def test_reference_audit_reports_conflicts_overlap_and_counts():
    from darija_eval.dataset import FrozenSplit, dataset_fingerprint, reference_audit
    rows = [
        Example(10, "same", "positive", "Arabic", "it"),
        Example(11, " same ", "negative", "Arabizi", "it"),
        Example(12, "unique", "neutral", "Arabic", "food"),
    ]
    split = FrozenSplit((0, 2), (1,), "sentiment", 42, dataset_fingerprint(rows))
    audit = reference_audit(rows, split)
    assert audit["overall"]["n"] == 3
    assert audit["split"]["eval"]["sentiment"] == {"negative": 1}
    assert audit["duplicates"]["group_count"] == 1
    assert audit["duplicates"]["rows_in_groups"] == 2
    assert audit["duplicates"]["groups"][0]["ids"] == [10, 11]
    assert audit["duplicates"]["conflicting_label_group_count"] == 1
    assert audit["duplicates"]["cross_split_group_count"] == 1
    assert audit["reference_labels"]["explicit_aliases"] == {"negative ": "negative"}
    assert "unique" not in json.dumps(audit)


def test_reference_audit_rejects_wrong_split_fingerprint():
    from darija_eval.dataset import FrozenSplit, reference_audit
    with pytest.raises(ValueError, match="does not match"):
        reference_audit(examples(), FrozenSplit((0,), (1,), "sentiment", 42, "wrong"))

def test_select_examples_rejects_negative_index_silently_wrapping() -> None:
    rows = examples()
    with pytest.raises(ValueError, match="out of range"):
        select_examples(rows, [-1])


def test_select_examples_rejects_out_of_range_and_non_int() -> None:
    rows = examples()
    with pytest.raises(ValueError, match="out of range"):
        select_examples(rows, [len(rows)])
    with pytest.raises(ValueError, match="invalid"):
        select_examples(rows, [True])
    with pytest.raises(ValueError, match="invalid"):
        select_examples(rows, ["0"])


def test_select_examples_accepts_valid_subset() -> None:
    rows = examples()
    assert [example.id for example in select_examples(rows, [0, 2])] == [0, 2]


def test_persisted_split_with_unknown_strategy_rejected(tmp_path):
    path = tmp_path / "split.json"
    load_or_create_split(examples(), path)
    payload = json.loads(path.read_text())
    payload["strategy"] = "coin-flip"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="strategy"):
        load_or_create_split(examples(), path)


def test_persisted_split_missing_key_rejected(tmp_path):
    path = tmp_path / "split.json"
    load_or_create_split(examples(), path)
    payload = json.loads(path.read_text())
    del payload["dev"]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="invalid"):
        load_or_create_split(examples(), path)


def test_persisted_split_non_object_or_non_array_rejected(tmp_path):
    path = tmp_path / "split.json"
    path.write_text(json.dumps([1, 2, 3]))
    with pytest.raises(ValueError, match="invalid"):
        load_or_create_split(examples(), path)
    load_or_create_split(examples(), tmp_path / "other.json")
    payload = json.loads((tmp_path / "other.json").read_text())
    payload["dev"] = "not-an-array"
    (tmp_path / "other.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="invalid"):
        load_or_create_split(examples(), tmp_path / "other.json")


def test_validate_dataset_rejects_empty() -> None:
    dataset = Dataset.from_dict({"review": [], "label": [], "topic": [], "writing_style": []})
    with pytest.raises(ValueError, match="empty"):
        validate_dataset(dataset)
