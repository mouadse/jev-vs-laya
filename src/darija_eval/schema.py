import hashlib
import json
from collections.abc import Sequence


SCHEMA_VERSION = "v2"

SENTIMENT_INSTRUCTIONS = (
    "What is the reviewer's sentiment toward the product or service in `review`, "
    "a Moroccan Darija review? Judge the overall attitude, not isolated words. "
    "Unfamiliar language alone does not make the sentiment neutral."
)

SENTIMENT_CRITERIA = {
    "positive": "Praise, satisfaction, approval, or recommendation.",
    "neutral": "Factual statements, balanced positive and negative opinions, or questions without an expressed attitude.",
    "negative": "Criticism, dissatisfaction, complaints, or disappointment."
}


def sentiment_question_dict(option_order: Sequence[str] | None = None) -> dict:
    order = tuple(SENTIMENT_CRITERIA if option_order is None else option_order)
    if len(order) != len(SENTIMENT_CRITERIA) or set(order) != set(SENTIMENT_CRITERIA):
        raise ValueError("option_order must contain every sentiment label exactly once")
    return {
        "type": "choice",
        "instructions": SENTIMENT_INSTRUCTIONS,
        "criteria": {label: SENTIMENT_CRITERIA[label] for label in order},
    }


def configuration_fingerprint(configuration: dict) -> str:
    """Canonical JSON objects; sequence order remains behaviorally significant."""
    payload = json.dumps(
        configuration, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def question_fingerprint(question: dict) -> str:
    """Encode choice positions explicitly before canonicalizing object keys."""
    return configuration_fingerprint({
        "identity_version": 2,
        "question": {**question, "criteria": list(question["criteria"].items())},
    })
