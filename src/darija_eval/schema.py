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


def sentiment_question_dict() -> dict:
    return {
        "type": "choice",
        "instructions": SENTIMENT_INSTRUCTIONS,
        "criteria": dict(SENTIMENT_CRITERIA),
    }

