SCHEMA_VERSION = "v1"

SENTIMENT_INSTRUCTIONS = (
    "Classify the sentiment expressed in this Moroccan Darija review. "
    "Judge the writer's attitude toward the product or service, not merely "
    "whether individual words sound positive or negative."
)

SENTIMENT_CRITERIA = {
    "positive": "The reviewer expresses a favorable, satisfied, approving, or recommending opinion.",
    "neutral": "The reviewer is primarily factual, mixed, unclear, or expresses neither clearly positive nor clearly negative sentiment.",
    "negative": "The reviewer expresses dissatisfaction, criticism, disappointment, rejection, or a clearly unfavorable opinion.",
}


def sentiment_question_dict() -> dict:
    return {
        "type": "choice",
        "instructions": SENTIMENT_INSTRUCTIONS,
        "criteria": dict(SENTIMENT_CRITERIA),
    }

