from darija_eval.statistics import paired_statistics


def test_paired_statistics_known_exact_test_and_interval():
    left = {"gold": "positive", "predicted": "positive", "writing_style": "Arabic"}
    right = {**left, "predicted": "negative"}
    result = paired_statistics([(left, right)] * 4, resamples=100)
    assert result["mcnemar_exact"]["p_value"] == 0.125
    assert result["accuracy_delta_ci95"] == [-1.0, -1.0]


def test_paired_statistics_identical_predictions():
    row = {"gold": "positive", "predicted": "neutral", "writing_style": "Arabizi"}
    result = paired_statistics([(row, row)] * 5, resamples=100)
    assert result["mcnemar_exact"]["p_value"] == 1.0
    assert result["macro_f1_delta_ci95"] == [0.0, 0.0]
