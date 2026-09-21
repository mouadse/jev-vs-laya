"""Paired uncertainty conditional on the observed sentiment/style strata."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .metrics import LABELS


def paired_statistics(pairs: list[tuple[dict, dict]], *, seed: int = 42, resamples: int = 5000) -> dict:
    if not pairs:
        raise ValueError("paired statistics require at least one example")
    groups = defaultdict(list)
    for index, (left, _) in enumerate(pairs):
        groups[(left["gold"], left["writing_style"])].append(index)
    rng = np.random.default_rng(seed)
    indices = np.concatenate([
        rng.choice(group, size=(resamples, len(group)), replace=True)
        for group in groups.values()
    ], axis=1)
    gold = np.array([LABELS.index(left["gold"]) for left, _ in pairs])
    predictions = [np.array([LABELS.index(pair[i]["predicted"]) for pair in pairs]) for i in (0, 1)]
    accuracy = [(pred[indices] == gold[indices]).mean(axis=1) for pred in predictions]
    f1 = []
    for pred in predictions:
        class_f1 = []
        for label in range(len(LABELS)):
            actual, guessed = gold[indices] == label, pred[indices] == label
            numerator = 2 * (actual & guessed).sum(axis=1)
            denominator = actual.sum(axis=1) + guessed.sum(axis=1)
            class_f1.append(np.divide(numerator, denominator, out=np.zeros(resamples), where=denominator > 0))
        f1.append(np.mean(class_f1, axis=0))
    correct = [pred == gold for pred in predictions]
    jev_only = int((correct[0] & ~correct[1]).sum())
    laya_only = int((~correct[0] & correct[1]).sum())
    discordant = jev_only + laya_only
    pvalue = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(jev_only, laya_only) + 1)) / 2**discordant) if discordant else 1.0
    return {
        "direction": "Laya minus Jev",
        "accuracy_delta_ci95": np.quantile(accuracy[1] - accuracy[0], [0.025, 0.975]).tolist(),
        "macro_f1_delta_ci95": np.quantile(f1[1] - f1[0], [0.025, 0.975]).tolist(),
        "bootstrap": {"method": "paired percentile, stratified by gold sentiment and writing style", "seed": seed, "resamples": resamples},
        "mcnemar_exact": {"p_value": pvalue, "jev_only_correct": jev_only, "laya_only_correct": laya_only, "discordant_n": discordant},
        "limitations": "Exploratory intervals assume independent reviews within strata. They do not include label uncertainty, model stochasticity, or dataset shift. Subgroup comparisons are descriptive, not causal.",
    }
