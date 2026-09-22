# Evaluation reference and review protocol

The question is agreement with the sentiment annotations in `ohidaoui/darija-reviews`
under one shared, versioned, zero-shot three-class schema. It is not proof of production
accuracy or human-level Darija understanding. Dataset annotations are the reference;
their independent adjudication has not been established by this project.

## Preserve the evidence

- Keep source rows, reference labels and the seed-42 dev/eval membership unchanged.
- Normalize only the explicitly documented `negative ` alias. Unknown or null labels
  stop prediction after raw dataset statistics are shown.
- Record source/selection fingerprints, selected IDs, schema content hash, requested
  and resolved model IDs, concurrency, failure counts and cached status.
- Recompute reports from raw predictions with `darija-eval reanalyse`. Preserve the
  original run, raw responses and timings. Historical metadata cannot be invented.
- Compare matched ID/text/reference/style/topic/schema records; never silently drop
  failed or unmatched examples to improve a score. Failed API calls are operational
  failures, not sentiment errors.
- Validate available manifests and provenance against saved evidence before analysis:
  selected IDs must account for successes and failures, and the selection fingerprint
  must match their text/reference/style/topic records in the original selection order.
  A comparison must not attach a local frozen-split identity to conflicting recorded
  dataset or split fingerprints. These checks detect inconsistency, not independently
  authenticated provenance; legacy runs may lack the metadata needed for them.

## Review reference labels independently

For a future annotation-quality study, have two fluent Darija reviewers independently
label reviews with model names, predictions, confidence and existing labels hidden.
Include Arabic and Arabizi and a random sample across all sentiments/topics; do not
select only model errors. Record positive/neutral/negative, an ambiguity flag, and a
brief rationale. Use a third reviewer to adjudicate disagreements. Preserve original
labels alongside any separately versioned adjudicated reference and report agreement
and changes. Do not silently replace the current benchmark's ground truth.

Before review, agree how to handle mixed opinions, unclear targets, recommendations,
factual questions, sarcasm, elongated words and code switching. Schema v2 reserves
neutral for factual statements, balanced opinions and questions without an expressed
attitude; unfamiliar language alone is not neutral. Schema v1 included "unclear"
in its neutral criterion. A reference whose rules differ measures task alignment as
well as language understanding. This protocol is proposed future work; no human
adjudication has been performed here.

## Interpret measured differences

Report class support, confusion matrices and macro F1 alongside accuracy. Compare
accuracy to the observed majority-label reference, clearly marked as descriptive.
Subgroup sample sizes can be small even above the topic reporting threshold of 10.
Arabic and Arabizi are different reviews with different topic/class mixtures, so
their accuracy difference is observational, not a controlled transliteration test.

Wilson intervals describe uncertainty in a proportion. The comparison's paired
bootstrap resamples the same review indices for both models within sentiment/style
strata, preserving this sample composition. Exact McNemar uses only cases where one
model is right and the other wrong. All assume review independence; duplicates or
shared authors/products can weaken that assumption. Unknown source clustering is a
remaining limitation. Neither method corrects label noise or multiple exploratory
subgroup comparisons. Methods: [bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html),
[binomial test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html).

Confidence is the reported probability of the predicted class. Show threshold support
and intervals as well as coverage; select operating thresholds on dev, then freeze
them before a new independent test. The currently inspected eval set must not become
a prompt/threshold optimization loop. Further confirmatory claims need new independent
data collected without consulting model predictions.

Reports include a predicted-class reliability diagram with ten equal-width bins
(not separate class-wise calibration). Empty bins have no plotted point. Selective
risk is the error rate among accepted predictions. Risk–coverage endpoints accept
whole equal-confidence groups, making the result invariant to row ordering.
Reported AURC uses grouped right-step integration: sum of each increase in successful
coverage times its endpoint risk. This is not arbitrary per-rank tie breaking.
Accuracy at 50% and 80% coverage uses the first endpoint reaching the target and
discloses actual coverage, which may overshoot. Requested coverage also counts API
failures in the denominator. These are descriptive statistics, not fitted thresholds.

New inference manifests embed the reference audit computed from the validated source
dataset and frozen split. Duplicate counts describe the full source dataset, not a
limited run's subset. Historical reports without this audit do not invent counts or
claim independently confirmed annotation errors.

Timing includes client/network/queue/retry overhead and may mix warm/cold observations.
Cached rows retain the latency of their original request. Do not infer throughput
from the wall-clock time of a cached run or claim a compute-speed winner from endpoints
with different infrastructure. Laya's server inference timing is a separate measure.
