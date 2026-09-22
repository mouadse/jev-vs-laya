# Improving the Darija sentiment system

**Research date:** 22 September 2026. **Scope:** the current Jev/Laya benchmark, informed by implementation-level reviews of KEV and Laya. This is a proposal, not a claim that new models have been trained or score gains achieved.

## Recommendation

Keep Jev as the reference to beat. Add KEV-4B as a third backend, measure its unmodified Darija performance, then warm-start a small, supervised Darija adaptation. In parallel, test a bounded set of schema and input variants on development data. Treat multilingual Laya as a lower-cost challenger that needs domain adaptation, rather than expecting an SDK upgrade or calibration alone to close its current quality gap.

The most promising investment is **better domain supervision for neutral sentiment and Arabizi**, followed by controlled training experiments. The repositories provide useful mechanisms; neither establishes a Darija improvement for this project.

For this report, “better score” means **higher three-class macro F1**, with accuracy, per-class recall, Arabic/Arabizi results, probability quality, and serving cost reported alongside it. Macro F1 prevents the positive majority from dominating the decision. Keep the existing zero-shot comparison as one track and label supervised/adapted systems as a separate track.

## 1. What the current evidence actually says

### Verified baseline

The following results were recomputed from saved prediction records using the project's validators and comparison functions, rather than copied from report summaries:

- [Jev v2 evidence](results/jev_eval_20260921_234315_802617/predictions.jsonl), resolved model `jev-1.13.0`.
- [Laya v2 evidence](results/laya_eval_20260921_233519_938576/predictions.jsonl), multilingual weights `1c5edc17a7acd8701df6fc341c0d179f1c62c982`, runtime `laya==0.3.4`.
- Both cover all **171 frozen eval rows**, with no unaccounted examples or API failures. IDs, text, reference labels, style, topic, schema fingerprint, and recorded dataset/split provenance agree. The [frozen partition](data/splits/seed_42.json) contains 680 development rows and 171 evaluation rows.

| Measure | Jev | Laya |
| --- | ---: | ---: |
| Accuracy | **79.53% (136/171)** | 53.80% (92/171) |
| Accuracy, 95% Wilson interval | 72.87–84.90% | 46.33–61.11% |
| Macro F1 | **0.7183** | 0.5000 |
| Positive F1, support 92 | 0.8791 | 0.6667 |
| Neutral F1, support 25 | 0.5000 | 0.3333 |
| Negative F1, support 54 | 0.7759 | 0.5000 |
| Arabic accuracy, support 127 | 85.04% | 60.63% |
| Arabizi accuracy, support 44 | 63.64% | 34.09% |
| ECE, ten bins; lower is better | 0.0408 | 0.2275 |
| Multiclass Brier, range 0–2; lower is better | 0.3198 | 0.6199 |

Jev's paired advantage is **25.73 percentage points of accuracy**, with a 95% stratified bootstrap interval of **17.54–33.92 points**. Its macro-F1 advantage is **0.2183**, interval **0.1185–0.3129**. These are exploratory, dataset-conditional intervals from 5,000 paired resamples, not guarantees under new labels, repeated model sampling, or distribution shift. The observed positive-majority reference is 53.80% accuracy; Laya equals that accuracy while producing a different mix of labels.

### Where improvement is needed

1. **Neutral interpretation:** Jev recovers only 11 of 25 neutral references. The other 14 split equally between positive and negative. Laya has a different failure: it predicts neutral 53 times, only 13 correctly—24.53% precision. One shared “increase neutral” rule would therefore be inappropriate.
2. **Arabizi:** Jev makes 16 errors among 44 Arabizi reviews, versus 19 among 127 Arabic reviews. This identifies a useful investigation area, but the two groups contain different reviews and class/topic mixtures. It does not prove transliteration will fix the gap.
3. **Complementarity is limited:** 81 examples are correct for both models, 55 only for Jev, 11 only for Laya, and 24 for neither. A label-informed oracle selecting between these exact outputs reaches 147/171, or 85.96%. That is a ceiling for selecting between these two frozen predictions, **not an achievable ensemble result**. Blind averaging can lose Jev's 55 unique wins.

Sources: [metric computation](src/darija_eval/metrics.py), [validated reanalysis](src/darija_eval/artifacts.py), [paired comparison](src/darija_eval/compare.py), and the prediction records linked above.

### Limits that affect the next experiment

The eval set has already been inspected repeatedly. Preserve it as historical regression evidence; obtain a new independent confirmation set before claiming that changes discovered through this analysis generalize. The best earlier v1 Jev run was 80.12% accuracy / 0.7241 macro F1, while the current v2 run is 79.53% / 0.7183. Do not claim that v2 established a gain, or select historical runs opportunistically.

These are agreement scores with the supplied annotations. Independent human adjudication is not established. Embedded source audits report five exact duplicate groups, no conflicting labels, and no cross-dev/eval duplicate groups; this review checked that the selected eval texts are unique, but did not redownload and reconstruct the full source audit or investigate fuzzy/author/product overlap.

All 171 records in the selected Jev run are cached; Laya's selected run records fresh calls. Jev's cached wall time is not a current speed measurement. Existing latency also mixes different infrastructure, network, queueing, and possibly retries. See the [evaluation protocol](docs/evaluation.md).

## 2. What to take from each repository

The source snapshots reviewed were:

| Repository | Pinned source revision | Important distinction |
| --- | --- | --- |
| [KEV](https://github.com/jaredpalmer/kev/tree/90990a5fac2995b9faa3190f7d437e84f2067768) | `90990a5fac2995b9faa3190f7d437e84f2067768` | A separate Qwen-based decision model; not hosted Jev or a Laya checkpoint. |
| [Laya](https://github.com/NandhaKishorM/laya/tree/573e5b62696ba441230cd6be71d593331b5d23af) | `573e5b62696ba441230cd6be71d593331b5d23af` | Source declares 0.3.5; the current deployment pins 0.3.4. Main-branch features are not verified deployed behavior. |

### KEV: the stronger starting point for a trainable challenger

KEV combines a frozen Qwen backbone with a trainable LoRA adapter and pointer head that scores the available options. Its current family includes Qwen3.5-based 0.8B, 4B, and 9B models. The useful feature here is a real custom-data training path: `--init_from` restores both the existing adapter and decision head, with compatibility checks, so domain adaptation does not start from a randomly initialized decision head. Sources: [model implementation](https://github.com/jaredpalmer/kev/blob/90990a5fac2995b9faa3190f7d437e84f2067768/kev/model.py), [checkpoint loading](https://github.com/jaredpalmer/kev/blob/90990a5fac2995b9faa3190f7d437e84f2067768/kev/checkpoint.py).

Its TypeSafe-shaped HTTP API makes it a plausible third backend behind this project's existing `Prediction` contract. Start with 4B as an engineering choice; it is not a demonstrated Darija winner. Upstream transfer scores use different tasks and cannot be compared numerically with our 171 reviews. Sources: [server](https://github.com/jaredpalmer/kev/blob/90990a5fac2995b9faa3190f7d437e84f2067768/kev/serve.py), [4B model card](https://github.com/jaredpalmer/kev/blob/90990a5fac2995b9faa3190f7d437e84f2067768/docs/model-cards/kev-4b.md).

Three training details matter:

- Shuffle label positions without changing their meaning. KEV supports this, but also defaults to adding distractors/none-of-the-above variants. Disable those option-set changes for the initial strict three-class sentiment experiment.
- Borrow its **contrastive-data method**: one meaningful change flips the answer, irrelevant changes preserve it, and related examples stay in the same split. For Darija, use fluent reviewers to validate negation and sentiment changes; the upstream generators concern explicit policies whose labels can be computed, which is not generally true of natural sentiment.
- Use simple supervised cross-entropy as the first control. KEV's latest completed calibration-loss screen did not promote any tested continuation, smoothing, Brier, or focal-loss variant. That is evidence for disciplined ablations, not for copying a complicated loss recipe.

Sources: [training code](https://github.com/jaredpalmer/kev/blob/90990a5fac2995b9faa3190f7d437e84f2067768/kev/train.py), [contrastive generators](https://github.com/jaredpalmer/kev/blob/90990a5fac2995b9faa3190f7d437e84f2067768/kev/contrastive.py), [completed loss-screen evidence](https://github.com/jaredpalmer/kev/blob/90990a5fac2995b9faa3190f7d437e84f2067768/runs/calibration-screen-review-v1/report.json).

### Laya: adapt the multilingual model, but extract ideas selectively

Laya uses a bidirectional encoder, option-scoring layers, and an action head. Each question becomes a complete sequence containing instructions, options, and state. Multiple questions are batched, but the state is repeated: six question variants do not have the compute cost of one. Token budgets also constrain instructions, option descriptions, and review text. Instrument actual truncation before adding long examples. Source: [sequence construction and model](https://github.com/NandhaKishorM/laya/blob/573e5b62696ba441230cd6be71d593331b5d23af/laya/common.py).

**Keep the explicit multilingual checkpoint.** The automatic router's language heuristic returned `is_english=True` when executed on `had lproduit khayb bzaf`, `zwin bzzaf 3jbni`, and mixed-script `had lproduit khayb بزاف`. Adopting its default router would send these examples to the English checkpoint. This is a reproduced routing finding, not a model-accuracy experiment. Sources: [language detector](https://github.com/NandhaKishorM/laya/blob/573e5b62696ba441230cd6be71d593331b5d23af/laya/lang.py), [router](https://github.com/NandhaKishorM/laya/blob/573e5b62696ba441230cd6be71d593331b5d23af/laya/router.py).

The supplied fine-tuning notebook is an English typed-decisions adaptation, not a multilingual Darija recipe. Its actual objective combines RL and cross-entropy; its calibration examples are taken from the training pool; the shown adaptation assigns zero weight to the action-head loss. Do not run it unchanged or treat its action probability as a validated abstention mechanism. A multilingual supervised training path with separate validation/calibration needs implementation. Source: [fine-tuning notebook](https://github.com/NandhaKishorM/laya/blob/573e5b62696ba441230cd6be71d593331b5d23af/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb).

Laya's research documentation explicitly says Jev was not run in that repository; its cited Jev numbers come from third parties. Our matched local results are the relevant comparison. Shortlisting is also irrelevant for three exhaustive sentiment classes. Sources: [research methodology](https://github.com/NandhaKishorM/laya/blob/573e5b62696ba441230cd6be71d593331b5d23af/research/README.md), [shortlisting implementation](https://github.com/NandhaKishorM/laya/blob/573e5b62696ba441230cd6be71d593331b5d23af/laya/shortlist.py).

## 3. Make the next experiments trustworthy first

The existing pipeline already validates predictions, resumes through caches, records manifests, computes uncertainty, and emits standalone HTML. Extend these components rather than replace them.

| Required change | Why it matters | Verification before experiments |
| --- | --- | --- |
| Represent criteria order explicitly in experiment identity. | Both backend schema fingerprints currently use `json.dumps(..., sort_keys=True)`. Option permutations therefore have the same hash even though inference can depend on order. | Two permutations get different cache identities; repeating the same permutation hits its own cache. Probability vectors map back to semantic labels. |
| Add an experiment/configuration fingerprint. | Cache identity must cover checkpoint revision, preprocessing, examples/retrieval corpus, option order, calibration and decision policy. Otherwise a modified system can reuse baseline outputs. | Change one behavior-affecting field at a time and verify invalidation; preserve old records. Separate raw-inference identity from derived calibration identity where practical. |
| Make Laya schema selection explicit and verifiable. | Its endpoint currently accepts only `review` and uses server-side schema constants. It does not emit a schema hash; the client tolerates that missing field. Client-side prompt changes alone do not prove remote prompt changes. | A versioned candidate endpoint returns the actual loaded schema/configuration hash; mismatches fail before scoring. Keep historical compatibility separate. |
| Add a distinct KEV backend and model-agnostic comparison. | CLI selection and comparison currently require Jev/Laya. Labeling KEV as Jev would corrupt provenance. | All backend pairs validate matching examples; old reports remain readable; record base, adapter, code revision, and serving configuration. |
| Log token admission/truncation and full-precision probabilities/logits where available. | A character limit is not a tokenizer limit, and rounded probabilities constrain calibration. | Long Arabic/Arabizi inputs expose token counts and truncation; no silent removal of difficult rows. Preserve supplied probabilities when logits are unavailable. |

Relevant local files: [schema](src/darija_eval/schema.py), [cache](src/darija_eval/cache.py), [Jev backend](src/darija_eval/backends/jev.py), [Laya backend](src/darija_eval/backends/laya.py), [Laya endpoint](src/darija_eval/modal_laya.py), [CLI](src/darija_eval/cli.py), [comparison](src/darija_eval/compare.py).

Use a separate KEV serving environment: its pinned project requires Python 3.12+ and a newer Transformers range than the existing Laya image. The HTTP boundary avoids forcing incompatible serving dependencies into one runtime. Source: [KEV dependencies](https://github.com/jaredpalmer/kev/blob/90990a5fac2995b9faa3190f7d437e84f2067768/pyproject.toml).

## 4. Prioritized experiments

### P1 — Review the task and build targeted supervision

Use the existing [blind review protocol](docs/evaluation.md): two fluent Darija annotators, hidden model outputs and original labels, then adjudication. Include random examples as well as dev error categories. Clarify balanced opinion versus dominant sentiment, rhetorical questions, sarcasm, negation, and code-switching. Preserve original benchmark labels; do not relabel examples simply to agree with a model.

Build learning curves from increasing amounts of reviewed training data, with deliberate coverage of neutral and Arabizi examples. Add reviewer-checked original/transliterated pairs and minimal negation pairs. Keep every translation, paraphrase, and duplicate family within one partition. Never assign “neutral” merely because text is unfamiliar or evidence is missing.

For pilots, use grouped cross-validation within the existing 680-row development pool, reserving a separate calibration subset when fitting probabilities. Training labels, model-selection labels, and calibration labels must have distinct roles. Small strata mean unstable estimates; report fold variation rather than pretending all 680 rows can serve every role independently.

Collect an independent, blinded confirmation panel from the intended review domain. A planning target of 500–1,000 reviews, with adequate neutral and Arabizi support, is reasonable for budgeting—not a statistical power guarantee. Choose final sample size from the minimum useful improvement and expected paired discordance before evaluation. If deliberately oversampling rare groups, also report results under the intended population mixture.

The existing [dataset survey](docs/dataset_survey.md) identifies AfriSenti Moroccan data as a candidate external resource. Use an independently versioned dataset adapter and verify its current data/label mapping before adoption. Tweets and product reviews are different domains: supplementary tweet training or evaluation must not silently replace the review objective. The [original AfriSenti repository](https://github.com/afrisenti-semeval/afrisent-semeval-2023) is the primary starting point.

### P2 — Run cheap, bounded inference experiments

Run each variant separately against the same development baseline before combining winners:

| Experiment | Hypothesis | Measure and tradeoff |
| --- | --- | --- |
| Short, reviewer-approved criteria clarification | Some errors reflect disagreement about neutral/mixed sentiment. | Macro F1 and both neutral precision/recall; no change to label ontology. |
| Six permutations of the three options | Position sensitivity contributes avoidable errors. | Flip rate, mapped probability variation, and F1 of the six-way mean; up to six sequence evaluations, not free accuracy. |
| Original text plus a reviewed normalization/transliteration | A second rendering helps interpret Arabizi. | Paired original/transformed performance and errors introduced by the transform; preserve negation, emoji, digits and the original text. |
| Small training-only exemplar context | Relevant examples clarify the task or language. | Start with a small fixed example set, then retrieval only if justified; include prompt length, truncation, latency, and retrieval leakage checks. |

TypeSafe's Choice API supports meaningful option descriptions and structured instructions. That permits experiments; it does not establish that longer prompts improve this task. Source: [Choice documentation](https://docs.typesafe.ai/primitives/choice).

Do not pick the best option order using inspected eval labels. For ensembles, average distributions **after aligning by class name**, not option position. Record every component prediction and measured cost. Neither Jev nor Laya is a text-generation transliterator: any transformation is a separate, versioned component.

### P3 — Warm-start KEV on Darija

Compare these arms under the same input schema and grouped development protocol:

1. Unmodified, pinned KEV-4B.
2. The same checkpoint adapted on ordinary reviewed Darija training examples.
3. The same adaptation with targeted Arabizi/neutral/contrastive data, controlling training steps or reporting the changed budget.

Use KEV's native request-plus-label JSONL format, with the same `state={"review": ...}` and question structure as serving. Keep labels in training targets, never in inference state. Start with low-learning-rate LoRA/head continuation and ordinary cross-entropy; evaluate a small learning curve before a broad hyperparameter search.

Initial recipe settings to inspect and freeze:

- `--init_from`: an immutable released checkpoint snapshot, including adapter and head.
- `--base` and `--base_revision`: exactly those recorded by that checkpoint.
- Initial `--lr 2e-5`, one or two epochs as bounded candidates, with batch/accumulation determined by a measured memory smoke test.
- `--p_none 0 --p_none_distract 0 --p_distract 0 --p_none_pair 0` for the first three-class control; retain label-order shuffling.
- Record admitted/rejected training examples: upstream training filters by token budgets, so reconcile requested versus used records.

These are proposed starting settings, not an executed recipe or proven optimum. The arguments and warm-start path exist in the pinned [trainer](https://github.com/jaredpalmer/kev/blob/90990a5fac2995b9faa3190f7d437e84f2067768/kev/train.py). If retaining general decision capability matters, compare a small replay mixture against task-only training on a separate retention set. Try 0.8B for cost or 9B for capacity only after the 4B experiments reveal a useful reason.

### P4 — Adapt multilingual Laya as a compact challenger

Start from the currently pinned multilingual weights. Compare a frozen-encoder/scorer-training arm with a low-learning-rate encoder fine-tune under supervised cross-entropy. Use the same reviewed training examples and model-selection protocol as KEV. Implement a reproducible training/export path; the English notebook is not ready-made for this.

Prioritize neutral precision and Arabizi errors. If probabilities show a stable class bias, a small regularized class-bias adjustment is another development-only experiment. Unlike scalar temperature, class-specific biases can change the predicted label; consequently they are supervised decision tuning and can hurt other classes. Do not infer or set the correction from the current eval confusion matrix.

Only investigate the upstream RL objective after a supervised baseline succeeds. That keeps any improvement attributable to data, model adaptation, or objective choice rather than changing all three together.

### P5 — Calibrate, then consider selective fallback

Fit a positive scalar temperature on reserved calibration data after model selection. KEV persists a temperature; current Laya main also applies configured temperatures. For each row, scalar temperature preserves the argmax, so **accuracy and macro F1 remain unchanged**. It can improve NLL/Brier/reliability and can alter confidence ordering between rows; measure risk–coverage again. Sources: [KEV calibration](https://github.com/jaredpalmer/kev/blob/90990a5fac2995b9faa3190f7d437e84f2067768/scripts/calibrate_checkpoint.py), [Laya inference](https://github.com/NandhaKishorM/laya/blob/573e5b62696ba441230cd6be71d593331b5d23af/laya/agent.py).

Continue defining benchmark confidence as the predicted class's probability. Upstream SDK confidence fields have different formulas and are not interchangeable empirical correctness rates. Preserve raw probabilities and calibration parameters separately.

Verify that serving applies the fitted temperature unchanged. Inspected Laya main clamps temperatures to `[0.5, 5]`, while its research fitter searches `[0.2, 10]`; constrain the fit or explicitly reconcile the runtime behavior. This main-source finding is not a verification of deployed 0.3.4. Sources: [runtime bounds](https://github.com/NandhaKishorM/laya/blob/573e5b62696ba441230cd6be71d593331b5d23af/laya/common.py), [research fitter](https://github.com/NandhaKishorM/laya/blob/573e5b62696ba441230cd6be71d593331b5d23af/research/scripts/build_benchmark_nb.py).

If an adapted local model is useful, test a local-model → Jev fallback using held-out or out-of-fold development predictions. Report the final label on **every** review, fraction escalated, total requests, total cost, and p50/p95 latency. Derive routing thresholds without eval labels. A system that abstains must report accepted accuracy and coverage separately; dropping hard examples is not a higher full-set score. The current weaker Laya model is not automatically a useful fallback for Jev.

## 5. Promotion rules and implementation order

Before running candidates, register a minimum useful gain and the cost constraint. A proposed starting gate is **+0.03 absolute macro F1** over the chosen baseline on development data, followed by independent confirmation whose paired 95% interval excludes zero. This is an engineering target, not a forecast. Freeze acceptable accuracy/class/style regressions in advance, inspect their uncertainty, and decline to claim subgroup safety when support is insufficient.

For a lower-cost replacement, use a separately registered quality non-inferiority margin and measured cost improvement; do not switch between quality and cost goals after seeing results. Repeat promising training candidates across seeds, but keep all variants and failed attempts visible. Perform one final confirmatory comparison after selecting the candidate; the old 171-row set remains a regression check.

| Stage | Concrete deliverable | Exit check |
| --- | --- | --- |
| 1. Experiment integrity | Ordered schema/config identity, verifiable endpoint schema, KEV backend, generic comparison | Cache-isolation and provenance tests; old Jev/Laya comparisons still work. |
| 2. Data and cheap probes | Reviewed dev taxonomy, grouped partitions, bounded inference ablations | Validated predictions and standalone HTML for every run; no test-derived tuning. |
| 3. Trainable challenger | KEV warm-start learning curve; then multilingual Laya CE baseline | Held-out dev improvement, recorded training admission, seed consistency and measured resource use. |
| 4. Probability/serving work | Independent calibration; optional fallback policy | Full-population quality plus calibration, coverage, cost, fresh latency and failure accounting. |
| 5. Confirmation | Frozen candidate on new blinded review data | Paired comparison against a contemporaneous pinned baseline; publish negative results too. |

Do not begin with a larger GPU, option shortlisting, a generic agent workflow, automatic language routing, or a complicated RL loss. None addresses the measured quality gap as directly as supervised domain data. Both repositories declare Apache-2.0 for code; preserve notices when reusing it and track model/data provenance independently of the code license. Sources: [KEV license](https://github.com/jaredpalmer/kev/blob/90990a5fac2995b9faa3190f7d437e84f2067768/LICENSE), [Laya license](https://github.com/NandhaKishorM/laya/blob/573e5b62696ba441230cd6be71d593331b5d23af/LICENSE).

## 6. Verification and reproducibility of this report

Local source reviewed at commit `18e191257f1e7297cea0bdf6340b6c7711accc94`. The evidence review used existing `artifacts.load_run` and `compare.compare_runs`; the latter produced a standalone comparison HTML in a temporary result directory. No new model inference, fine-tuning, endpoint deployment, or runtime upgrade was performed.

Prediction SHA-256 checksums:

```text
Jev:  da2e9c48913cf680484c9c57a75449c4ceeebbb5e33715370db8f6d6af18b765
Laya: b3d7d731cbae423ffeb75e93d2b763bbb69f58324d63a6611c6ddb4fa6e41a25
Shared schema: 2e37db5a57c6837d97d4a43e26c639ba4343a58655dbfc74f3c56bf6af9a27f5
```

To reproduce the paired baseline offline through the existing CLI, using the saved runs:

```bash
uv run darija-eval compare \
  results/jev_eval_20260921_234315_802617 \
  results/laya_eval_20260921_233519_938576
```

This recomputes from validated evidence and writes comparison JSON, paired predictions, and standalone HTML. It does not call either model. Every proposed future evaluation, calibration comparison, and model-comparison run must likewise include a readable standalone `report.html` in its result directory.
