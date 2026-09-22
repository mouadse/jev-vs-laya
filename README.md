# Jev vs Laya: The At-Home Darija Showdown

Can an open-weight model you control match a hosted AI service? This reproducible
zero-shot benchmark pits TypeSafe Jev against self-hosted Laya on informal Moroccan
Darija sentiment—across Arabic script and Arabizi, with `positive`, `neutral`, and
`negative` labels. No model is trained or fine-tuned.

## Setup

```bash
uv sync
cp .env.example .env
# Put the keys in .env, or export them in the shell:
export TYPESAFE_API_KEY="..."
export HF_TOKEN="..."  # authenticates dataset and Laya model downloads
export LAYA_ENDPOINT_URL="https://...modal.run"
export KEV_ENDPOINT_URL="https://...modal.run"
```

The CLI loads these variables from `.env`. The checked-in `.env.example` points to
the deployed Laya endpoint but contains no credentials.

The benchmark loads the `test` split of
[`ohidaoui/darija-reviews`](https://huggingface.co/datasets/ohidaoui/darija-reviews).
The source dataset has no row ID, so source row indices are used as stable IDs.
The first data load freezes a deterministic seed-42 split under `data/splits/`:
80% dev and 20% eval, stratified by normalized sentiment and writing style when
possible.

The dataset currently contains four raw labels. The four rows labeled `negative `
with a trailing space are explicitly normalized to `negative`. Any other unknown,
blank, or null label stops the run.

## Deploy Laya on Modal

The deployment uses the repository's multilingual checkpoint directly for both
Arabic-script Darija and Arabizi. Keeping one checkpoint makes the writing-style
comparison meaningful. The weights and source revision are pinned, and the service
runs on one L4 with scale-to-zero behavior.

Create a Modal secret named `hf-secret` containing `HF_TOKEN`, then run:

```bash
uv run modal run -m darija_eval.modal_laya::download_model
uv run modal deploy -m darija_eval.modal_laya --strategy recreate
```

The first command stores only the multilingual checkpoint in a Modal Volume. Put
the deployed `predict` URL in `LAYA_ENDPOINT_URL`. The endpoint accepts only a
`review` string; sentiment labels, writing style, and topic are never sent to Laya.

Schema v2 shortens the criteria, explicitly references `review`, and no longer treats
unfamiliar language as neutral. It is shared by Jev and Laya; redeploy Laya after a
schema change. Version and question-content changes invalidate prediction cache keys.
Old v1 runs remain readable, but cannot be paired with v2 runs by `compare`.
No temperature, decision thresholds, transliteration or option-order ensemble has
been fitted or enabled; prompt changes alone do not establish an accuracy gain.

## Deploy Kev on Modal

Kev runs as a separate Modal app because it needs Python 3.12 and a newer
Transformers range than the Laya image. The deployment pins the `kev-4b`
adapter, its Qwen3.5-4B base revision, and the KEV source revision, and serves
the checkpoint-carried temperature on one L4 with scale-to-zero behavior.

```bash
uv run modal run -m darija_eval.modal_kev::download_model
uv run modal deploy -m darija_eval.modal_kev --strategy recreate
```

Put the deployed `predict` URL in `KEV_ENDPOINT_URL`. Like Laya, the endpoint
accepts only a `review` string and returns the label distribution, except it
also returns a verifiable schema-content hash that the client requires.

## Commands

```bash
uv run darija-eval inspect
uv run darija-eval inspect --json-output results/reference_audit.json
uv run darija-eval demo --backend jev
uv run darija-eval demo --backend laya
uv run darija-eval demo --backend kev
uv run darija-eval dev-eval --backend laya
uv run darija-eval dev-eval --backend laya --limit 20
uv run darija-eval eval --backend laya
uv run darija-eval eval --backend kev
uv run darija-eval compare results/jev_eval_<timestamp> results/laya_eval_<timestamp>
uv run darija-eval compare results/jev_eval_<timestamp> results/laya_eval_<timestamp> results/kev_eval_<timestamp>
uv run darija-eval reanalyse results/jev_eval_<timestamp>
```

Omit `--backend` to use Jev. `compare` accepts two or three eval-run directories
from distinct backends, in any order (the backend is named `laya`, not `yala`).
Run the full frozen eval once per backend first; demo/dev runs cannot be compared.
The command rejects failed or mismatched runs, recomputes metrics from prediction
records, and aligns every backend by ID. It checks IDs against the local frozen
split and identifies limited samples. Three-run reports include all three pairwise
comparisons, a shared scoreboard, and an all-model error explorer. Each comparison
writes `comparison.json`, `paired_predictions.jsonl`, and standalone `report.html`.
`reanalyse` creates updated metrics and HTML from saved evidence without contacting
any model; original files stay intact.

When provenance is present, reanalysis checks selected IDs and review/reference
fingerprints, schema fingerprints, and manifest/metrics agreement before scoring.
Comparisons also reject recorded dataset or split fingerprints that conflict with
each other or the local frozen split. Reanalysis preserves the original manifest.
Legacy runs without this metadata remain readable, but missing provenance cannot
be independently verified.

Use `--concurrency` (default 5) and `--max-retries` (default 3) on prediction
commands to adjust remote API behavior. Successful calls are cached under
`data/cache/`; the cache key includes backend, requested model, exact review text,
schema version, and a content hash of the question. Interrupted runs therefore reuse
completed predictions. Older caches without content hashes remain on disk but cannot
be safely reused by the new keys; use `reanalyse` to update old reports without calls.
`jev-latest` is a mutable service alias: inspect resolved model IDs when comparing
runs across dates. The deployed Laya endpoint verifies model/version identity;
historical responses do not establish a content-hash proof of the remote schema.
Malformed or invalid cache entries are treated as misses and replaced only after a
successful fresh prediction. Invalid fresh predictions are recorded as API failures,
never cached or scored. Missing probabilities remain supported: label-based metrics
are still computed, while calibration metrics are skipped.

Each run writes a timestamped directory under `results/` with `predictions.jsonl`,
`failures.jsonl`, `api_failures.jsonl`, `metrics.json`, `summary.md`, and a
standalone `report.html`. Open `report.html` directly in a browser for the visual
summary and interactive error explorer.
New inference runs also record `manifest.json` before calls begin, with exact IDs,
dataset/selection and schema fingerprints, split scope, and concurrency.
`failures.jsonl` contains model misclassifications; transport/authentication/schema
failures are counted separately in `api_failures.jsonl` and excluded from quality
metrics. Probability-dependent calibration metrics are emitted only when the backend
supplies a complete class distribution. Laya exposes its raw softmax distribution,
which this benchmark records without post-hoc calibration. The comparison report
also separates Laya GPU inference time from client round-trip latency.

**Keep the frozen 20% eval split untouched during prompt and schema development.**
Use `demo` and `dev-eval` while iterating. All three backends share the same dataset,
schema, evaluation, metrics, cache, and artifact pipeline.

## Reading the analysis

Accuracy includes a 95% Wilson interval. A descriptive majority-label reference
shows how much accuracy can come from class imbalance; it is explicitly selected
from the scored sample, not claimed as a dev-selected model. Macro F1 always uses
all three labels, including absent classes. Style/topic groups show support and
uncertainty; their differences do not isolate the effect of transliteration.

Comparison reports include a paired, sentiment/style-stratified bootstrap interval
(5,000 resamples, seed 42) for accuracy and macro-F1 differences, plus an exact
McNemar test on discordant pairs. These are exploratory, conditional on this dataset;
they do not capture uncertain annotations, repeated model sampling, or dataset shift.
Three-backend reports compute these statistics for every pair on the same matched
sample; intervals and p-values are not adjusted for multiple comparisons.
Confidence tables show accepted counts and accuracy intervals so a tiny high-confidence
subset is not mistaken for demonstrated safety. ECE uses ten equal-width bins;
multiclass Brier is the sum over classes (range 0–2). Cached latency is historical;
only explicitly fresh requests enter current-run latency statistics.

Individual reports also include a predicted-class reliability diagram and a
tie-aware risk–coverage curve with AURC and accuracy at 50%/80% coverage. Equal-confidence
groups are accepted whole; actual coverage is shown when it exceeds the target.
A prominent reference-label caveat includes full-source duplicate audit counts for
new runs; historical runs without embedded audit data show qualified disclosures.

See [evaluation reference and review protocol](docs/evaluation.md) for reference-label
handling and rules for future experiments. Do not relabel an error merely because a
model disagrees, or tune prompts/thresholds on the already inspected eval results.
