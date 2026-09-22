# Jev vs Kev vs Laya — Darija Sentiment Benchmark

Can a self-hosted open-weight model match a hosted AI service on informal
Moroccan Darija? This reproducible, zero-shot benchmark pits TypeSafe **Jev**
(hosted API) against self-hosted **Kev-9B** and **Laya** (both on Modal GPUs)
on Arabic-script and Arabizi reviews. No model is trained or fine-tuned.

## Results at a glance

171 paired reviews from the frozen eval split (Arabic N=127, Arabizi N=44).
Same reviews, same labels, same schema for all three systems.

| Model | Accuracy (95% CI) | Macro F1 | Arabic | Arabizi |
|---|---|---|---|---|
| **Jev** `jev-1.13.0` | **79.5%** (72.9–84.9) | 71.7% | 84.3% | 65.9% |
| **Kev-9B** `kev-9b` + Qwen3.5-9B | 71.9% (64.8–78.1) | 61.1% | 78.0% | 54.5% |
| **Laya** `multilingual` | 53.8% (46.3–61.1) | 50.0% | 60.6% | 34.1% |
| Always-positive reference | 53.8% | — | — | — |

```text
Jev  ████████████████░░░░  79.5%
Kev  ██████████████░░░░░░  71.9%
Laya ███████████░░░░░░░░░  53.8%
```

Head-to-head gaps (paired bootstrap 95% CI, McNemar exact p):

| Pair | Gap | 95% CI | p |
|---|---|---|---|
| Kev − Jev | −7.6 pp | −12.9 to −2.3 | 0.019 |
| Laya − Jev | −25.7 pp | −33.9 to −17.5 | 6.2e-08 |
| Laya − Kev | −18.1 pp | −26.3 to −9.4 | 1.9e-04 |

**Takeaways:**

- **Jev wins overall**, and the gap to Kev is supported but modest (−7.6 pp).
- **Laya ties the always-positive baseline (53.8%)** — not viable for this task as configured.
- **Arabizi is the hard part for everyone**: −18 pp (Jev), −23 pp (Kev), −27 pp (Laya) vs Arabic script.
- **Neutral is the weak class**: F1 50.0 / 35.0 / 33.3 for Jev / Kev / Laya.
- **Jev confidence is actionable**: ≥0.80 → 90.2% accuracy at 65.5% coverage;
  ≥0.95 → 95.8% at 41.5% coverage. Kev tracks closely; Laya is poorly calibrated (ECE 0.23 vs 0.07).

> Full interactive report with per-review error explorer:
> [`results/compare_jev_kev_laya_20260922_105535_729323/report.html`](results/compare_jev_kev_laya_20260922_105535_729323/report.html)
> (clone the repo and open it in a browser — GitHub does not render local HTML).
> Machine-readable results: `comparison.json`, `paired_predictions.jsonl` in the same directory.

## How it works

- **Data**: `test` split of
  [`ohidaoui/darija-reviews`](https://huggingface.co/datasets/ohidaoui/darija-reviews).
  Source has no row ID, so source row indices are stable IDs. First load freezes a
  deterministic seed-42 split under `data/splits/`: 80% dev, 20% eval, stratified
  by sentiment and writing style when possible.
- **Labels**: `positive` / `neutral` / `negative`. The four rows labeled
  `negative ` (trailing space) are normalized to `negative`; any other unknown,
  blank, or null label stops the run.
- **Shared schema v2** for all backends; version/content changes invalidate cache
  keys. Endpoints receive only the `review` string — never labels, style, or topic.
- **Stats**: accuracy with 95% Wilson interval; paired sentiment/style-stratified
  bootstrap intervals (5,000 resamples, seed 42) plus exact McNemar tests for
  pairwise gaps. Exploratory, conditional on this dataset — no adjustment for
  multiple comparisons, no label-uncertainty or shift modeling.
- **Calibration**: ECE (10 equal-width bins), multiclass Brier (range 0–2),
  log-loss, and thresholded selective-accuracy tables with accepted counts.
- **Latency honesty**: recorded round-trip times mix infrastructure, queueing, and
  cached rows — not a controlled speed test. GPU-only inference is reported
  separately where available (Kev p50/p95 84/109 ms, Laya 57/140 ms).

See [evaluation reference and review protocol](docs/evaluation.md) for label
handling and the rules for future experiments — including not tuning on the
already-inspected eval results.

## Reproduce

```bash
uv sync
cp .env.example .env
# Put the keys in .env, or export them in the shell:
export TYPESAFE_API_KEY="..."
export HF_TOKEN="..."  # authenticates dataset and model downloads
export LAYA_ENDPOINT_URL="https://...modal.run"
export KEV_ENDPOINT_URL="https://...modal.run"
```

The CLI loads `.env` (so `HF_TOKEN` authenticates Hugging Face downloads).
`.env.example` points at the deployed endpoints but contains no credentials.

### Deploy Laya on Modal

One multilingual checkpoint serves both Arabic-script Darija and Arabizi, so the
writing-style comparison stays meaningful. Weights and source revision are pinned;
the service runs on one L4 with scale-to-zero.

Create a Modal secret named `hf-secret` containing `HF_TOKEN`, then:

```bash
uv run modal run -m darija_eval.modal_laya::download_model
uv run modal deploy -m darija_eval.modal_laya --strategy recreate
```

Put the deployed `predict` URL in `LAYA_ENDPOINT_URL`. Redeploy Laya after any
schema change.

### Deploy Kev on Modal

Kev runs as a separate Modal app (Python 3.12, newer Transformers than the Laya
image). The deployment pins the `kev-9b` adapter, its Qwen3.5-9B base revision,
and the KEV source revision, serving the checkpoint-carried temperature in FP32
on one L40S with scale-to-zero. The 9B checkpoint exceeds L4 memory with the
existing FP32 loader.

```bash
uv run modal run -m darija_eval.modal_kev::download_model
uv run modal deploy -m darija_eval.modal_kev --strategy recreate
```

Put the deployed `predict` URL in `KEV_ENDPOINT_URL`. Unlike Laya, Kev also
returns a verifiable schema-content hash that the client requires. The endpoint
URL and `--backend kev` stay unchanged across checkpoints; the model identity
changes, so 4B cache entries are not reused for 9B (historical 4B results and
weights remain intact).

### Commands

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
from distinct backends or distinct resolved models of the same backend, in any
order. A saved KEV-4B run and a new KEV-9B run display as separate model
variants; identical backend/model pairs are rejected. Run the full frozen eval
for each model first; demo/dev runs cannot be compared.

To compare saved 4B predictions with a new 9B evaluation:

```bash
uv run darija-eval eval --backend kev
uv run darija-eval compare \
  results/kev_eval_20260922_101711_853681 \
  results/kev_eval_<new_9b_timestamp>
```

`reanalyse` rebuilds metrics and HTML from saved evidence without contacting any
model; original files stay intact.

### Caching, provenance, artifacts

- `--concurrency` (default 5) and `--max-retries` (default 3) tune remote calls.
  Successful calls are cached under `data/cache/`; keys include backend, model,
  exact review text, schema version, and question-content hash, so interrupted
  runs resume. Malformed cache entries are treated as misses; invalid fresh
  predictions are recorded as API failures, never cached or scored.
- `jev-latest` is a mutable service alias — inspect resolved model IDs when
  comparing runs across dates.
- Each run writes a timestamped directory under `results/` with
  `predictions.jsonl`, `failures.jsonl`, `api_failures.jsonl`, `metrics.json`,
  `manifest.json`, `summary.md`, and standalone `report.html`. Comparisons add
  `comparison.json` and `paired_predictions.jsonl`.
- Reanalysis and comparison verify ID, review/reference, schema, and
  dataset/split fingerprints where provenance is present; legacy runs without
  metadata remain readable but unverifiable.

**Keep the frozen 20% eval split untouched during prompt and schema development.**
Use `demo` and `dev-eval` while iterating. All three backends share the same
dataset, schema, evaluation, metrics, cache, and artifact pipeline.

## Limitations

- N=171 (Arabizi N=44): subgroup intervals are wide; style groups mix different
  reviews and topics, so gaps do not isolate transliteration effects.
- Reference labels are dataset annotations, not independently adjudicated.
- These saved predictions have been inspected — further prompt development
  belongs on dev, with a new independent test set for confirmatory claims.
- No temperature, threshold, transliteration, or ensemble fitting; prompt changes
  alone do not establish a gain.
