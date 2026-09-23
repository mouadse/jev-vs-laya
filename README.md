# Darija Sentiment Benchmark

### Jev vs Kev-9B vs Laya on Moroccan Arabic reviews

Can a self-hosted model match a hosted AI service on informal Moroccan Darija? I built an end-to-end, zero-shot benchmark using [Darija Reviews](https://huggingface.co/datasets/ohidaoui/darija-reviews) to answer that question with the **same reviews, labels, and three-class task** for every model. It evaluates positive, neutral, and negative sentiment in both Arabic script and **Arabizi** (Darija written in Latin characters and numerals), then turns the saved predictions into an interactive HTML report.

**Python 3.11+ · TypeSafe Jev · Modal GPUs · traceable evaluation · statistical comparison**

## Results

The frozen holdout contains **171 reviews**: 127 Arabic-script and 44 Arabizi. All three models returned predictions for all 171 reviews in the saved comparison run of **22 September 2026**.

| Model | Accuracy (95% interval) | Macro F1 | Arabic | Arabizi |
| :--- | ---: | ---: | ---: | ---: |
| **Jev** · TypeSafe API, `jev-1.13.0` | **79.5%** (72.9–84.9) | **71.7%** | **84.3%** | **65.9%** |
| **Kev-9B** · Qwen3.5-9B on Modal | 71.9% (64.8–78.1) | 61.1% | 78.0% | 54.5% |
| **Laya** · multilingual checkpoint on Modal | 53.8% (46.3–61.1) | 50.0% | 60.6% | 34.1% |
| Always predict positive · descriptive reference | 53.8% | — | — | — |

Jev led Kev by **7.6 percentage points** on these paired reviews (exploratory 95% paired bootstrap interval: **2.3–12.9 points**). Laya matched the always-positive reference on overall accuracy. Every model scored lower on the Arabizi subset, and neutral sentiment was the weakest class.

These are agreement scores against the dataset's original annotations, **not** independently adjudicated labels or estimates of production performance. The style groups contain different reviews, so their score gap does not isolate the effect of script. Accuracy intervals use the Wilson method; the paired interval uses 5,000 sentiment/style-stratified bootstrap resamples. See the [evaluation protocol](docs/evaluation.md) for the full interpretation.

> **About the evidence:** The table comes from a saved local three-model comparison. Generated reports and prediction records live in `results/`, which is ignored by Git; they are created when you run the commands below. The numbers are shown here, but the underlying run artifacts are not downloadable from this repository.

## What I built

| Part | What it does |
| :--- | :--- |
| [Dataset and frozen split](src/darija_eval/dataset.py) | Validates all 851 source reviews, documents the four known `negative ` label aliases, rejects unknown labels, and preserves a seed-42 split of 680 development and 171 evaluation reviews. |
| [Backend adapters](src/darija_eval/backends) | Send **only the review text** to Jev, djev, Kev, or Laya through one versioned sentiment schema; reference labels and metadata stay out of inference requests. |
| [Evaluation pipeline](src/darija_eval/evaluate.py) | Runs concurrent requests with retries, separates API failures from classification errors, records model identity, and resumes successful predictions from an experiment-aware cache. |
| [Analysis and comparison](src/darija_eval/compare.py) | Checks that runs refer to the same examples and compatible schemas, then reports paired differences, class and writing-style results, uncertainty, and confidence calibration. |
| [Standalone reports](src/darija_eval/report.py) | Generates readable HTML with charts, confusion matrices, and a searchable per-review error explorer alongside machine-readable JSON and JSONL. |

The [frozen split](data/splits/seed_42.json), model and schema fingerprints, and saved manifests make comparisons traceable. `reanalyse` can rebuild metrics and HTML from prediction records **without calling a model again**. No model is trained or fine-tuned in this project.

## djev game demos

The pinned [djev-run](https://github.com/taeold/djev-run) image includes three browser games backed by `POST /v1/systemone`. **The Modal apps are currently stopped**; the links below point to the last deployed `mouadse` workspace and work only after that workspace [redeploys the fast L40S server](#run-it) and `/health` returns 200. The images are credited upstream examples, not screenshots of this Modal deployment.

### Snake

Choose a safe direction from the live board state. Press **Start** (or Space) to let djev play; **R** resets. [Open Snake after deployment](https://mouadse--djev-run-l40s-fast-djevfastserver.eu-west.modal.direct/snake).

<a href="https://github.com/taeold/djev-run"><img src="https://github.com/user-attachments/assets/2e9a5321-f8a9-4734-b6f2-4d6f47193390" width="640" alt="Upstream djev-run Snake demo with board, move probabilities, and live decision payload"></a>

### Dino

The T-Rex runs continuously while djev chooses **jump**, **duck**, or **run**. Press **Start Autopilot** (or Space); **R** resets and **H** toggles hitboxes. [Open Dino after deployment](https://mouadse--djev-run-l40s-fast-djevfastserver.eu-west.modal.direct/dino).

<a href="https://github.com/virajbhartiya/laya-vs-jev"><img src="https://raw.githubusercontent.com/virajbhartiya/laya-vs-jev/main/docs/assets/trex-arena-window.png" width="640" alt="Original Laya vs Jev two-player T-Rex arena with decisions and survival metrics"></a>

*The Dino image shows the [original two-player T-Rex arena](https://github.com/virajbhartiya/laya-vs-jev), not djev-run's single-player page.*

### Tetris

djev selects a legal placement for each piece; the UI shows its probabilities and board-health reads. It starts in autopilot; switch to **Manual (Arrows + Space)** to play yourself. [Open Tetris after deployment](https://mouadse--djev-run-l40s-fast-djevfastserver.eu-west.modal.direct/tetris).

<a href="https://github.com/taeold/djev-run"><img src="https://github.com/user-attachments/assets/664f12cc-be17-4181-8a9a-5106e64c2f61" width="640" alt="Upstream djev-run Tetris demo with board, placement probabilities, and model telemetry"></a>

The djev-run demos adapt [laya-coreml's Snake](https://github.com/mizorewww/laya-coreml), [laya-vs-jev's Dino](https://github.com/virajbhartiya/laya-vs-jev), and [jev-tetris](https://github.com/trungdq88/jev-tetris). These are gameplay showcases, not results from the Darija sentiment holdout above.

## Run it

Install [uv](https://docs.astral.sh/uv/) and use Python 3.11 or newer. Copy [.env.example](.env.example) to `.env`, then set `TYPESAFE_API_KEY` for Jev. `HF_TOKEN` authenticates Hugging Face downloads; the Modal backends also need their own deployed endpoint URLs. `.env` is ignored by Git.

```bash
uv sync
cp .env.example .env
# Edit .env with your credentials and endpoint URLs.

uv run darija-eval inspect                 # Validate data and show the frozen split
uv run darija-eval demo --backend jev      # Ten development examples
uv run darija-eval dev-eval --backend jev  # Development split
uv run darija-eval eval --backend jev      # Frozen holdout
```

Each model run prints its output directory and writes a standalone `report.html` there. Open that file in a browser to explore errors and metrics. Keep prompt and schema iteration on `demo` or `dev-eval`; the holdout has already been inspected and should be treated as historical evaluation evidence.

To evaluate and compare all three models, deploy the two Modal services, set `LAYA_ENDPOINT_URL` and `KEV_ENDPOINT_URL` in `.env`, then run:

```bash
uv run darija-eval eval --backend laya
uv run darija-eval eval --backend kev
uv run darija-eval compare path/to/jev_eval path/to/kev_eval path/to/laya_eval
uv run darija-eval reanalyse path/to/jev_eval
```

Replace the example paths with the actual directories printed by each run. `compare` accepts two or three full evaluation runs; it rejects mismatched examples and duplicate backend/model pairs. Saved KEV-4B and KEV-9B runs count as distinct model variants.

<details>
<summary><strong>Deploy the self-hosted backends on Modal</strong></summary>

Create a Modal secret named `hf-secret` containing `HF_TOKEN`. Laya uses an L4 GPU; Kev-9B uses an L40S. Each service scales to zero when idle.

```bash
uv run modal run -m darija_eval.modal_laya::download_model
uv run modal deploy -m darija_eval.modal_laya --strategy recreate

uv run modal run -m darija_eval.modal_kev::download_model
uv run modal deploy -m darija_eval.modal_kev --strategy recreate
```

Set the deployed `predict` URLs in `.env`. The endpoint implementations are in [modal_laya.py](src/darija_eval/modal_laya.py) and [modal_kev.py](src/darija_eval/modal_kev.py).

</details>

<details>
<summary><strong>Deploy the djev game server on Modal</strong></summary>

The [djev-run](https://github.com/taeold/djev-run) image and NVIDIA checkpoint are pinned in [modal_djev.py](src/darija_eval/modal_djev.py). The [fast Modal Server](src/darija_eval/modal_djev_fast.py) serves `/v1/systemone` and all three games on one L40S GPU, scaling to zero after 120 seconds idle. Create the `hf-secret` Modal secret with `HF_TOKEN`; download the weights once, then deploy:

```bash
uv run modal run -m darija_eval.modal_djev::download_model
uv run modal deploy -m darija_eval.modal_djev_fast --strategy recreate
```

Set `DJEV_URL` to the URL printed by `modal deploy`. Wait for `/health` to return 200 before opening a game; Modal returns HTTP 503 while the GPU starts:

```bash
curl "$DJEV_URL/health"
xdg-open "$DJEV_URL/snake"
xdg-open "$DJEV_URL/tetris"
xdg-open "$DJEV_URL/dino"
```

The original `@modal.web_server` variant is available with `uv run modal deploy -m darija_eval.modal_djev --strategy recreate`, but deploy only the variant you intend to use to avoid a second GPU service.

The last fast endpoint was `https://mouadse--djev-run-l40s-fast-djevfastserver.eu-west.modal.direct`, but **it is currently stopped** and must be redeployed before the game links above work. In a matched seven-state Snake probe, median warm request time from this machine fell from 709 ms on the original web-server endpoint to 228 ms on the Modal Server; one Server request took 946 ms. The first full startup took 111 seconds in the deployment check. Keeping one container warm avoids the cold start but incurs continuous L40S charges.

The fast server config uses European compute and routing: a 100-pair synthetic Snake probe measured median/p95 latency of 182/188 ms versus 264/345 ms through the US deployment, with 100/100 matching move choices. The probe used seven board positions with varying seeds; it does not establish general gameplay quality. European compute pinning adds a 15% resource-price premium, and scarce regional L40S capacity can add several minutes to a cold start.

The fast deployment sets `OMP_NUM_THREADS=1`. A 30-state warm Snake probe recorded locally at `results/djev_omp_probe_20260923/report.html` (ignored by Git) measured median request time of 183 ms before and 175–176 ms after; the p95 improvement was not consistent across candidate runs. Different containers and uncontrolled placement limit attribution, and this probe does not measure Dino or Tetris improvements.

The API uses `model: "dgemma"` in `/v1/systemone` requests. Set `DJEV_ENDPOINT_URL` in `.env` to the deployed root URL and run `uv run darija-eval eval --backend djev` to evaluate it. The existing `--backend jev` calls the hosted TypeSafe service and remains a separate run.

</details>

## Scope and next steps

- The holdout is small, especially **Arabizi (44)** and **neutral (25)**. Subgroup estimates are uncertain.
- Source annotations have not been independently reviewed. A [dataset survey](docs/dataset_survey.md) identifies a larger candidate for a future, separately versioned study.
- `jev-latest` is a mutable service alias. The saved run resolved to `jev-1.13.0`; future runs may resolve differently.
- Cached client timings include the original network and service time, so they are **not** a controlled inference-speed comparison.

The [evaluation protocol](docs/evaluation.md) records the remaining provenance and annotation limits and outlines how to collect a new independent confirmation set.
