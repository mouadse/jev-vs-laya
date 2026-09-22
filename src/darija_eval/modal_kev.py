from __future__ import annotations

import os
import time

import modal

from .schema import SCHEMA_VERSION, question_fingerprint, sentiment_question_dict

APP_NAME = "kev-darija-sentiment"
ADAPTER_REPO = "jaredpalmer/kev-4b"
ADAPTER_REVISION = "485ace8703592fcf405488b262449990824cfed1"
BASE_REPO = "Qwen/Qwen3.5-4B-Base"
BASE_REVISION = "1001bb4d826a52d1f399e183466143f4da7b741b"
KEV_CODE_REVISION = "90990a5fac2995b9faa3190f7d437e84f2067768"
MODEL_IDENTIFIER = (
    f"{ADAPTER_REPO}@{ADAPTER_REVISION}"
    f"+{BASE_REPO}@{BASE_REVISION}"
    f"+kev@{KEV_CODE_REVISION}"
)
GPU = "L4"
QUESTION = sentiment_question_dict()
SCHEMA_FINGERPRINT = question_fingerprint(QUESTION)

def checkpoint_base_matches(meta_base: str | None, meta_base_revision: str | None) -> bool:
    """Fail-closed check that a checkpoint's recorded base matches the pinned base.

    Short and full SHAs compare by prefix in either direction (the model card
    abbreviates the base revision), but a missing revision never matches: an
    empty string is a prefix of everything, so it must be rejected explicitly.
    """
    revision = meta_base_revision or ""
    return (
        meta_base == BASE_REPO
        and bool(revision)
        and (
            revision == BASE_REVISION
            or BASE_REVISION.startswith(revision)
            or revision.startswith(BASE_REVISION)
        )
    )

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name("kev-darija-models", create_if_missing=True)
hf_secret = modal.Secret.from_name("hf-secret", required_keys=["HF_TOKEN"])

download_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub[hf_xet]>=0.34")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

# Separate image from Laya: KEV needs Python 3.12+ and transformers>=5.17.
inference_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        f"kev @ git+https://github.com/jaredpalmer/kev@{KEV_CODE_REVISION}",
        "fastapi[standard]>=0.115",
    )
    .env(
        {
            "HF_HUB_CACHE": "/models/hf-cache",
            "USE_TF": "0",
            "USE_TORCH": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_python_source("darija_eval")
)


@app.function(
    image=download_image,
    volumes={"/models": model_volume},
    secrets=[hf_secret],
    timeout=3600,
)
def download_model() -> str:
    from huggingface_hub import snapshot_download

    cache = "/models/hf-cache"
    os.environ["HF_HUB_CACHE"] = cache
    adapter_path = snapshot_download(
        repo_id=ADAPTER_REPO,
        revision=ADAPTER_REVISION,
        allow_patterns=["*.json", "*.safetensors", "*.pt", "*.txt", "*.jinja"],
        token=os.environ["HF_TOKEN"],
    )
    base_path = snapshot_download(
        repo_id=BASE_REPO,
        revision=BASE_REVISION,
        allow_patterns=["*.json", "*.safetensors", "*.txt"],
        token=os.environ["HF_TOKEN"],
    )
    model_volume.commit()
    return f"{adapter_path} {base_path}"


@app.cls(
    image=inference_image,
    gpu=GPU,
    max_containers=1,
    scaledown_window=300,
    timeout=600,
    startup_timeout=600,
    volumes={"/models": model_volume},
    secrets=[hf_secret],
)
class KevService:
    @modal.enter()
    def load(self) -> None:
        import torch

        from kev.api import Choice
        from kev.checkpoint import Checkpoint, LoadOptions
        from kev.serve import Server

        checkpoint = Checkpoint(f"{ADAPTER_REPO}@{ADAPTER_REVISION}")
        if not checkpoint_base_matches(checkpoint.meta.base, checkpoint.meta.base_revision):
            raise RuntimeError(
                f"checkpoint base {checkpoint.meta.base}@{checkpoint.meta.base_revision} "
                f"does not match pinned {BASE_REPO}@{BASE_REVISION}"
            )
        tokenizer, model = checkpoint.load("cuda", LoadOptions())
        self.server = Server(checkpoint=checkpoint, tok=tokenizer, model=model, device="cuda")
        self.question = Choice(**QUESTION)
        self.temperature = float(model.head.temperature)
        print(f"Loaded {MODEL_IDENTIFIER} on {torch.cuda.get_device_name(0)}")

    @modal.fastapi_endpoint(method="POST", docs=True)
    def predict(self, payload: dict) -> dict:
        from fastapi import HTTPException

        from kev.api import SystemOneRequest, output_tokens, to_answers, to_record
        from kev.serve import prepare

        review = payload.get("review") if isinstance(payload, dict) else None
        if not isinstance(review, str) or not review.strip():
            raise HTTPException(status_code=422, detail="review must be a non-empty string")
        if len(review) > 5000:
            raise HTTPException(status_code=422, detail="review is too long")
        started = time.perf_counter()
        request = prepare(
            SystemOneRequest(state={"review": review}, questions={"sentiment": self.question})
        )
        rec, meta = to_record(request)
        distributions, info = self.server.probs(rec)
        full = [float(value) for value in distributions[0]]
        keys = meta[0]["keys"]
        if len(keys) != len(full):
            raise HTTPException(status_code=500, detail="model output does not match the question options")
        label = keys[max(range(len(full)), key=lambda i: full[i])]
        rounded = to_answers(distributions, meta)["sentiment"]
        inference_ms = (time.perf_counter() - started) * 1000
        checkpoint_meta = self.server.checkpoint.meta
        return {
            "label": label,
            "probabilities": dict(zip(keys, full)),
            "model": MODEL_IDENTIFIER,
            "schema_version": SCHEMA_VERSION,
            "schema_fingerprint": SCHEMA_FINGERPRINT,
            "inference_ms": inference_ms,
            "raw": {
                "usage": {
                    "input_tokens": info["tokens"],
                    "output_tokens": output_tokens(self.server.tok, {"sentiment": rounded}),
                },
                "server_latency_ms": info["latency_ms"],
                "temperature": self.temperature,
                "base": checkpoint_meta.base,
                "base_revision": checkpoint_meta.base_revision,
                "lora": checkpoint_meta.lora,
                "gpu": GPU,
                "probabilities_rounded": rounded["probabilities"],
            },
        }

    @modal.fastapi_endpoint(method="GET", docs=True)
    def health(self) -> dict:
        return {"status": "ok", "model": MODEL_IDENTIFIER, "gpu": GPU}
