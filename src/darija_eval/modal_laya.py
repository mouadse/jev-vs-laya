from __future__ import annotations

import os
import time
from pathlib import Path

import modal

from .schema import SCHEMA_VERSION, sentiment_question_dict

APP_NAME = "laya-darija-sentiment"
MODEL_REPO = "convaiinnovations/laya"
MODEL_REVISION = "1c5edc17a7acd8701df6fc341c0d179f1c62c982"
MODEL_DIR = Path("/models/laya-multilingual")
MODEL_IDENTIFIER = f"{MODEL_REPO}:multilingual@{MODEL_REVISION}+laya-0.3.4"

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name("laya-darija-models", create_if_missing=True)
hf_secret = modal.Secret.from_name("hf-secret", required_keys=["HF_TOKEN"])

download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub[hf_xet]>=0.34")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

inference_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "laya==0.3.4",
        "transformers==5.0.0",
        "fastapi[standard]>=0.115",
    )
    .env(
        {
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
    timeout=1800,
)
def download_model() -> str:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=MODEL_REPO,
        revision=MODEL_REVISION,
        allow_patterns=["multilingual/*"],
        local_dir="/models/laya-bundle",
        token=os.environ["HF_TOKEN"],
    )
    source = Path("/models/laya-bundle/multilingual")
    if MODEL_DIR.exists():
        import shutil

        shutil.rmtree(MODEL_DIR)
    source.rename(MODEL_DIR)
    model_volume.commit()
    return str(MODEL_DIR)


@app.cls(
    image=inference_image,
    gpu="L4",
    max_containers=1,
    scaledown_window=300,
    timeout=600,
    startup_timeout=600,
    volumes={"/models": model_volume},
)
class LayaService:
    @modal.enter()
    def load(self) -> None:
        import laya
        import torch

        if not MODEL_DIR.exists():
            raise RuntimeError("model weights are missing; run download_model first")
        self.agent = laya.load(str(MODEL_DIR), device="cuda")
        print(f"Loaded {MODEL_IDENTIFIER} on {torch.cuda.get_device_name(0)}")

    @modal.fastapi_endpoint(method="POST", docs=True)
    def predict(self, payload: dict) -> dict:
        from fastapi import HTTPException

        review = payload.get("review") if isinstance(payload, dict) else None
        if not isinstance(review, str) or not review.strip():
            raise HTTPException(status_code=422, detail="review must be a non-empty string")
        if len(review) > 5000:
            raise HTTPException(status_code=422, detail="review is too long")
        started = time.perf_counter()
        result = self.agent.predict(
            {"review": review}, {"sentiment": sentiment_question_dict()}
        )
        inference_ms = (time.perf_counter() - started) * 1000
        answer = result["answers"]["sentiment"]
        return {
            "label": answer["choice"],
            "probabilities": answer["probabilities"],
            "model": MODEL_IDENTIFIER,
            "schema_version": SCHEMA_VERSION,
            "inference_ms": inference_ms,
            "raw": result,
        }

    @modal.fastapi_endpoint(method="GET", docs=True)
    def health(self) -> dict:
        return {"status": "ok", "model": MODEL_IDENTIFIER, "gpu": "L4"}

