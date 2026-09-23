from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import modal

APP_NAME = "djev-run-l40s"
MODEL_REPO = "nvidia/diffusiongemma-26B-A4B-it-NVFP4"
MODEL_REVISION = "ec4ff3df205028f4e81c954c2227f9312b3ec2ea"
IMAGE = (
    "ghcr.io/taeold/djev-run@"
    "sha256:fa646d1637c4411a8ea7a74cd3786fba5f9703585c7c912a0dcda6d68dde5f5c"
)
MODEL_IDENTIFIER = f"{MODEL_REPO}@{MODEL_REVISION}+{IMAGE}"
MODEL_DIR = Path("/mnt/gcs/dgemma")

app = modal.App(APP_NAME)
model_volume = modal.Volume.from_name("djev-run-models", create_if_missing=True)
hf_secret = modal.Secret.from_name("hf-secret", required_keys=["HF_TOKEN"])

download_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub[hf_xet]>=0.34")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)
serve_image = modal.Image.from_registry(IMAGE, add_python="3.12").entrypoint([])


@app.function(
    image=download_image,
    volumes={"/mnt/gcs": model_volume},
    secrets=[hf_secret],
    timeout=3600,
)
def download_model() -> str:
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=MODEL_REPO,
        revision=MODEL_REVISION,
        local_dir=str(MODEL_DIR),
        token=os.environ["HF_TOKEN"],
    )
    model_volume.commit()
    return path


@app.function(
    image=serve_image,
    gpu="L40S",
    cpu=8,
    memory=65536,
    max_containers=1,
    scaledown_window=120,
    timeout=900,
    volumes={"/mnt/gcs": model_volume},
    env={
        "PORT": "8080",
        "CANVAS": "128",
        "MAX_SEQS": "4",
        "MAX_MODEL_LEN": "4096",
        "GPU_UTIL": "0.85",
        "KV_CACHE_GB": "2",
        "DISABLE_MM": "1",
        "TEST_PAGE": "1",
        "TORCH_COMPILE_DISABLE": "1",
        "CUDA_MODULE_LOADING": "LAZY",
    },
)
@modal.web_server(8080, startup_timeout=900)
def serve() -> None:
    if not (MODEL_DIR / "model.safetensors.index.json").exists():
        raise RuntimeError("model weights are missing; run download_model first")

    # The Cloud Run entrypoint stages weights in /dev/shm. Modal's default disk
    # quota has room for this 18 GiB staging copy.
    entrypoint = Path("/entrypoint.sh").read_text()
    original_path = "/dev/shm/dgemma"
    if original_path not in entrypoint:
        raise RuntimeError("the pinned djev image has an unexpected entrypoint")
    original_health = '{"status": "ok", "mode": "inproc-vllm"}'
    if original_health not in entrypoint:
        raise RuntimeError("the pinned djev image has an unexpected health response")
    modal_entrypoint = Path("/tmp/djev-modal-entrypoint.sh")
    health = {
        "status": "ok",
        "mode": "inproc-vllm",
        "model": "dgemma",
        "model_revision": MODEL_REVISION,
        "image": IMAGE,
    }
    modal_entrypoint.write_text(
        entrypoint.replace(original_path, "/tmp/dgemma")
        .replace(original_health, json.dumps(health))
    )
    env = os.environ.copy()
    # Modal injects its own Python 3.12, while the upstream image installs
    # vLLM and its dependencies in Debian's dist-packages directory.
    env["PYTHONPATH"] = "/usr/local/lib/python3.12/dist-packages"
    subprocess.Popen(["bash", str(modal_entrypoint)], env=env)
