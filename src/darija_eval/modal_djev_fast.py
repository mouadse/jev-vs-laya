"""Low-latency Modal Server for the pinned djev image and its built-in demos."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import modal

from .modal_djev import IMAGE, MODEL_REVISION, model_volume, serve_image

app = modal.App("djev-run-l40s-fast")


@app.server(
    image=serve_image,
    gpu="L40S",
    cpu=8,
    memory=65536,
    max_containers=1,
    scaledown_window=120,
    startup_timeout=900,
    port=8080,
    unauthenticated=True,
    routing_region="eu-west",
    compute_region="eu",
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
class DjevFastServer:
    @modal.enter()
    def start(self) -> None:
        if not Path("/mnt/gcs/dgemma/model.safetensors.index.json").exists():
            raise RuntimeError("model weights are missing; run download_model first")

        entrypoint = Path("/entrypoint.sh").read_text()
        original_path = "/dev/shm/dgemma"
        original_health = '{"status": "ok", "mode": "inproc-vllm"}'
        if original_path not in entrypoint or original_health not in entrypoint:
            raise RuntimeError("the pinned djev image has an unexpected entrypoint")
        health = {
            "status": "ok",
            "mode": "inproc-vllm",
            "model": "dgemma",
            "model_revision": MODEL_REVISION,
            "image": IMAGE,
        }
        patched = Path("/tmp/djev-modal-entrypoint.sh")
        patched.write_text(
            entrypoint.replace(original_path, "/tmp/dgemma")
            .replace(original_health, json.dumps(health))
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = "/usr/local/lib/python3.12/dist-packages"
        subprocess.Popen(["bash", str(patched)], env=env)
