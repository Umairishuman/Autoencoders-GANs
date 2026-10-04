"""Runtime configuration (environment variables, with defaults for local development)."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = Path(os.environ.get("MODELS_DIR", REPO_ROOT / "AI" / "models"))
SAMPLES_DIR = Path(os.environ.get("SAMPLES_DIR", REPO_ROOT / "backend" / "samples"))
MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", "10"))
CORS_ORIGINS = [o.strip() for o in os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000,http://localhost:8080").split(",") if o.strip()]
WANDB_URL = os.environ.get("WANDB_URL", "https://forge.coreweave.com/wandb/umairishuman-national-university-of-computer-and-emergin/genai-assignment1")
ORT_THREADS = int(os.environ.get("ORT_THREADS", "0"))          # 0 = onnxruntime default
WARMUP = os.environ.get("WARMUP", "1") == "1"

IMG_SIZE = 128
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}

# file name -> role. The sidecar JSONs are optional but are shown in /api/models when present.
MODEL_FILES = {
    "universal": "task1_universal_dae.onnx",
    "classifier": "task2_classifier.onnx",
    "expert_salt_pepper": "task2_expert_salt_pepper.onnx",
    "expert_gaussian_blur": "task2_expert_gaussian_blur.onnx",
    "expert_occlusion": "task2_expert_occlusion.onnx",
    "soft_moe": "task3_soft_moe.onnx",
    "sketch": "task4_face2sketch_generator.onnx",
}
SIDECARS = {
    "universal": "task1_universal_dae.json",
    "hard_routing": "task2_hard_routing.json",
    "soft_moe": "task3_soft_moe.json",
    "sketch": "task4_face2sketch_generator.json",
}
# tensor inside the soft-MoE graph that holds the stacked branch outputs [B, 4, 3, H, W]
# (torch.stack([x, A_salt(x), A_blur(x), A_occ(x)], 1) in moe.SoftMoE.forward)
MOE_BRANCH_TENSOR = os.environ.get("MOE_BRANCH_TENSOR", "/moe/Concat_output_0")
