"""Restoration Lab — FastAPI backend serving the four ONNX workspaces.

Endpoints (all JSON; images are returned as PNG data URLs):
  GET  /api/health                  liveness + which models are loaded
  GET  /api/models                  model files, shapes, warm-up latency, training sidecars (hyper-parameters, parity, metrics)
  GET  /api/samples?kind=pets|faces sample images bundled with the app;  GET /api/samples/{kind}/{name} returns the file
  POST /api/corrupt                 preview a corruption (no model)
  POST /api/restore/universal       Task 1  universal denoising autoencoder
  POST /api/restore/hard            Task 2  classifier -> identity bypass or one specialist
  POST /api/restore/soft            Task 3  soft mixture-of-experts (weights, per-branch outputs, optional hard comparison)
  POST /api/sketch                  Task 4  face photo + style (0, 1, 2 or 'all') -> sketch
Interactive documentation: /docs
"""
from __future__ import annotations

import logging
import sys
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import ValidationError

from . import config

if str(config.REPO_ROOT / "src") not in sys.path:        # shared training modules (pet_data.py) for local runs
    sys.path.insert(0, str(config.REPO_ROOT / "src"))

import onnxruntime as ort  # noqa: E402

from . import imaging as I  # noqa: E402
from . import pipelines as P  # noqa: E402
from .registry import REGISTRY, ModelUnavailable  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")
STARTED = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.time()
    REGISTRY.load()
    log.info("models loaded in %.1fs: %s", time.time() - t0, REGISTRY.status())
    yield


app = FastAPI(title="Restoration Lab API", version="1.0.0", lifespan=lifespan,
              description="Generative AI Assignment 1 — universal restoration, hard routing, soft mixture-of-experts and "
                          "style-conditioned face-to-sketch generation, served from ONNX Runtime.")
app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS, allow_credentials=False,
                   allow_methods=["GET", "POST"], allow_headers=["*"])


# ---------------------------------------------------------------------------------------- errors
@app.exception_handler(I.ImageError)
async def _image_error(_: Request, e: I.ImageError):
    return JSONResponse(status_code=e.status, content={"error": "invalid_image", "detail": str(e)})


@app.exception_handler(ModelUnavailable)
async def _model_error(_: Request, e: ModelUnavailable):
    return JSONResponse(status_code=503, content={"error": "model_unavailable", "detail": str(e), "model": e.role})


@app.exception_handler(ValidationError)
async def _validation_error(_: Request, e: ValidationError):
    return JSONResponse(status_code=422, content={"error": "invalid_parameters",
                                                  "detail": [{"field": ".".join(map(str, x["loc"])), "message": x["msg"]} for x in e.errors()]})


# ---------------------------------------------------------------------------------------- helpers
SAMPLE_KINDS = ("pets", "faces")
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _samples(kind: str) -> list:
    d = config.SAMPLES_DIR / kind
    return sorted(p for p in d.glob("*") if p.suffix.lower() in IMG_EXT) if d.is_dir() else []


async def _read_source(file: Optional[UploadFile], sample: Optional[str], kind: str) -> tuple[bytes, dict]:
    if file is not None and sample:
        raise I.ImageError("Send either a file or a sample name, not both.", 400)
    if file is not None:
        data = await file.read()
        I.validate_upload(data, file.content_type)
        return data, {"source": "upload", "filename": file.filename, "bytes": len(data)}
    if sample:
        match = [p for p in _samples(kind) if p.name == sample]
        if not match:
            raise I.ImageError(f"Unknown {kind} sample '{sample}'. See GET /api/samples?kind={kind}.", 404)
        data = match[0].read_bytes()
        return data, {"source": "sample", "filename": sample, "bytes": len(data)}
    raise I.ImageError("No image: upload a file (field 'file') or choose a sample (field 'sample').", 400)


def _spec(corruption, severity, p, kernel_size, sigma, n_boxes, coverage, seed) -> P.CorruptionSpec:
    return P.CorruptionSpec(type=corruption, severity=severity, p=p, kernel_size=kernel_size, sigma=sigma,
                            n_boxes=n_boxes, coverage=coverage, seed=seed)


def _finish(result: dict, src: dict, t0: float, t_pre: float) -> dict:
    result["source"] = src
    result.setdefault("timing_ms", {})
    result["timing_ms"]["preprocess"] = round(t_pre * 1000, 2)
    result["timing_ms"]["total"] = round((time.perf_counter() - t0) * 1000, 2)
    result["image_size"] = [config.IMG_SIZE, config.IMG_SIZE]
    return result


# Shared form fields for the restoration endpoints
CORRUPTION_DOC = "none | clean | salt_pepper | gaussian_blur | occlusion"
SEVERITY_DOC = "low | medium | high (fixed test severities) | random (training distribution) | custom (use the parameter fields)"


# ---------------------------------------------------------------------------------------- endpoints
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/api/health", tags=["system"])
def health():
    st = REGISTRY.status()
    return {**st, "uptime_s": round(time.time() - STARTED, 1), "onnxruntime": ort.__version__,
            "providers": ort.get_available_providers(), "device": "CPU"}


@app.get("/api/models", tags=["system"])
def models():
    return {"models": {k: m.info() for k, m in REGISTRY.models.items()}, "sidecars": REGISTRY.sidecars,
            "experiment_tracking": config.WANDB_URL}


@app.get("/api/samples", tags=["samples"])
def list_samples(kind: str = "pets"):
    if kind not in SAMPLE_KINDS:
        raise HTTPException(400, f"kind must be one of {SAMPLE_KINDS}")
    return {"kind": kind, "samples": [{"name": p.name, "url": f"/api/samples/{kind}/{p.name}"} for p in _samples(kind)]}


@app.get("/api/samples/{kind}/{name}", tags=["samples"])
def get_sample(kind: str, name: str):
    if kind not in SAMPLE_KINDS:
        raise HTTPException(400, f"kind must be one of {SAMPLE_KINDS}")
    match = [p for p in _samples(kind) if p.name == name]
    if not match:
        raise HTTPException(404, "sample not found")
    return FileResponse(match[0])


@app.post("/api/corrupt", tags=["restoration"])
async def corrupt(file: Optional[UploadFile] = File(None), sample: Optional[str] = Form(None),
                  corruption: str = Form("salt_pepper", description=CORRUPTION_DOC), severity: str = Form("medium", description=SEVERITY_DOC),
                  p: Optional[float] = Form(None), kernel_size: Optional[int] = Form(None), sigma: Optional[float] = Form(None),
                  n_boxes: Optional[int] = Form(None), coverage: Optional[float] = Form(None), seed: Optional[int] = Form(None)):
    """Apply a corruption with the training pipeline's own code and return it (no model is run)."""
    t0 = time.perf_counter()
    data, src = await _read_source(file, sample, "pets")
    img = I.preprocess_pet(data)
    prep = P.prepare(img, _spec(corruption, severity, p, kernel_size, sigma, n_boxes, coverage, seed))
    res = {"images": {"input": I.png_data_url(prep["input"]), "reference": I.png_data_url(img)}, "corruption": prep["corruption"]}
    return _finish(res, src, t0, time.perf_counter() - t0)


@app.post("/api/restore/universal", tags=["restoration"])
async def restore_universal(file: Optional[UploadFile] = File(None), sample: Optional[str] = Form(None),
                            corruption: str = Form("none", description=CORRUPTION_DOC), severity: str = Form("medium", description=SEVERITY_DOC),
                            p: Optional[float] = Form(None), kernel_size: Optional[int] = Form(None), sigma: Optional[float] = Form(None),
                            n_boxes: Optional[int] = Form(None), coverage: Optional[float] = Form(None), seed: Optional[int] = Form(None)):
    """Task 1 — one autoencoder for every condition, without being told the corruption."""
    t0 = time.perf_counter()
    data, src = await _read_source(file, sample, "pets")
    img = I.preprocess_pet(data); t_pre = time.perf_counter() - t0
    return _finish(P.universal(img, _spec(corruption, severity, p, kernel_size, sigma, n_boxes, coverage, seed)), src, t0, t_pre)


@app.post("/api/restore/hard", tags=["restoration"])
async def restore_hard(file: Optional[UploadFile] = File(None), sample: Optional[str] = Form(None),
                       corruption: str = Form("none", description=CORRUPTION_DOC), severity: str = Form("medium", description=SEVERITY_DOC),
                       p: Optional[float] = Form(None), kernel_size: Optional[int] = Form(None), sigma: Optional[float] = Form(None),
                       n_boxes: Optional[int] = Form(None), coverage: Optional[float] = Form(None), seed: Optional[int] = Form(None)):
    """Task 2 — the classifier predicts the corruption; clean inputs bypass restoration, otherwise one specialist restores."""
    t0 = time.perf_counter()
    data, src = await _read_source(file, sample, "pets")
    img = I.preprocess_pet(data); t_pre = time.perf_counter() - t0
    return _finish(P.hard_routing(img, _spec(corruption, severity, p, kernel_size, sigma, n_boxes, coverage, seed)), src, t0, t_pre)


@app.post("/api/restore/soft", tags=["restoration"])
async def restore_soft(file: Optional[UploadFile] = File(None), sample: Optional[str] = Form(None),
                       corruption: str = Form("none", description=CORRUPTION_DOC), severity: str = Form("medium", description=SEVERITY_DOC),
                       p: Optional[float] = Form(None), kernel_size: Optional[int] = Form(None), sigma: Optional[float] = Form(None),
                       n_boxes: Optional[int] = Form(None), coverage: Optional[float] = Form(None), seed: Optional[int] = Form(None),
                       compare_hard: bool = Form(False), include_branches: bool = Form(True)):
    """Task 3 — all four branches weighted by the gate: output = w0·x + w1·A_salt(x) + w2·A_blur(x) + w3·A_occ(x)."""
    t0 = time.perf_counter()
    data, src = await _read_source(file, sample, "pets")
    img = I.preprocess_pet(data); t_pre = time.perf_counter() - t0
    spec = _spec(corruption, severity, p, kernel_size, sigma, n_boxes, coverage, seed)
    return _finish(P.soft_moe(img, spec, compare_hard=compare_hard, include_branches=include_branches), src, t0, t_pre)


@app.post("/api/sketch", tags=["generation"])
async def sketch(file: Optional[UploadFile] = File(None), sample: Optional[str] = Form(None),
                 style: str = Form("0", description="0, 1, 2 (= Style 1, 2, 3) or 'all'")):
    """Task 4 — style-conditioned face-to-sketch generation (upload or webcam capture)."""
    t0 = time.perf_counter()
    s = style.strip().lower()
    if s == "all":
        styles = [0, 1, 2]
    elif s in {"0", "1", "2"}:
        styles = [int(s)]
    elif s in {"style 1", "style 2", "style 3"}:
        styles = [int(s[-1]) - 1]
    else:
        raise I.ImageError("style must be 0, 1, 2 or 'all'", 422)
    data, src = await _read_source(file, sample, "faces")
    photo = I.preprocess_face(data); t_pre = time.perf_counter() - t0
    return _finish(P.face_to_sketch(photo, styles), src, t0, t_pre)
