"""Inference pipelines for the four workspaces. Pure NumPy + onnxruntime (no PyTorch in the backend).

Corruptions are produced by pet_data.py, the same module that generated the training, validation and test corruptions.
"""
from __future__ import annotations

import secrets
import time
from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel, Field

import pet_data as pdt

from . import imaging as I
from . import metrics as Mx
from .registry import REGISTRY

CLASS_NAMES = list(pdt.CLASS_NAMES)                      # clean, salt_pepper, gaussian_blur, occlusion
LABEL = {n: i for i, n in enumerate(CLASS_NAMES)}
EXPERT_ROLE = {1: "expert_salt_pepper", 2: "expert_gaussian_blur", 3: "expert_occlusion"}
BRANCHES = ["identity", "A_salt", "A_blur", "A_occ"]
BRANCH_LABELS = ["Identity (clean)", "Salt-and-pepper expert", "Blur expert", "Occlusion expert"]


class CorruptionSpec(BaseModel):
    type: Literal["none", "clean", "salt_pepper", "gaussian_blur", "occlusion"] = Field(
        "none", description="'none' = use the image as uploaded (already corrupted, no reference available)")
    severity: Literal["low", "medium", "high", "custom", "random"] = Field(
        "medium", description="low/medium/high = the fixed test-set severities; random = sampled from the training distribution")
    p: Optional[float] = Field(None, ge=0.001, le=0.5, description="salt-and-pepper probability (custom)")
    kernel_size: Optional[Literal[3, 5, 7]] = None
    sigma: Optional[float] = Field(None, ge=0.1, le=5.0)
    n_boxes: Optional[int] = Field(None, ge=1, le=3)
    coverage: Optional[float] = Field(None, ge=0.02, le=0.6)
    seed: Optional[int] = Field(None, ge=0, le=2**31 - 2)


def build_corruption(spec: CorruptionSpec) -> tuple[Optional[dict], list[str]]:
    """Returns (params dict accepted by pet_data.apply_corruption, warnings). None for type 'none'."""
    if spec.type == "none":
        return None, []
    label = LABEL[spec.type]
    seed = spec.seed if spec.seed is not None else secrets.randbelow(2**31 - 1)
    warn = []
    if label == 0:
        return {"label": 0, "type": "clean", "seed": seed}, warn
    if spec.severity in ("low", "medium", "high"):
        return pdt.fixed_params(label, spec.severity, seed), warn
    if spec.severity == "random":
        return pdt.sample_params(label, np.random.default_rng(seed)), warn
    # custom
    base = {"label": label, "type": spec.type, "seed": seed}
    if label == pdt.SALT_PEPPER:
        p = spec.p if spec.p is not None else 0.08
        if not pdt.SP_PROB_RANGE[0] <= p <= pdt.SP_PROB_RANGE[1]:
            warn.append(f"p={p} is outside the training range {pdt.SP_PROB_RANGE}")
        return {**base, "p": float(p)}, warn
    if label == pdt.BLUR:
        k = spec.kernel_size or 5
        s = spec.sigma if spec.sigma is not None else 1.5
        if not pdt.BLUR_SIGMA_RANGE[0] <= s <= pdt.BLUR_SIGMA_RANGE[1]:
            warn.append(f"sigma={s} is outside the training range {pdt.BLUR_SIGMA_RANGE}")
        return {**base, "kernel_size": int(k), "sigma": float(s)}, warn
    n = spec.n_boxes or 2
    cov = spec.coverage if spec.coverage is not None else 0.2
    if not pdt.OCC_COVERAGE_RANGE[0] <= cov <= pdt.OCC_COVERAGE_RANGE[1]:
        warn.append(f"coverage={cov} is outside the training range {pdt.OCC_COVERAGE_RANGE}")
    try:
        boxes, actual = pdt.sample_occlusion_boxes(np.random.default_rng(seed), n, cov)
    except RuntimeError:
        boxes, actual = pdt.sample_occlusion_boxes(np.random.default_rng(seed), n, cov, accept=lambda a: abs(a - cov) <= 0.05)
        warn.append("requested coverage could only be matched within ±5%")
    return {**base, "n_boxes": n, "coverage_target": cov, "coverage": actual, "boxes": boxes}, warn


def prepare(img: np.ndarray, spec: CorruptionSpec) -> dict:
    """img = preprocessed upload/sample (HWC [0,1]). If a corruption is requested it is applied and img becomes the reference."""
    params, warn = build_corruption(spec)
    if params is None:
        return {"input": img, "reference": None, "corruption": {"applied": False, "type": "none",
                "description": "image used as uploaded (assumed already corrupted)", "params": None, "warnings": []}}
    corrupted = pdt.apply_corruption(img, params).astype(np.float32)
    jsonable = {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in params.items()}
    return {"input": corrupted, "reference": img, "corruption": {
        "applied": True, "type": params["type"], "label": int(params["label"]), "severity": spec.severity,
        "description": pdt.describe_params(params), "params": jsonable, "warnings": warn,
        "in_training_distribution": not warn}}


def _images_and_metrics(prep: dict, output: np.ndarray) -> tuple[dict, Optional[dict]]:
    ref = prep["reference"]
    images = {"input": I.png_data_url(prep["input"]), "output": I.png_data_url(output)}
    if ref is None:
        return images, None
    images["reference"] = I.png_data_url(ref)
    images["error_map"] = I.png_data_url(I.error_map(output, ref))
    images["input_error_map"] = I.png_data_url(I.error_map(prep["input"], ref))
    m_in, m_out = Mx.quality(prep["input"], ref), Mx.quality(output, ref)
    gain = None if (m_in["psnr"] is None or m_out["psnr"] is None) else round(m_out["psnr"] - m_in["psnr"], 3)
    return images, {"input_vs_reference": m_in, "output_vs_reference": m_out, "psnr_gain_db": gain,
                    "ssim_gain": round(m_out["ssim"] - m_in["ssim"], 4)}


# ---------------------------------------------------------------------------------------------- Task 1
def universal(img: np.ndarray, spec: CorruptionSpec) -> dict:
    prep = prepare(img, spec)
    (out,), ms = REGISTRY.run("universal", {"input": I.to_nchw(prep["input"])})
    output = I.from_nchw(out)
    images, metrics = _images_and_metrics(prep, output)
    sc = REGISTRY.sidecars.get("universal", {})
    return {"workspace": "universal_restoration", "images": images, "metrics": metrics, "corruption": prep["corruption"],
            "timing_ms": {"inference": round(ms, 2)},
            "model": {"name": "Universal denoising autoencoder", "file": "task1_universal_dae.onnx",
                      "code_dim": sc.get("code_dim"), "compression_ratio": sc.get("compression_ratio")}}


# ---------------------------------------------------------------------------------------------- Task 2
def hard_routing(img: np.ndarray, spec: CorruptionSpec, _prep: dict | None = None) -> dict:
    prep = _prep or prepare(img, spec)
    x = I.to_nchw(prep["input"])
    (logits, probs), clf_ms = REGISTRY.run("classifier", {"input": x})
    probs = probs[0].astype(float)
    pred = int(np.argmax(probs))
    if pred == 0:                                       # identity bypass: a clean input is not processed by any expert
        output, exp_ms, role = prep["input"].copy(), 0.0, "identity"
    else:
        role = EXPERT_ROLE[pred]
        (out,), exp_ms = REGISTRY.run(role, {"input": x})
        output = I.from_nchw(out)
    images, metrics = _images_and_metrics(prep, output)
    applied = prep["corruption"].get("label")
    return {"workspace": "hard_routed_restoration", "images": images, "metrics": metrics, "corruption": prep["corruption"],
            "classifier": {"probabilities": {n: round(float(p), 5) for n, p in zip(CLASS_NAMES, probs)},
                           "logits": {n: round(float(v), 4) for n, v in zip(CLASS_NAMES, logits[0])},
                           "predicted": CLASS_NAMES[pred], "confidence": round(float(probs[pred]), 5)},
            "routing": {"selected_branch": BRANCHES[pred], "selected_label": BRANCH_LABELS[pred], "model_role": role,
                        "identity_bypass": pred == 0,
                        "applied_corruption": None if applied is None else CLASS_NAMES[applied],
                        "prediction_matches_applied": None if applied is None else bool(applied == pred)},
            "timing_ms": {"classifier": round(clf_ms, 2), "expert": round(exp_ms, 2), "inference": round(clf_ms + exp_ms, 2)},
            "model": {"name": "Corruption classifier + specialist autoencoders",
                      "files": ["task2_classifier.onnx"] + [f"task2_expert_{n}.onnx" for n in CLASS_NAMES[1:]]}}


# ---------------------------------------------------------------------------------------------- Task 3
def soft_moe(img: np.ndarray, spec: CorruptionSpec, compare_hard: bool = False, include_branches: bool = True) -> dict:
    prep = prepare(img, spec)
    lm = REGISTRY.get("soft_moe")
    want = ["output", "weights", "logits"] + (["branches"] if include_branches and "branches" in lm.extra_outputs else [])
    res, ms = REGISTRY.run("soft_moe", {"input": I.to_nchw(prep["input"])}, want)
    out, w, logits = res[0], res[1][0].astype(float), res[2][0]
    output = I.from_nchw(out)
    images, metrics = _images_and_metrics(prep, output)
    order = np.argsort(-w)
    dominant = int(order[0])
    result = {"workspace": "soft_moe_restoration", "images": images, "metrics": metrics, "corruption": prep["corruption"],
              "routing": {"weights": {b: round(float(v), 5) for b, v in zip(BRANCHES, w)},
                          "weights_list": [round(float(v), 5) for v in w], "branch_labels": BRANCH_LABELS,
                          "dominant_branch": BRANCHES[dominant], "dominant_label": BRANCH_LABELS[dominant],
                          "dominant_weight": round(float(w[dominant]), 5),
                          "mode": "dominant" if w[dominant] > 0.9 else ("distributed" if w[dominant] < 0.6 else "mixed"),
                          "ranking": [BRANCHES[i] for i in order],
                          "entropy": round(float(-(w * np.log(np.clip(w, 1e-9, 1))).sum()), 4),
                          "gate_logits": {b: round(float(v), 4) for b, v in zip(BRANCHES, logits)},
                          "tau": REGISTRY.sidecars.get("soft_moe", {}).get("tau"),
                          "equation": "output = " + " + ".join(f"{v:.2f}·{b}" for b, v in zip(BRANCHES, w))},
              "timing_ms": {"inference": round(ms, 2)},
              "model": {"name": "Jointly trained soft mixture-of-experts", "file": "task3_soft_moe.onnx"}}
    if len(res) > 3:                                     # each branch's own output, as computed inside the mixture
        br = res[3][0]
        result["branch_images"] = {b: I.png_data_url(np.clip(br[k].transpose(1, 2, 0), 0, 1)) for k, b in enumerate(BRANCHES)}
        if prep["reference"] is not None:
            result["branch_metrics"] = {b: Mx.quality(np.clip(br[k].transpose(1, 2, 0), 0, 1), prep["reference"])
                                        for k, b in enumerate(BRANCHES)}
    if compare_hard:
        h = hard_routing(img, spec, _prep=prep)
        result["hard_routing"] = {"output": h["images"]["output"], "metrics": h["metrics"], "routing": h["routing"],
                                  "classifier": h["classifier"], "timing_ms": h["timing_ms"]}
    return result


# ---------------------------------------------------------------------------------------------- Task 4
STYLE_NAMES = ["Style 1", "Style 2", "Style 3"]


def face_to_sketch(photo: np.ndarray, styles: list[int]) -> dict:
    x = np.repeat(I.to_nchw(photo), len(styles), axis=0)
    (sk,), ms = REGISTRY.run("sketch", {"photo": x, "style": np.array(styles, dtype=np.int64)})
    sketches = [{"style": s, "style_name": STYLE_NAMES[s], "image": I.png_data_url(np.clip(sk[i].transpose(1, 2, 0), 0, 1))}
                for i, s in enumerate(styles)]
    return {"workspace": "face_to_sketch", "images": {"photo": I.png_data_url(photo)}, "sketches": sketches,
            "timing_ms": {"inference": round(ms, 2), "per_sketch": round(ms / len(styles), 2)},
            "model": {"name": "Style-conditioned conditional GAN generator (U-Net)", "file": "task4_face2sketch_generator.onnx"}}
