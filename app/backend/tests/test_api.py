"""API tests against the real ONNX models (run: cd backend && pytest -q).

Checks every endpoint, that responses equal direct onnxruntime calls on the same input (to PNG precision),
that corruptions are produced by pet_data exactly as in training, and that bad requests fail cleanly.
"""
from __future__ import annotations

import base64
import io
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "backend"))


def _make_samples(d: Path):
    rng = np.random.default_rng(0)
    for kind in ("pets", "faces"):
        (d / kind).mkdir(parents=True, exist_ok=True)
        for i in range(2):
            a = (rng.random((180 + 20 * i, 150, 3)) * 255).astype(np.uint8)
            a[40:120, 30:120] = [200, 150, 90]
            Image.fromarray(a).save(d / kind / f"sample_{i}.jpg")


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    samples = tmp_path_factory.mktemp("samples")
    _make_samples(samples)
    os.environ["SAMPLES_DIR"] = str(samples)
    os.environ.setdefault("MODELS_DIR", str(ROOT / "models"))
    from fastapi.testclient import TestClient
    import importlib
    import app.config as cfg
    importlib.reload(cfg)
    from app.main import app
    with TestClient(app) as c:
        yield c


def png_bytes(arr_uint8: np.ndarray) -> bytes:
    buf = io.BytesIO(); Image.fromarray(arr_uint8).save(buf, format="PNG"); return buf.getvalue()


def decode(url: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB"), dtype=np.float32) / 255


@pytest.fixture(scope="session")
def pet_png():
    rng = np.random.default_rng(1)
    a = np.zeros((128, 128, 3), np.uint8); a[..., 0] = np.linspace(30, 220, 128)[None]; a[..., 1] = 120
    a[30:90, 40:100] = [230, 200, 160]
    a = np.clip(a + rng.normal(0, 4, a.shape), 0, 255).astype(np.uint8)
    return png_bytes(a)


def post(client, url, png, **form):
    return client.post(url, files={"file": ("x.png", png, "image/png")}, data={k: str(v) for k, v in form.items()})


# ----------------------------------------------------------------------------- system
def test_health_all_models_loaded(client):
    r = client.get("/api/health").json()
    assert r["status"] == "ok", r
    assert r["models_loaded"] == 7
    assert all(m["loaded"] for m in r["models"].values())


def test_models_and_moe_branches_exposed(client):
    r = client.get("/api/models").json()
    assert r["models"]["soft_moe"]["extra_outputs"] == ["branches"]
    assert set(r["sidecars"]) == {"universal", "hard_routing", "soft_moe", "sketch"}


def test_samples(client):
    r = client.get("/api/samples?kind=pets").json()
    assert len(r["samples"]) == 2
    assert client.get(r["samples"][0]["url"]).status_code == 200


# ----------------------------------------------------------------------------- corruption = training code
def test_corruption_matches_pet_data(client, pet_png):
    import pet_data as pdt
    from app import imaging as I
    r = post(client, "/api/corrupt", pet_png, corruption="gaussian_blur", severity="high", seed=7).json()
    assert r["corruption"]["params"]["kernel_size"] == 7 and r["corruption"]["params"]["sigma"] == 2.5
    ref = pdt.apply_corruption(I.preprocess_pet(pet_png), pdt.fixed_params(2, "high", 7))
    assert np.abs(decode(r["images"]["input"]) - ref).max() <= 1 / 255 + 1e-6


@pytest.mark.parametrize("ctype,sev", [("salt_pepper", "low"), ("salt_pepper", "high"), ("occlusion", "medium"),
                                       ("occlusion", "random"), ("gaussian_blur", "random"), ("clean", "medium")])
def test_corruption_variants(client, pet_png, ctype, sev):
    r = post(client, "/api/corrupt", pet_png, corruption=ctype, severity=sev, seed=3)
    assert r.status_code == 200, r.text
    c = r.json()["corruption"]
    assert c["applied"] and c["type"] == ctype


def test_custom_occlusion_coverage(client, pet_png):
    c = post(client, "/api/corrupt", pet_png, corruption="occlusion", severity="custom", n_boxes=3, coverage=0.3, seed=1).json()["corruption"]
    assert len(c["params"]["boxes"]) == 3 and abs(c["params"]["coverage"] - 0.3) <= 0.05


# ----------------------------------------------------------------------------- task 1
def test_universal_matches_direct_onnx(client, pet_png):
    from app import imaging as I
    from app.registry import REGISTRY
    r = post(client, "/api/restore/universal", pet_png, corruption="none").json()
    x = I.preprocess_pet(pet_png)
    (direct,), _ = REGISTRY.run("universal", {"input": I.to_nchw(x)})
    assert np.abs(decode(r["images"]["output"]) - I.from_nchw(direct)).max() <= 1 / 255 + 1e-6
    assert r["metrics"] is None and r["timing_ms"]["inference"] > 0


def test_universal_with_corruption_reports_metrics(client, pet_png):
    r = post(client, "/api/restore/universal", pet_png, corruption="salt_pepper", severity="high", seed=5).json()
    m = r["metrics"]
    assert {"reference", "error_map"} <= set(r["images"])
    assert m["output_vs_reference"]["ssim"] > m["input_vs_reference"]["ssim"], m       # restoration should help on s&p


# ----------------------------------------------------------------------------- task 2
def test_hard_routing_probabilities_and_bypass(client, pet_png):
    r = post(client, "/api/restore/hard", pet_png, corruption="salt_pepper", severity="medium", seed=2).json()
    probs = r["classifier"]["probabilities"]
    assert abs(sum(probs.values()) - 1) < 1e-4
    assert r["classifier"]["predicted"] == max(probs, key=probs.get)
    assert r["routing"]["prediction_matches_applied"] in (True, False)
    if r["routing"]["identity_bypass"]:
        assert r["timing_ms"]["expert"] == 0.0


# ----------------------------------------------------------------------------- task 3
def test_soft_moe_weights_branches_and_mixture(client, pet_png):
    r = post(client, "/api/restore/soft", pet_png, corruption="gaussian_blur", severity="medium", seed=4, compare_hard="true").json()
    w = r["routing"]["weights_list"]
    assert abs(sum(w) - 1) < 1e-4 and len(w) == 4
    assert set(r["branch_images"]) == {"identity", "A_salt", "A_blur", "A_occ"}
    # the exposed branches recombine into the model's output: sum_k w_k * branch_k == output
    br = [decode(r["branch_images"][b]) for b in ("identity", "A_salt", "A_blur", "A_occ")]
    mix = np.clip(sum(wk * bk for wk, bk in zip(w, br)), 0, 1)
    assert np.abs(mix - decode(r["images"]["output"])).max() < 3 / 255
    assert "hard_routing" in r and r["routing"]["mode"] in ("dominant", "distributed", "mixed")


# ----------------------------------------------------------------------------- task 4
def test_sketch_single_and_all_styles(client):
    face = np.full((160, 140, 4), 255, np.uint8); face[30:130, 30:110, :3] = [210, 170, 150]; face[..., 3] = 255; face[:10, :, 3] = 0
    png = png_bytes(face)
    one = post(client, "/api/sketch", png, style="1").json()
    assert len(one["sketches"]) == 1 and one["sketches"][0]["style_name"] == "Style 2"
    allr = post(client, "/api/sketch", png, style="all").json()
    imgs = [decode(s["image"]) for s in allr["sketches"]]
    assert len(imgs) == 3 and np.abs(imgs[0] - imgs[2]).mean() > 1e-3          # styles produce different sketches


def test_sketch_matches_direct_onnx(client):
    from app import imaging as I
    from app.registry import REGISTRY
    face = (np.random.default_rng(3).random((128, 128, 3)) * 255).astype(np.uint8); png = png_bytes(face)
    r = post(client, "/api/sketch", png, style="2").json()
    (direct,), _ = REGISTRY.run("sketch", {"photo": I.to_nchw(I.preprocess_face(png)), "style": np.array([2], np.int64)})
    assert np.abs(decode(r["sketches"][0]["image"]) - I.from_nchw(direct)).max() <= 1 / 255 + 1e-6


# ----------------------------------------------------------------------------- errors
def test_errors(client, pet_png):
    assert client.post("/api/restore/universal").status_code == 400                                   # no image
    assert client.post("/api/restore/universal", files={"file": ("x.txt", b"hello", "text/plain")}).status_code == 415
    assert client.post("/api/restore/universal", files={"file": ("x.png", b"\x89PNGbroken", "image/png")}).status_code == 415
    assert post(client, "/api/restore/universal", pet_png, corruption="fog").status_code == 422
    assert post(client, "/api/restore/universal", pet_png, corruption="salt_pepper", severity="custom", p=0.9).status_code == 422
    assert post(client, "/api/sketch", pet_png, style="7").status_code == 422
    assert client.post("/api/restore/hard", data={"sample": "nope.jpg"}).status_code == 404
    tiny = png_bytes(np.zeros((8, 8, 3), np.uint8))
    assert post(client, "/api/restore/hard", tiny).status_code == 422


def test_sample_source(client):
    r = client.post("/api/restore/universal", data={"sample": "sample_0.jpg", "corruption": "occlusion", "severity": "low", "seed": "1"})
    assert r.status_code == 200 and r.json()["source"]["source"] == "sample"
