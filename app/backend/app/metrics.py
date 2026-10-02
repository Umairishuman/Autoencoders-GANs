"""PSNR / SSIM in NumPy, matching restoration.py (11x11 Gaussian window, sigma 1.5, 'valid' region, data range 1)."""
from __future__ import annotations

import numpy as np

_g = np.exp(-0.5 * ((np.arange(11) - 5) / 1.5) ** 2)
_G = (_g / _g.sum()).astype(np.float64)


def _filt(img: np.ndarray) -> np.ndarray:
    """Separable 11x11 Gaussian filter, 'valid' convolution, per channel. img: (H, W, C)."""
    h, w = img.shape[:2]
    rows = sum(_G[i] * img[:, i:w - 10 + i] for i in range(11))
    return sum(_G[i] * rows[i:h - 10 + i] for i in range(11))


def psnr(a: np.ndarray, b: np.ndarray) -> float | None:
    """None when the images are identical (infinite PSNR)."""
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return None if mse < 1e-10 else 10 * np.log10(1.0 / mse)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64), b.astype(np.float64)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_a, mu_b = _filt(a), _filt(b)
    saa = _filt(a * a) - mu_a ** 2
    sbb = _filt(b * b) - mu_b ** 2
    sab = _filt(a * b) - mu_a * mu_b
    m = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2))
    return float(m.mean())


def quality(img: np.ndarray, reference: np.ndarray) -> dict:
    p = psnr(img, reference)
    return {"psnr": None if p is None else round(p, 3), "ssim": round(ssim(img, reference), 4),
            "mae": round(float(np.abs(img - reference).mean()), 5), "identical": p is None}
