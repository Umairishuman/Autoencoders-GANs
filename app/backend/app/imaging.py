"""Upload validation, training-identical preprocessing, and image encoding."""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image, UnidentifiedImageError

from . import config

Image.MAX_IMAGE_PIXELS = 50_000_000          # decompression-bomb guard


class ImageError(ValueError):
    """Raised for invalid uploads; mapped to HTTP 4xx by the API."""

    def __init__(self, message: str, status: int = 422):
        super().__init__(message)
        self.status = status


def validate_upload(data: bytes, content_type: str | None) -> None:
    if not data:
        raise ImageError("Empty file.", 400)
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise ImageError(f"File too large ({len(data) / 1e6:.1f} MB > {config.MAX_UPLOAD_MB:.0f} MB).", 413)
    if content_type and content_type.split(";")[0].strip().lower() not in config.ALLOWED_TYPES | {"application/octet-stream"}:
        raise ImageError(f"Unsupported file type '{content_type}'. Use JPEG, PNG, WebP or BMP.", 415)
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        with Image.open(io.BytesIO(data)) as im:
            if min(im.size) < 16:
                raise ImageError(f"Image too small ({im.size[0]}x{im.size[1]}); need at least 16x16 pixels.")
    except UnidentifiedImageError:
        raise ImageError("The file is not a readable image.", 415)
    except Image.DecompressionBombError:
        raise ImageError("Image has too many pixels.", 413)


def _resize_rgb(im: Image.Image, size: int) -> np.ndarray:
    return np.asarray(im.resize((size, size), Image.Resampling.BICUBIC), dtype=np.uint8).copy()


def preprocess_pet(data: bytes, size: int = config.IMG_SIZE) -> np.ndarray:
    """Tasks 1-3: identical to pet_data.load_standardised_image (convert('RGB') -> bicubic resize). Returns float32 HWC [0,1]."""
    with Image.open(io.BytesIO(data)) as im:
        arr = _resize_rgb(im.convert("RGB"), size)
    return arr.astype(np.float32) / 255.0


def preprocess_face(data: bytes, size: int = config.IMG_SIZE) -> np.ndarray:
    """Task 4: identical to sketch_gan.load_rgb (transparent pixels composited on white). Returns float32 HWC [0,1]."""
    with Image.open(io.BytesIO(data)) as im:
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, im)
        arr = _resize_rgb(im.convert("RGB"), size)
    return arr.astype(np.float32) / 255.0


def to_nchw(img_hwc: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(img_hwc.transpose(2, 0, 1)[None], dtype=np.float32)


def from_nchw(t: np.ndarray) -> np.ndarray:
    return np.clip(t[0].transpose(1, 2, 0), 0.0, 1.0)


def png_data_url(img_hwc01: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray((np.clip(img_hwc01, 0, 1) * 255 + 0.5).astype(np.uint8)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# 'magma'-like colour map (anchor points sampled from matplotlib's magma) for error maps, without matplotlib
_MAGMA = np.array([[0, 0, 4], [28, 16, 68], [79, 18, 123], [129, 37, 129], [181, 54, 122],
                   [229, 80, 100], [251, 135, 97], [254, 194, 135], [252, 253, 191]], dtype=np.float32) / 255.0


def error_map(output: np.ndarray, reference: np.ndarray, vmax: float = 0.5) -> np.ndarray:
    """|output - reference| averaged over RGB, mapped through a magma-like colour scale (0 .. vmax)."""
    e = np.clip(np.abs(output - reference).mean(-1) / vmax, 0, 1) * (len(_MAGMA) - 1)
    lo = np.floor(e).astype(int)
    hi = np.minimum(lo + 1, len(_MAGMA) - 1)
    f = (e - lo)[..., None]
    return _MAGMA[lo] * (1 - f) + _MAGMA[hi] * f
