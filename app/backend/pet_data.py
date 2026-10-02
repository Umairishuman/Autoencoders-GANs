"""
pet_data.py — shared data pipeline for Tasks 1–3 (Oxford-IIIT Pet restoration).

Single source of truth for:
  * locating the Oxford-IIIT Pet dataset (Kaggle / torchvision layouts),
  * loading + standardising clean images (RGB, 128x128) with an on-disk cache
    of the *clean* resized images only (no corrupted copies are ever written),
  * the deterministic 80/20 train/val split of the official `trainval` list (seed 42),
  * the four input conditions and their corruption functions (pure NumPy, so the
    FastAPI backend can import them without PyTorch),
  * validation / test corruption manifests (deterministic, fully parameterised),
  * PyTorch datasets, a class-balanced batch sampler and DataLoader builders.

Label convention (used everywhere, incl. the Task 3 gate weight order w0..w3):
    0 = clean, 1 = salt_pepper, 2 = gaussian_blur, 3 = occlusion
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

# --------------------------------------------------------------------------------------
# Constants (the assignment's corruption specification)
# --------------------------------------------------------------------------------------
IMG_SIZE = 128
SPLIT_SEED = 42
VAL_FRACTION = 0.20

CLEAN, SALT_PEPPER, BLUR, OCCLUSION = 0, 1, 2, 3
CLASS_NAMES = ("clean", "salt_pepper", "gaussian_blur", "occlusion")
NUM_CLASSES = 4

SP_PROB_RANGE = (0.02, 0.15)          # uniform
BLUR_KERNELS = (3, 5, 7)              # uniform choice
BLUR_SIGMA_RANGE = (0.5, 2.5)         # uniform
OCC_BOX_RANGE = (1, 3)                # inclusive, uniform
OCC_COVERAGE_RANGE = (0.10, 0.35)     # union of boxes / image area, uniform
OCC_ASPECT_RANGE = (0.5, 2.0)         # box width/height, log-uniform (design choice)
OCC_MIN_SIDE = 4                      # px (design choice)
TEST_OCC_TOLERANCE = 0.01             # |actual - target| coverage for fixed test boxes

SEVERITY_LEVELS = ("low", "medium", "high")
TEST_SEVERITIES = {
    SALT_PEPPER: {"low": {"p": 0.03}, "medium": {"p": 0.08}, "high": {"p": 0.15}},
    BLUR: {"low": {"kernel_size": 3, "sigma": 0.7},
           "medium": {"kernel_size": 5, "sigma": 1.5},
           "high": {"kernel_size": 7, "sigma": 2.5}},
    OCCLUSION: {"low": {"n_boxes": 1, "coverage": 0.10},
                "medium": {"n_boxes": 2, "coverage": 0.20},
                "high": {"n_boxes": 3, "coverage": 0.35}},
}

MANIFEST_VERSION = 1


def corruption_config() -> dict:
    """The full corruption configuration as a JSON-serialisable dict (for W&B / report)."""
    return {
        "image_size": IMG_SIZE,
        "class_names": list(CLASS_NAMES),
        "train_class_probs": [0.25] * 4,
        "salt_pepper": {"p_uniform": list(SP_PROB_RANGE), "salt_vs_pepper": 0.5,
                        "granularity": "per-pixel (all 3 channels)"},
        "gaussian_blur": {"kernel_sizes": list(BLUR_KERNELS), "sigma_uniform": list(BLUR_SIGMA_RANGE),
                          "padding": "reflect", "separable": True},
        "occlusion": {"n_boxes": list(OCC_BOX_RANGE), "coverage_uniform": list(OCC_COVERAGE_RANGE),
                      "aspect_log_uniform": list(OCC_ASPECT_RANGE), "min_side_px": OCC_MIN_SIDE,
                      "boxes_overlap": False, "fill": 0.0},
        "test_severities": {CLASS_NAMES[k]: v for k, v in TEST_SEVERITIES.items()},
        "test_occlusion_tolerance": TEST_OCC_TOLERANCE,
    }


# --------------------------------------------------------------------------------------
# Dataset discovery and loading
# --------------------------------------------------------------------------------------
def read_split_list(path: str | Path) -> list[str]:
    """Read an official list file (e.g. 'Abyssinian_100 1 1 1'). Breed/species ids are discarded."""
    names = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.append(line.split()[0])
    return names


def find_pet_dataset(search_root: str | Path = "/kaggle/input") -> dict:
    """Locate annotations (trainval.txt + test.txt) and the images folder under `search_root`.

    Works for the common Kaggle upload (images/images, annotations/annotations) and the
    torchvision layout (oxford-iiit-pet/images, oxford-iiit-pet/annotations).
    """
    search_root = Path(search_root)
    ann_dir = None
    for tv in sorted(search_root.rglob("trainval.txt")):
        if (tv.parent / "test.txt").exists():
            ann_dir = tv.parent
            break
    if ann_dir is None:
        raise FileNotFoundError(
            f"No folder containing both trainval.txt and test.txt under {search_root}. "
            "The official split lists are required; set IMAGES_DIR / ANNOT_DIR manually.")
    first = read_split_list(ann_dir / "trainval.txt")[0]
    hits = sorted(search_root.rglob(f"{first}.jpg"))
    if not hits:
        raise FileNotFoundError(f"Could not find image '{first}.jpg' under {search_root}.")
    return {"annotations_dir": ann_dir, "images_dir": hits[0].parent}


def load_standardised_image(path: str | Path, size: int = IMG_SIZE) -> tuple[np.ndarray, str]:
    """Open any image, force 3-channel RGB, resize to size x size (bicubic, antialiased).

    Returns (uint8 HxWx3 array, original PIL mode). The backend must use this same function.
    """
    with Image.open(path) as im:
        mode = im.mode
        im = im.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
        return np.asarray(im, dtype=np.uint8).copy(), mode


def load_clean_images(names: Sequence[str], images_dir: str | Path, cache_path: str | Path | None = None,
                      size: int = IMG_SIZE, num_threads: int = 8, verbose: bool = True) -> dict:
    """Load + standardise every image in `names`. Caches the CLEAN resized uint8 array to .npz.

    Returns dict(images=(N,H,W,3) uint8, names=[...kept...], failed=[...], modes=Counter).
    Images that cannot be decoded are reported and dropped (the split is made afterwards).
    """
    names = list(names)
    if cache_path is not None and Path(cache_path).exists():
        z = np.load(cache_path, allow_pickle=False)
        if list(z["requested"]) == names and int(z["size"]) == size:
            out = {"images": z["images"], "names": list(z["names"]), "failed": list(z["failed"]),
                   "modes": Counter(json.loads(str(z["modes"]))), "from_cache": True}
            if verbose:
                print(f"Loaded {len(out['names'])} cached images from {cache_path}")
            return out

    images_dir = Path(images_dir)

    def _load(name):
        try:
            arr, mode = load_standardised_image(images_dir / f"{name}.jpg", size)
            return name, arr, mode, None
        except Exception as e:  # corrupted / unreadable file
            return name, None, None, repr(e)

    with ThreadPoolExecutor(max_workers=num_threads) as ex:
        results = list(ex.map(_load, names))

    kept = [(n, a, m) for n, a, m, err in results if err is None]
    failed = [(n, err) for n, a, m, err in results if err is not None]
    imgs = np.stack([a for _, a, _ in kept]) if kept else np.zeros((0, size, size, 3), np.uint8)
    out = {"images": imgs, "names": [n for n, _, _ in kept], "failed": [n for n, _ in failed],
           "modes": Counter(m for _, _, m in kept), "from_cache": False}
    if verbose:
        print(f"Loaded {len(kept)}/{len(names)} images; failed: {len(failed)}")
        for n, err in failed:
            print(f"  ! {n}: {err}")
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, images=imgs, names=np.array(out["names"]), requested=np.array(names),
                 failed=np.array(out["failed"], dtype=str), size=size, modes=json.dumps(dict(out["modes"])))
    return out


# --------------------------------------------------------------------------------------
# Deterministic split
# --------------------------------------------------------------------------------------
def make_split(names: Sequence[str], val_fraction: float = VAL_FRACTION, seed: int = SPLIT_SEED) -> dict:
    """80/20 random split of the official trainval list with numpy default_rng(seed)."""
    names = list(names)
    n = len(names)
    perm = np.random.default_rng(seed).permutation(n)
    n_val = int(round(val_fraction * n))
    n_train = n - n_val
    train = [names[i] for i in sorted(perm[:n_train])]
    val = [names[i] for i in sorted(perm[n_train:])]
    return {"seed": seed, "val_fraction": val_fraction, "method": "numpy.default_rng(seed).permutation",
            "n_total": n, "train": train, "val": val}


# --------------------------------------------------------------------------------------
# Corruption functions (pure NumPy; img = float32 HxWx3 in [0,1])
# --------------------------------------------------------------------------------------
def salt_and_pepper(img: np.ndarray, p: float, seed: int) -> np.ndarray:
    """Each pixel is selected with prob p; selected pixels become white or black (50/50)."""
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    selected = rng.random((h, w)) < p
    salt = rng.random((h, w)) < 0.5
    out = img.copy()
    out[selected & salt] = 1.0
    out[selected & ~salt] = 0.0
    return out


def gaussian_kernel1d(kernel_size: int, sigma: float) -> np.ndarray:
    x = np.arange(kernel_size, dtype=np.float64) - (kernel_size - 1) / 2.0
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return (k / k.sum()).astype(np.float32)


def gaussian_blur(img: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    """Separable Gaussian blur with reflect padding (matches torchvision's gaussian_blur)."""
    k = gaussian_kernel1d(kernel_size, sigma)
    r = kernel_size // 2
    h, w = img.shape[:2]
    padded = np.pad(img, ((r, r), (r, r), (0, 0)), mode="reflect")
    horiz = sum(k[i] * padded[:, i:i + w] for i in range(kernel_size))        # (h+2r, w, 3)
    out = sum(k[i] * horiz[i:i + h] for i in range(kernel_size))              # (h, w, 3)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def occlude(img: np.ndarray, boxes: Iterable[Sequence[int]], fill: float = 0.0) -> np.ndarray:
    """Black out boxes given as [x0, y0, x1, y1] (x1/y1 exclusive)."""
    out = img.copy()
    for x0, y0, x1, y1 in boxes:
        out[y0:y1, x0:x1] = fill
    return out


def boxes_coverage(boxes, h: int = IMG_SIZE, w: int = IMG_SIZE) -> float:
    mask = np.zeros((h, w), bool)
    for x0, y0, x1, y1 in boxes:
        mask[y0:y1, x0:x1] = True
    return float(mask.mean())


def sample_occlusion_boxes(rng: np.random.Generator, n_boxes: int, coverage: float,
                           h: int = IMG_SIZE, w: int = IMG_SIZE,
                           accept=None, max_attempts: int = 1000) -> tuple[list[list[int]], float]:
    """Sample `n_boxes` non-overlapping boxes whose union covers ~`coverage` of the image.

    The total area is split between boxes (each gets at least half of an equal share), each
    box gets a log-uniform aspect ratio, and boxes are placed at random non-overlapping
    positions. Non-overlap makes the union area equal to the sum of box areas, so the jointly
    covered fraction is controlled precisely. `accept(actual)->bool` decides if rounding is OK.
    """
    if accept is None:
        accept = lambda a: abs(a - coverage) <= TEST_OCC_TOLERANCE  # noqa: E731
    total = coverage * h * w
    lo_ar, hi_ar = np.log(OCC_ASPECT_RANGE[0]), np.log(OCC_ASPECT_RANGE[1])
    for _ in range(max_attempts):
        shares = 0.5 / n_boxes + 0.5 * rng.dirichlet([2.0] * n_boxes)
        boxes, ok = [], True
        for area in shares * total:
            ar = float(np.exp(rng.uniform(lo_ar, hi_ar)))
            bw = int(np.clip(round(np.sqrt(area * ar)), OCC_MIN_SIDE, w))
            bh = int(np.clip(round(area / bw), OCC_MIN_SIDE, h))
            for _try in range(200):
                x0 = int(rng.integers(0, w - bw + 1))
                y0 = int(rng.integers(0, h - bh + 1))
                cand = [x0, y0, x0 + bw, y0 + bh]
                if all(cand[2] <= b[0] or b[2] <= cand[0] or cand[3] <= b[1] or b[3] <= cand[1]
                       for b in boxes):
                    boxes.append(cand)
                    break
            else:
                ok = False
                break
        if ok:
            actual = boxes_coverage(boxes, h, w)
            if accept(actual):
                return boxes, actual
    raise RuntimeError(f"Could not place {n_boxes} boxes covering {coverage:.3f}")


def _new_seed(rng: np.random.Generator) -> int:
    return int(rng.integers(0, 2**31 - 1))


def sample_params(label: int, rng: np.random.Generator) -> dict:
    """Sample corruption parameters from the TRAINING distribution. Fully reproducible via 'seed'."""
    seed = _new_seed(rng)
    r = np.random.default_rng(seed)
    label = int(label)
    if label == CLEAN:
        return {"label": CLEAN, "type": "clean", "seed": seed}
    if label == SALT_PEPPER:
        return {"label": SALT_PEPPER, "type": "salt_pepper", "seed": seed,
                "p": float(r.uniform(*SP_PROB_RANGE))}
    if label == BLUR:
        return {"label": BLUR, "type": "gaussian_blur", "seed": seed,
                "kernel_size": int(r.choice(BLUR_KERNELS)), "sigma": float(r.uniform(*BLUR_SIGMA_RANGE))}
    if label == OCCLUSION:
        n = int(r.integers(OCC_BOX_RANGE[0], OCC_BOX_RANGE[1] + 1))
        cov = float(r.uniform(*OCC_COVERAGE_RANGE))
        lo, hi = OCC_COVERAGE_RANGE
        boxes, actual = sample_occlusion_boxes(r, n, cov, accept=lambda a: lo <= a <= hi)
        return {"label": OCCLUSION, "type": "occlusion", "seed": seed, "n_boxes": n,
                "coverage_target": cov, "coverage": actual, "boxes": boxes}
    raise ValueError(f"Unknown label {label}")


def fixed_params(label: int, level: str, seed: int) -> dict:
    """Parameters for the fixed TEST severities (low/medium/high)."""
    label = int(label)
    if label == CLEAN:
        return {"label": CLEAN, "type": "clean", "seed": seed}
    spec = TEST_SEVERITIES[label][level]
    base = {"label": label, "type": CLASS_NAMES[label], "seed": seed}
    if label == SALT_PEPPER:
        return {**base, "p": spec["p"]}
    if label == BLUR:
        return {**base, "kernel_size": spec["kernel_size"], "sigma": spec["sigma"]}
    boxes, actual = sample_occlusion_boxes(np.random.default_rng(seed), spec["n_boxes"], spec["coverage"])
    return {**base, "n_boxes": spec["n_boxes"], "coverage_target": spec["coverage"],
            "coverage": actual, "boxes": boxes}


def apply_corruption(img: np.ndarray, params: dict) -> np.ndarray:
    """Deterministically apply a parameter dict (from sample_params / fixed_params / a manifest)."""
    label = int(params["label"])
    if label == CLEAN:
        return img.copy()
    if label == SALT_PEPPER:
        return salt_and_pepper(img, params["p"], params["seed"])
    if label == BLUR:
        return gaussian_blur(img, params["kernel_size"], params["sigma"])
    if label == OCCLUSION:
        return occlude(img, params["boxes"])
    raise ValueError(f"Unknown label {label}")


def describe_params(params: dict) -> str:
    """Short human-readable description (plot titles, API responses)."""
    t = int(params["label"])
    if t == CLEAN:
        return "clean"
    if t == SALT_PEPPER:
        return f"s&p p={params['p']:.3f}"
    if t == BLUR:
        return f"blur k={params['kernel_size']} σ={params['sigma']:.2f}"
    return f"occl {params['n_boxes']}box {100 * params['coverage']:.1f}%"


# --------------------------------------------------------------------------------------
# Manifests
# --------------------------------------------------------------------------------------
def build_val_manifest(val_names: Sequence[str], seed: int = SPLIT_SEED) -> dict:
    """Exactly balanced (25% per condition) validation manifest; params from the training ranges."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    n = len(val_names)
    labels = rng.permutation(np.resize(np.arange(NUM_CLASSES), n))
    records = []
    for i, (name, label) in enumerate(zip(val_names, labels)):
        p = sample_params(int(label), rng)
        records.append({"id": f"{name}__{p['type']}", "image": name, "index": i,
                        "severity_level": "sampled" if label != CLEAN else "none", **p})
    return {"meta": {"kind": "validation", "version": MANIFEST_VERSION, "seed": seed,
                     "seed_sequence": [seed, 1], "n_images": n, "n_records": len(records),
                     "config": corruption_config()},
            "records": records}


def build_test_manifest(test_names: Sequence[str], seed: int = SPLIT_SEED) -> dict:
    """Every test image × {clean, 3 corruptions × 3 fixed severities} = 10 records per image."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
    records = []
    for i, name in enumerate(test_names):
        s = _new_seed(rng)
        records.append({"id": f"{name}__clean", "image": name, "index": i,
                        "severity_level": "none", **fixed_params(CLEAN, "none", s)})
        for label in (SALT_PEPPER, BLUR, OCCLUSION):
            for level in SEVERITY_LEVELS:
                s = _new_seed(rng)
                p = fixed_params(label, level, s)
                records.append({"id": f"{name}__{p['type']}__{level}", "image": name, "index": i,
                                "severity_level": level, **p})
    return {"meta": {"kind": "test", "version": MANIFEST_VERSION, "seed": seed,
                     "seed_sequence": [seed, 2], "n_images": len(test_names), "n_records": len(records),
                     "config": corruption_config()},
            "records": records}


def save_json(obj, path: str | Path, indent=None) -> str:
    """Write JSON deterministically and return its SHA-256 (use it to prove files are identical)."""
    text = json.dumps(obj, indent=indent, sort_keys=True, separators=None if indent else (",", ":"))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def load_json(path: str | Path):
    return json.loads(Path(path).read_text())


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------------------
# PyTorch datasets / samplers / loaders  (torch imported lazily so the backend stays light)
# --------------------------------------------------------------------------------------
try:
    import torch
    from torch.utils.data import DataLoader, Dataset, Sampler, get_worker_info
except ImportError:  # backend without torch: only the NumPy functions above are needed
    torch = None
    Dataset = object
    Sampler = object


def to_chw_tensor(img_hwc: np.ndarray):
    return torch.from_numpy(np.ascontiguousarray(img_hwc.transpose(2, 0, 1), dtype=np.float32))


class RuntimeCorruptionDataset(Dataset):
    """Training dataset: a NEW corruption type + severity is sampled on every __getitem__.

    Returns (corrupted_image, clean_image, corruption_label):
        (3,128,128) float32 [0,1], (3,128,128) float32 [0,1], int64 scalar.
    `classes` restricts which conditions are drawn (e.g. (1,) for the Task 2 salt-and-pepper
    specialist); they are drawn uniformly. `item` may also be an (index, label) tuple, which is
    how BalancedCorruptionBatchSampler forces exactly balanced batches.
    """

    def __init__(self, images_u8: np.ndarray, classes: Sequence[int] = (0, 1, 2, 3), seed: int | None = None):
        self.images = images_u8
        self.classes = np.asarray(classes, dtype=np.int64)
        self.seed = seed
        self._rng, self._rng_key = None, None

    def __len__(self):
        return len(self.images)

    def _get_rng(self) -> np.random.Generator:
        # One independent generator per process/worker. Worker seeds come from torch
        # (base_seed + worker_id, re-drawn every epoch), so torch.manual_seed controls everything
        # and forked workers never share a random stream.
        info = get_worker_info()
        key = (os.getpid(), info.seed if info is not None else None)
        if key != self._rng_key:
            if info is not None:
                s = info.seed % 2**32
            else:
                s = self.seed if self.seed is not None else int(torch.initial_seed() % 2**32)
            self._rng, self._rng_key = np.random.default_rng(s), key
        return self._rng

    def get_with_params(self, item):
        rng = self._get_rng()
        if isinstance(item, (tuple, list)):
            idx, label = int(item[0]), int(item[1])
        else:
            idx, label = int(item), int(rng.choice(self.classes))
        clean = self.images[idx].astype(np.float32) / 255.0
        params = sample_params(label, rng)
        corrupted = apply_corruption(clean, params)
        return corrupted, clean, params

    def __getitem__(self, item):
        corrupted, clean, params = self.get_with_params(item)
        return to_chw_tensor(corrupted), to_chw_tensor(clean), torch.tensor(params["label"], dtype=torch.long)


class ManifestCorruptionDataset(Dataset):
    """Validation/test dataset: corruption parameters are read from a fixed manifest.

    `name_to_row` maps image name -> row in `images_u8`. With return_index=True a 4th element
    (the record index) is returned so results can be grouped by type/severity afterwards.
    """

    def __init__(self, images_u8: np.ndarray, name_to_row: dict, records: list[dict],
                 classes: Sequence[int] | None = None, return_index: bool = False):
        self.images = images_u8
        self.name_to_row = name_to_row
        self.records = [r for r in records if classes is None or r["label"] in classes]
        self.return_index = return_index

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        clean = self.images[self.name_to_row[rec["image"]]].astype(np.float32) / 255.0
        corrupted = apply_corruption(clean, rec)
        out = (to_chw_tensor(corrupted), to_chw_tensor(clean), torch.tensor(rec["label"], dtype=torch.long))
        return out + (torch.tensor(i),) if self.return_index else out


class BalancedCorruptionBatchSampler(Sampler):
    """Yields batches of (image_index, label) with EXACTLY equal counts per class
    (when batch_size is a multiple of the number of classes). Intended for the Task 2 classifier.
    """

    def __init__(self, n_items: int, batch_size: int, classes: Sequence[int] = (0, 1, 2, 3),
                 drop_last: bool = True, seed: int | None = None):
        self.n, self.bs, self.classes, self.drop_last = n_items, batch_size, np.asarray(classes), drop_last
        self.seed, self.epoch = seed, 0

    def __len__(self):
        return self.n // self.bs if self.drop_last else -(-self.n // self.bs)

    def __iter__(self):
        base = self.seed if self.seed is not None else int(torch.randint(0, 2**31 - 1, ()).item())
        rng = np.random.default_rng([base, self.epoch])
        self.epoch += 1
        perm = rng.permutation(self.n)
        for start in range(0, len(self) * self.bs, self.bs):
            idx = perm[start:start + self.bs]
            labels = rng.permutation(np.resize(self.classes, len(idx)))
            yield [(int(i), int(l)) for i, l in zip(idx, labels)]


def default_num_workers() -> int:
    return max(0, min(4, (os.cpu_count() or 2)))


def make_train_loader(images_u8, batch_size=32, classes=(0, 1, 2, 3), balanced=False,
                      num_workers=None, seed=None, pin_memory=True, drop_last=True):
    num_workers = default_num_workers() if num_workers is None else num_workers
    ds = RuntimeCorruptionDataset(images_u8, classes=classes, seed=seed)
    kw = dict(num_workers=num_workers, pin_memory=pin_memory and torch.cuda.is_available(),
              persistent_workers=num_workers > 0)
    if balanced:
        bs = BalancedCorruptionBatchSampler(len(ds), batch_size, classes, drop_last=drop_last, seed=seed)
        return DataLoader(ds, batch_sampler=bs, **kw)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=drop_last, **kw)


def make_eval_loader(images_u8, name_to_row, records, batch_size=64, classes=None,
                     return_index=False, num_workers=None, pin_memory=True):
    num_workers = default_num_workers() if num_workers is None else num_workers
    ds = ManifestCorruptionDataset(images_u8, name_to_row, records, classes=classes, return_index=return_index)
    # Not persistent: eval workers exit after each pass, so idle loaders don't hold RAM
    # (each worker gradually copies the Python objects it touches — "copy-on-read").
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                      pin_memory=pin_memory and torch.cuda.is_available(), persistent_workers=False)


class PetData:
    """Everything a Task 1–3 notebook needs, built by prepare_data()."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def load_test_images(self, cache_dir=None):
        """FINAL EVALUATION ONLY: decode the quarantined official test images (cached)."""
        cache = Path(cache_dir or self.cache_dir) / "test_128.npz"
        out = load_clean_images(self.test_names, self.images_dir, cache_path=cache)
        return out["images"], {n: i for i, n in enumerate(out["names"])}, out["failed"]

    def summary(self) -> dict:
        return {"source": self.source, "images_dir": str(self.images_dir), "n_train": len(self.train_images),
                "n_val": len(self.val_images), "n_test": len(self.test_names),
                "n_val_records": len(self.val_manifest["records"]),
                "n_test_records": len(self.test_manifest["records"]),
                "split_sha256": self.hashes.get("split"), "val_manifest_sha256": self.hashes.get("val"),
                "test_manifest_sha256": self.hashes.get("test")}


def find_prep_outputs(search_root="/kaggle/input") -> Path | None:
    """Locate an attached output of notebook 00 (folder containing manifests/val_manifest.json)."""
    for p in sorted(Path(search_root).rglob("val_manifest.json")):
        root = p.parent.parent
        if (root / "manifests" / "test_manifest.json").exists() and (root / "artifacts" / "split_seed42.json").exists():
            return root
    return None


def prepare_data(input_root="/kaggle/input", work_dir="/kaggle/working", images_dir=None, annot_dir=None,
                 seed: int = SPLIT_SEED, verbose: bool = True) -> PetData:
    """Return split, clean train/val arrays and both manifests.

    1. If notebook 00's output is attached under `input_root`, its split + manifests are reused
       (guarantees byte-identical data across Tasks 1–3).
    2. Otherwise everything is regenerated deterministically (same seeds -> same files/hashes)
       and saved under `work_dir`.
    The official Oxford-IIIT Pet dataset must be attached in both cases (for the pixels).
    """
    input_root, work_dir = Path(input_root), Path(work_dir)
    cache_dir = work_dir / "cache"
    if images_dir is None or annot_dir is None:
        found = find_pet_dataset(input_root)
        images_dir = Path(images_dir or found["images_dir"])
        annot_dir = Path(annot_dir or found["annotations_dir"])
    test_names = read_split_list(Path(annot_dir) / "test.txt")

    prep = find_prep_outputs(input_root)
    if prep is not None:
        source = f"attached notebook-00 output: {prep}"
        split = load_json(prep / "artifacts" / "split_seed42.json")
        val_manifest = load_json(prep / "manifests" / "val_manifest.json")
        test_manifest = load_json(prep / "manifests" / "test_manifest.json")
        hashes = {"split": sha256_file(prep / "artifacts" / "split_seed42.json"),
                  "val": sha256_file(prep / "manifests" / "val_manifest.json"),
                  "test": sha256_file(prep / "manifests" / "test_manifest.json")}
        wanted = split["train"] + split["val"]
        cached = prep / "cache" / "trainval_128.npz"
        tv = None
        if cached.exists():
            z = load_clean_images(read_split_list(Path(annot_dir) / "trainval.txt"), images_dir,
                                  cache_path=cached, verbose=False)
            tv = z if set(wanted) <= set(z["names"]) else None
        if tv is None:
            tv = load_clean_images(wanted, images_dir, cache_path=cache_dir / "trainval_128.npz", verbose=verbose)
    else:
        source = "regenerated deterministically (notebook-00 output not attached)"
        trainval_names = read_split_list(Path(annot_dir) / "trainval.txt")
        tv = load_clean_images(trainval_names, images_dir, cache_path=cache_dir / "trainval_128.npz", verbose=verbose)
        split = make_split(tv["names"], VAL_FRACTION, seed)
        split["excluded_unreadable"] = tv["failed"]
        val_manifest = build_val_manifest(split["val"], seed)
        test_manifest = build_test_manifest(test_names, seed)
        hashes = {"split": save_json(split, work_dir / "artifacts" / "split_seed42.json", indent=1),
                  "val": save_json(val_manifest, work_dir / "manifests" / "val_manifest.json"),
                  "test": save_json(test_manifest, work_dir / "manifests" / "test_manifest.json")}

    row = {n: i for i, n in enumerate(tv["names"])}
    train_images = tv["images"][[row[n] for n in split["train"]]]
    val_images = tv["images"][[row[n] for n in split["val"]]]
    assert not set(split["train"]) & set(split["val"]) and not (set(split["train"]) | set(split["val"])) & set(test_names)
    data = PetData(source=source, images_dir=images_dir, annot_dir=annot_dir, cache_dir=cache_dir, split=split,
                   train_images=train_images, val_images=val_images,
                   val_name_to_row={n: i for i, n in enumerate(split["val"])},
                   val_manifest=val_manifest, test_manifest=test_manifest, test_names=test_names, hashes=hashes)
    if verbose:
        for k, v in data.summary().items():
            print(f"  {k:22s} {v}")
    return data


def seed_everything(seed: int = SPLIT_SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
