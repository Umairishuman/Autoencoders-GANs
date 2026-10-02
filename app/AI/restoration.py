"""
restoration.py — shared restoration components (Task 1, reused by Task 2 specialists and Task 3).

Contents
  * SSIM (Gaussian window, Wang et al. 2004) + PSNR, per-image and batched, differentiable
  * L1 + SSIM loss:  L = alpha * L1 + (1 - alpha) * (1 - SSIM)
  * ConvAutoencoder: conv encoder -> compressed spatial latent (1x1-conv bottleneck + dropout)
                     -> conv decoder, with optional LIMITED skip connections (deepest levels only)
  * train_one_epoch / evaluate (AMP-aware, per-corruption metrics)
  * ONNX export, PyTorch-vs-ONNX parity check, latency measurement
  * plotting helpers for restoration grids (target / input / output / |error|)
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CLASS_NAMES = ("clean", "salt_pepper", "gaussian_blur", "occlusion")


# ======================================================================================
# Metrics
# ======================================================================================
def _gaussian_window(size: int, sigma: float, channels: int, device, dtype):
    x = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w2d = g[:, None] * g[None, :]
    return w2d.expand(channels, 1, size, size).contiguous()


def ssim_map(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, sigma: float = 1.5,
             data_range: float = 1.0) -> torch.Tensor:
    """SSIM map (B,C,H',W') with an 11x11 Gaussian window (sigma 1.5), 'valid' region only.
    Matches skimage.structural_similarity(gaussian_weights=True, sigma=1.5,
    use_sample_covariance=False, data_range=1)."""
    x, y = x.float(), y.float()
    c = x.shape[1]
    w = _gaussian_window(window_size, sigma, c, x.device, x.dtype)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mu_x = F.conv2d(x, w, groups=c)
    mu_y = F.conv2d(y, w, groups=c)
    sxx = F.conv2d(x * x, w, groups=c) - mu_x ** 2
    syy = F.conv2d(y * y, w, groups=c) - mu_y ** 2
    sxy = F.conv2d(x * y, w, groups=c) - mu_x * mu_y
    return ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))


def ssim_per_image(x, y, **kw) -> torch.Tensor:
    return ssim_map(x, y, **kw).mean(dim=(1, 2, 3))


def psnr_per_image(x, y, data_range: float = 1.0) -> torch.Tensor:
    mse = ((x.float() - y.float()) ** 2).mean(dim=(1, 2, 3)).clamp_min(1e-10)
    return 10 * torch.log10(data_range ** 2 / mse)


def mae_per_image(x, y) -> torch.Tensor:
    return (x.float() - y.float()).abs().mean(dim=(1, 2, 3))


def val_score(psnr_mean: float, ssim_mean: float) -> float:
    """Optuna objective (alpha-independent, so the study cannot 'win' by changing the loss scale):
    equal-weight combination of structural similarity and reconstruction fidelity (PSNR/40, capped)."""
    return 0.5 * float(ssim_mean) + 0.5 * min(float(psnr_mean) / 40.0, 1.0)


# ======================================================================================
# Loss
# ======================================================================================
class L1SSIMLoss(nn.Module):
    """alpha * L1 + (1 - alpha) * (1 - SSIM). Always computed in float32 (AMP-safe)."""

    def __init__(self, alpha: float = 0.8):
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, pred, target):
        pred, target = pred.float(), target.float()
        l1 = F.l1_loss(pred, target)
        s = ssim_map(pred, target).mean()
        total = self.alpha * l1 + (1 - self.alpha) * (1 - s)
        return total, {"l1": l1.detach(), "ssim": s.detach()}


# ======================================================================================
# Model
# ======================================================================================
def norm(c: int) -> nn.GroupNorm:
    # GroupNorm instead of BatchNorm: no running statistics, so train- and eval-mode behave the same
    # at any batch size (Optuna samples batch sizes 16-64), and frozen experts in Task 3 cannot drift.
    return nn.GroupNorm(min(8, c), c)


def conv_block(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1, bias=False), norm(cout), nn.SiLU(inplace=True),
        nn.Conv2d(cout, cout, 3, 1, 1, bias=False), norm(cout), nn.SiLU(inplace=True),
    )


class ConvAutoencoder(nn.Module):
    """Convolutional denoising autoencoder with a genuine compressed code.

    Encoder: 128 -> 64 -> 32 -> 16 -> 8 (stride-2 conv blocks), channels grow base x (1,2,4,8,8).
    Latent:  1x1 conv to `latent_channels` at 8x8 (e.g. 16 ch -> 1,024 values), then code dropout.
    Decoder: bilinear upsample + conv blocks (no transposed-conv checkerboard artefacts), sigmoid output.

    Skip connections (skip_levels = 0, 1 or 2) are BOTTLENECKED: the encoder feature at the
    16x16 (and 32x32) level is squeezed by a 1x1 conv to `skip_channels` channels and passes the same
    code dropout before reaching the decoder. The information the decoder receives is therefore
    bounded by the total code size

        code_dim = latent_dim + sum_over_skipped_levels(skip_channels * H_l * W_l),

    which is what `compression_ratio` reports (input values / code values). Full- and half-resolution
    features are never skipped. skip_channels=None reproduces the original full-width skips
    (e.g. 384 x 16 x 16 = 98,304 values, larger than the image); it is kept only for the ablation,
    which showed that such skips let the decoder ignore the latent.
    """

    def __init__(self, base_channels: int = 32, latent_channels: int = 32, dropout: float = 0.1,
                 skip_levels: int = 0, skip_channels: int | None = 4, depth: int = 4, max_mult: int = 8,
                 img_size: int = 128):
        super().__init__()
        assert 0 <= skip_levels <= 2, "only the two deepest levels may be skipped"
        self.config = dict(base_channels=base_channels, latent_channels=latent_channels, dropout=dropout,
                           skip_levels=skip_levels, skip_channels=skip_channels, depth=depth,
                           max_mult=max_mult, img_size=img_size)
        chs = [min(base_channels * 2 ** i, base_channels * max_mult) for i in range(depth + 1)]
        self.chs, self.depth, self.skip_levels, self.img_size = chs, depth, skip_levels, img_size
        self.stem = conv_block(3, chs[0], 1)
        self.down = nn.ModuleList([conv_block(chs[i], chs[i + 1], 2) for i in range(depth)])
        self.to_latent = nn.Conv2d(chs[-1], latent_channels, 1)
        self.code_dropout = nn.Dropout(dropout)
        self.from_latent = nn.Sequential(nn.Conv2d(latent_channels, chs[-1], 1, bias=False),
                                         norm(chs[-1]), nn.SiLU(inplace=True))
        # decoder stage j upsamples to resolution level (depth - j - 1); skips only for j < skip_levels
        self.skip_proj = nn.ModuleList()
        self.skip_shapes = []
        self.up = nn.ModuleList()
        for j in range(depth):
            lvl = depth - j - 1
            extra = 0
            if j < skip_levels:
                width = chs[lvl] if skip_channels is None else skip_channels
                self.skip_proj.append(nn.Identity() if skip_channels is None else nn.Conv2d(chs[lvl], skip_channels, 1))
                res = img_size // 2 ** lvl
                self.skip_shapes.append((width, res, res))
                extra = width
            self.up.append(conv_block(chs[lvl + 1] + extra, chs[lvl], 1))
        self.head = nn.Conv2d(chs[0], 3, 3, 1, 1)
        self.latent_shape = (latent_channels, img_size // 2 ** depth, img_size // 2 ** depth)

    # ---- code-size accounting ------------------------------------------------------------
    @property
    def latent_dim(self) -> int:
        return int(np.prod(self.latent_shape))

    @property
    def skip_dim(self) -> int:
        return int(sum(np.prod(s) for s in self.skip_shapes))

    @property
    def code_dim(self) -> int:
        """Total number of values the decoder receives from the encoder (latent + skip codes)."""
        return self.latent_dim + self.skip_dim

    @property
    def compression_ratio(self) -> float:
        return 3 * self.img_size ** 2 / self.code_dim

    # ---- forward -------------------------------------------------------------------------
    def encode(self, x):
        """Returns (latent, skip_codes). skip_codes[j] feeds decoder stage j (deepest first)."""
        feats = []
        h = self.stem(x)
        for blk in self.down:
            feats.append(h)
            h = blk(h)
        skips = [proj(feats[self.depth - j - 1]) for j, proj in enumerate(self.skip_proj)]
        return self.to_latent(h), skips

    def decode(self, z, skips=None):
        h = self.from_latent(self.code_dropout(z))
        for j, blk in enumerate(self.up):
            h = F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False)
            if j < self.skip_levels:
                s = skips[j] if skips is not None else h.new_zeros(h.shape[0], *self.skip_shapes[j])
                h = torch.cat([h, self.code_dropout(s)], dim=1)
            h = blk(h)
        return torch.sigmoid(self.head(h))

    def forward(self, x):
        z, skips = self.encode(x)
        return self.decode(z, skips)


@torch.no_grad()
def code_dependence(model, loader, device, max_batches=None) -> dict:
    """How much the output depends on each part of the code (evaluated on a manifest loader).

    full:        normal forward pass
    zero_latent: latent replaced by zeros (skips intact)   -> large drop = the latent is used
    zero_skips:  skip codes replaced by zeros (latent intact) -> drop = what the skips contribute
    """
    model.eval()
    acc = {k: {"psnr": [], "ssim": []} for k in ("full", "zero_latent", "zero_skips")}
    for b, batch in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        x, y = batch[0].to(device), batch[1].to(device)
        z, skips = model.encode(x)
        outs = {"full": model.decode(z, skips), "zero_latent": model.decode(torch.zeros_like(z), skips)}
        if model.skip_levels:
            outs["zero_skips"] = model.decode(z, [torch.zeros_like(s) for s in skips])
        for k, o in outs.items():
            o = o.float().clamp(0, 1)
            acc[k]["psnr"].append(psnr_per_image(o, y).cpu())
            acc[k]["ssim"].append(ssim_per_image(o, y).cpu())
    out = {}
    for k, d in acc.items():
        if d["psnr"]:
            out[f"{k}_psnr"] = torch.cat(d["psnr"]).mean().item()
            out[f"{k}_ssim"] = torch.cat(d["ssim"]).mean().item()
    out["latent_drop_db"] = out["full_psnr"] - out["zero_latent_psnr"]
    if "zero_skips_psnr" in out:
        out["skip_drop_db"] = out["full_psnr"] - out["zero_skips_psnr"]
    return out


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ======================================================================================
# Training / evaluation
# ======================================================================================
def make_optimizer_and_scheduler(model, lr, weight_decay, steps_per_epoch, epochs, warmup_epochs=1):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total = max(1, steps_per_epoch * epochs)
    warm = max(1, int(steps_per_epoch * warmup_epochs))

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        t = (step - warm) / max(1, total - warm)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def make_scaler(device, amp=True):
    """One GradScaler per training run (fp16 AMP on CUDA only; a no-op on CPU)."""
    return torch.amp.GradScaler("cuda", enabled=bool(amp and device.type == "cuda"))


def train_one_epoch(model, loader, loss_fn, optimizer, scheduler, device, scaler=None, amp=True,
                    grad_clip=1.0, max_batches=None) -> dict:
    model.train()
    use_amp = amp and device.type == "cuda"
    if scaler is None:
        scaler = make_scaler(device, amp)
    tot = {"loss": 0.0, "l1": 0.0, "ssim": 0.0}
    n = 0
    for b, batch in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        x, y = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            pred = model(x)
        loss, parts = loss_fn(pred, y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at batch {b}")
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        bs = x.shape[0]
        tot["loss"] += loss.item() * bs
        tot["l1"] += parts["l1"].item() * bs
        tot["ssim"] += parts["ssim"].item() * bs
        n += bs
    return {k: v / max(n, 1) for k, v in tot.items()}


@torch.no_grad()
def evaluate(model, loader, device, loss_fn=None, amp=True, per_image=False, include_input=False) -> dict:
    """Metrics over a (manifest) loader. Returns means overall and per corruption type.
    per_image=True also returns numpy arrays aligned with the dataset order.
    include_input=True also scores the corrupted input itself (the 'do nothing' baseline)."""
    model.eval()
    use_amp = amp and device.type == "cuda"
    keys = ["psnr", "ssim", "mae"] + (["in_psnr", "in_ssim", "in_mae"] if include_input else [])
    acc = {k: [] for k in keys}
    labels, losses = [], []
    for batch in loader:
        x, y, lab = batch[0].to(device), batch[1].to(device), batch[2]
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            pred = model(x)
        pred = pred.float().clamp(0, 1)
        if loss_fn is not None:
            losses.append(loss_fn(pred, y)[0].item() * x.shape[0])
        acc["psnr"].append(psnr_per_image(pred, y).cpu())
        acc["ssim"].append(ssim_per_image(pred, y).cpu())
        acc["mae"].append(mae_per_image(pred, y).cpu())
        if include_input:
            acc["in_psnr"].append(psnr_per_image(x, y).cpu())
            acc["in_ssim"].append(ssim_per_image(x, y).cpu())
            acc["in_mae"].append(mae_per_image(x, y).cpu())
        labels.append(lab.cpu())
    arr = {k: torch.cat(v).numpy() for k, v in acc.items()}
    labels = torch.cat(labels).numpy()
    out = {k: float(v.mean()) for k, v in arr.items()}
    for c, name in enumerate(CLASS_NAMES):
        m = labels == c
        if m.any():
            for k, v in arr.items():
                out[f"{k}/{name}"] = float(v[m].mean())
    out["score"] = val_score(out["psnr"], out["ssim"])
    if loss_fn is not None:
        out["loss"] = sum(losses) / len(labels)
    if per_image:
        out["per_image"] = {**arr, "label": labels}
    return out


@torch.no_grad()
def predict_numpy(model, imgs_hwc: np.ndarray, device, batch_size=64) -> np.ndarray:
    """(N,H,W,3) float [0,1] -> restored (N,H,W,3) float [0,1]."""
    model.eval()
    outs = []
    for i in range(0, len(imgs_hwc), batch_size):
        x = torch.from_numpy(np.ascontiguousarray(imgs_hwc[i:i + batch_size].transpose(0, 3, 1, 2))).float().to(device)
        outs.append(model(x).float().clamp(0, 1).cpu().numpy().transpose(0, 2, 3, 1))
    return np.concatenate(outs)


# ======================================================================================
# ONNX
# ======================================================================================
def export_onnx(model, path, img_size=128, opset=17, input_name="input", output_name="output"):
    model = model.eval().cpu()
    dummy = torch.rand(1, 3, img_size, img_size)
    kw = dict(input_names=[input_name], output_names=[output_name], opset_version=opset,
              dynamic_axes={input_name: {0: "batch"}, output_name: {0: "batch"}})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        try:
            torch.onnx.export(model, dummy, str(path), dynamo=False, **kw)
        except TypeError:          # torch < 2.5 has no `dynamo` kwarg
            torch.onnx.export(model, dummy, str(path), **kw)
    try:
        import onnx
        onnx.checker.check_model(onnx.load(str(path)))
    except ImportError:
        pass
    return path


def verify_onnx(model, path, x: torch.Tensor, atol=1e-4) -> dict:
    """Compare PyTorch (CPU, eval) and onnxruntime outputs on the same batch."""
    import onnxruntime as ort
    model = model.eval().cpu()
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = model(x.cpu().float()).numpy()
    out = sess.run(None, {sess.get_inputs()[0].name: x.cpu().float().numpy()})[0]
    diff = np.abs(out - ref)
    p_ref = psnr_per_image(torch.from_numpy(ref), torch.from_numpy(out))
    return {"max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()),
            "min_psnr_torch_vs_onnx": float(p_ref.min()), "n_images": int(len(x)),
            "passed": bool(diff.max() < atol)}


def onnx_latency_ms(path, img_size=128, batch=1, n_warmup=10, n_runs=100) -> dict:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    x = np.random.rand(batch, 3, img_size, img_size).astype(np.float32)
    for _ in range(n_warmup):
        sess.run(None, {name: x})
    ts = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, {name: x})
        ts.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": float(np.median(ts)), "p90_ms": float(np.percentile(ts, 90)), "batch": batch}


# ======================================================================================
# Plotting
# ======================================================================================
def restoration_grid(rows: list[dict], path=None, title=None, err_vmax=0.5):
    """rows: dicts with 'clean', 'corrupted', 'restored' (HxWx3 float [0,1]) and optional 'label'.
    Columns: clean target | corrupted input | reconstruction | absolute error map."""
    import matplotlib.pyplot as plt
    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(9.2, 2.35 * n), squeeze=False)
    for i, r in enumerate(rows):
        err = np.abs(r["restored"] - r["clean"]).mean(-1)
        panels = [r["clean"], r["corrupted"], r["restored"]]
        for j, im in enumerate(panels):
            axes[i, j].imshow(np.clip(im, 0, 1))
        hm = axes[i, 3].imshow(err, cmap="magma", vmin=0, vmax=err_vmax)
        for ax in axes[i]:
            ax.set_xticks([]); ax.set_yticks([])
        axes[i, 0].set_ylabel(r.get("label", ""), fontsize=7.5, rotation=0, ha="right", va="center", labelpad=4)
        if "in_text" in r:
            axes[i, 1].set_xlabel(r["in_text"], fontsize=7.5)
        if "out_text" in r:
            axes[i, 2].set_xlabel(r["out_text"], fontsize=7.5)
    for j, t in enumerate(["clean target", "corrupted input", "reconstruction", "|error| (mean RGB)"]):
        axes[0, j].set_title(t, fontsize=9)
    fig.colorbar(hm, ax=axes[:, 3].tolist(), fraction=0.046, pad=0.02, shrink=0.6)
    if title:
        fig.suptitle(title, fontsize=11)
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig
