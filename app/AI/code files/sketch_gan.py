"""
sketch_gan.py — Task 4: style-conditioned face-to-sketch generation with a conditional GAN (FS2K).

    y_hat = G(x, s)            x: photo, s in {0,1,2}: sketch style (FS2K "style" field)
    D(x, y, s) -> patch logits (PatchGAN)
    L_D = 0.5 * [BCE(D(x,y,s), 1) + BCE(D(x,G(x,s),s), 0)]
    L_G = BCE(D(x,G(x,s),s), 1) + lambda_L1 * L1(y, G(x,s))

Style conditioning (learned embeddings, separate tables in G and D):
  * Generator: the style embedding is concatenated to the 1x1 bottleneck AND drives conditional instance
    normalisation (per-style gamma/beta) in every decoder block, the multi-style mechanism of
    Dumoulin et al. 2017 ("A learned representation for artistic style").
  * Discriminator: its own embedding is broadcast to a spatial map and concatenated to (photo, sketch), so D
    judges whether a sketch is real FOR THAT STYLE.
Images are in [-1, 1] inside the networks; the ONNX wrapper takes/returns [0, 1].
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

STYLE_NAMES = ("Style 1", "Style 2", "Style 3")
IMG = 128
JITTER = 142          # train images are stored at 142x142 and randomly cropped to 128 (pix2pix-style jitter)


# --------------------------------------------------------------------------------------
# Dataset discovery / loading
# --------------------------------------------------------------------------------------
def find_fs2k(search_root="/kaggle/input") -> Path:
    """Folder that contains anno_train.json, anno_test.json, photo/ and sketch/."""
    for p in sorted(Path(search_root).rglob("anno_train.json")):
        r = p.parent
        if (r / "anno_test.json").exists() and (r / "photo").is_dir() and (r / "sketch").is_dir():
            return r
    raise FileNotFoundError(f"FS2K not found under {search_root}: need anno_train.json, anno_test.json, photo/, sketch/")


def _with_ext(base: Path) -> Path | None:
    for ext in (".jpg", ".png", ".jpeg", ".JPG", ".PNG"):
        if base.with_suffix(ext).exists():
            return base.with_suffix(ext)
    return None


def read_annotations(root: Path, split: str) -> list[dict]:
    """[{name, photo, sketch, style}] for the official split. Mapping follows FS2K's own split script:
    photo/photoK/imageNNNN.{jpg,png}  <->  sketch/sketchK/sketchNNNN.{jpg,png}"""
    entries = json.loads((root / f"anno_{split}.json").read_text())
    out, missing = [], []
    for e in entries:
        name = e["image_name"]
        photo = _with_ext(root / "photo" / name)
        sketch = _with_ext(root / "sketch" / name.replace("image", "sketch").replace("photo", "sketch"))
        if photo is None or sketch is None:
            missing.append(name); continue
        out.append({"name": name, "photo": str(photo), "sketch": str(sketch), "style": int(e["style"])})
    if missing:
        print(f"[{split}] {len(missing)} entries without photo/sketch file, e.g. {missing[:3]}")
    return out


def stratified_val_split(records, frac=0.15, seed=42):
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(records))
    tr, va = train_test_split(idx, test_size=frac, random_state=seed, stratify=[r["style"] for r in records])
    return [records[i] for i in sorted(tr)], [records[i] for i in sorted(va)]


def load_rgb(path, size):
    with Image.open(path) as im:
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))       # transparent -> white (sketch paper)
            im = Image.alpha_composite(bg, im)
        return np.asarray(im.convert("RGB").resize((size, size), Image.Resampling.BICUBIC), dtype=np.uint8).copy()


def load_pairs(records, size, cache=None):
    """Both members of a pair are resized identically -> pixel correspondence preserved."""
    if cache is not None and Path(cache).exists():
        z = np.load(cache)
        if list(z["names"]) == [r["name"] for r in records] and int(z["size"]) == size:
            return z["photos"], z["sketches"]
    photos = np.stack([load_rgb(r["photo"], size) for r in records])
    sketches = np.stack([load_rgb(r["sketch"], size) for r in records])
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, photos=photos, sketches=sketches, names=np.array([r["name"] for r in records]), size=size)
    return photos, sketches


class PairDataset(torch.utils.data.Dataset):
    """Returns (photo, sketch, style) in [-1,1]. With augment=True the SAME random crop and flip are applied to
    both images of the pair (spatial correspondence must be preserved for paired training)."""

    def __init__(self, photos, sketches, styles, augment=False, size=IMG):
        self.p, self.s, self.st, self.aug, self.size = photos, sketches, np.asarray(styles), augment, size

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        p, s = self.p[i], self.s[i]
        if self.aug:
            H = p.shape[0]
            y0, x0 = np.random.randint(0, H - self.size + 1, size=2)
            p, s = p[y0:y0 + self.size, x0:x0 + self.size], s[y0:y0 + self.size, x0:x0 + self.size]
            if np.random.rand() < 0.5:
                p, s = p[:, ::-1], s[:, ::-1]
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1))).float() / 127.5 - 1.0
        return t(p), t(s), torch.tensor(int(self.st[i]))


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------
class CondInstanceNorm(nn.Module):
    """InstanceNorm whose scale/shift are predicted from the style embedding (conditional IN)."""

    def __init__(self, ch, emb_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(ch, affine=False)
        self.gamma = nn.Linear(emb_dim, ch)
        self.beta = nn.Linear(emb_dim, ch)
        nn.init.zeros_(self.gamma.weight); nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight); nn.init.zeros_(self.beta.bias)

    def forward(self, x, e):
        return self.norm(x) * (1 + self.gamma(e)[:, :, None, None]) + self.beta(e)[:, :, None, None]


class UNetGenerator(nn.Module):
    """pix2pix 'unet_128' (7 downsamplings 128 -> 1) with style conditioning."""

    def __init__(self, base_channels=64, emb_dim=16, dropout=0.5, n_styles=3):
        super().__init__()
        self.config = dict(base_channels=base_channels, emb_dim=emb_dim, dropout=dropout, n_styles=n_styles)
        g = base_channels
        self.style_emb = nn.Embedding(n_styles, emb_dim)
        enc = [g, 2 * g, 4 * g, 8 * g, 8 * g, 8 * g, 8 * g]
        self.down = nn.ModuleList()
        cin = 3
        for i, c in enumerate(enc):
            layers = [] if i == 0 else [nn.LeakyReLU(0.2, inplace=True)]
            layers.append(nn.Conv2d(cin, c, 4, 2, 1, bias=False))
            if 0 < i < len(enc) - 1:
                layers.append(nn.InstanceNorm2d(c, affine=True))
            self.down.append(nn.Sequential(*layers))
            cin = c
        dec_out = [8 * g, 8 * g, 8 * g, 4 * g, 2 * g, g]           # u7 .. u2
        skips = enc[::-1][1:]                                        # d6 .. d1 channels
        self.up_conv, self.up_norm, self.up_drop = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        cin = enc[-1] + emb_dim                                      # bottleneck + style embedding
        for i, c in enumerate(dec_out):
            self.up_conv.append(nn.ConvTranspose2d(cin, c, 4, 2, 1, bias=False))
            self.up_norm.append(CondInstanceNorm(c, emb_dim))
            self.up_drop.append(nn.Dropout(dropout if i < 3 else 0.0))
            cin = c + skips[i]
        self.final = nn.ConvTranspose2d(cin, 3, 4, 2, 1)

    def forward(self, x, style):
        e = self.style_emb(style)
        feats, h = [], x
        for d in self.down:
            h = d(h)
            feats.append(h)
        h = torch.cat([feats[-1], e[:, :, None, None].expand(-1, -1, h.shape[2], h.shape[3])], 1)
        for i in range(len(self.up_conv)):
            h = self.up_conv[i](F.relu(h))
            h = self.up_drop[i](self.up_norm[i](h, e))
            h = torch.cat([h, feats[-2 - i]], 1)
        return torch.tanh(self.final(F.relu(h)))


class PatchDiscriminator(nn.Module):
    """70x70-style PatchGAN on (photo, sketch, style map) -> 14x14 patch logits for 128x128 inputs."""

    def __init__(self, base_channels=64, emb_dim=16, n_styles=3):
        super().__init__()
        self.config = dict(base_channels=base_channels, emb_dim=emb_dim, n_styles=n_styles)
        d = base_channels
        self.style_emb = nn.Embedding(n_styles, emb_dim)
        self.net = nn.Sequential(
            nn.Conv2d(6 + emb_dim, d, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(d, 2 * d, 4, 2, 1, bias=False), nn.InstanceNorm2d(2 * d, affine=True), nn.LeakyReLU(0.2, True),
            nn.Conv2d(2 * d, 4 * d, 4, 2, 1, bias=False), nn.InstanceNorm2d(4 * d, affine=True), nn.LeakyReLU(0.2, True),
            nn.Conv2d(4 * d, 8 * d, 4, 1, 1, bias=False), nn.InstanceNorm2d(8 * d, affine=True), nn.LeakyReLU(0.2, True),
            nn.Conv2d(8 * d, 1, 4, 1, 1))

    def forward(self, photo, sketch, style):
        e = self.style_emb(style)[:, :, None, None].expand(-1, -1, photo.shape[2], photo.shape[3])
        return self.net(torch.cat([photo, sketch, e], 1))


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.InstanceNorm2d) and m.affine:
        nn.init.normal_(m.weight, 1.0, 0.02); nn.init.zeros_(m.bias)


class GeneratorExport(nn.Module):
    """ONNX wrapper: photo [B,3,128,128] in [0,1] + style int64 [B] (0,1,2) -> sketch [B,3,128,128] in [0,1]."""

    def __init__(self, G):
        super().__init__()
        self.G = G

    def forward(self, photo, style):
        return (self.G(photo * 2 - 1, style) + 1) / 2


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------
def train_one_epoch(G, D, loader, opt_g, opt_d, device, lambda_l1, scaler_g, scaler_d, amp=True):
    G.train(); D.train()
    use_amp = amp and device.type == "cuda"
    bce = nn.BCEWithLogitsLoss()
    agg = {k: 0.0 for k in ("d_real", "d_fake", "d_loss", "g_adv", "g_l1", "g_loss", "d_acc_real", "d_acc_fake")}
    n = 0
    for x, y, s in loader:
        x, y, s = x.to(device, non_blocking=True), y.to(device, non_blocking=True), s.to(device, non_blocking=True)
        # ---- discriminator
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            fake = G(x, s)
            lr_real = D(x, y, s)
            lr_fake = D(x, fake.detach(), s)
        d_real = bce(lr_real.float(), torch.ones_like(lr_real, dtype=torch.float32))
        d_fake = bce(lr_fake.float(), torch.zeros_like(lr_fake, dtype=torch.float32))
        d_loss = 0.5 * (d_real + d_fake)
        opt_d.zero_grad(set_to_none=True)
        scaler_d.scale(d_loss).backward(); scaler_d.step(opt_d); scaler_d.update()
        # ---- generator
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            lg_fake = D(x, fake, s)
        g_adv = bce(lg_fake.float(), torch.ones_like(lg_fake, dtype=torch.float32))
        g_l1 = F.l1_loss(fake.float(), y.float())
        g_loss = g_adv + lambda_l1 * g_l1
        if not torch.isfinite(g_loss) or not torch.isfinite(d_loss):
            raise FloatingPointError("non-finite GAN loss")
        opt_g.zero_grad(set_to_none=True)
        scaler_g.scale(g_loss).backward(); scaler_g.step(opt_g); scaler_g.update()
        b = len(s)
        for k, v in (("d_real", d_real), ("d_fake", d_fake), ("d_loss", d_loss), ("g_adv", g_adv), ("g_l1", g_l1), ("g_loss", g_loss)):
            agg[k] += v.item() * b
        agg["d_acc_real"] += (lr_real.float() > 0).float().mean().item() * b
        agg["d_acc_fake"] += (lr_fake.float() < 0).float().mean().item() * b
        n += b
    return {k: v / max(n, 1) for k, v in agg.items()}


def linear_decay(total_epochs):
    """pix2pix schedule: constant lr for the first half, then linear decay to 0."""
    half = total_epochs // 2
    return lambda ep: 1.0 if ep < half else max(0.0, 1.0 - (ep - half) / max(1, total_epochs - half))


@torch.no_grad()
def generate(G, photos_m11, styles, device, bs=64):
    """photos in [-1,1] (N,3,H,W) tensor, styles (N,) -> sketches in [0,1] numpy (N,3,H,W)."""
    G.eval()
    outs = []
    for i in range(0, len(photos_m11), bs):
        o = G(photos_m11[i:i + bs].to(device), styles[i:i + bs].to(device))
        outs.append(((o.float() + 1) / 2).clamp(0, 1).cpu())
    return torch.cat(outs)


class Metrics:
    """L1 / PSNR / SSIM always; LPIPS if the lpips package + AlexNet weights are available."""

    def __init__(self, device):
        import restoration as R
        self.R, self.device, self.lpips = R, device, None
        try:
            import lpips
            self.lpips = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        except Exception as e:
            self.lpips_error = repr(e)

    @torch.no_grad()
    def per_image(self, fake01, real01):
        R = self.R
        out = {"l1": (fake01 - real01).abs().mean(dim=(1, 2, 3)).numpy(),
               "psnr": R.psnr_per_image(fake01, real01).numpy(), "ssim": R.ssim_per_image(fake01, real01).numpy()}
        if self.lpips is not None:
            vals = []
            for i in range(0, len(fake01), 64):
                a = fake01[i:i + 64].to(self.device) * 2 - 1; b = real01[i:i + 64].to(self.device) * 2 - 1
                vals.append(self.lpips(a, b).flatten().cpu())
            out["lpips"] = torch.cat(vals).numpy()
        return out


def val_score(m: dict) -> float:
    """Optuna objective: structure (SSIM) + perceptual similarity (1 - LPIPS); SSIM only if LPIPS is unavailable."""
    if "lpips" in m:
        return float(0.5 * np.mean(m["ssim"]) + 0.5 * (1 - np.mean(m["lpips"])))
    return float(np.mean(m["ssim"]))
