"""
classifier.py — corruption classifier for Task 2 (reused as the gating network in Task 3).

Classes (same order everywhere, incl. the Task 3 gate weights w0..w3):
    0 = clean, 1 = salt_pepper, 2 = gaussian_blur, 3 = occlusion

Design notes
  * The first conv block runs at FULL resolution (128x128): salt-and-pepper noise and slight blur are
    pixel-level cues that are destroyed by early downsampling.
  * GroupNorm (no running statistics): train/eval behave identically, which matters when the network is
    later fine-tuned as the Task 3 gate with small batches.
  * Global AVERAGE + global MAX pooling are concatenated: max pooling responds to rare extreme activations
    (isolated black/white impulses, hard edges of black occlusion boxes); average pooling to global
    statistics (overall loss of high frequencies under blur).
  * forward() returns LOGITS; softmax / temperature are applied by the caller (Task 3: softmax(G(x)/tau)).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CLASS_NAMES = ("clean", "salt_pepper", "gaussian_blur", "occlusion")
CHANNEL_CONFIGS = {
    "16-32-64-128": (16, 32, 64, 128),
    "32-64-128-256": (32, 64, 128, 256),
    "32-64-128-256-256": (32, 64, 128, 256, 256),
}


def _gn(c):
    return nn.GroupNorm(min(8, c), c)


class CorruptionClassifier(nn.Module):
    def __init__(self, channels=(32, 64, 128, 256), dropout: float = 0.2, num_classes: int = 4):
        super().__init__()
        if isinstance(channels, str):
            channels = CHANNEL_CONFIGS[channels]
        channels = tuple(int(c) for c in channels)
        self.config = dict(channels=list(channels), dropout=dropout, num_classes=num_classes)
        layers, cin = [], 3
        for i, c in enumerate(channels):
            stride = 1 if i == 0 else 2            # stage 0 keeps full resolution
            layers += [nn.Conv2d(cin, c, 3, stride, 1, bias=False), _gn(c), nn.SiLU(inplace=True),
                       nn.Conv2d(c, c, 3, 1, 1, bias=False), _gn(c), nn.SiLU(inplace=True)]
            cin = c
        self.features = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * cin, num_classes)

    def forward(self, x):
        h = self.features(x)
        h = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)
        return self.fc(self.dropout(h))


class ClassifierWithProbs(nn.Module):
    """ONNX wrapper: returns (logits, probabilities)."""

    def __init__(self, clf):
        super().__init__()
        self.clf = clf

    def forward(self, x):
        logits = self.clf(x)
        return logits, torch.softmax(logits, dim=1)


# --------------------------------------------------------------------------------------
# Training / evaluation
# --------------------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, scheduler, device, scaler, amp=True, grad_clip=1.0, label_smoothing=0.0):
    model.train()
    use_amp = amp and device.type == "cuda"
    tot_loss, correct, n = 0.0, 0, 0
    for x, _, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        loss = F.cross_entropy(logits.float(), y, label_smoothing=label_smoothing)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite classifier loss")
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        tot_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return {"loss": tot_loss / max(n, 1), "acc": correct / max(n, 1)}


@torch.no_grad()
def predict(model, loader, device, amp=True):
    """Returns (labels, probs[N,4], logits[N,4]) over a manifest loader (dataset order)."""
    model.eval()
    use_amp = amp and device.type == "cuda"
    labels, logits = [], []
    for batch in loader:
        x = batch[0].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            lg = model(x)
        logits.append(lg.float().cpu())
        labels.append(batch[2])
    logits = torch.cat(logits)
    return torch.cat(labels).numpy(), torch.softmax(logits, 1).numpy(), logits.numpy()


def classification_metrics(y_true, probs) -> dict:
    """Accuracy, macro P/R/F1, per-class P/R/F1/support, row-normalised confusion matrix, CE loss."""
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
    y_pred = probs.argmax(1)
    labels = list(range(len(CLASS_NAMES)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    ce = float(-np.log(np.clip(probs[np.arange(len(y_true)), y_true], 1e-12, 1)).mean())
    out = {"acc": float(accuracy_score(y_true, y_pred)), "macro_precision": float(mp), "macro_recall": float(mr),
           "macro_f1": float(mf), "loss": ce, "cm": cm, "cm_norm": cm_norm}
    for i, n in enumerate(CLASS_NAMES):
        out[f"precision/{n}"], out[f"recall/{n}"], out[f"f1/{n}"], out[f"support/{n}"] = float(p[i]), float(r[i]), float(f[i]), int(s[i])
    return out


def per_class_table(m: dict):
    import pandas as pd
    rows = [{"class": n, "precision": m[f"precision/{n}"], "recall": m[f"recall/{n}"], "f1": m[f"f1/{n}"],
             "support": m[f"support/{n}"]} for n in CLASS_NAMES]
    rows.append({"class": "macro avg", "precision": m["macro_precision"], "recall": m["macro_recall"],
                 "f1": m["macro_f1"], "support": int(sum(m[f"support/{n}"] for n in CLASS_NAMES))})
    rows.append({"class": "accuracy", "precision": np.nan, "recall": np.nan, "f1": m["acc"],
                 "support": int(sum(m[f"support/{n}"] for n in CLASS_NAMES))})
    return pd.DataFrame(rows).set_index("class")


def plot_confusion(cm_norm, cm_counts=None, title="", path=None, ax=None):
    import matplotlib.pyplot as plt
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    short = ["clean", "s&p", "blur", "occl."]
    ax.set_xticks(range(4)); ax.set_xticklabels(short); ax.set_yticks(range(4)); ax.set_yticklabels(short)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    for i in range(4):
        for j in range(4):
            txt = f"{cm_norm[i, j]:.2f}" + (f"\n({cm_counts[i, j]})" if cm_counts is not None else "")
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if cm_norm[i, j] > 0.6 else "black")
    ax.set_title(title, fontsize=10)
    if own:
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        if path:
            fig.savefig(path, dpi=150)
        return fig
    return im
