"""
moe.py — Task 3: jointly trained soft mixture-of-experts restoration.

    w      = softmax(G(x~) / tau)                          (gate G = Task 2 classifier, 4 logits)
    x_hat  = w0 * x~ + w1 * A_salt(x~) + w2 * A_blur(x~) + w3 * A_occ(x~)
    L_MoE  = l1 * L1 + ls * (1 - SSIM) + lc * CE + lb * L_balance
    L_balance = sum_k (mean_batch(w_k) - 1/4)^2           (computed on exactly balanced batches)

Branch order everywhere: 0 identity (clean), 1 salt_pepper, 2 gaussian_blur, 3 occlusion.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import restoration as R

BRANCHES = ("identity", "A_salt", "A_blur", "A_occ")
CLASS_NAMES = ("clean", "salt_pepper", "gaussian_blur", "occlusion")


class SoftMoE(nn.Module):
    def __init__(self, gate: nn.Module, experts: list[nn.Module], tau: float = 1.0):
        super().__init__()
        assert len(experts) == 3
        self.gate = gate
        self.experts = nn.ModuleList(experts)
        self.register_buffer("tau", torch.tensor(float(tau)))

    @classmethod
    def from_task2(cls, classifier: nn.Module, specialists: dict, tau: float = 1.0):
        """Deep-copies the trained Task 2 classifier (-> gate) and specialists 1,2,3 (-> experts)."""
        return cls(copy.deepcopy(classifier), [copy.deepcopy(specialists[k]) for k in (1, 2, 3)], tau)

    def forward(self, x):
        logits = self.gate(x)
        w = torch.softmax(logits.float() / self.tau, dim=1)                     # (B, 4)
        branches = torch.stack([x] + [e(x) for e in self.experts], dim=1)      # (B, 4, 3, H, W)
        out = (w[:, :, None, None, None].to(branches.dtype) * branches).sum(1)
        return out, w, logits

    # --- training phases --------------------------------------------------------------
    def freeze_experts(self, frozen: bool = True):
        for p in self.experts.parameters():
            p.requires_grad_(not frozen)

    def train_phase(self, experts_frozen: bool):
        """Gate always trains; frozen experts are put in eval mode (no code dropout)."""
        self.train()
        if experts_frozen:
            self.experts.eval()


class MoEExport(nn.Module):
    """ONNX wrapper: input -> (output image, routing weights, gate logits)."""

    def __init__(self, moe: SoftMoE):
        super().__init__()
        self.moe = moe

    def forward(self, x):
        out, w, logits = self.moe(x)
        return out.clamp(0, 1), w, logits


def balance_loss(w: torch.Tensor) -> torch.Tensor:
    return ((w.mean(0) - 1.0 / w.shape[1]) ** 2).sum()


def moe_loss(out, target, w, logits, labels, tau, l1=0.8, ls=0.2, lc=0.1, lb=0.01):
    out, target = out.float(), target.float()
    L1 = F.l1_loss(out, target)
    ssim = R.ssim_map(out, target).mean()
    # CE on the routing distribution itself (logits / tau), so the classification term shapes the actual weights
    ce = F.cross_entropy(logits.float() / tau, labels)
    bal = balance_loss(w)
    total = l1 * L1 + ls * (1 - ssim) + lc * ce + lb * bal
    return total, {"l1": L1.detach(), "ssim": ssim.detach(), "ce": ce.detach(), "balance": bal.detach()}


def train_one_epoch(moe, loader, optimizer, scheduler, device, scaler, lam, experts_frozen, amp=True, grad_clip=1.0):
    moe.train_phase(experts_frozen)
    use_amp = amp and device.type == "cuda"
    agg = {k: 0.0 for k in ("loss", "l1", "ssim", "ce", "balance", "gate_acc")}
    n = 0
    for x, y, lab in loader:
        x, y, lab = x.to(device, non_blocking=True), y.to(device, non_blocking=True), lab.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            out, w, logits = moe(x)
        loss, parts = moe_loss(out, y, w, logits, lab, moe.tau, **lam)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite MoE loss")
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in moe.parameters() if p.requires_grad], grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        b = len(lab)
        agg["loss"] += loss.item() * b
        for k in ("l1", "ssim", "ce", "balance"):
            agg[k] += parts[k].item() * b
        agg["gate_acc"] += (w.argmax(1) == lab).float().sum().item()
        n += b
    return {k: v / max(n, 1) for k, v in agg.items()}


@torch.no_grad()
def evaluate(moe, loader, device, amp=True, per_image=False):
    """Reconstruction metrics overall/per class, mean routing weights per true class, gate accuracy."""
    moe.eval()
    use_amp = amp and device.type == "cuda"
    P, S, M, W, L = [], [], [], [], []
    for batch in loader:
        x, y, lab = batch[0].to(device), batch[1].to(device), batch[2]
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            out, w, _ = moe(x)
        out = out.float().clamp(0, 1)
        P.append(R.psnr_per_image(out, y).cpu()); S.append(R.ssim_per_image(out, y).cpu()); M.append(R.mae_per_image(out, y).cpu())
        W.append(w.float().cpu()); L.append(lab)
    P, S, M, W, L = (torch.cat(t).numpy() for t in (P, S, M, W, L))
    res = {"psnr": float(P.mean()), "ssim": float(S.mean()), "mae": float(M.mean()),
           "gate_acc": float((W.argmax(1) == L).mean()), "weight_matrix": np.zeros((4, 4))}
    for c, n in enumerate(CLASS_NAMES):
        m = L == c
        if m.any():
            res[f"psnr/{n}"], res[f"ssim/{n}"], res[f"mae/{n}"] = float(P[m].mean()), float(S[m].mean()), float(M[m].mean())
            res["weight_matrix"][c] = W[m].mean(0)
    res["score"] = balanced_score(res)
    res["branch_usage"] = W.mean(0)                    # average weight per branch over all inputs
    if per_image:
        res["per_image"] = {"psnr": P, "ssim": S, "mae": M, "w": W, "label": L}
    return res


def balanced_score(res) -> float:
    """Mean over the 4 conditions of 0.5*SSIM + 0.5*min(PSNR/40, 1): every condition counts equally, and near-perfect
    clean reconstructions (very high PSNR) cannot dominate the objective."""
    vals = [R.val_score(res[f"psnr/{n}"], res[f"ssim/{n}"]) for n in CLASS_NAMES if f"psnr/{n}" in res]
    return float(np.mean(vals))


def collapse_check(res, min_usage=0.03, min_gate_acc=0.4) -> str | None:
    """Routing collapse = a branch that is (almost) never used, or a gate that no longer tracks the corruption."""
    u = res["branch_usage"]
    if u.min() < min_usage:
        return f"branch {BRANCHES[int(u.argmin())]} inactive (mean weight {u.min():.3f})"
    if res["gate_acc"] < min_gate_acc:
        return f"gate accuracy {res['gate_acc']:.2f} < {min_gate_acc}"
    return None
