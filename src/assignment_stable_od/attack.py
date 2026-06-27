"""Core attack primitives: losses, image I/O, PGD loop."""

from __future__ import annotations
from contextlib import nullcontext
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .pruning import temporary_random_pruning

# ImageNet normalization constants used by MMDet
_MEAN    = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_STD     = np.array([58.395,  57.12,  57.375], dtype=np.float32)
_MEAN_T  = torch.from_numpy(_MEAN).view(1, 3, 1, 1)
_STD_T   = torch.from_numpy(_STD).view(1, 3, 1, 1)
_STD_AVG = float(_STD.mean())   # ≈ 57.6, for pixel → normalised conversion


# ── Attack configuration ───────────────────────────────────────────────────────

@dataclass
class AttackConfig:
    """All parameters governing one PGD attack run."""

    # PGD budget
    epsilon_px:   float = 8.0
    n_iters:      int   = 40
    step_size_px: float = 2.0
    momentum:     float = 0.9
    seed_base:    int   = 0
    device:       str   = "cuda:0"

    # Loss
    loss_type: str   = "osfd"   # "osfd" | "rpn"
    osfd_k:    float = 3.0      # amplification factor in OSFD Eq. 2

    # Pruning (RaPA-style DropConnect)
    n_masks:       int               = 1
    pruning_scope: str | None        = "backbone"
    pruning_rate:  float             = 0.0
    pruning_types: list[str] | None  = field(default=None)
    # None → ["Normalization", "Linear"] inside temporary_random_pruning


# ── Image I/O ─────────────────────────────────────────────────────────────────

def load_image_bgr(img_path) -> np.ndarray:
    """Load and resize BGR image to MMDet standard (short side 800, long ≤ 1333)."""
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    scale = 800 / min(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    if max(nh, nw) > 1333:
        s2 = 1333 / max(nh, nw)
        nh, nw = int(round(nh * s2)), int(round(nw * s2))
    return cv2.resize(img, (nw, nh))


def bgr_to_tensor(img_bgr: np.ndarray, device: str = "cuda:0") -> torch.Tensor:
    """uint8 BGR HWC → normalised float RGB [1, 3, H, W]."""
    rgb = img_bgr[:, :, ::-1].astype(np.float32)
    t   = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    return (t - _MEAN_T.to(device)) / _STD_T.to(device)


def tensor_to_bgr(img_t: torch.Tensor) -> np.ndarray:
    """Normalised float RGB [1, 3, H, W] → uint8 BGR HWC."""
    pixel = (img_t * _STD_T.to(img_t.device) + _MEAN_T.to(img_t.device)).clamp(0, 255)
    arr   = pixel.squeeze(0).permute(1, 2, 0).cpu().byte().numpy()
    return arr[:, :, ::-1].copy()


def px_to_norm(px: float) -> float:
    """Convert L_inf pixel budget to normalised-space scalar (mean-std approximation)."""
    return px / _STD_AVG


# ── Losses ────────────────────────────────────────────────────────────────────

def rpn_suppression_loss(model, img_t: torch.Tensor) -> torch.Tensor:
    """Minimise RPN sigmoid objectness: backbone → neck → rpn_head."""
    feats = model.backbone(img_t)
    feats = model.neck(feats)
    cls_scores, _ = model.rpn_head(feats)
    return sum(torch.sigmoid(s).mean() for s in cls_scores)


def osfd_feature_loss(
    model,
    img_adv_t: torch.Tensor,
    clean_feats: list[torch.Tensor],
    k: float = 3.0,
) -> torch.Tensor:
    """OSFD backbone feature distortion (Ding et al. AAAI 2024, Eq. 2).

    Maximises MSE(f_adv, k·f_clean) across all backbone stages.
    k=3 amplifies the target for significant (high-valued) features in object
    regions, suppressing them while elevating vicinal (background) features.
    Returns negated loss: minimising it = maximising feature distortion.
    """
    feats_adv = model.backbone(img_adv_t)
    return -sum(
        F.mse_loss(f_adv, (k * f_cln).detach())
        for f_adv, f_cln in zip(feats_adv, clean_feats)
    )


# ── PGD loop ──────────────────────────────────────────────────────────────────

def _grad_single_pass(
    model,
    img_t: torch.Tensor,
    delta: torch.Tensor,
    cfg: AttackConfig,
    seed: int,
    clean_feats: list[torch.Tensor] | None,
) -> torch.Tensor:
    """One forward-backward pass, optionally with random weight masking.

    Backward runs while masks are active → gradient correctly reflects pruned model.
    """
    x = (img_t + delta).requires_grad_(True)
    ctx = (
        temporary_random_pruning(
            model, cfg.pruning_rate,
            scope=cfg.pruning_scope, seed=seed, type_list=cfg.pruning_types,
        )
        if cfg.pruning_scope and cfg.pruning_rate > 0
        else nullcontext()
    )
    with ctx:
        if cfg.loss_type == "osfd":
            loss = osfd_feature_loss(model, x, clean_feats, k=cfg.osfd_k)
        elif cfg.loss_type == "rpn":
            loss = rpn_suppression_loss(model, x)
        else:
            raise ValueError(f"Unknown loss_type={cfg.loss_type!r}. Use 'osfd' or 'rpn'.")
        loss.backward()

    grad = x.grad.detach()
    model.zero_grad()
    return grad


def pgd_attack(
    model,
    img_bgr: np.ndarray,
    cfg: AttackConfig,
    aux_model=None,
) -> np.ndarray:
    """MIM-style PGD attack. Inputs and outputs are uint8 BGR HWC numpy arrays.

    cfg.aux_model:  Second surrogate for cross-backbone gradient averaging (E3c).
                    Gradients from both models are averaged per mask iteration.
    """
    model.eval()
    if aux_model is not None:
        aux_model.eval()

    device = cfg.device
    eps_n  = px_to_norm(cfg.epsilon_px)
    step_n = px_to_norm(cfg.step_size_px)
    img_t  = bgr_to_tensor(img_bgr, device)
    delta  = torch.empty_like(img_t).uniform_(-eps_n, eps_n)
    g_mom  = torch.zeros_like(img_t)
    n_srcs = 2 if aux_model is not None else 1

    # Pre-compute clean backbone features once per image (OSFD only)
    clean_feats = aux_clean_feats = None
    if cfg.loss_type == "osfd":
        with torch.no_grad():
            clean_feats = [f.detach() for f in model.backbone(img_t)]
        if aux_model is not None:
            with torch.no_grad():
                aux_clean_feats = [f.detach() for f in aux_model.backbone(img_t)]

    for step in range(cfg.n_iters):
        grad = torch.zeros_like(img_t)

        for m in range(cfg.n_masks):
            seed = cfg.seed_base + step * cfg.n_masks + m
            grad += _grad_single_pass(model, img_t, delta, cfg, seed, clean_feats)
            if aux_model is not None:
                # Offset seed space to avoid correlation with primary model masks
                aux_seed = cfg.seed_base + 100_000 + step * cfg.n_masks + m
                grad += _grad_single_pass(aux_model, img_t, delta, cfg, aux_seed, aux_clean_feats)

        grad  /= cfg.n_masks * n_srcs
        g_norm = grad.abs().mean().clamp_min(1e-12)
        g_mom  = cfg.momentum * g_mom + grad / g_norm

        with torch.no_grad():
            delta = (delta - step_n * g_mom.sign()).clamp(-eps_n, eps_n)

    return tensor_to_bgr((img_t + delta).detach())
