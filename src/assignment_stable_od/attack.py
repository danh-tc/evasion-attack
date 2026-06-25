"""Core attack primitives: loss, preprocessing, PGD loop."""

from __future__ import annotations

from contextlib import nullcontext

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .pruning import temporary_random_pruning

# Normalisation constants matching COCO-pretrained MMDet models (RGB order).
_MEAN = np.array([123.675, 116.28,  103.53],  dtype=np.float32)
_STD  = np.array([58.395,  57.12,   57.375],  dtype=np.float32)
_MEAN_T = torch.from_numpy(_MEAN).view(1, 3, 1, 1)
_STD_T  = torch.from_numpy(_STD ).view(1, 3, 1, 1)


# ── Image I/O helpers ─────────────────────────────────────────────────────────

def load_image_bgr(img_path) -> np.ndarray:
    """Load + resize BGR image to MMDet standard (short side 800, long ≤1333)."""
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    scale = 800 / min(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    if max(nh, nw) > 1333:
        s2 = 1333 / max(nh, nw)
        nh, nw = int(round(nh * s2)), int(round(nw * s2))
    return cv2.resize(img, (nw, nh))          # HWC BGR uint8


def bgr_to_tensor(img_bgr: np.ndarray, device: str = "cuda:0") -> torch.Tensor:
    """uint8 BGR HWC → normalised float RGB [1,3,H,W]."""
    img_rgb = img_bgr[:, :, ::-1].astype(np.float32)
    t = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    mean = _MEAN_T.to(device)
    std  = _STD_T.to(device)
    return (t - mean) / std


def tensor_to_bgr(img_t: torch.Tensor) -> np.ndarray:
    """Normalised float RGB [1,3,H,W] → uint8 BGR HWC (clamps to [0,255])."""
    mean = _MEAN_T.to(img_t.device)
    std  = _STD_T.to(img_t.device)
    pixel = (img_t * std + mean).clamp(0, 255)
    arr = pixel.squeeze(0).permute(1, 2, 0).cpu().byte().numpy()
    return arr[:, :, ::-1].copy()             # RGB → BGR


def load_and_preprocess(img_path, device: str = "cuda:0") -> torch.Tensor:
    return bgr_to_tensor(load_image_bgr(img_path), device)


# ── Epsilon / step-size unit conversion ──────────────────────────────────────
# We work with epsilon in *pixel* units [0, 255].
# In normalised space the per-channel epsilon is epsilon_px / std_c.
# We use the mean std ≈ 57.6 as a single scalar approximation so that
# delta can be a single unconstrained tensor and we apply one clamp.

_STD_MEAN = float(_STD.mean())   # ≈ 57.6


def px_to_norm(value_px: float) -> float:
    """Convert L_inf budget in pixel units to normalised-space scalar."""
    return value_px / _STD_MEAN


# ── Differentiable loss ───────────────────────────────────────────────────────

def rpn_suppression_loss(model, img_t: torch.Tensor) -> torch.Tensor:
    """Minimise RPN objectness — pushes model to produce 0 detections.

    Differentiable w.r.t. img_t through backbone → neck → RPN.
    """
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
    """OSFD-style backbone feature distortion loss (RaPA+OSFD combination).

    Maximises L2 distance between adversarial and clean backbone features.
    Operates at backbone level only — architecture-agnostic, no RPN head
    required → works on DINO/DETR targets that lack rpn_head.

    clean_feats should be pre-computed once before the attack loop with
    torch.no_grad() to avoid storing the computation graph.

    Note: k is retained for API compatibility but unused; the objective is
    to maximise ||f_adv - f_clean||, not to target k*f_clean.
    """
    feats_adv = model.backbone(img_adv_t)
    # Negate MSE so that minimising the loss maximises feature distortion.
    return -sum(
        F.mse_loss(f_adv, f_clean.detach())
        for f_adv, f_clean in zip(feats_adv, clean_feats)
    )


# ── PGD with optional pruning diversity ───────────────────────────────────────

def _compute_grad(model, img_t, delta, pruning_scope, pruning_rate, seed,
                  use_feature_loss, clean_feats, feature_k):
    """Single forward-backward pass, returns detached gradient."""
    x_adv = (img_t + delta).requires_grad_(True)
    ctx = (
        temporary_random_pruning(model, pruning_rate, scope=pruning_scope, seed=seed)
        if pruning_scope and pruning_rate > 0 else nullcontext()
    )
    with ctx:
        if use_feature_loss:
            loss = osfd_feature_loss(model, x_adv, clean_feats, k=feature_k)
        else:
            loss = rpn_suppression_loss(model, x_adv)
        loss.backward()
    grad = x_adv.grad.detach()
    model.zero_grad()
    return grad


def pgd_attack(
    model,
    img_bgr: np.ndarray,
    *,
    epsilon_px: float,
    n_iters: int,
    step_size_px: float,
    n_masks: int = 1,
    pruning_scope: str | None = None,
    pruning_rate: float = 0.0,
    momentum: float = 0.9,
    seed_base: int = 0,
    device: str = "cuda:0",
    aux_model=None,
    use_osfd: bool = False,
    osfd_k: float = 3.0,
) -> np.ndarray:
    """MIM-style PGD.  Inputs and outputs are uint8 BGR HWC arrays.

    Args:
        img_bgr:      Clean image (uint8 BGR, already resized to model input).
        epsilon_px:   L_inf budget in pixel units (e.g. 8 for 8/255 attack).
        step_size_px: PGD step size in pixel units.
        n_masks:      Masks to average gradients over per iteration (per model).
        pruning_scope/rate: Optional scope/rate for RaPA-OD.
        aux_model:    Second surrogate for cross-backbone gradient averaging
                      (Direction A). When provided, each step averages gradients
                      from both model and aux_model with independent pruning masks.

    Returns:
        Adversarial image as uint8 BGR numpy array.
    """
    model.eval()
    if aux_model is not None:
        aux_model.eval()
    eps_n  = px_to_norm(epsilon_px)
    step_n = px_to_norm(step_size_px)

    img_t = bgr_to_tensor(img_bgr, device)
    delta = torch.empty_like(img_t).uniform_(-eps_n, eps_n)
    g_mom = torch.zeros_like(img_t)

    n_sources = 2 if aux_model is not None else 1

    # Pre-compute clean backbone features once (OSFD loss only)
    clean_feats = None
    if use_osfd:
        with torch.no_grad():
            clean_feats = [f.detach() for f in model.backbone(img_t)]

    for step in range(n_iters):
        grad_accum = torch.zeros_like(img_t)

        for mask_idx in range(n_masks):
            seed = seed_base + step * n_masks + mask_idx
            grad_accum = grad_accum + _compute_grad(
                model, img_t, delta, pruning_scope, pruning_rate, seed,
                use_osfd, clean_feats, osfd_k,
            )

            if aux_model is not None:
                seed_aux = seed_base + 10000 + step * n_masks + mask_idx
                grad_accum = grad_accum + _compute_grad(
                    aux_model, img_t, delta, pruning_scope, pruning_rate, seed_aux,
                    False, None, osfd_k,
                )

        grad_accum = grad_accum / (n_masks * n_sources)

        grad_norm = grad_accum.abs().mean().clamp_min(1e-12)
        g_mom     = momentum * g_mom + grad_accum / grad_norm

        with torch.no_grad():
            delta = delta - step_n * g_mom.sign()
            delta = delta.clamp(-eps_n, eps_n)

    img_adv_t = (img_t + delta).detach()
    return tensor_to_bgr(img_adv_t)
