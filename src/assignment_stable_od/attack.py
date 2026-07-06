"""Core attack primitives: losses, image I/O, PGD loop."""

from __future__ import annotations
from contextlib import nullcontext
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import rotate as tvf_rotate

from .pruning import temporary_random_pruning

# ImageNet normalization constants used by MMDet
_MEAN    = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_STD     = np.array([58.395,  57.12,  57.375], dtype=np.float32)
_MEAN_T  = torch.from_numpy(_MEAN).view(1, 3, 1, 1)
_STD_T   = torch.from_numpy(_STD).view(1, 3, 1, 1)
_STD_AVG = float(_STD.mean())   # ≈ 57.6, for pixel → normalised conversion

# Valid pixel range [0, 255] expressed in normalised space, per channel
# (used by RRB's pad/clamp steps, which operate on raw pixel bounds in the
# original OSFD implementation).
_PIXEL_LO_T = torch.from_numpy((0.0 - _MEAN) / _STD).view(1, 3, 1, 1)
_PIXEL_HI_T = torch.from_numpy((255.0 - _MEAN) / _STD).view(1, 3, 1, 1)


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

    # E3a — low-frequency gradient constraint
    low_freq_keep: float = 0.0   # fraction of freq bandwidth to keep; 0.0 = disabled

    # E3b — patch masking (zero random patches before feature extraction)
    patch_mask_size:  int = 0    # patch side length in pixels; 0 = disabled
    patch_mask_count: int = 4    # number of patches to zero per forward pass

    # E1c — RRB augmentation (OSFD's T(.): Rotation, Resizing, Blur)
    # Defaults per Ding et al. AAAI24 "Parameters" section / attack_faster_rcnn.yaml
    rrb_enabled:  bool  = False
    rrb_theta:    float = 7.0    # max rotation angle, degrees
    rrb_l_s:      int   = 10     # rotation-axis jitter around box/image center, px
    rrb_rho:      float = 0.8    # resize aggressiveness relative to object size
    rrb_s_max:    float = 1.10   # max resize scale factor
    rrb_sigma_px: float = 6.0    # Gaussian noise std ("blur" in the paper), pixel units


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


# ── E3 extensions ─────────────────────────────────────────────────────────────

def low_freq_filter(grad: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    """Low-pass filter on gradient via 2D rFFT (E3a).

    Keeps the lowest keep_ratio fraction of spatial frequencies, zeroing
    high-freq components that encode texture/fine-detail rather than shape.
    Operates independently per channel.
    """
    G = torch.fft.rfft2(grad)              # [..., H, W//2+1], complex
    H, W_h = G.shape[-2], G.shape[-1]
    # Half-bandwidth: keep k rows from top and k rows from bottom (symmetric)
    h_k = max(1, round(H * keep_ratio * 0.5))
    w_k = max(1, round(W_h * keep_ratio))
    mask = torch.zeros(H, W_h, device=grad.device, dtype=grad.real.dtype)
    mask[:h_k, :w_k] = 1.0
    mask[-h_k:, :w_k] = 1.0               # negative-frequency rows (wrap-around)
    return torch.fft.irfft2(G * mask, s=grad.shape[-2:])


def patch_mask_image(
    img_t: torch.Tensor,
    patch_size: int,
    n_patches: int,
    seed: int,
) -> torch.Tensor:
    """Zero n_patches random patches via differentiable multiply (E3b).

    Multiplying by a 0/1 mask is autograd-safe: gradient in masked regions
    is zeroed, so the PGD update ignores those spatial locations this step.
    Forces the perturbation to be effective across diverse spatial regions.
    """
    H, W = img_t.shape[-2], img_t.shape[-1]
    mask = torch.ones(1, 1, H, W, device=img_t.device)
    rng  = torch.Generator(device="cpu")   # randint only supports CPU generators
    rng.manual_seed(seed)
    ps = patch_size
    for _ in range(n_patches):
        y0 = int(torch.randint(0, max(1, H - ps), (1,), generator=rng).item())
        x0 = int(torch.randint(0, max(1, W - ps), (1,), generator=rng).item())
        mask[..., y0:y0 + ps, x0:x0 + ps] = 0.0
    return img_t * mask


# ── E1c: RRB augmentation (OSFD's T(.)) ──────────────────────────────────────

def _rrb_rotate(
    img_t: torch.Tensor,
    gt_boxes: torch.Tensor | None,
    theta: float,
    l_s: int,
    seed: int,
) -> torch.Tensor:
    """Random axis rotation: rotate around a randomly chosen GT-box center
    (jittered by ±l_s px) or the image center if no boxes are available.
    """
    H, W = img_t.shape[-2], img_t.shape[-1]
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)

    centers = [(W / 2.0, H / 2.0)]
    if gt_boxes is not None and gt_boxes.numel() > 0:
        cx = ((gt_boxes[:, 0] + gt_boxes[:, 2]) / 2.0).tolist()
        cy = ((gt_boxes[:, 1] + gt_boxes[:, 3]) / 2.0).tolist()
        centers += list(zip(cx, cy))

    idx    = int(torch.randint(0, len(centers), (1,), generator=rng).item())
    cx, cy = centers[idx]
    if l_s > 0:
        cx += int(torch.randint(-l_s, l_s + 1, (1,), generator=rng).item())
        cy += int(torch.randint(-l_s, l_s + 1, (1,), generator=rng).item())
    cx = float(min(max(cx, 0), W - 1))
    cy = float(min(max(cy, 0), H - 1))

    angle = float(torch.empty(1).uniform_(-theta, theta, generator=rng).item())
    return tvf_rotate(img_t, angle, center=[cx, cy], fill=0.0)


def _rrb_resize(
    img_t: torch.Tensor,
    gt_boxes: torch.Tensor | None,
    rho: float,
    s_max: float,
    seed: int,
) -> torch.Tensor:
    """Adaptive random resizing: scale+pad relative to a randomly chosen GT
    box's size, then resize back to the original resolution. No-op if no
    boxes are available (paper's formulation requires a reference box).
    """
    if gt_boxes is None or gt_boxes.numel() == 0:
        return img_t

    H, W = img_t.shape[-2], img_t.shape[-1]
    rng  = torch.Generator(device="cpu")
    rng.manual_seed(seed)

    idx   = int(torch.randint(0, gt_boxes.shape[0], (1,), generator=rng).item())
    box   = gt_boxes[idx]
    box_w = float((box[2] - box[0]).item())
    box_h = float((box[3] - box[1]).item())

    scale_h = min(1.0 + rho * (box_h / H), s_max)
    scale_w = min(1.0 + rho * (box_w / W), s_max)
    max_h, max_w = int(scale_h * H), int(scale_w * W)
    new_h = int(torch.randint(H, max_h + 1, (1,), generator=rng).item())
    new_w = int(torch.randint(W, max_w + 1, (1,), generator=rng).item())

    rescaled = F.interpolate(img_t, size=(new_h, new_w), mode="bilinear", align_corners=True)
    rem_h, rem_w = max_h - new_h, max_w - new_w
    pad_top  = int(torch.randint(0, rem_h + 1, (1,), generator=rng).item()) if rem_h > 0 else 0
    pad_left = int(torch.randint(0, rem_w + 1, (1,), generator=rng).item()) if rem_w > 0 else 0
    # Pad with normalised-zero (≈ mean pixel) rather than raw-pixel-zero
    # (black), avoiding an artificial dark border that could skew the
    # surrogate's BatchNorm statistics; a deliberate simplification of the
    # original pixel-space implementation.
    padded = F.pad(rescaled, (pad_left, rem_w - pad_left, pad_top, rem_h - pad_top),
                    mode="constant", value=0.0)
    return F.interpolate(padded, size=(H, W), mode="bilinear", align_corners=True)


def _rrb_noise(img_t: torch.Tensor, sigma_norm: float, seed: int) -> torch.Tensor:
    """Additive Gaussian noise (the paper calls this 'Gaussian blur', but its
    own reference implementation is elementwise noise, not a spatial blur
    kernel), clamped to the valid normalised pixel range.
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    noise = torch.randn(img_t.shape, generator=g).to(img_t.device, img_t.dtype)
    out   = img_t + sigma_norm * noise
    lo, hi = _PIXEL_LO_T.to(img_t.device), _PIXEL_HI_T.to(img_t.device)
    return torch.max(torch.min(out, hi), lo)


def rrb_augment(
    img_t: torch.Tensor,
    gt_boxes: torch.Tensor | None,
    cfg: "AttackConfig",
    seed: int,
) -> torch.Tensor:
    """OSFD's RRB augmentation T(.) (Ding et al. AAAI24): rotate around an
    object-centred axis, adaptively resize relative to object scale, then
    add Gaussian noise. Sequential composition (rotate -> resize -> noise)
    matches the reference implementation, which applies rotation then
    resizing to the same evolving tensor before blurring
    (OSFD/attack/base/RRB.py::preprocess_data).

    Simplification: the reference code additionally concatenates a
    rotate-only view with this rotate+resize view into one mini-batch each
    step; here a single composed view is sampled per forward pass instead,
    since the n_masks/n_iters loop already performs many independent
    stochastic passes whose gradients are averaged — equivalent diversity
    without needing an explicit doubled batch.
    """
    out = _rrb_rotate(img_t, gt_boxes, cfg.rrb_theta, cfg.rrb_l_s, seed)
    out = _rrb_resize(out, gt_boxes, cfg.rrb_rho, cfg.rrb_s_max, seed + 1)
    out = _rrb_noise(out, px_to_norm(cfg.rrb_sigma_px), seed + 2)
    return out


# ── PGD loop ──────────────────────────────────────────────────────────────────

def _grad_single_pass(
    model,
    img_t: torch.Tensor,
    delta: torch.Tensor,
    cfg: AttackConfig,
    seed: int,
    clean_feats: list[torch.Tensor] | None,
    gt_boxes: torch.Tensor | None = None,
) -> torch.Tensor:
    """One forward-backward pass, optionally with random weight masking.

    E1c RRB augmentation and E3b patch masking are applied to the input
    before forward (gradient flows through both, so the PGD update accounts
    for their effect). E3a low-freq filter is applied to the gradient after
    backward.
    """
    x = (img_t + delta).requires_grad_(True)
    x_fwd = x
    # E3b: zero random patches before feature extraction
    if cfg.patch_mask_size > 0:
        x_fwd = patch_mask_image(x_fwd, cfg.patch_mask_size, cfg.patch_mask_count,
                                  seed=seed + 300_000)
    # E1c: OSFD's RRB augmentation T(.) — rotate, adaptively resize, add noise
    if cfg.rrb_enabled:
        x_fwd = rrb_augment(x_fwd, gt_boxes, cfg, seed=seed + 500_000)
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
            loss = osfd_feature_loss(model, x_fwd, clean_feats, k=cfg.osfd_k)
        elif cfg.loss_type == "rpn":
            loss = rpn_suppression_loss(model, x_fwd)
        else:
            raise ValueError(f"Unknown loss_type={cfg.loss_type!r}. Use 'osfd' or 'rpn'.")
        loss.backward()

    grad = x.grad.detach()
    # E3a: project gradient onto low-frequency subspace
    if cfg.low_freq_keep > 0.0:
        grad = low_freq_filter(grad, cfg.low_freq_keep)
    model.zero_grad()
    return grad


def pgd_attack(
    model,
    img_bgr: np.ndarray,
    cfg: AttackConfig,
    aux_model=None,
    gt_boxes: np.ndarray | None = None,
) -> np.ndarray:
    """MIM-style PGD attack. Inputs and outputs are uint8 BGR HWC numpy arrays.

    cfg.aux_model:  Second surrogate for cross-backbone gradient averaging (E3c).
                    Gradients from both models are averaged per mask iteration.
    gt_boxes:       [N, 4] xyxy GT boxes in img_bgr's pixel coordinate frame.
                    Only used by E1c's RRB augmentation to pick rotation axes
                    and a reference box for adaptive resizing; the loss
                    functions themselves remain label-free. Ignored unless
                    cfg.rrb_enabled.
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

    gt_boxes_t = None
    if cfg.rrb_enabled and gt_boxes is not None and len(gt_boxes) > 0:
        gt_boxes_t = torch.as_tensor(gt_boxes, dtype=torch.float32, device=device)

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
            grad += _grad_single_pass(model, img_t, delta, cfg, seed, clean_feats, gt_boxes_t)
            if aux_model is not None:
                # Offset seed space to avoid correlation with primary model masks
                aux_seed = cfg.seed_base + 100_000 + step * cfg.n_masks + m
                grad += _grad_single_pass(aux_model, img_t, delta, cfg, aux_seed,
                                           aux_clean_feats, gt_boxes_t)

        grad  /= cfg.n_masks * n_srcs
        g_norm = grad.abs().mean().clamp_min(1e-12)
        g_mom  = cfg.momentum * g_mom + grad / g_norm

        with torch.no_grad():
            delta = (delta - step_n * g_mom.sign()).clamp(-eps_n, eps_n)

    return tensor_to_bgr((img_t + delta).detach())
