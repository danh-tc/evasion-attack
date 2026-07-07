"""RRB (Random axis Rotation + adaptive Resizing + gaussian Blur) augmentation.

Ported from reference-source/OSFD/attack/base/RRB.py, stripped of its mmcv
1.x `IFGSM`/pipeline base-class dependencies down to standalone tensor
functions. Defaults match reference-source/OSFD/config/attack_faster_rcnn.yaml:
theta=7.0, l_s=10, rho=0.8, s_max=1.10, sigma=6.0.

Exploits the detector backbone's "limited equivariance": augmenting before
feature extraction is approximately equivalent to augmenting the features
directly (paper Eq. 8), so these transforms diversify the gradient signal
without breaking the object-aware suppress/amplify structure of the OSFD
loss, as long as ground-truth box centers are used to bias rotation/resize
toward object regions.
"""
from __future__ import annotations

import random

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import rotate


def random_axis_rotation(imgs: torch.Tensor, gt_bboxes: list[torch.Tensor], theta: float = 7.0, l_s: int = 10) -> torch.Tensor:
    """Rotate each image by a random small angle around a randomly chosen
    object-bbox center (or the image center), per paper "Random axis rotation"."""
    device = imgs.device
    result = []
    for idx in range(imgs.shape[0]):
        single = imgs[idx].unsqueeze(0)
        boxes = gt_bboxes[idx % len(gt_bboxes)]
        h, w = single.shape[-2], single.shape[-1]
        img_center = torch.tensor([[w // 2, h // 2]], device=device, dtype=torch.float32)
        if boxes.numel() > 0:
            box_centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            centers = torch.cat([box_centers, img_center], dim=0)
        else:
            centers = img_center
        if l_s > 0:
            centers = centers + torch.randint_like(centers, low=-l_s, high=l_s)
        cx, cy = centers[random.randrange(centers.shape[0])]
        angle = random.random() * 2 * theta - theta
        result.append(rotate(single, angle, center=[int(cx), int(cy)]))
    return torch.cat(result, dim=0)


def adaptive_random_resizing(
    imgs: torch.Tensor,
    gt_bboxes: list[torch.Tensor],
    rho: float = 0.8,
    s_max: float = 1.10,
) -> torch.Tensor:
    """Resize-and-pad each image with a scale tied to a random object bbox's
    size (larger objects -> more augmentation headroom), per paper Eq. 3-4."""
    result = []
    for idx in range(imgs.shape[0]):
        single = imgs[idx].unsqueeze(0)
        h0, w0 = single.shape[-2], single.shape[-1]
        boxes = gt_bboxes[idx % len(gt_bboxes)]
        if boxes.numel() > 0:
            box = boxes[random.randrange(boxes.shape[0])]
            box_w = float(box[2] - box[0])
            box_h = float(box[3] - box[1])
        else:
            box_w = box_h = 0.0

        scale_h = min(1 + rho * (box_h / h0), s_max)
        scale_w = min(1 + rho * (box_w / w0), s_max)
        new_h = random.randint(h0, max(h0, int(scale_h * h0)))
        new_w = random.randint(w0, max(w0, int(scale_w * w0)))
        rescaled = F.interpolate(single, size=(new_h, new_w), mode="bilinear", align_corners=True)

        rem_h = int(scale_h * h0) - new_h
        rem_w = int(scale_w * w0) - new_w
        pad_top = random.randint(0, max(rem_h, 0))
        pad_left = random.randint(0, max(rem_w, 0))
        padded = F.pad(rescaled, (pad_left, max(rem_w, 0) - pad_left, pad_top, max(rem_h, 0) - pad_top),
                        mode="constant", value=0.0)
        padded = F.interpolate(padded, size=(h0, w0), mode="bilinear", align_corners=True)
        result.append(padded)
    return torch.cat(result, dim=0)


def gaussian_blur(imgs: torch.Tensor, sigma: float = 6.0, pixel_max: float = 255.0) -> torch.Tensor:
    """Additive Gaussian noise (as in the original OSFD code — not a
    convolutional blur kernel, despite the name)."""
    return torch.clamp(imgs + torch.randn_like(imgs) * sigma, 0.0, pixel_max)


def rrb_augment(
    imgs: torch.Tensor,
    gt_bboxes: list[torch.Tensor],
    theta: float = 7.0,
    l_s: int = 10,
    rho: float = 0.8,
    s_max: float = 1.10,
    sigma: float = 6.0,
    prob: float = 1.0,
    pixel_max: float = 255.0,
) -> torch.Tensor:
    """Apply rotation then adaptive resizing (each independently gated by
    `prob`), then additive Gaussian noise, preserving batch size.

    Deviation from vendored OSFD: the original `RRB.preprocess_data`
    concatenates a rotation-only view and a rotation+resizing view along the
    batch dimension (2x batch), relying on its mmcv 1.x pipeline to also
    duplicate ground-truth/clean-feature bookkeeping for the doubled batch.
    We don't replicate that plumbing here — `osfd_loss` compares
    same-shaped clean/adversarial batches — so this applies both transforms
    sequentially to one view instead of producing two. If doubled-view
    ensembling turns out to matter, revisit by also duplicating
    `clean_img`/`gt_bboxes` before calling this function.
    """
    current = imgs
    if random.random() < prob:
        current = random_axis_rotation(current, gt_bboxes, theta=theta, l_s=l_s)
    if random.random() < prob:
        current = adaptive_random_resizing(current, gt_bboxes, rho=rho, s_max=s_max)
    return gaussian_blur(current, sigma=sigma, pixel_max=pixel_max)
