"""Differentiable image preprocessing matching mmdet's DetDataPreprocessor.

mmdet 3.x models expect input already normalized/padded by
`model.data_preprocessor` before reaching `model.backbone`. That module's
own `forward()` is written for batches of uint8 tensors with no grad
tracking, which doesn't suit an attack that needs gradients w.r.t. pixel
values. Instead we read the same (mean, std, pad_size_divisor) from
`model.cfg.model.data_preprocessor` and replicate the arithmetic ourselves,
directly on a float tensor with `requires_grad=True`, so autograd flows
through cleanly.

ASSUMPTION: images loaded by evasion_attack.data are RGB-ordered (e.g. via
PIL `Image.open(...).convert("RGB")`), and (mean, std) in the mmdet config
are given in RGB order (the standard mmdetection convention — e.g.
mean=[123.675, 116.28, 103.53] — with `bgr_to_rgb=True` telling the
*original* preprocessor to convert cv2's BGR to RGB first). Since our
loader already produces RGB, no channel swap is needed here. If a target
model's config uses `to_rgb=False`/BGR-order mean-std (uncommon in mmdet
3.x model zoo but worth checking per-model), this will silently mismatch —
verify against `scripts/check_env.py` output before trusting numbers.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class PreprocessConfig:
    mean: torch.Tensor  # shape [3, 1, 1]
    std: torch.Tensor  # shape [3, 1, 1]
    pad_size_divisor: int


def get_preprocess_config(model, device) -> PreprocessConfig:
    dp_cfg = model.cfg.model.data_preprocessor
    mean = torch.tensor(dp_cfg.get("mean", [123.675, 116.28, 103.53]), device=device).view(3, 1, 1)
    std = torch.tensor(dp_cfg.get("std", [58.395, 57.12, 57.375]), device=device).view(3, 1, 1)
    pad_size_divisor = int(dp_cfg.get("pad_size_divisor", 1))
    return PreprocessConfig(mean=mean, std=std, pad_size_divisor=pad_size_divisor)


def pad_to_divisor(img: torch.Tensor, divisor: int, pad_val: float = 0.0) -> torch.Tensor:
    if divisor <= 1:
        return img
    h, w = img.shape[-2], img.shape[-1]
    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor
    if pad_h == 0 and pad_w == 0:
        return img
    return F.pad(img, (0, pad_w, 0, pad_h), mode="constant", value=pad_val)


def normalize(img_pixel_space: torch.Tensor, cfg: PreprocessConfig) -> torch.Tensor:
    """img_pixel_space: float tensor in [0, 255], NCHW, RGB channel order."""
    return (img_pixel_space - cfg.mean) / cfg.std


def prepare_for_backbone(img_pixel_space: torch.Tensor, cfg: PreprocessConfig) -> torch.Tensor:
    """Normalize then pad — the exact order `DetDataPreprocessor` uses."""
    normalized = normalize(img_pixel_space, cfg)
    return pad_to_divisor(normalized, cfg.pad_size_divisor)
