"""OSFD multi-stage backbone feature loss, ported to mmdet 3.x.

Ported from reference-source/OSFD/attack/ours/OSFD.py (loss) and
attack/Attack.py (stage extraction). The original paper's loss (Eq. 2):

    L = sum_i (1/N_i) * sum_j (F_i,j(T(x_adv)) - k * F_i,j(x))^2

i.e. a per-stage mean-squared-error between the adversarial (augmented)
features and k-amplified clean features, summed across backbone stages.
`k >= 1` suppresses significant (object) features and amplifies vicinal
(boundary) features in expectation — see paper Section "Explanation of
OSFD" / Figure 3.

Unlike the vendored mmdet 2.x code, mmdet 3.x's ResNet backbone is a plain
nn.Module: `model.backbone(x)` directly returns a tuple of per-stage feature
maps, no neck/head involved, so no mmdet Runner/pipeline machinery is needed
here — just call the backbone as a function.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class OSFDLossOutput:
    total: torch.Tensor
    per_stage: list = field(default_factory=list)  # list[torch.Tensor], one per backbone stage


def osfd_loss(
    backbone: nn.Module,
    clean_img: torch.Tensor,
    adv_img: torch.Tensor,
    k: float = 3.0,
    stage_weights: list[float] | None = None,
    no_grad_clean: bool = True,
) -> OSFDLossOutput:
    """Compute the OSFD feature-distortion loss between clean and adversarial images.

    Args:
        backbone: e.g. `model.backbone` of an mmdet detector (ResNet, Swin, ...).
            Must return a tuple/list of per-stage feature tensors.
        clean_img: preprocessed (normalized, padded) clean image batch, NCHW.
        adv_img: preprocessed adversarial image batch (already passed through
            the RRB augmentation T(.) by the caller), NCHW.
        k: benign-feature amplification factor (paper default k=3.0).
        stage_weights: optional per-stage weight w_i (method F / stage
            weighting). Defaults to uniform (1.0 each), matching methods A-E.
        no_grad_clean: the clean branch is only a reference target — we never
            need gradients w.r.t. backbone params, only w.r.t. the
            perturbation flowing through the adversarial branch. Wrapping the
            clean forward pass in `torch.no_grad()` (default) roughly halves
            attack memory/compute versus tracking both branches.

    Returns:
        OSFDLossOutput with `.total` (scalar, to maximize via gradient ascent
        on the perturbation) and `.per_stage` (list of per-stage scalar
        losses, needed for Stage 4's temporal stage-weighting diagnostics).
    """
    if no_grad_clean:
        with torch.no_grad():
            feats_clean = backbone(clean_img)
    else:
        feats_clean = backbone(clean_img)
    feats_adv = backbone(adv_img)
    if len(feats_clean) != len(feats_adv):
        raise ValueError(
            f"Backbone returned {len(feats_clean)} clean stages but "
            f"{len(feats_adv)} adversarial stages — RRB augmentation must not "
            f"change spatial resolution in a way that changes stage count."
        )

    n_stages = len(feats_clean)
    if stage_weights is None:
        stage_weights = [1.0] * n_stages
    if len(stage_weights) != n_stages:
        raise ValueError(f"stage_weights has {len(stage_weights)} entries, backbone has {n_stages} stages")

    per_stage = []
    for f_clean, f_adv in zip(feats_clean, feats_adv):
        # F.mse_loss(..., reduction="mean") == (1/N_i) * sum_j (.)^2, matching Eq. 2.
        per_stage.append(F.mse_loss(f_adv, k * f_clean, reduction="mean"))

    total = sum(w * s for w, s in zip(stage_weights, per_stage))
    return OSFDLossOutput(total=total, per_stage=per_stage)
