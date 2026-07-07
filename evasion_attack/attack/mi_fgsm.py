"""MI-FGSM attack loop with pluggable random-parameter masking.

Implements the update rule from PLAN_EXPERIMENTS.md / the RaPA paper (Eq. in
Algorithm 1, generalized with momentum a la MI-FGSM):

    g_t = (1/S) * sum_{s=1..S} grad_delta L_OSFD(delta_t; M_{t,s}, T_{t,s})
    v_t = mu * v_{t-1} + g_t / ||g_t||_1
    delta_{t+1} = clip_eps(delta_t + alpha * sign(v_t))

S=1 (method B) reduces to a single masked forward/backward per iteration,
relying on momentum to act as a temporal ensemble across iterations instead
of an explicit multi-mask average within one iteration (S=5, method C) —
this is exactly the hypothesis Stage 1 tests, not an assumed truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from evasion_attack.attack.methods import MethodConfig
from evasion_attack.augment.rrb import rrb_augment
from evasion_attack.hooks.dropconnect import DropConnectMasker
from evasion_attack.losses.osfd import osfd_loss
from evasion_attack.utils.preprocess import get_preprocess_config, prepare_for_backbone


@dataclass
class AttackConfig:
    epsilon: float = 8.0  # L_inf budget, pixel units [0, 255]
    alpha: float = 2.0  # step size, pixel units
    n_iters: int = 40
    momentum: float = 0.9
    seed: int | None = None


@dataclass
class AttackResult:
    delta: torch.Tensor
    loss_history: list = field(default_factory=list)  # one float per iteration (S-averaged)
    n_masked_layers: int = 0


def run_mi_fgsm_attack(
    model,
    img: torch.Tensor,
    gt_bboxes: list[torch.Tensor],
    method: MethodConfig,
    attack_cfg: AttackConfig,
) -> AttackResult:
    """Run the OA-TRaPA attack against one image batch.

    Args:
        model: an mmdet detector (from `init_detector`), in eval mode, with
            all params `requires_grad_(False)` EXCEPT we never need backbone
            param grads at all — only grad w.r.t. `delta`. Caller does not
            need to freeze params; `torch.autograd.grad` below only asks for
            the gradient w.r.t. `delta`, so backbone params never accumulate
            `.grad` regardless of their `requires_grad` flag.
        img: clean image batch, float tensor in [0, 255], NCHW, RGB order,
            on the same device as `model`.
        gt_bboxes: list of per-image [N_i, 4] xyxy tensors (pixel coords, same
            scale as `img`), used by RRB to bias rotation/resize toward
            objects.
        method: one of METHOD_REGISTRY's configs (A/B/C).
        attack_cfg: epsilon/alpha/iters/momentum.
    """
    if attack_cfg.seed is not None:
        torch.manual_seed(attack_cfg.seed)

    device = img.device
    backbone = model.backbone
    preprocess_cfg = get_preprocess_config(model, device)

    masker = None
    n_masked_layers = 0
    if method.drop_prob > 0.0:
        masker = DropConnectMasker(backbone, method.drop_prob, type_list=method.mask_type_list)
        n_masked_layers = masker.attach()
        if n_masked_layers == 0:
            raise RuntimeError(
                f"Method '{method.method_id}' requested masking (drop_prob="
                f"{method.drop_prob}) but found 0 maskable layers of type "
                f"{method.mask_type_list} in the backbone — check the "
                f"surrogate architecture."
            )

    delta = torch.zeros_like(img)
    velocity = torch.zeros_like(img)
    loss_history = []

    rrb_kwargs = dict(
        theta=method.rrb.theta, l_s=method.rrb.l_s, rho=method.rrb.rho,
        s_max=method.rrb.s_max, sigma=method.rrb.sigma, prob=method.rrb.prob,
    )

    try:
        for _t in range(attack_cfg.n_iters):
            grad_accum = torch.zeros_like(img)
            iter_loss = 0.0
            for _s in range(method.s_masks):
                if masker is not None:
                    masker.resample()

                delta_var = delta.clone().detach().requires_grad_(True)
                adv_img = torch.clamp(img + delta_var, 0.0, 255.0)
                aug_img = rrb_augment(adv_img, gt_bboxes, **rrb_kwargs)

                clean_input = prepare_for_backbone(img, preprocess_cfg)
                adv_input = prepare_for_backbone(aug_img, preprocess_cfg)

                out = osfd_loss(backbone, clean_input, adv_input, k=method.k)
                (grad,) = torch.autograd.grad(out.total, delta_var)

                grad_accum = grad_accum + grad
                iter_loss += float(out.total.detach())

            grad_avg = grad_accum / method.s_masks
            l1_norm = grad_avg.abs().sum(dim=tuple(range(1, grad_avg.dim())), keepdim=True) + 1e-12
            velocity = attack_cfg.momentum * velocity + grad_avg / l1_norm

            delta = delta + attack_cfg.alpha * velocity.sign()
            delta = torch.clamp(delta, -attack_cfg.epsilon, attack_cfg.epsilon)
            delta = torch.clamp(img + delta, 0.0, 255.0) - img
            delta = delta.detach()

            loss_history.append(iter_loss / method.s_masks)
    finally:
        if masker is not None:
            masker.detach()

    return AttackResult(delta=delta, loss_history=loss_history, n_masked_layers=n_masked_layers)
