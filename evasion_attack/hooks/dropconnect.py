"""Random parameter pruning (DropConnect-style) masking hooks.

Ported from reference-source/RaPA/core/attacker/DropConnect.py. RaPA applies
Bernoulli masks to Linear + Normalization layer weight/bias. Our attack loss
(OSFD) only runs `model.backbone(x)`, and a ResNet-50 backbone has no
nn.Linear layers, so for method A/B/C the effective target set is just
BatchNorm2d affine params (weight=gamma, bias=beta). The module is written
generally (type_list, name filter) so later stages (D/E/F) can widen the
scope to structured/importance-guided masks.

IMPORTANT: masks are NOT resampled on every forward call. The OSFD loss
requires the benign branch F_i(x; M) and the adversarial branch
F_i(T(x+delta); M) to see the *same* mask M within one loss sample (see
PLAN_EXPERIMENTS.md, "Hard constraint"). Call `resample()` explicitly once
per iteration (or once per S-sample when S>1), then run both forward passes;
the hooks just re-apply whatever mask is currently stored.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def is_maskable_layer(module: nn.Module, type_list=("Normalization", "Linear")) -> bool:
    if "Linear" in type_list and isinstance(module, nn.Linear):
        return True
    if "Normalization" in type_list and isinstance(module, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
        return True
    return False


def _apply_mask_pre_hook(module: nn.Module, _inputs):
    if not hasattr(module, "_dc_current_weight_mask"):
        raise RuntimeError(
            "DropConnect hook fired before resample() was called — "
            "call masker.resample() at the start of each iteration/sample."
        )
    module.weight.data = module._dc_original_weight * module._dc_current_weight_mask
    if module.bias is not None:
        module.bias.data = module._dc_original_bias * module._dc_current_bias_mask


def _restore_weight_post_hook(module: nn.Module, _inputs, _output):
    module.weight.data = module._dc_original_weight
    if module.bias is not None:
        module.bias.data = module._dc_original_bias


class DropConnectMasker:
    """Attaches/detaches Bernoulli DropConnect hooks on a scoped submodule.

    Usage:
        masker = DropConnectMasker(model.backbone, drop_prob=0.05)
        masker.attach()
        for t in range(n_iters):
            masker.resample()                 # fresh mask M_t
            feats_clean = backbone(clean_img)  # sees M_t
            feats_adv = backbone(adv_img)      # sees the SAME M_t
            ...
        masker.detach()
    """

    def __init__(self, scope: nn.Module, drop_prob: float, type_list=("Normalization",)):
        self.scope = scope
        self.drop_prob = drop_prob
        self.type_list = type_list
        self._handles: list = []
        self._modules: list[nn.Module] = []

    def attach(self) -> int:
        self.detach()
        for _name, module in self.scope.named_modules():
            if is_maskable_layer(module, self.type_list):
                module._dc_original_weight = module.weight.data.clone()
                if module.bias is not None:
                    module._dc_original_bias = module.bias.data.clone()
                h1 = module.register_forward_pre_hook(_apply_mask_pre_hook)
                h2 = module.register_forward_hook(_restore_weight_post_hook)
                self._handles.extend([h1, h2])
                self._modules.append(module)
        return len(self._modules)

    def resample(self) -> None:
        """Draw a fresh Bernoulli mask for every maskable layer in scope."""
        for module in self._modules:
            module._dc_current_weight_mask = torch.bernoulli(
                (1.0 - self.drop_prob) * torch.ones_like(module.weight)
            )
            if module.bias is not None:
                module._dc_current_bias_mask = torch.bernoulli(
                    (1.0 - self.drop_prob) * torch.ones_like(module.bias)
                )

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        for module in self._modules:
            for attr in ("_dc_original_weight", "_dc_original_bias",
                         "_dc_current_weight_mask", "_dc_current_bias_mask"):
                if hasattr(module, attr):
                    delattr(module, attr)
        self._modules = []

    @property
    def num_layers(self) -> int:
        return len(self._modules)

    def __enter__(self):
        self.attach()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.detach()
