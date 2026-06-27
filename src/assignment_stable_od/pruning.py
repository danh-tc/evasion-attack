"""Temporary random weight pruning for attack diversity (RaPA-style).

Default targets Normalization + Linear layers, matching the RaPA paper
(Su et al. CVPR 2026): BatchNorm/LayerNorm parameters are most effective
for improving adversarial transferability over Conv layers.
"""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator

import torch
from torch import nn
from torch.nn.utils import prune


_TYPE_MAP: dict[str, tuple[type, ...]] = {
    "Normalization": (nn.BatchNorm2d, nn.LayerNorm),
    "Linear":        (nn.Linear,),
    "Conv":          (nn.Conv2d,),
}


def eligible_modules(
    model: nn.Module,
    scope: str = "all",
    type_list: list[str] | None = None,
) -> list[tuple[str, nn.Module]]:
    """Return (name, module) pairs eligible for random unstructured pruning.

    Args:
        scope:     Module name prefix, e.g. "backbone". "all" disables filtering.
        type_list: Layer types to include. Subset of ["Normalization", "Linear", "Conv"].
                   Default ["Normalization", "Linear"] matches RaPA paper recommendation.
    """
    if type_list is None:
        type_list = ["Normalization", "Linear"]

    unknown = set(type_list) - set(_TYPE_MAP)
    if unknown:
        raise ValueError(f"Unknown type_list entries: {unknown}. Valid: {list(_TYPE_MAP)}")

    allowed = tuple(cls for key in type_list for cls in _TYPE_MAP[key])

    out: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if scope != "all" and not name.startswith(scope):
            continue
        if (
            isinstance(module, allowed)
            and hasattr(module, "weight")
            and module.weight is not None
            and module.weight.numel() > 0
        ):
            out.append((name, module))
    return out


@contextmanager
def temporary_random_pruning(
    model: nn.Module,
    amount: float,
    *,
    scope: str = "all",
    seed: int = 0,
    type_list: list[str] | None = None,
) -> Iterator[int]:
    """Apply random unstructured masks, then restore exact original weights.

    The context manager ensures loss.backward() executes while masks are active,
    so gradients w.r.t. the input image correctly reflect the pruned model.

    Yields the number of modules that were pruned.
    """
    if not 0.0 <= amount < 1.0:
        raise ValueError(f"amount must be in [0, 1), got {amount}")

    modules = eligible_modules(model, scope, type_list)
    if not modules:
        raise ValueError(
            f"No eligible modules found — scope={scope!r}, type_list={type_list}"
        )

    cuda_devices = sorted(
        {m.weight.device.index for _, m in modules if m.weight.is_cuda}
    )
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        for _, module in modules:
            prune.random_unstructured(module, name="weight", amount=amount)

    try:
        yield len(modules)
    finally:
        with torch.no_grad():
            for _, module in modules:
                orig = module.weight_orig.detach().clone()
                prune.remove(module, "weight")
                module.weight.copy_(orig)
