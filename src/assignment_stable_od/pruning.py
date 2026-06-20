"""Temporary random weight pruning for inference experiments."""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator

import torch
from torch import nn
from torch.nn.utils import prune


def eligible_modules(model: nn.Module, scope: str = "all") -> list[tuple[str, nn.Module]]:
    modules: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if scope != "all" and not name.startswith(scope):
            continue
        if isinstance(module, (nn.Conv2d, nn.Linear)) and module.weight.numel() > 0:
            modules.append((name, module))
    return modules


@contextmanager
def temporary_random_pruning(
    model: nn.Module,
    amount: float,
    *,
    scope: str = "all",
    seed: int = 0,
) -> Iterator[int]:
    """Apply random unstructured masks, then restore the exact original weights."""
    if not 0.0 <= amount < 1.0:
        raise ValueError("amount must be in [0, 1)")

    modules = eligible_modules(model, scope)
    if not modules:
        raise ValueError(f"no eligible Conv2d/Linear modules found for scope={scope!r}")

    devices = sorted(
        {module.weight.device.index for _, module in modules if module.weight.is_cuda}
    )
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for _, module in modules:
            prune.random_unstructured(module, name="weight", amount=amount)

    try:
        yield len(modules)
    finally:
        with torch.no_grad():
            for _, module in modules:
                original = module.weight_orig.detach().clone()
                prune.remove(module, "weight")
                module.weight.copy_(original)

