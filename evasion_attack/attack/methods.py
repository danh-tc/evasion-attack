"""Method registry for the staged experiment plan (PLAN_EXPERIMENTS.md).

Only A, B, C are implemented (Stage 1). D, E, F are placeholders — their
substance (balanced cyclic group masking, feature-guided importance bias,
temporal stage weighting) is deliberately not built yet, per the plan's
gating rule: don't invest in Stage 2-4 machinery until Stage 1 confirms the
core hypothesis (masking helps cross-family transfer, and S=1 stands in for
S=5). Implementing them speculatively now would be exactly the kind of
premature abstraction the plan is trying to avoid.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RRBConfig:
    theta: float = 7.0
    l_s: int = 10
    rho: float = 0.8
    s_max: float = 1.10
    sigma: float = 6.0
    prob: float = 1.0


@dataclass
class MethodConfig:
    method_id: str
    description: str
    drop_prob: float  # 0.0 == no masking (method A)
    s_masks: int  # number of masks sampled per iteration
    k: float = 3.0  # OSFD amplification factor
    mask_type_list: tuple[str, ...] = ("Normalization", "Linear")
    rrb: RRBConfig = field(default_factory=RRBConfig)


# drop_prob=0.05 chosen as RaPA's own ResNet-50 default (paper Sec 4.1) as a
# starting point; PLAN_EXPERIMENTS Stage 1 sweeps this if the probe is
# ambiguous, but not before the first pass/fail check.
METHOD_REGISTRY: dict[str, MethodConfig] = {
    "A": MethodConfig(
        method_id="A",
        description="OSFD baseline, no masking (control)",
        drop_prob=0.0,
        s_masks=1,
    ),
    "B": MethodConfig(
        method_id="B",
        description="OSFD + Bernoulli DropConnect on BN affine, S=1 (momentum carries history)",
        drop_prob=0.05,
        s_masks=1,
    ),
    "C": MethodConfig(
        method_id="C",
        description="OSFD + Bernoulli DropConnect on BN affine, S=5 (explicit ensemble, RaPA-style)",
        drop_prob=0.05,
        s_masks=5,
    ),
}


def get_method(method_id: str) -> MethodConfig:
    try:
        return METHOD_REGISTRY[method_id]
    except KeyError:
        implemented = ", ".join(sorted(METHOD_REGISTRY))
        raise KeyError(
            f"Method '{method_id}' is not implemented yet. Implemented: {implemented}. "
            f"D/E/F are Stage 2-4 and gated behind Stage 1 results — see PLAN_EXPERIMENTS.md."
        ) from None
