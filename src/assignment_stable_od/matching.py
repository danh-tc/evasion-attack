"""Prediction matching metrics used by the early pruning pilot."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torchvision.ops import box_iou


@dataclass(frozen=True)
class StabilityMetrics:
    reference_count: int
    candidate_count: int
    matched_count: int
    stable_count: int
    match_rate: float
    stability_rate: float
    mean_matched_iou: float
    mean_score_ratio: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def match_predictions(
    reference: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
    *,
    score_threshold: float = 0.3,
    stability_iou: float = 0.5,
) -> StabilityMetrics:
    """Greedily match same-class predictions and measure correspondence stability."""
    ref_keep = reference["scores"] >= score_threshold
    cand_keep = candidate["scores"] >= score_threshold
    ref = {key: value[ref_keep] for key, value in reference.items()}
    cand = {key: value[cand_keep] for key, value in candidate.items()}

    ref_count = int(ref["scores"].numel())
    cand_count = int(cand["scores"].numel())
    if ref_count == 0:
        return StabilityMetrics(0, cand_count, 0, 0, 1.0, 1.0, 1.0, 1.0)
    if cand_count == 0:
        return StabilityMetrics(ref_count, 0, 0, 0, 0.0, 0.0, 0.0, 0.0)

    ious = box_iou(ref["bboxes"], cand["bboxes"])
    same_class = ref["labels"][:, None] == cand["labels"][None, :]
    ious = ious.masked_fill(~same_class, -1.0)

    pairs: list[tuple[float, int, int]] = []
    for ref_idx, cand_idx in (ious >= 0).nonzero(as_tuple=False).tolist():
        pairs.append((float(ious[ref_idx, cand_idx]), ref_idx, cand_idx))
    pairs.sort(reverse=True)

    used_ref: set[int] = set()
    used_cand: set[int] = set()
    matched_ious: list[float] = []
    score_ratios: list[float] = []
    for iou, ref_idx, cand_idx in pairs:
        if ref_idx in used_ref or cand_idx in used_cand:
            continue
        used_ref.add(ref_idx)
        used_cand.add(cand_idx)
        matched_ious.append(iou)
        denominator = float(ref["scores"][ref_idx].clamp_min(1e-12))
        score_ratios.append(float(cand["scores"][cand_idx]) / denominator)

    matched_count = len(matched_ious)
    stable_count = sum(iou >= stability_iou for iou in matched_ious)
    return StabilityMetrics(
        reference_count=ref_count,
        candidate_count=cand_count,
        matched_count=matched_count,
        stable_count=stable_count,
        match_rate=matched_count / ref_count,
        stability_rate=stable_count / ref_count,
        mean_matched_iou=sum(matched_ious) / matched_count if matched_count else 0.0,
        mean_score_ratio=sum(score_ratios) / matched_count if matched_count else 0.0,
    )

