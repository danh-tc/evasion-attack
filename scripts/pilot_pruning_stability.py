#!/usr/bin/env python3
"""Small pseudo-label pilot for pruning-induced assignment instability."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from mmdet.apis import inference_detector, init_detector
from mmdet.utils import register_all_modules

from assignment_stable_od.matching import match_predictions
from assignment_stable_od.pruning import temporary_random_pruning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rates", type=float, nargs="+", default=[0.0, 0.1, 0.3, 0.5])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--scope", default="all")
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--stability-iou", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=Path("results/pilot_stability.csv"))
    return parser.parse_args()


def tensors(result) -> dict[str, torch.Tensor]:
    instances = result.pred_instances
    return {
        "bboxes": instances.bboxes.detach(),
        "labels": instances.labels.detach(),
        "scores": instances.scores.detach(),
    }


def main() -> None:
    args = parse_args()
    register_all_modules()
    model = init_detector(str(args.config), str(args.checkpoint), device="cuda:0")
    reference = tensors(inference_detector(model, str(args.image)))
    rows: list[dict[str, int | float | str]] = []

    for rate in args.rates:
        for seed in args.seeds:
            with temporary_random_pruning(model, rate, scope=args.scope, seed=seed) as count:
                candidate = tensors(inference_detector(model, str(args.image)))
            metrics = match_predictions(
                reference,
                candidate,
                score_threshold=args.score_threshold,
                stability_iou=args.stability_iou,
            )
            row = {
                "image": str(args.image),
                "scope": args.scope,
                "pruning_rate": rate,
                "seed": seed,
                "pruned_modules": count,
                **metrics.to_dict(),
            }
            rows.append(row)
            print(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()

