#!/usr/bin/env python3
"""Pruning-stability pilot: sweep pruning rate × seeds × images, plot signal.

This is the GO/NO-GO experiment.  We check whether random weight pruning
corrupts the correspondence between clean-model predictions and pruned-model
predictions.  If stability_rate drops sharply with pruning rate, the
assignment-corrupting-diversity failure mode is real and worth studying further.

Usage:
    python scripts/run_pilot.py                          # defaults
    python scripts/run_pilot.py --n-images 20 --rates 0 0.1 0.2 0.3 0.4 0.5
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from mmdet.apis import inference_detector, init_detector
from mmdet.utils import register_all_modules

from assignment_stable_od.matching import match_predictions
from assignment_stable_od.pruning import temporary_random_pruning


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path,
                   default=Path("data/manifests/dev_300.json"))
    p.add_argument("--coco-root", type=Path, default=Path("data/coco"))
    p.add_argument("--config", type=Path,
                   default=Path("checkpoints/faster-rcnn_r50_fpn_1x_coco.py"))
    p.add_argument("--checkpoint", type=Path,
                   default=Path("checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"))
    p.add_argument("--n-images", type=int, default=15,
                   help="Number of images to sample from manifest (first N)")
    p.add_argument("--rates", type=float, nargs="+",
                   default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--scope", default="all",
                   help="Module scope: all | backbone | neck | rpn_head | roi_head")
    p.add_argument("--score-threshold", type=float, default=0.3)
    p.add_argument("--stability-iou", type=float, default=0.5)
    p.add_argument("--out-dir", type=Path, default=Path("results/pilot"))
    return p.parse_args()


def tensors_from_result(result) -> dict[str, torch.Tensor]:
    inst = result.pred_instances
    return {
        "bboxes": inst.bboxes.detach(),
        "labels": inst.labels.detach(),
        "scores": inst.scores.detach(),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load image list ───────────────────────────────────────────────────────
    manifest = json.loads(args.manifest.read_text())
    image_ids = manifest["image_ids"][: args.n_images]

    ann_path = args.coco_root / "annotations/instances_val2017.json"
    ann = json.loads(ann_path.read_text())
    id2file = {img["id"]: img["file_name"] for img in ann["images"]}
    image_paths = [args.coco_root / "val2017" / id2file[i] for i in image_ids]

    print(f"Images  : {len(image_paths)}")
    print(f"Rates   : {args.rates}")
    print(f"Seeds   : {args.seeds}")
    print(f"Scope   : {args.scope}")
    print()

    # ── Load model ────────────────────────────────────────────────────────────
    register_all_modules()
    model = init_detector(str(args.config), str(args.checkpoint), device="cuda:0")

    # ── Run sweep ─────────────────────────────────────────────────────────────
    rows: list[dict] = []
    total = len(image_paths) * len(args.rates) * len(args.seeds)
    done = 0

    for img_path in image_paths:
        # Reference: clean model predictions
        reference = tensors_from_result(inference_detector(model, str(img_path)))
        n_ref = int((reference["scores"] >= args.score_threshold).sum())

        for rate in args.rates:
            for seed in args.seeds:
                if rate == 0.0:
                    # No pruning: compare model with itself → perfect stability
                    candidate = reference
                    n_modules = 0
                else:
                    with temporary_random_pruning(
                        model, rate, scope=args.scope, seed=seed
                    ) as n_modules:
                        candidate = tensors_from_result(
                            inference_detector(model, str(img_path))
                        )

                metrics = match_predictions(
                    reference,
                    candidate,
                    score_threshold=args.score_threshold,
                    stability_iou=args.stability_iou,
                )
                row = {
                    "image": img_path.name,
                    "scope": args.scope,
                    "pruning_rate": rate,
                    "seed": seed,
                    "n_ref_preds": n_ref,
                    "n_modules": n_modules,
                    **metrics.to_dict(),
                }
                rows.append(row)
                done += 1

        pct = 100 * done / total
        print(f"  {img_path.name}  [{done}/{total}  {pct:.0f}%]")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = args.out_dir / f"stability_{args.scope}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {csv_path}")

    # ── Aggregate and print table ─────────────────────────────────────────────
    import pandas as pd
    df = pd.DataFrame(rows)
    agg = (
        df.groupby("pruning_rate")[
            ["stability_rate", "match_rate", "mean_matched_iou", "mean_score_ratio"]
        ]
        .agg(["mean", "std"])
        .round(3)
    )
    print("\n── Mean ± Std across images × seeds ─────────────────────────────")
    print(agg.to_string())

    # ── Plot ──────────────────────────────────────────────────────────────────
    rates = sorted(df["pruning_rate"].unique())
    metrics_to_plot = {
        "stability_rate": "Stability Rate\n(fraction of clean preds still matched at IoU≥0.5)",
        "match_rate": "Match Rate\n(fraction of clean preds with any same-class match)",
        "mean_matched_iou": "Mean Matched IoU",
        "mean_score_ratio": "Mean Score Ratio\n(pruned score / clean score)",
    }

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(
        f"Pruning Stability Pilot — Faster R-CNN R50-FPN\n"
        f"scope={args.scope}  n_images={len(image_paths)}  seeds={args.seeds}",
        fontsize=11,
    )

    for ax, (col, ylabel) in zip(axes.flat, metrics_to_plot.items()):
        means, stds = [], []
        for r in rates:
            vals = df[df["pruning_rate"] == r][col].values
            means.append(vals.mean())
            stds.append(vals.std())
        means, stds = np.array(means), np.array(stds)
        ax.plot(rates, means, "o-", color="steelblue", linewidth=2, markersize=6)
        ax.fill_between(rates, means - stds, means + stds, alpha=0.2, color="steelblue")
        ax.set_xlabel("Pruning Rate")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(rates)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = args.out_dir / f"stability_{args.scope}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot  : {plot_path}")


if __name__ == "__main__":
    main()
