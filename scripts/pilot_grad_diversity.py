#!/usr/bin/env python3
"""Gradient diversity pilot: measure cosine similarity of input gradients
across pruned model variants vs the original model.

For each image we compute dL/dx (gradient w.r.t. the input image) under:
  - the clean model (reference)
  - S pruned variants at various rates and scopes

We then report:
  - cos_sim_vs_clean:  similarity of pruned gradient to clean gradient
  - cos_sim_pairwise:  mean similarity between pairs of pruned gradients

The key question: do stable masks (high stability_rate) produce gradients
that are diverse from each other yet aligned with the clean gradient?
If yes, those masks are useful for transfer. If cos_sim_vs_clean is near 0
or negative, the gradient is pointing in a useless direction.

Usage:
    python scripts/pilot_grad_diversity.py
    python scripts/pilot_grad_diversity.py --scope rpn_head --rates 0.1 0.3 0.5 0.7
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from mmdet.apis import init_detector
from mmdet.utils import register_all_modules

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
    p.add_argument("--n-images", type=int, default=10)
    p.add_argument("--scopes", nargs="+",
                   default=["all", "backbone", "neck", "rpn_head", "roi_head"])
    p.add_argument("--rates", type=float, nargs="+",
                   default=[0.05, 0.1, 0.2, 0.3, 0.5])
    p.add_argument("--n-masks", type=int, default=3,
                   help="Number of random masks (seeds) per rate")
    p.add_argument("--out-dir", type=Path, default=Path("results/pilot"))
    return p.parse_args()


# ── Loss: object-hiding (suppress all predictions) ───────────────────────────

def object_hiding_loss(model, img_tensor: torch.Tensor) -> torch.Tensor:
    """Simple surrogate loss: maximise the sum of all RPN objectness logits.

    This is a cheap approximation — we want a loss whose gradient tells us
    'how to suppress detections'.  For feasibility we don't need a perfect
    loss, just one that produces a meaningful gradient wrt the input image.
    """
    # Forward through backbone + neck only (cheap, avoids RoI align complexity)
    feats = model.backbone(img_tensor)
    feats = model.neck(feats)
    # RPN cls scores: list of tensors [B, num_anchors, H, W]
    rpn_cls_scores, _ = model.rpn_head.forward_single(feats[0]) \
        if hasattr(model.rpn_head, 'forward_single') else (None, None)

    if rpn_cls_scores is None:
        # Fallback: sum all feature map activations
        loss = sum(f.abs().mean() for f in feats)
    else:
        # Maximise objectness → minimise negative objectness
        loss = -torch.sigmoid(rpn_cls_scores).mean()
    return loss


def compute_input_gradient(
    model, img_path: Path, device: str = "cuda:0"
) -> torch.Tensor:
    """Return the normalised gradient of the loss w.r.t. the input image."""
    from mmcv.transforms import Compose
    from mmdet.utils import register_all_modules as _reg

    # Load and preprocess with mmdet pipeline
    import mmcv
    img_bgr = mmcv.imread(str(img_path))

    # Simple resize + normalise matching the model's pipeline
    # mean/std from COCO-trained Faster R-CNN
    mean = torch.tensor([123.675, 116.28, 103.53], device=device).view(3, 1, 1)
    std  = torch.tensor([58.395,  57.12,  57.375], device=device).view(3, 1, 1)

    # Resize to 800×max_side keeping aspect ratio (standard MMDet setting)
    h, w = img_bgr.shape[:2]
    scale = 800 / min(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    # Cap long side at 1333
    if max(new_h, new_w) > 1333:
        scale2 = 1333 / max(new_h, new_w)
        new_h, new_w = int(new_h * scale2), int(new_w * scale2)

    import cv2
    img_resized = cv2.resize(img_bgr, (new_w, new_h))
    img_rgb = img_resized[:, :, ::-1].copy()

    img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float().to(device)
    img_t = (img_t - mean) / std
    img_t = img_t.unsqueeze(0).requires_grad_(True)

    loss = object_hiding_loss(model, img_t)
    loss.backward()

    grad = img_t.grad.detach().flatten()
    grad = F.normalize(grad, dim=0)
    return grad


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest.read_text())
    image_ids = manifest["image_ids"][: args.n_images]
    ann = json.loads((args.coco_root / "annotations/instances_val2017.json").read_text())
    id2file = {img["id"]: img["file_name"] for img in ann["images"]}
    image_paths = [args.coco_root / "val2017" / id2file[i] for i in image_ids]

    register_all_modules()
    model = init_detector(str(args.config), str(args.checkpoint), device="cuda:0")
    model.eval()

    rows: list[dict] = []
    seeds = list(range(args.n_masks))

    for scope in args.scopes:
        print(f"\n=== scope: {scope} ===")
        for img_path in image_paths:
            # Clean gradient (reference)
            model.zero_grad()
            g_clean = compute_input_gradient(model, img_path)

            for rate in args.rates:
                grads_pruned: list[torch.Tensor] = []
                for seed in seeds:
                    model.zero_grad()
                    if rate == 0.0:
                        g = g_clean
                    else:
                        with temporary_random_pruning(model, rate, scope=scope, seed=seed):
                            try:
                                g = compute_input_gradient(model, img_path)
                            except Exception:
                                g = torch.zeros_like(g_clean)
                    grads_pruned.append(g)

                # cos sim: each pruned mask vs clean
                sims_vs_clean = [cosine_sim(g, g_clean) for g in grads_pruned]
                # cos sim: between pairs of pruned masks
                if len(grads_pruned) >= 2:
                    sims_pairwise = [
                        cosine_sim(a, b)
                        for a, b in combinations(grads_pruned, 2)
                    ]
                else:
                    sims_pairwise = [1.0]

                row = {
                    "scope": scope,
                    "image": img_path.name,
                    "pruning_rate": rate,
                    "cos_sim_vs_clean_mean": float(np.mean(sims_vs_clean)),
                    "cos_sim_vs_clean_std": float(np.std(sims_vs_clean)),
                    "cos_sim_pairwise_mean": float(np.mean(sims_pairwise)),
                    "cos_sim_pairwise_std": float(np.std(sims_pairwise)),
                }
                rows.append(row)

            print(f"  {img_path.name} done")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = args.out_dir / "grad_diversity.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {csv_path}")

    # ── Print summary ─────────────────────────────────────────────────────────
    import pandas as pd
    df = pd.DataFrame(rows)
    for scope in args.scopes:
        sub = df[df["scope"] == scope]
        agg = sub.groupby("pruning_rate")[
            ["cos_sim_vs_clean_mean", "cos_sim_pairwise_mean"]
        ].mean().round(3)
        print(f"\nscope={scope}")
        print(agg.to_string())

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Gradient Diversity Pilot — Faster R-CNN R50-FPN", fontsize=12)
    colors = plt.cm.tab10(np.linspace(0, 1, len(args.scopes)))

    for ax, col, title in zip(
        axes,
        ["cos_sim_vs_clean_mean", "cos_sim_pairwise_mean"],
        ["Cosine Sim: pruned vs clean\n(1=same direction, 0=orthogonal)",
         "Cosine Sim: between masks\n(lower = more diverse)"],
    ):
        for scope, color in zip(args.scopes, colors):
            sub = df[df["scope"] == scope]
            agg = sub.groupby("pruning_rate")[col].mean()
            ax.plot(agg.index, agg.values, "o-", label=scope,
                    color=color, linewidth=2, markersize=6)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Pruning Rate")
        ax.set_ylabel(title, fontsize=9)
        ax.set_ylim(-0.2, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = args.out_dir / "grad_diversity.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot  : {plot_path}")


if __name__ == "__main__":
    main()
