#!/usr/bin/env python3
"""Attack any image (outside COCO) and visualize clean vs adversarial.

Usage:
    python scripts/attack_any_image.py --image /path/to/image.jpg
    python scripts/attack_any_image.py --image /path/to/image.jpg --out result.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from mmdet.apis import inference_detector, init_detector
from mmdet.utils import register_all_modules

from assignment_stable_od.attack import load_image_bgr, pgd_attack

DEVICE       = "cuda:0"
SCORE_THRESH = 0.3
FONT         = cv2.FONT_HERSHEY_SIMPLEX


def draw_boxes(img, preds, classes, colour):
    out = img.copy()
    boxes  = preds.bboxes.cpu().numpy()
    labels = preds.labels.cpu().numpy()
    scores = preds.scores.cpu().numpy()
    keep   = scores >= SCORE_THRESH
    for box, lbl, sc in zip(boxes[keep], labels[keep], scores[keep]):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        text = f"{classes[lbl]} {sc:.2f}"
        (tw, th), _ = cv2.getTextSize(text, FONT, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw, y1), colour, -1)
        cv2.putText(out, text, (x1, y1 - 3), FONT, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return out


def add_label(img, text, colour):
    bar = np.zeros((28, img.shape[1], 3), dtype=np.uint8)
    bar[:] = colour
    cv2.putText(bar, text, (6, 19), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def adversarial_at_orig_scale(adv_r, bgr_r, orig):
    h_o, w_o = orig.shape[:2]
    h_r, w_r = bgr_r.shape[:2]
    delta = adv_r.astype(np.float32) - bgr_r.astype(np.float32)
    if (h_o, w_o) != (h_r, w_r):
        delta = cv2.resize(delta, (w_o, h_o), interpolation=cv2.INTER_LINEAR)
    return np.clip(orig.astype(np.float32) + delta, 0, 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image",     type=Path, required=True)
    p.add_argument("--out",       type=Path, default=None)
    p.add_argument("--epsilon",   type=float, default=8.0)
    p.add_argument("--n-iters",   type=int,   default=40)
    p.add_argument("--step-size", type=float, default=2.0)
    args = p.parse_args()

    if not args.image.exists():
        print(f"ERROR: {args.image} not found"); return

    out_path = args.out or args.image.parent / f"{args.image.stem}_attacked.jpg"

    register_all_modules()

    print("Loading Faster R-CNN surrogate...")
    surrogate = init_detector(
        "checkpoints/faster-rcnn_r50_fpn_1x_coco.py",
        "checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth",
        device=DEVICE)
    surrogate.eval()
    classes = list(surrogate.dataset_meta["classes"])

    print(f"Attacking: {args.image.name}  ε={args.epsilon}px  {args.n_iters} iters")

    img_orig = cv2.imread(str(args.image))
    img_bgr  = load_image_bgr(args.image)

    # ── Attack ────────────────────────────────────────────────────────────
    img_adv_r = pgd_attack(
        surrogate, img_bgr,
        epsilon_px=args.epsilon, n_iters=args.n_iters,
        step_size_px=args.step_size, n_masks=2,
        pruning_scope="backbone", pruning_rate=0.05,
        device=DEVICE,
    )
    img_adv = adversarial_at_orig_scale(img_adv_r, img_bgr, img_orig)

    # ── Inference ─────────────────────────────────────────────────────────
    p_clean = inference_detector(surrogate, img_orig).pred_instances
    p_adv   = inference_detector(surrogate, img_adv ).pred_instances
    nc = int((p_clean.scores >= SCORE_THRESH).sum())
    na = int((p_adv.scores   >= SCORE_THRESH).sum())
    print(f"Detections: {nc} → {na}  (removed {nc - na})")

    # ── Visualize ─────────────────────────────────────────────────────────
    W = min(img_orig.shape[1], 800)
    H = int(img_orig.shape[0] * W / img_orig.shape[1])

    def panel(img, preds, colour, title):
        d = draw_boxes(img, preds, classes, colour)
        d = cv2.resize(d, (W, H))
        return add_label(d, title, colour)

    left  = panel(img_orig, p_clean, (30, 160, 30), f"CLEAN — {nc} detections")
    right = panel(img_adv,  p_adv,   (0,   0, 180), f"ADVERSARIAL (ε={args.epsilon}px) — {na} detections")

    # perturbation (amplified ×5 for visibility)
    delta_amp = np.clip(
        128 + 5 * (img_adv.astype(np.int32) - img_orig.astype(np.int32)), 0, 255
    ).astype(np.uint8)
    delta_amp = cv2.resize(delta_amp, (W, H))
    delta_panel = add_label(delta_amp, "Perturbation (×5 amplified)", (60, 60, 60))

    # pad delta to same height, place beside a blank
    blank = np.zeros_like(delta_panel)
    row2  = np.hstack([delta_panel, blank])
    grid  = np.vstack([np.hstack([left, right]), row2])

    cv2.imwrite(str(out_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 93])
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
