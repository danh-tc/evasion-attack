#!/usr/bin/env python3
"""Visualize clean vs adversarial detections side by side.

Saves comparison images to results/viz/:
  {img_id}_clean.jpg   — clean image with surrogate + target detections
  {img_id}_adv.jpg     — adversarial image with surrogate + target detections
  {img_id}_compare.jpg — 2×2 grid: [sur_clean | sur_adv] / [tgt_clean | tgt_adv]

Usage:
    python scripts/visualize_attack.py --n-images 5
    python scripts/visualize_attack.py --image-ids 885 9483 23899
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from mmdet.apis import inference_detector, init_detector
from mmdet.utils import register_all_modules
from pycocotools.coco import COCO

from assignment_stable_od.attack import load_image_bgr, pgd_attack

DEVICE       = "cuda:0"
SCORE_THRESH = 0.3

# colours per model (BGR)
COLOUR_SUR = (0,   200, 0)    # green  — surrogate
COLOUR_TGT = (0,   120, 255)  # orange — target

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.45
FONT_THICK = 1
BOX_THICK  = 2


def draw_boxes(img_bgr: np.ndarray, preds, classes: list[str],
               colour: tuple, score_thresh: float = SCORE_THRESH) -> np.ndarray:
    out = img_bgr.copy()
    boxes  = preds.bboxes.cpu().numpy()
    labels = preds.labels.cpu().numpy()
    scores = preds.scores.cpu().numpy()

    keep = scores >= score_thresh
    boxes, labels, scores = boxes[keep], labels[keep], scores[keep]

    for box, lbl, sc in zip(boxes, labels, scores):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, BOX_THICK)
        text = f"{classes[lbl]} {sc:.2f}"
        (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICK)
        cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw, y1), colour, -1)
        cv2.putText(out, text, (x1, y1 - 3), FONT, FONT_SCALE,
                    (255, 255, 255), FONT_THICK, cv2.LINE_AA)
    return out


def add_label_bar(img: np.ndarray, text: str, colour: tuple) -> np.ndarray:
    bar = np.zeros((28, img.shape[1], 3), dtype=np.uint8)
    bar[:] = colour
    cv2.putText(bar, text, (6, 19), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def resize_to(img: np.ndarray, h: int, w: int) -> np.ndarray:
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


def adversarial_at_orig_scale(img_adv_r, img_bgr_r, img_orig):
    h_o, w_o = img_orig.shape[:2]
    h_r, w_r = img_bgr_r.shape[:2]
    delta = img_adv_r.astype(np.float32) - img_bgr_r.astype(np.float32)
    if (h_o, w_o) != (h_r, w_r):
        delta = cv2.resize(delta, (w_o, h_o), interpolation=cv2.INTER_LINEAR)
    return np.clip(img_orig.astype(np.float32) + delta, 0, 255).astype(np.uint8)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest",   type=Path, default=Path("data/manifests/dev_300.json"))
    p.add_argument("--coco-root",  type=Path, default=Path("data/coco"))
    p.add_argument("--n-images",   type=int,  default=5)
    p.add_argument("--image-ids",  type=int,  nargs="+", default=None,
                   help="Specific COCO image IDs to visualize (overrides --n-images)")
    p.add_argument("--epsilon",    type=float, default=8.0)
    p.add_argument("--n-iters",    type=int,   default=40)
    p.add_argument("--step-size",  type=float, default=2.0)
    p.add_argument("--out-dir",    type=Path,  default=Path("results/viz"))
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest  = json.loads(args.manifest.read_text())
    coco_gt   = COCO(str(args.coco_root / "annotations/instances_val2017.json"))

    if args.image_ids:
        image_ids = args.image_ids
    else:
        image_ids = manifest["image_ids"][:args.n_images]

    id2file = {i: coco_gt.loadImgs(i)[0]["file_name"] for i in image_ids}

    register_all_modules()

    print("Loading surrogate (Faster R-CNN R50)...")
    surrogate = init_detector(
        "checkpoints/faster-rcnn_r50_fpn_1x_coco.py",
        "checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth",
        device=DEVICE)
    surrogate.eval()
    sur_classes = list(surrogate.dataset_meta["classes"])

    print("Loading target (Deformable DETR R50)...")
    target = init_detector(
        "checkpoints/deformable-detr_r50_16xb2-50e_coco.py",
        "checkpoints/deformable-detr_r50_16xb2-50e_coco_20221029_210934-6bc7d21b.pth",
        device=DEVICE)
    target.eval()
    tgt_classes = list(target.dataset_meta["classes"])

    print(f"\nVisualizing {len(image_ids)} images | ε={args.epsilon}px | {args.n_iters} iters")
    print(f"Output: {args.out_dir}/\n")

    for img_id in image_ids:
        fname    = id2file[img_id]
        img_path = args.coco_root / "val2017" / fname
        print(f"  {fname} ...", end=" ", flush=True)

        img_orig = cv2.imread(str(img_path))
        img_bgr  = load_image_bgr(img_path)

        # ── Attack ────────────────────────────────────────────────────────
        img_adv_r = pgd_attack(
            surrogate, img_bgr,
            epsilon_px=args.epsilon, n_iters=args.n_iters,
            step_size_px=args.step_size, n_masks=2,
            pruning_scope="backbone", pruning_rate=0.05,
            device=DEVICE,
        )
        img_adv = adversarial_at_orig_scale(img_adv_r, img_bgr, img_orig)

        # ── Inference ─────────────────────────────────────────────────────
        sur_clean = inference_detector(surrogate, img_orig).pred_instances
        sur_adv   = inference_detector(surrogate, img_adv ).pred_instances
        tgt_clean = inference_detector(target,    img_orig).pred_instances
        tgt_adv   = inference_detector(target,    img_adv ).pred_instances

        nc_sur = int((sur_clean.scores >= SCORE_THRESH).sum())
        na_sur = int((sur_adv.scores   >= SCORE_THRESH).sum())
        nc_tgt = int((tgt_clean.scores >= SCORE_THRESH).sum())
        na_tgt = int((tgt_adv.scores   >= SCORE_THRESH).sum())
        print(f"sur {nc_sur}→{na_sur}  tgt {nc_tgt}→{na_tgt}")

        # ── Draw ──────────────────────────────────────────────────────────
        H, W = img_orig.shape[:2]
        # cap display width at 640px per panel
        dw = min(W, 640)
        dh = int(H * dw / W)

        def panel(img, preds, classes, colour, title):
            drawn = draw_boxes(img, preds, classes, colour)
            drawn = resize_to(drawn, dh, dw)
            return add_label_bar(drawn, title, colour)

        p_sur_c = panel(img_orig, sur_clean, sur_classes, (30,140,30),
                        f"Faster R-CNN CLEAN  [{nc_sur} dets]")
        p_sur_a = panel(img_adv,  sur_adv,   sur_classes, (0,0,180),
                        f"Faster R-CNN ADV    [{na_sur} dets]")
        p_tgt_c = panel(img_orig, tgt_clean, tgt_classes, (30,100,180),
                        f"Def-DETR CLEAN      [{nc_tgt} dets]")
        p_tgt_a = panel(img_adv,  tgt_adv,   tgt_classes, (0,0,120),
                        f"Def-DETR ADV        [{na_tgt} dets]")

        # 2×2 grid
        row1 = np.hstack([p_sur_c, p_sur_a])
        row2 = np.hstack([p_tgt_c, p_tgt_a])

        # add noise magnitude overlay bottom-right of adv panels
        delta_vis = np.clip(
            (img_adv.astype(np.int32) - img_orig.astype(np.int32) + 128), 0, 255
        ).astype(np.uint8)
        delta_vis = resize_to(delta_vis, dh, dw)
        delta_vis = add_label_bar(delta_vis, "Perturbation (×1, +128)", (60,60,60))

        # pad delta panel to match row width
        pad_w = row1.shape[1] - dw
        pad   = np.zeros((delta_vis.shape[0], pad_w, 3), dtype=np.uint8)
        row3  = np.hstack([delta_vis, pad])

        grid = np.vstack([row1, row2, row3])

        out_path = args.out_dir / f"{img_id:012d}_compare.jpg"
        cv2.imwrite(str(out_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 92])

    print(f"\nDone — saved to {args.out_dir}/")
    print("Files:", ", ".join(
        f.name for f in sorted(args.out_dir.glob("*_compare.jpg"))
    ))


if __name__ == "__main__":
    main()
