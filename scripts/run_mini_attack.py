#!/usr/bin/env python3
"""Feasibility demo: PGD baseline vs global RaPA-OD vs scope-aware RaPA-OD.

Attacks 10 images and reports:
  - White-box ASR on Faster R-CNN (surrogate)
  - Transfer ASR on RetinaNet  (target, zero-query)

ASR = fraction of GT objects that disappear after attack.
'Disappear' = no prediction with same class AND IoU≥0.5 AND score≥0.3.

Usage:
    python scripts/run_mini_attack.py
    python scripts/run_mini_attack.py --n-images 15 --n-iters 40 --epsilon 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from mmdet.apis import inference_detector, init_detector
from mmdet.utils import register_all_modules
from pycocotools.coco import COCO
from torchvision.ops import box_iou
from tqdm import tqdm

from assignment_stable_od.attack import load_image_bgr, pgd_attack

DEVICE       = "cuda:0"
SCORE_THRESH = 0.3
MATCH_IOU    = 0.5

CONFIGS = [
    dict(name="pgd_baseline",    label="PGD (no pruning)",
         n_masks=1, scope=None,       rate=0.0),
    dict(name="rapa_global_01",  label="RaPA-OD  all    p=0.1  [corrupting]",
         n_masks=3, scope="all",      rate=0.1),
    dict(name="rapa_rpn_03",     label="RaPA-OD  rpn    p=0.3  [stable]",
         n_masks=3, scope="rpn_head", rate=0.3),
    dict(name="rapa_neck_03",    label="RaPA-OD  neck   p=0.3  [stable]",
         n_masks=3, scope="neck",     rate=0.3),
]


# ── GT matching ───────────────────────────────────────────────────────────────

def count_matched_gt(
    gt_boxes:    torch.Tensor,
    gt_labels:   torch.Tensor,
    pred_boxes:  torch.Tensor,
    pred_labels: torch.Tensor,
    pred_scores: torch.Tensor,
) -> int:
    keep = pred_scores >= SCORE_THRESH
    pred_boxes, pred_labels = pred_boxes[keep], pred_labels[keep]
    if gt_boxes.numel() == 0 or pred_boxes.numel() == 0:
        return 0
    ious       = box_iou(gt_boxes, pred_boxes)
    same_class = gt_labels[:, None] == pred_labels[None, :]
    return int(((ious >= MATCH_IOU) & same_class).any(dim=1).sum())


def preds_from_bgr(model, img_bgr: np.ndarray):
    """Run inference on a uint8 BGR HWC numpy image (original resolution)."""
    result = inference_detector(model, img_bgr)
    return result.pred_instances


def adversarial_at_orig_scale(
    img_adv_resized: np.ndarray,
    img_bgr_resized: np.ndarray,
    img_orig:        np.ndarray,
) -> np.ndarray:
    """Scale the adversarial perturbation back to original image resolution.

    Attack is computed on a pre-resized image.  Rescale the delta and add
    it to the original image so that inference_detector returns predictions
    in original image coordinates (matching COCO GT box coordinates).
    """
    h_orig, w_orig = img_orig.shape[:2]
    h_res,  w_res  = img_bgr_resized.shape[:2]
    delta = img_adv_resized.astype(np.float32) - img_bgr_resized.astype(np.float32)
    if (h_orig, w_orig) != (h_res, w_res):
        delta = cv2.resize(delta, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    return np.clip(img_orig.astype(np.float32) + delta, 0, 255).astype(np.uint8)


def gt_for_image(
    coco_gt: COCO, img_id: int, cat_to_label: dict[str, int]
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
    boxes, labels = [], []
    for ann in anns:
        name = coco_gt.loadCats(ann["category_id"])[0]["name"]
        if name not in cat_to_label:
            continue
        x, y, w, h = ann["bbox"]
        boxes.append([x, y, x + w, y + h])
        labels.append(cat_to_label[name])
    if not boxes:
        return None, None
    return (
        torch.tensor(boxes,  dtype=torch.float32, device=DEVICE),
        torch.tensor(labels, dtype=torch.long,    device=DEVICE),
    )


def build_cat_map(model, coco_gt: COCO) -> dict[str, int]:
    classes = model.dataset_meta["classes"]
    m: dict[str, int] = {}
    for i, cls in enumerate(classes):
        m[cls] = i
        m[cls.replace("_", " ")] = i
    return m


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest",   type=Path,
                   default=Path("data/manifests/dev_300.json"))
    p.add_argument("--coco-root",  type=Path, default=Path("data/coco"))
    p.add_argument("--sur-config", type=Path,
                   default=Path("checkpoints/faster-rcnn_r50_fpn_1x_coco.py"))
    p.add_argument("--sur-ckpt",   type=Path,
                   default=Path("checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"))
    p.add_argument("--tgt-config", type=Path,
                   default=Path("checkpoints/retinanet_r50_fpn_1x_coco.py"))
    p.add_argument("--tgt-ckpt",   type=Path,
                   default=Path("checkpoints/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth"))
    p.add_argument("--n-images",    type=int,   default=10)
    p.add_argument("--epsilon",     type=float, default=8.0,
                   help="L_inf budget in pixel units [0-255]  (default: 8)")
    p.add_argument("--n-iters",     type=int,   default=20)
    p.add_argument("--step-size",   type=float, default=2.0,
                   help="PGD step size in pixel units         (default: 2)")
    p.add_argument("--momentum",    type=float, default=0.9)
    p.add_argument("--out",         type=Path,
                   default=Path("results/mini_attack/results.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    manifest  = json.loads(args.manifest.read_text())
    image_ids = manifest["image_ids"][: args.n_images]
    coco_gt   = COCO(str(args.coco_root / "annotations/instances_val2017.json"))
    id2file   = {i: coco_gt.loadImgs(i)[0]["file_name"] for i in image_ids}

    register_all_modules()
    surrogate = init_detector(str(args.sur_config), str(args.sur_ckpt), device=DEVICE)
    target    = init_detector(str(args.tgt_config), str(args.tgt_ckpt), device=DEVICE)
    surrogate.eval();  target.eval()

    sur_cat = build_cat_map(surrogate, coco_gt)
    tgt_cat = build_cat_map(target,    coco_gt)

    print(f"Surrogate : {args.sur_config.stem}")
    print(f"Target    : {args.tgt_config.stem}")
    print(f"Images    : {len(image_ids)}")
    print(f"ε={args.epsilon}px  iters={args.n_iters}  step={args.step_size}px\n")

    all_results: dict = {}

    bar_fmt = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    sep = "─" * 72

    for cfg_idx, cfg in enumerate(CONFIGS):
        label = cfg["label"]
        print(f"\n{'═'*72}")
        print(f"  [{cfg_idx+1}/{len(CONFIGS)}] {label}")
        print(f"{'═'*72}")
        wb_list, trf_list = [], []
        t_cfg = time.perf_counter()

        pbar = tqdm(
            image_ids,
            desc="  attack",
            unit="img",
            bar_format=bar_fmt,
            file=sys.stdout,
            dynamic_ncols=True,
        )
        for img_id in pbar:
            img_path = args.coco_root / "val2017" / id2file[img_id]
            img_orig = cv2.imread(str(img_path))
            img_bgr  = load_image_bgr(img_path)

            gt_sur, lbl_sur = gt_for_image(coco_gt, img_id, sur_cat)
            gt_tgt, lbl_tgt = gt_for_image(coco_gt, img_id, tgt_cat)
            if gt_sur is None or gt_tgt is None:
                pbar.write(f"  skip {id2file[img_id]} (no GT)")
                continue

            img_adv_resized = pgd_attack(
                surrogate, img_bgr,
                epsilon_px   = args.epsilon,
                n_iters      = args.n_iters,
                step_size_px = args.step_size,
                n_masks      = cfg["n_masks"],
                pruning_scope= cfg["scope"],
                pruning_rate = cfg["rate"],
                momentum     = args.momentum,
                device       = DEVICE,
            )
            img_adv = adversarial_at_orig_scale(img_adv_resized, img_bgr, img_orig)

            # White-box
            p_clean = preds_from_bgr(surrogate, img_orig)
            p_adv   = preds_from_bgr(surrogate, img_adv)
            mc = count_matched_gt(gt_sur, lbl_sur, p_clean.bboxes, p_clean.labels, p_clean.scores)
            ma = count_matched_gt(gt_sur, lbl_sur, p_adv.bboxes,   p_adv.labels,   p_adv.scores)
            wb = max(0.0, (mc - ma) / max(mc, 1))
            wb_list.append(wb)

            # Transfer
            q_clean = preds_from_bgr(target, img_orig)
            q_adv   = preds_from_bgr(target, img_adv)
            nc = count_matched_gt(gt_tgt, lbl_tgt, q_clean.bboxes, q_clean.labels, q_clean.scores)
            na = count_matched_gt(gt_tgt, lbl_tgt, q_adv.bboxes,   q_adv.labels,   q_adv.scores)
            trf = max(0.0, (nc - na) / max(nc, 1))
            trf_list.append(trf)

            mwb_run  = float(np.mean(wb_list))
            mtrf_run = float(np.mean(trf_list))
            pbar.set_postfix(
                wb=f"{wb:.2f}",
                trf=f"{trf:.2f}",
                WB_avg=f"{mwb_run:.3f}",
                TRF_avg=f"{mtrf_run:.3f}",
            )
            pbar.write(
                f"  {id2file[img_id]}  "
                f"wb={wb:.2f} trf={trf:.2f}  "
                f"(sur {mc:2d}→{ma:2d}  tgt {nc:2d}→{na:2d})  "
                f"run_avg  WB={mwb_run:.3f}  TRF={mtrf_run:.3f}"
            )

        pbar.close()
        elapsed = time.perf_counter() - t_cfg
        mwb  = float(np.mean(wb_list))  if wb_list  else 0.0
        mtrf = float(np.mean(trf_list)) if trf_list else 0.0
        print(f"{sep}")
        print(f"  FINAL  WB-ASR: {mwb:.3f}   TRF-ASR: {mtrf:.3f}"
              f"   ({len(wb_list)} images, {elapsed:.0f}s)")
        all_results[cfg["name"]] = dict(
            label=label,
            wb_asr=mwb, trf_asr=mtrf,
            wb_per_image=wb_list, trf_per_image=trf_list,
        )

    # ── Summary table ─────────────────────────────────────────────────────────
    w = 44
    print("=" * (w + 20))
    print(f"  {'Config':<{w}} {'WB-ASR':>7}  {'TRF-ASR':>7}")
    print("-" * (w + 20))
    for cfg in CONFIGS:
        r = all_results[cfg["name"]]
        print(f"  {cfg['label']:<{w}} {r['wb_asr']:>7.3f}  {r['trf_asr']:>7.3f}")
    print("=" * (w + 20))

    args.out.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
