#!/usr/bin/env python3
"""Smoke test for rapa_osfd_005 config.

Runs 3 configs (pgd_baseline, rapa_backbone_005, rapa_osfd_005) on N images
against a single target (retinanet_r50) so we can verify OSFD works and see
its relative effect without downloading all checkpoints.

Usage (from project root, venv active):
    python scripts/smoke_osfd.py --n-images 20 --n-iters 10
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
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from assignment_stable_od.attack import load_image_bgr, pgd_attack

DEVICE       = "cuda:0"
SCORE_THRESH = 0.3
MATCH_IOU    = 0.5

SURROGATE = dict(
    config=PROJECT_DIR / "checkpoints/faster-rcnn_r50_fpn_1x_coco.py",
    ckpt=PROJECT_DIR   / "checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth",
)
TARGET = dict(
    name="retinanet_r50",
    config=PROJECT_DIR / "checkpoints/retinanet_r50_fpn_1x_coco.py",
    ckpt=PROJECT_DIR   / "checkpoints/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth",
)

CONFIGS = [
    dict(name="pgd_baseline",      n_masks=1, scope=None,       rate=0.0,  use_osfd=False),
    dict(name="rapa_backbone_005", n_masks=2, scope="backbone", rate=0.05, use_osfd=False),
    dict(name="rapa_osfd_005",     n_masks=2, scope="backbone", rate=0.05, use_osfd=True),
]


def preds_from_bgr(model, img_bgr):
    return inference_detector(model, img_bgr).pred_instances


def count_disappeared(clean_bx, clean_lb, clean_sc, adv_preds,
                      score_thresh=SCORE_THRESH, iou_thresh=MATCH_IOU):
    """How many clean detections disappeared after the attack."""
    keep = clean_sc >= score_thresh
    if not keep.any():
        return 0, 0
    cb = torch.tensor(clean_bx[keep], dtype=torch.float32, device=DEVICE)
    cl = torch.tensor(clean_lb[keep], dtype=torch.long,    device=DEVICE)

    ab = adv_preds.bboxes
    al = adv_preds.labels
    asc = adv_preds.scores
    ak = asc >= score_thresh

    n_clean = len(cb)
    if not ak.any():
        return n_clean, n_clean

    from torchvision.ops import box_iou
    iou = box_iou(cb, ab[ak])
    matched = 0
    for i in range(len(cb)):
        row = iou[i]
        same_class = (al[ak] == cl[i])
        if same_class.any() and (row * same_class.float()).max() >= iou_thresh:
            matched += 1
    return n_clean, n_clean - matched


def build_cat_map(model):
    return {cls: i for i, cls in enumerate(model.dataset_meta["classes"])}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest",  type=Path, default=PROJECT_DIR / "data/manifests/dev_300.json")
    p.add_argument("--coco-root", type=Path, default=PROJECT_DIR / "data/coco")
    p.add_argument("--n-images",  type=int,  default=20)
    p.add_argument("--n-iters",   type=int,  default=10)
    p.add_argument("--epsilon",   type=float, default=8.0)
    p.add_argument("--step-size", type=float, default=2.0)
    p.add_argument("--out",       type=Path,
                   default=PROJECT_DIR / "results/multi_target/results_osfd_smoke.json")
    return p.parse_args()


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    coco_gt  = COCO(str(args.coco_root / "annotations/instances_val2017.json"))
    manifest = json.loads(args.manifest.read_text())
    img_ids  = manifest["image_ids"][:args.n_images]
    id2file  = {i: coco_gt.loadImgs(i)[0]["file_name"] for i in img_ids}
    img_dir  = args.coco_root / "val2017"

    register_all_modules()

    print("Loading surrogate (Faster R-CNN R50)...")
    sur = init_detector(str(SURROGATE["config"]), str(SURROGATE["ckpt"]), device=DEVICE)
    sur.eval()
    sur_cat = build_cat_map(sur)

    print(f"Loading target ({TARGET['name']})...")
    tgt = init_detector(str(TARGET["config"]), str(TARGET["ckpt"]), device=DEVICE)
    tgt.eval()
    tgt_cat = build_cat_map(tgt)

    print(f"\nImages={len(img_ids)}  iters={args.n_iters}  ε={args.epsilon}px\n")

    results = {}

    for cfg in CONFIGS:
        name  = cfg["name"]
        t0    = time.perf_counter()
        wb_asr, trf_asr = [], []

        for img_id in tqdm(img_ids, desc=f"  {name}", file=sys.stdout, dynamic_ncols=True):
            img_path = img_dir / id2file[img_id]
            img_orig = cv2.imread(str(img_path))
            if img_orig is None:
                print(f"    [WARN] missing image {id2file[img_id]}, skip")
                continue
            img_bgr = load_image_bgr(img_path)

            # clean preds
            p_sur_clean = preds_from_bgr(sur, img_orig)
            p_tgt_clean = preds_from_bgr(tgt, img_orig)
            clean_sur_bx = p_sur_clean.bboxes.cpu().numpy()
            clean_sur_lb = p_sur_clean.labels.cpu().numpy()
            clean_sur_sc = p_sur_clean.scores.cpu().numpy()
            clean_tgt_bx = p_tgt_clean.bboxes.cpu().numpy()
            clean_tgt_lb = p_tgt_clean.labels.cpu().numpy()
            clean_tgt_sc = p_tgt_clean.scores.cpu().numpy()

            # attack
            img_adv = pgd_attack(
                sur, img_bgr,
                epsilon_px=args.epsilon, n_iters=args.n_iters,
                step_size_px=args.step_size, n_masks=cfg["n_masks"],
                pruning_scope=cfg["scope"], pruning_rate=cfg["rate"],
                momentum=0.9, device=DEVICE,
                use_osfd=cfg["use_osfd"],
            )
            # scale delta back to original image size
            h_orig, w_orig = img_orig.shape[:2]
            h_adv,  w_adv  = img_adv.shape[:2]
            if (h_orig, w_orig) != (h_adv, w_adv):
                delta = img_adv.astype(np.float32) - cv2.resize(
                    img_orig, (w_adv, h_adv)).astype(np.float32)
                delta = cv2.resize(delta, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
                img_adv_full = np.clip(img_orig.astype(np.float32) + delta, 0, 255).astype(np.uint8)
            else:
                img_adv_full = img_adv

            # WB ASR
            p_sur_adv = preds_from_bgr(sur, img_adv_full)
            n_clean, n_disap = count_disappeared(
                clean_sur_bx, clean_sur_lb, clean_sur_sc, p_sur_adv)
            wb_asr.append(n_disap / max(n_clean, 1))

            # Transfer ASR
            p_tgt_adv = preds_from_bgr(tgt, img_adv_full)
            n_clean_t, n_disap_t = count_disappeared(
                clean_tgt_bx, clean_tgt_lb, clean_tgt_sc, p_tgt_adv)
            trf_asr.append(n_disap_t / max(n_clean_t, 1))

        elapsed = time.perf_counter() - t0
        wb_mean  = float(np.mean(wb_asr))  if wb_asr  else 0.0
        trf_mean = float(np.mean(trf_asr)) if trf_asr else 0.0
        results[name] = dict(wb_asr=wb_mean, trf_asr=trf_mean, elapsed_s=elapsed)
        print(f"  → WB={wb_mean:.3f}  TRF={trf_mean:.3f}  ({elapsed:.0f}s)\n")

    print("=" * 60)
    print(f"{'Config':<25} {'WB-ASR':>8} {'TRF-ASR':>8}")
    print("-" * 60)
    for name, r in results.items():
        print(f"  {name:<23} {r['wb_asr']:>8.3f} {r['trf_asr']:>8.3f}")
    print("=" * 60)

    args.out.write_text(json.dumps(
        {"meta": vars(args) | {"n_images": len(img_ids)}, "results": results},
        indent=2, default=str,
    ))
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
