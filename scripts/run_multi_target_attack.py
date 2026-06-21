#!/usr/bin/env python3
"""Multi-target feasibility: craft on Faster R-CNN, evaluate on 3 target architectures.

Targets:
  - RetinaNet  R50-FPN  (anchor-based one-stage, CNN)
  - FCOS       R50-FPN  (anchor-free one-stage, CNN)
  - Deformable DETR R50 (transformer)

Usage:
    python scripts/run_multi_target_attack.py \
        --n-images 20 --n-iters 20 --epsilon 8
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

# Best config from backbone sweep: backbone p=0.05, n_masks=2
CONFIGS = [
    dict(name="pgd_baseline",      label="PGD baseline",
         n_masks=1, scope=None,       rate=0.0),
    dict(name="rapa_backbone_005", label="RaPA-OD backbone p=0.05 [best]",
         n_masks=2, scope="backbone", rate=0.05),
]

TARGETS = [
    dict(name="retinanet_r50",
         config="checkpoints/retinanet_r50_fpn_1x_coco.py",
         ckpt="checkpoints/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth"),
    dict(name="retinanet_r101",
         config="checkpoints/retinanet_r101_fpn_1x_coco.py",
         ckpt="checkpoints/retinanet_r101_fpn_1x_coco_20200130-7a93545f.pth"),
    dict(name="fcos_r50",
         config="checkpoints/fcos_r50-caffe_fpn_gn-head_1x_coco.py",
         ckpt="checkpoints/fcos_r50_caffe_fpn_gn-head_1x_coco-821213aa.pth"),
    dict(name="deformable_detr",
         config="checkpoints/deformable-detr_r50_16xb2-50e_coco.py",
         ckpt="checkpoints/deformable-detr_r50_16xb2-50e_coco_20221029_210934-6bc7d21b.pth"),
]


def count_matched_gt(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores) -> int:
    keep = pred_scores >= SCORE_THRESH
    pred_boxes, pred_labels = pred_boxes[keep], pred_labels[keep]
    if gt_boxes.numel() == 0 or pred_boxes.numel() == 0:
        return 0
    ious = box_iou(gt_boxes, pred_boxes)
    same_class = gt_labels[:, None] == pred_labels[None, :]
    return int(((ious >= MATCH_IOU) & same_class).any(dim=1).sum())


def preds_from_bgr(model, img_bgr: np.ndarray):
    return inference_detector(model, img_bgr).pred_instances


def adversarial_at_orig_scale(img_adv_resized, img_bgr_resized, img_orig) -> np.ndarray:
    h_orig, w_orig = img_orig.shape[:2]
    h_res,  w_res  = img_bgr_resized.shape[:2]
    delta = img_adv_resized.astype(np.float32) - img_bgr_resized.astype(np.float32)
    if (h_orig, w_orig) != (h_res, w_res):
        delta = cv2.resize(delta, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    return np.clip(img_orig.astype(np.float32) + delta, 0, 255).astype(np.uint8)


def gt_for_image(coco_gt, img_id, cat_to_label):
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


def build_cat_map(model, coco_gt):
    m = {}
    for i, cls in enumerate(model.dataset_meta["classes"]):
        m[cls] = i
        m[cls.replace("_", " ")] = i
    return m


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest",  type=Path, default=Path("data/manifests/dev_300.json"))
    p.add_argument("--coco-root", type=Path, default=Path("data/coco"))
    p.add_argument("--sur-config", type=Path,
                   default=Path("checkpoints/faster-rcnn_r50_fpn_1x_coco.py"))
    p.add_argument("--sur-ckpt",   type=Path,
                   default=Path("checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"))
    p.add_argument("--n-images",   type=int,   default=20)
    p.add_argument("--epsilon",    type=float, default=8.0)
    p.add_argument("--n-iters",    type=int,   default=20)
    p.add_argument("--step-size",  type=float, default=2.0)
    p.add_argument("--momentum",   type=float, default=0.9)
    p.add_argument("--out",        type=Path,
                   default=Path("results/multi_target/results.json"))
    return p.parse_args()


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    manifest  = json.loads(args.manifest.read_text())
    image_ids = manifest["image_ids"][:args.n_images]
    coco_gt   = COCO(str(args.coco_root / "annotations/instances_val2017.json"))
    id2file   = {i: coco_gt.loadImgs(i)[0]["file_name"] for i in image_ids}

    register_all_modules()

    print("Loading surrogate...")
    surrogate = init_detector(str(args.sur_config), str(args.sur_ckpt), device=DEVICE)
    surrogate.eval()
    sur_cat = build_cat_map(surrogate, coco_gt)

    print("Loading targets...")
    targets = {}
    for t in TARGETS:
        m = init_detector(t["config"], t["ckpt"], device=DEVICE)
        m.eval()
        targets[t["name"]] = dict(model=m, cat=build_cat_map(m, coco_gt))
        print(f"  ✓ {t['name']}")

    print(f"\nSurrogate : {args.sur_config.stem}")
    print(f"Targets   : {', '.join(targets)}")
    print(f"Images    : {len(image_ids)}")
    print(f"ε={args.epsilon}px  iters={args.n_iters}  step={args.step_size}px")

    all_results = {}
    bar_fmt = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"

    for cfg_idx, cfg in enumerate(CONFIGS):
        label = cfg["label"]
        print(f"\n{'═'*76}")
        print(f"  [{cfg_idx+1}/{len(CONFIGS)}] {label}")
        print(f"{'═'*76}")

        wb_list = []
        trf_lists = {t["name"]: [] for t in TARGETS}
        t_cfg = time.perf_counter()

        pbar = tqdm(image_ids, desc="  attack", unit="img",
                    bar_format=bar_fmt, file=sys.stdout, dynamic_ncols=True)

        for img_id in pbar:
            img_path = args.coco_root / "val2017" / id2file[img_id]
            img_orig = cv2.imread(str(img_path))
            img_bgr  = load_image_bgr(img_path)

            gt_sur, lbl_sur = gt_for_image(coco_gt, img_id, sur_cat)
            if gt_sur is None:
                pbar.write(f"  skip {id2file[img_id]} (no GT on surrogate)")
                continue

            # check at least one target has GT
            tgt_gts = {}
            for tname, tinfo in targets.items():
                gt, lbl = gt_for_image(coco_gt, img_id, tinfo["cat"])
                tgt_gts[tname] = (gt, lbl)

            # ── Attack ──────────────────────────────────────────────────────
            img_adv_resized = pgd_attack(
                surrogate, img_bgr,
                epsilon_px=args.epsilon, n_iters=args.n_iters,
                step_size_px=args.step_size, n_masks=cfg["n_masks"],
                pruning_scope=cfg["scope"], pruning_rate=cfg["rate"],
                momentum=args.momentum, device=DEVICE,
            )
            img_adv = adversarial_at_orig_scale(img_adv_resized, img_bgr, img_orig)

            # ── White-box ────────────────────────────────────────────────────
            p_clean = preds_from_bgr(surrogate, img_orig)
            p_adv   = preds_from_bgr(surrogate, img_adv)
            mc = count_matched_gt(gt_sur, lbl_sur, p_clean.bboxes, p_clean.labels, p_clean.scores)
            ma = count_matched_gt(gt_sur, lbl_sur, p_adv.bboxes,   p_adv.labels,   p_adv.scores)
            wb = max(0.0, (mc - ma) / max(mc, 1))
            wb_list.append(wb)

            # ── Transfer (all targets) ───────────────────────────────────────
            trf_parts = []
            for tname, tinfo in targets.items():
                gt_t, lbl_t = tgt_gts[tname]
                if gt_t is None:
                    trf_parts.append(f"{tname}=N/A")
                    continue
                q_clean = preds_from_bgr(tinfo["model"], img_orig)
                q_adv   = preds_from_bgr(tinfo["model"], img_adv)
                nc = count_matched_gt(gt_t, lbl_t, q_clean.bboxes, q_clean.labels, q_clean.scores)
                na = count_matched_gt(gt_t, lbl_t, q_adv.bboxes,   q_adv.labels,   q_adv.scores)
                trf = max(0.0, (nc - na) / max(nc, 1))
                trf_lists[tname].append(trf)
                trf_parts.append(f"{tname}={trf:.2f}")

            mwb = float(np.mean(wb_list))
            trf_avgs = {k: float(np.mean(v)) if v else 0.0 for k, v in trf_lists.items()}
            pbar.write(
                f"  {id2file[img_id]}  wb={wb:.2f}  " +
                "  ".join(trf_parts) +
                f"  | avg WB={mwb:.3f}  " +
                "  ".join(f"{k}={v:.3f}" for k, v in trf_avgs.items())
            )

        elapsed = time.perf_counter() - t_cfg
        mwb  = float(np.mean(wb_list)) if wb_list else 0.0
        trf_finals = {k: float(np.mean(v)) if v else 0.0 for k, v in trf_lists.items()}

        print(f"{'─'*76}")
        print(f"  FINAL  WB={mwb:.3f}  " +
              "  ".join(f"TRF-{k}={v:.3f}" for k, v in trf_finals.items()) +
              f"  ({len(wb_list)} images, {elapsed:.0f}s)")

        all_results[cfg["name"]] = dict(
            label=label, wb_asr=mwb,
            trf_asr=trf_finals, wb_per_image=wb_list,
        )

    # ── Summary table ─────────────────────────────────────────────────────────
    tgt_names = [t["name"] for t in TARGETS]
    col_w = 38
    header = f"  {'Config':<{col_w}}  {'WB':>6}  " + "  ".join(f"{n.upper():>12}" for n in tgt_names)
    print(f"\n{'═'*len(header)}")
    print(header)
    print(f"{'─'*len(header)}")
    for cfg in CONFIGS:
        r = all_results[cfg["name"]]
        row = f"  {cfg['label']:<{col_w}}  {r['wb_asr']:>6.3f}  "
        row += "  ".join(f"{r['trf_asr'].get(n, 0):>12.3f}" for n in tgt_names)
        print(row)
    print(f"{'═'*len(header)}")

    args.out.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
