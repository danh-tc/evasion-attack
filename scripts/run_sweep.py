#!/usr/bin/env python3
"""E0 — Hyperparameter sweep: rate × n_masks grid on representative targets.

Loads all models once, then runs attack for each (rate, n_masks) combination.
Uses 3 representative targets (one per backbone family) for speed.

Usage:
    python scripts/run_sweep.py --n-images 20 \
        --rates 0.01 0.05 0.10 0.20 0.50 \
        --masks 1 2 3 5 \
        --loss osfd --k 3.0 --prune-types norm \
        --out results/e0_sweep.json
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import cv2
import numpy as np
import torch
from mmdet.apis import inference_detector, init_detector
from mmdet.utils import register_all_modules
from pycocotools.coco import COCO
from torchvision.ops import box_iou
from tqdm import tqdm

from assignment_stable_od.attack import AttackConfig, load_image_bgr, pgd_attack

DEVICE       = "cuda:0"
SCORE_THRESH = 0.3
MATCH_IOU    = 0.5

SURROGATE_CONFIG = "checkpoints/faster-rcnn_r50_fpn_1x_coco.py"
SURROGATE_CKPT   = "checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"

# One representative per backbone family
SWEEP_TARGETS = [
    {
        "name": "fcos_r50", "group": "A",
        "config": "checkpoints/fcos_r50-caffe_fpn_gn-head_1x_coco.py",
        "ckpt":   "checkpoints/fcos_r50_caffe_fpn_gn-head_1x_coco-821213aa.pth",
    },
    {
        "name": "yolov3_d53", "group": "B",
        "config": "/workspace/evasion-venv/lib/python3.10/site-packages/mmdet/.mim/configs/yolo/yolov3_d53_8xb8-ms-608-273e_coco.py",
        "ckpt":   "checkpoints/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth",
    },
    {
        "name": "mask_rcnn_swin_t", "group": "C",
        "config": "/workspace/evasion-venv/lib/python3.10/site-packages/mmdet/.mim/configs/swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py",
        "ckpt":   "checkpoints/mask_rcnn_swin-t_1x_coco_20210902_120937-9d6b7cfa.pth",
    },
]

_PRUNE_TYPE_MAP = {"norm": "Normalization", "linear": "Linear", "conv": "Conv"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_cat_map(model) -> dict[str, int]:
    m = {}
    for i, cls in enumerate(model.dataset_meta["classes"]):
        m[cls] = i
        m[cls.replace("_", " ")] = i
    return m


def count_matched_gt(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores) -> int:
    keep = pred_scores >= SCORE_THRESH
    pred_boxes, pred_labels = pred_boxes[keep], pred_labels[keep]
    if gt_boxes.numel() == 0 or pred_boxes.numel() == 0:
        return 0
    ious = box_iou(gt_boxes, pred_boxes)
    same_class = gt_labels[:, None] == pred_labels[None, :]
    return int(((ious >= MATCH_IOU) & same_class).any(dim=1).sum())


def gt_for_image(
    coco_gt: COCO, img_id: int, cat_map: dict
) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
    anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
    boxes, labels = [], []
    for ann in anns:
        name = coco_gt.loadCats(ann["category_id"])[0]["name"]
        if name not in cat_map:
            continue
        x, y, w, h = ann["bbox"]
        boxes.append([x, y, x + w, y + h])
        labels.append(cat_map[name])
    if not boxes:
        return None, None
    return (
        torch.tensor(boxes,  dtype=torch.float32, device=DEVICE),
        torch.tensor(labels, dtype=torch.long,    device=DEVICE),
    )


def adversarial_at_orig_scale(
    img_adv_resized: np.ndarray,
    img_bgr_resized: np.ndarray,
    img_orig: np.ndarray,
) -> np.ndarray:
    delta = img_adv_resized.astype(np.float32) - img_bgr_resized.astype(np.float32)
    h_o, w_o = img_orig.shape[:2]
    h_r, w_r = img_bgr_resized.shape[:2]
    if (h_o, w_o) != (h_r, w_r):
        delta = cv2.resize(delta, (w_o, h_o), interpolation=cv2.INTER_LINEAR)
    return np.clip(img_orig.astype(np.float32) + delta, 0, 255).astype(np.uint8)


def run_one_combo(
    surrogate,
    sur_cat_map: dict,
    targets: dict[str, dict],
    image_ids: list[int],
    clean_cache: dict[int, dict],
    id2file: dict[int, str],
    img_dir: Path,
    coco_gt: COCO,
    cfg: AttackConfig,
) -> dict[str, float]:
    """Attack all images with cfg; return per-model mean ASR (white-box and transfer)."""
    wb_list: list[float] = []
    trf_lists: dict[str, list[float]] = {n: [] for n in targets}

    for img_id in image_ids:
        img_path = img_dir / id2file[img_id]
        img_orig = cv2.imread(str(img_path))
        img_bgr  = load_image_bgr(img_path)

        gt_sur, lbl_sur = gt_for_image(coco_gt, img_id, sur_cat_map)
        if gt_sur is None:
            continue

        img_adv_r = pgd_attack(surrogate, img_bgr, cfg)
        img_adv   = adversarial_at_orig_scale(img_adv_r, img_bgr, img_orig)

        # White-box ASR
        bx, lb, sc = clean_cache[img_id]["surrogate"]
        mc = count_matched_gt(
            gt_sur, lbl_sur,
            torch.tensor(bx, dtype=torch.float32, device=DEVICE),
            torch.tensor(lb, dtype=torch.long,    device=DEVICE),
            torch.tensor(sc, dtype=torch.float32, device=DEVICE),
        )
        p_adv = inference_detector(surrogate, img_adv).pred_instances
        ma = count_matched_gt(gt_sur, lbl_sur, p_adv.bboxes, p_adv.labels, p_adv.scores)
        wb_list.append(max(0.0, (mc - ma) / max(mc, 1)))

        # Transfer ASR
        for tname, tinfo in targets.items():
            gt_t, lbl_t = gt_for_image(coco_gt, img_id, tinfo["cat"])
            if gt_t is None:
                continue
            bx, lb, sc = clean_cache[img_id][tname]
            nc = count_matched_gt(
                gt_t, lbl_t,
                torch.tensor(bx, dtype=torch.float32, device=DEVICE),
                torch.tensor(lb, dtype=torch.long,    device=DEVICE),
                torch.tensor(sc, dtype=torch.float32, device=DEVICE),
            )
            p_t = inference_detector(tinfo["model"], img_adv).pred_instances
            na = count_matched_gt(gt_t, lbl_t, p_t.bboxes, p_t.labels, p_t.scores)
            trf_lists[tname].append(max(0.0, (nc - na) / max(nc, 1)))

    return {
        "wb": float(np.mean(wb_list)) if wb_list else 0.0,
        **{k: float(np.mean(v)) if v else 0.0 for k, v in trf_lists.items()},
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="E0 — rate × n_masks hyperparameter sweep")

    p.add_argument("--manifest",   type=Path, default=Path("data/manifests/dev_300.json"))
    p.add_argument("--coco-root",  type=Path, default=Path("data/coco"))
    p.add_argument("--n-images",   type=int,  default=20, help="Images per combo (20 for smoke)")
    p.add_argument("--sur-config", type=Path, default=Path(SURROGATE_CONFIG))
    p.add_argument("--sur-ckpt",   type=Path, default=Path(SURROGATE_CKPT))

    p.add_argument("--rates", nargs="+", type=float, default=[0.01, 0.05, 0.10, 0.20, 0.50])
    p.add_argument("--masks", nargs="+", type=int,   default=[1, 2, 3, 5])

    p.add_argument("--loss",        choices=["osfd", "rpn"], default="osfd")
    p.add_argument("--k",           type=float, default=3.0)
    p.add_argument("--prune-scope", type=str,   default="backbone")
    p.add_argument("--prune-types", nargs="+",  default=["norm"], choices=list(_PRUNE_TYPE_MAP))
    p.add_argument("--n-iters",     type=int,   default=40)
    p.add_argument("--epsilon",     type=float, default=8.0)
    p.add_argument("--step-size",   type=float, default=2.0)
    p.add_argument("--momentum",    type=float, default=0.9)

    p.add_argument("--out", type=Path, default=Path("results/e0_sweep.json"))
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    manifest  = json.loads(args.manifest.read_text())
    image_ids = manifest["image_ids"][:args.n_images]
    coco_gt   = COCO(str(args.coco_root / "annotations/instances_val2017.json"))
    id2file   = {i: coco_gt.loadImgs(i)[0]["file_name"] for i in image_ids}
    img_dir   = args.coco_root / "val2017"

    register_all_modules()

    print("Loading surrogate...")
    surrogate   = init_detector(str(args.sur_config), str(args.sur_ckpt), device=DEVICE)
    surrogate.eval()
    sur_cat_map = build_cat_map(surrogate)

    print("Loading sweep targets (3 representative)...")
    targets: dict[str, dict] = {}
    for t in SWEEP_TARGETS:
        m = init_detector(t["config"], t["ckpt"], device=DEVICE)
        m.eval()
        targets[t["name"]] = {"model": m, "group": t["group"], "cat": build_cat_map(m)}
        print(f"  [Group {t['group']}] {t['name']}")

    # Pre-compute clean predictions once — reused across all combos
    print(f"\nPre-computing clean predictions ({len(image_ids)} images)...")
    clean_cache: dict[int, dict] = {}
    for img_id in tqdm(image_ids, desc="  clean", file=sys.stdout):
        img_path = img_dir / id2file[img_id]
        img_orig = cv2.imread(str(img_path))
        clean_cache[img_id] = {}

        p = inference_detector(surrogate, img_orig).pred_instances
        clean_cache[img_id]["surrogate"] = (
            p.bboxes.cpu().numpy(), p.labels.cpu().numpy(), p.scores.cpu().numpy()
        )
        for tname, tinfo in targets.items():
            p = inference_detector(tinfo["model"], img_orig).pred_instances
            clean_cache[img_id][tname] = (
                p.bboxes.cpu().numpy(), p.labels.cpu().numpy(), p.scores.cpu().numpy()
            )

    prune_types = [_PRUNE_TYPE_MAP[t] for t in args.prune_types]
    prune_scope = None if args.prune_scope == "none" else args.prune_scope

    grid     = list(product(args.rates, args.masks))
    n_combos = len(grid)
    print(f"\n{'═'*72}")
    print(f"  Sweep: {len(args.rates)} rates × {len(args.masks)} masks = {n_combos} combos")
    print(f"  loss={args.loss}  k={args.k}  prune_types={prune_types}  scope={prune_scope}")
    print(f"  images/combo={args.n_images}  ε={args.epsilon}px  iters={args.n_iters}")
    print(f"{'═'*72}")

    tgt_names = list(targets.keys())
    header = (f"  {'rate':>6}  {'masks':>5}  {'WB-ASR':>7}  " +
              "  ".join(f"{n:>18}" for n in tgt_names))
    print(header)
    print(f"{'─'*len(header)}")

    sweep_results: list[dict] = []
    for combo_idx, (rate, n_masks) in enumerate(grid):
        cfg = AttackConfig(
            epsilon_px=args.epsilon, n_iters=args.n_iters, step_size_px=args.step_size,
            momentum=args.momentum, loss_type=args.loss, osfd_k=args.k,
            n_masks=n_masks, pruning_scope=prune_scope,
            pruning_rate=rate, pruning_types=prune_types, device=DEVICE,
        )
        asr = run_one_combo(
            surrogate, sur_cat_map, targets, image_ids,
            clean_cache, id2file, img_dir, coco_gt, cfg,
        )
        row = {"rate": rate, "n_masks": n_masks, **asr}
        sweep_results.append(row)

        line = (f"  {rate:>6.2f}  {n_masks:>5}  {asr['wb']:>7.3f}  " +
                "  ".join(f"{asr.get(n, 0.0):>18.3f}" for n in tgt_names))
        print(f"[{combo_idx+1:>{len(str(n_combos))}}/{n_combos}] {line}")

    print(f"{'═'*len(header)}")

    # Best config per metric (capture `key` in default arg to avoid closure bug)
    for metric in ["wb"] + tgt_names:
        best = max(sweep_results, key=lambda r, m=metric: r.get(m, 0.0))
        print(f"  Best {metric}: rate={best['rate']:.2f}  n_masks={best['n_masks']}"
              f"  ASR={best.get(metric, 0.0):.3f}")

    output = {
        "meta": {
            "n_images": args.n_images,
            "loss": args.loss, "osfd_k": args.k,
            "prune_types": prune_types, "prune_scope": prune_scope,
            "epsilon_px": args.epsilon, "n_iters": args.n_iters,
            "rates": args.rates, "masks": args.masks,
        },
        "results": sweep_results,
    }
    args.out.write_text(json.dumps(output, indent=2))
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
