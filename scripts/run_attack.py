#!/usr/bin/env python3
"""Craft adversarial examples on surrogate, evaluate transfer on 6 target models.

Covers experiments E1a, E2a, E2b, E3c (see PLAN_EXPERIMENTS.md).

Usage examples:
    # E1a — PGD baseline
    python scripts/run_attack.py --loss rpn --n-masks 1 --rate 0 \
        --out results/e1a_pgd.json

    # E2a — RaPA (Norm) + RPN
    python scripts/run_attack.py --loss rpn --prune-types norm --n-masks 2 --rate 0.05 \
        --out results/e2a_rapa_rpn.json

    # E2b — RaPA (Norm) + OSFD k=3  [main baseline]
    python scripts/run_attack.py --loss osfd --k 3.0 --prune-types norm --n-masks 2 --rate 0.05 \
        --out results/e2b_rapa_osfd.json

    # E3c — dual surrogate (R50 + Swin-T)
    python scripts/run_attack.py --loss osfd --k 3.0 --prune-types norm --n-masks 2 --rate 0.05 \
        --aux-config checkpoints/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py \
        --aux-ckpt   checkpoints/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_20210902_120937-9d6b7cfa.pth \
        --out results/e3c_dual.json
"""
from __future__ import annotations

import argparse
import contextlib
import io
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
from pycocotools.cocoeval import COCOeval
from torchvision.ops import box_iou
from tqdm import tqdm

from assignment_stable_od.attack import AttackConfig, load_image_bgr, pgd_attack

DEVICE        = "cuda:0"
SCORE_THRESH  = 0.3    # for per-image ASR matching
MATCH_IOU     = 0.5
COCO_SCORE_TH = 0.05   # low threshold for COCO mAP accumulation

SURROGATE_CONFIG = "checkpoints/faster-rcnn_r50_fpn_1x_coco.py"
SURROGATE_CKPT   = "checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"

TARGETS = [
    # Group A — In-family (ResNet-50 backbone)
    {
        "name": "fcos_r50", "group": "A",
        "config": "checkpoints/fcos_r50-caffe_fpn_gn-head_1x_coco.py",
        "ckpt":   "checkpoints/fcos_r50_caffe_fpn_gn-head_1x_coco-821213aa.pth",
    },
    {
        "name": "deformable_detr", "group": "A",
        "config": "checkpoints/deformable-detr_r50_16xb2-50e_coco.py",
        "ckpt":   "checkpoints/deformable-detr_r50_16xb2-50e_coco_20221029_210934-6bc7d21b.pth",
    },
    # Group B — Near-family (non-ResNet CNN backbone)
    {
        "name": "yolov3_d53", "group": "B",
        "config": "/workspace/evasion-venv/lib/python3.10/site-packages/mmdet/.mim/configs/yolo/yolov3_d53_8xb8-ms-608-273e_coco.py",
        "ckpt":   "checkpoints/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth",
    },
    {
        "name": "yolox_l", "group": "B",
        "config": "/workspace/evasion-venv/lib/python3.10/site-packages/mmdet/.mim/configs/yolox/yolox_l_8xb8-300e_coco.py",
        "ckpt":   "checkpoints/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth",
    },
    # Group C — Cross-family (Swin ViT backbone)
    {
        "name": "mask_rcnn_swin_t", "group": "C",
        "config": "/workspace/evasion-venv/lib/python3.10/site-packages/mmdet/.mim/configs/swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py",
        "ckpt":   "checkpoints/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_20210902_120937-9d6b7cfa.pth",
    },
    {
        "name": "dino_swin_l", "group": "C",
        "config": "/workspace/evasion-venv/lib/python3.10/site-packages/mmdet/.mim/configs/dino/dino-5scale_swin-l_8xb2-12e_coco.py",
        "ckpt":   "checkpoints/dino-5scale_swin-l_8xb2-12e_coco_20230228_072924-a654145f.pth",
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


def label_to_cat_id(model, coco_gt: COCO) -> list[int]:
    mapping = []
    for cls in model.dataset_meta["classes"]:
        ids = coco_gt.getCatIds(catNms=[cls])
        if not ids:
            ids = coco_gt.getCatIds(catNms=[cls.replace("_", " ")])
        mapping.append(ids[0])
    return mapping


def model_role(mname: str, targets: dict) -> str:
    if mname == "surrogate":
        return "WB"
    info = targets.get(mname)
    return f"G{info['group']}" if info else "??"


def preds_to_coco(preds, img_id: int, cat_ids: list[int]) -> list[dict]:
    keep   = preds.scores >= COCO_SCORE_TH
    boxes  = preds.bboxes[keep].cpu().numpy()
    scores = preds.scores[keep].cpu().numpy()
    labels = preds.labels[keep].cpu().numpy()
    out = []
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        out.append({
            "image_id":    img_id,
            "category_id": cat_ids[int(label)],
            "bbox":        [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            "score":       float(score),
        })
    return out


def compute_coco_ap(coco_gt: COCO, results: list[dict], image_ids: list[int]) -> dict:
    if not results:
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0}
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(results)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.params.imgIds = list(image_ids)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return {"AP": float(ev.stats[0]), "AP50": float(ev.stats[1]), "AP75": float(ev.stats[2])}


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
    h_orig, w_orig = img_orig.shape[:2]
    h_res,  w_res  = img_bgr_resized.shape[:2]
    if (h_orig, w_orig) != (h_res, w_res):
        delta = cv2.resize(delta, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    return np.clip(img_orig.astype(np.float32) + delta, 0, 255).astype(np.uint8)


def _to_device_tensors(bx, lb, sc):
    return (
        torch.tensor(bx, dtype=torch.float32, device=DEVICE),
        torch.tensor(lb, dtype=torch.long,    device=DEVICE),
        torch.tensor(sc, dtype=torch.float32, device=DEVICE),
    )


# ── Evaluation passes ─────────────────────────────────────────────────────────

def precompute_clean(
    surrogate,
    sur_cat_ids: list[int],
    targets: dict,
    image_ids: list[int],
    id2file: dict,
    img_dir: Path,
    bar_fmt: str,
) -> tuple[dict, dict]:
    """Run inference on all clean images; return (clean_cache, clean_coco)."""
    clean_cache: dict[int, dict] = {}
    clean_coco:  dict[str, list] = {"surrogate": [], **{n: [] for n in targets}}

    for img_id in tqdm(image_ids, desc="  clean", unit="img",
                       bar_format=bar_fmt, file=sys.stdout, dynamic_ncols=True):
        img_orig = cv2.imread(str(img_dir / id2file[img_id]))
        clean_cache[img_id] = {}

        p = inference_detector(surrogate, img_orig).pred_instances
        clean_cache[img_id]["surrogate"] = (
            p.bboxes.cpu().numpy(), p.labels.cpu().numpy(), p.scores.cpu().numpy()
        )
        clean_coco["surrogate"].extend(preds_to_coco(p, img_id, sur_cat_ids))

        for tname, tinfo in targets.items():
            p = inference_detector(tinfo["model"], img_orig).pred_instances
            clean_cache[img_id][tname] = (
                p.bboxes.cpu().numpy(), p.labels.cpu().numpy(), p.scores.cpu().numpy()
            )
            clean_coco[tname].extend(preds_to_coco(p, img_id, tinfo["cat_ids"]))

    return clean_cache, clean_coco


def eval_one_image(
    img_id: int,
    img_adv: np.ndarray,
    surrogate,
    sur_cat_ids: list[int],
    sur_cat_map: dict,
    targets: dict,
    clean_cache: dict,
    coco_gt: COCO,
) -> tuple[float | None, dict[str, float], dict[str, list]]:
    """Evaluate WB and TRF ASR for one adversarial image.

    Returns (wb_asr, trf_asr_per_target, adv_coco_records).
    Returns (None, ...) if surrogate has no GT for this image.
    """
    gt_sur, lbl_sur = gt_for_image(coco_gt, img_id, sur_cat_map)
    if gt_sur is None:
        return None, {}, {}

    # White-box
    bx, lb, sc = clean_cache[img_id]["surrogate"]
    mc = count_matched_gt(gt_sur, lbl_sur, *_to_device_tensors(bx, lb, sc))
    p_sur = inference_detector(surrogate, img_adv).pred_instances
    ma = count_matched_gt(gt_sur, lbl_sur, p_sur.bboxes, p_sur.labels, p_sur.scores)
    wb = max(0.0, (mc - ma) / max(mc, 1))

    adv_records: dict[str, list] = {"surrogate": preds_to_coco(p_sur, img_id, sur_cat_ids)}
    trf_asr: dict[str, float] = {}

    for tname, tinfo in targets.items():
        gt_t, lbl_t = gt_for_image(coco_gt, img_id, tinfo["cat"])
        if gt_t is None:
            continue
        bx, lb, sc = clean_cache[img_id][tname]
        nc = count_matched_gt(gt_t, lbl_t, *_to_device_tensors(bx, lb, sc))
        p_t = inference_detector(tinfo["model"], img_adv).pred_instances
        na = count_matched_gt(gt_t, lbl_t, p_t.bboxes, p_t.labels, p_t.scores)
        trf_asr[tname]    = max(0.0, (nc - na) / max(nc, 1))
        adv_records[tname] = preds_to_coco(p_t, img_id, tinfo["cat_ids"])

    return wb, trf_asr, adv_records


def run_attack_loop(
    surrogate,
    aux_surrogate,
    targets: dict,
    image_ids: list[int],
    id2file: dict,
    img_dir: Path,
    coco_gt: COCO,
    sur_cat_ids: list[int],
    sur_cat_map: dict,
    clean_cache: dict,
    cfg: AttackConfig,
    bar_fmt: str,
) -> tuple[list[float], dict[str, list], dict[str, list], float]:
    """Attack all images and accumulate per-image ASR and COCO records.

    Returns (wb_list, trf_lists, adv_coco, elapsed_seconds).
    """
    wb_list:   list[float]     = []
    trf_lists: dict[str, list] = {n: [] for n in targets}
    adv_coco:  dict[str, list] = {"surrogate": [], **{n: [] for n in targets}}
    t_start = time.perf_counter()

    pbar = tqdm(image_ids, desc="  attack", unit="img",
                bar_format=bar_fmt, file=sys.stdout, dynamic_ncols=True)

    for img_id in pbar:
        img_path = img_dir / id2file[img_id]
        img_orig = cv2.imread(str(img_path))
        img_bgr  = load_image_bgr(img_path)

        img_adv_r = pgd_attack(surrogate, img_bgr, cfg, aux_model=aux_surrogate)
        img_adv   = adversarial_at_orig_scale(img_adv_r, img_bgr, img_orig)

        wb, trf_asr, adv_records = eval_one_image(
            img_id, img_adv, surrogate, sur_cat_ids, sur_cat_map,
            targets, clean_cache, coco_gt,
        )
        if wb is None:
            pbar.write(f"  skip {id2file[img_id]} (no GT)")
            continue

        wb_list.append(wb)
        for tname, v in trf_asr.items():
            trf_lists[tname].append(v)
        for mname, recs in adv_records.items():
            adv_coco[mname].extend(recs)

        mwb      = float(np.mean(wb_list))
        trf_avgs = {k: float(np.mean(v)) if v else 0.0 for k, v in trf_lists.items()}
        trf_str  = "  ".join(f"{k}={trf_asr.get(k, 0):.2f}" for k in targets)
        avg_str  = "  ".join(f"{k}={v:.3f}" for k, v in trf_avgs.items())
        pbar.write(f"  {id2file[img_id]}  wb={wb:.2f}  {trf_str}  | avg WB={mwb:.3f}  {avg_str}")

    return wb_list, trf_lists, adv_coco, time.perf_counter() - t_start


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="RaPA-OD: craft adversarial images and evaluate transfer")

    p.add_argument("--manifest",   type=Path, default=Path("data/manifests/dev_300.json"))
    p.add_argument("--coco-root",  type=Path, default=Path("data/coco"))
    p.add_argument("--n-images",   type=int,  default=300)
    p.add_argument("--sur-config", type=Path, default=Path(SURROGATE_CONFIG))
    p.add_argument("--sur-ckpt",   type=Path, default=Path(SURROGATE_CKPT))
    p.add_argument("--aux-config", type=Path, default=None, help="Aux surrogate config (E3c)")
    p.add_argument("--aux-ckpt",   type=Path, default=None, help="Aux surrogate checkpoint (E3c)")

    p.add_argument("--epsilon",    type=float, default=8.0,  help="L_inf budget in pixels")
    p.add_argument("--n-iters",    type=int,   default=40)
    p.add_argument("--step-size",  type=float, default=2.0,  help="PGD step size in pixels")
    p.add_argument("--momentum",   type=float, default=0.9)
    p.add_argument("--loss",       choices=["osfd", "rpn"], default="osfd")
    p.add_argument("--k",          type=float, default=3.0,
                   help="OSFD amplification factor k (default 3.0 per paper)")
    p.add_argument("--n-masks",    type=int,   default=1,
                   help="Masks per iteration (S in RaPA). 1 = no diversity.")
    p.add_argument("--rate",       type=float, default=0.0,
                   help="Pruning rate in [0, 1). 0 = no pruning.")
    p.add_argument("--prune-scope", type=str,  default="backbone",
                   help="Module prefix to prune. 'none' disables pruning.")
    p.add_argument("--prune-types", nargs="+", default=["norm"],
                   choices=list(_PRUNE_TYPE_MAP),
                   help="Layer types: norm=BN+LN, linear=Linear, conv=Conv2d")
    p.add_argument("--out",        type=Path,  default=Path("results/attack_result.json"))
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

    prune_scope = None if args.prune_scope == "none" else args.prune_scope
    prune_types = [_PRUNE_TYPE_MAP[t] for t in args.prune_types] if args.rate > 0 else None
    cfg = AttackConfig(
        epsilon_px=args.epsilon, n_iters=args.n_iters, step_size_px=args.step_size,
        momentum=args.momentum, loss_type=args.loss, osfd_k=args.k,
        n_masks=args.n_masks, pruning_scope=prune_scope,
        pruning_rate=args.rate, pruning_types=prune_types, device=DEVICE,
    )

    print("Loading surrogate...")
    surrogate   = init_detector(str(args.sur_config), str(args.sur_ckpt), device=DEVICE)
    surrogate.eval()
    sur_cat_map = build_cat_map(surrogate)
    sur_cat_ids = label_to_cat_id(surrogate, coco_gt)

    aux_surrogate = None
    if args.aux_config and args.aux_ckpt:
        print("Loading aux surrogate (E3c)...")
        aux_surrogate = init_detector(str(args.aux_config), str(args.aux_ckpt), device=DEVICE)
        aux_surrogate.eval()
        print(f"  aux: {args.aux_config.stem}")

    print("Loading targets...")
    targets: dict[str, dict] = {}
    for t in TARGETS:
        m = init_detector(t["config"], t["ckpt"], device=DEVICE)
        m.eval()
        targets[t["name"]] = {
            "model":   m,
            "group":   t["group"],
            "cat":     build_cat_map(m),
            "cat_ids": label_to_cat_id(m, coco_gt),
        }
        print(f"  [Group {t['group']}] {t['name']}")

    print(f"\nSurrogate : {args.sur_config.stem}")
    print(f"Images    : {len(image_ids)}")
    print(f"Loss      : {cfg.loss_type}  k={cfg.osfd_k}")
    print(f"Pruning   : scope={cfg.pruning_scope}  rate={cfg.pruning_rate}"
          f"  types={cfg.pruning_types}  n_masks={cfg.n_masks}")
    print(f"Budget    : eps={cfg.epsilon_px}px  iters={cfg.n_iters}  step={cfg.step_size_px}px")

    bar_fmt = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"

    # ── Clean predictions ─────────────────────────────────────────────────────
    n_models = 1 + len(targets)
    print(f"\n{'═'*76}")
    print(f"  Pre-computing clean predictions ({n_models} models x {len(image_ids)} images)...")
    print(f"{'═'*76}")

    clean_cache, clean_coco = precompute_clean(
        surrogate, sur_cat_ids, targets, image_ids, id2file, img_dir, bar_fmt
    )

    print("\nClean mAP:")
    clean_map: dict[str, dict] = {}
    for mname, results in clean_coco.items():
        ap = compute_coco_ap(coco_gt, results, image_ids)
        clean_map[mname] = ap
        role = model_role(mname, targets)
        print(f"  [{role}] {mname:<24} AP={ap['AP']:.4f}  AP50={ap['AP50']:.4f}  AP75={ap['AP75']:.4f}")

    # ── Attack loop ───────────────────────────────────────────────────────────
    print(f"\n{'═'*76}")
    print(f"  Attacking {len(image_ids)} images...")
    print(f"{'═'*76}")

    wb_list, trf_lists, adv_coco, elapsed = run_attack_loop(
        surrogate, aux_surrogate, targets, image_ids, id2file, img_dir,
        coco_gt, sur_cat_ids, sur_cat_map, clean_cache, cfg, bar_fmt,
    )
    mwb        = float(np.mean(wb_list)) if wb_list else 0.0
    trf_finals = {k: float(np.mean(v)) if v else 0.0 for k, v in trf_lists.items()}

    # ── Adversarial mAP ───────────────────────────────────────────────────────
    print("\nComputing adversarial mAP...")
    adv_map: dict[str, dict] = {}
    for mname, results in adv_coco.items():
        ap = compute_coco_ap(coco_gt, results, image_ids)
        adv_map[mname] = ap
        c_ap = clean_map[mname]["AP"]
        role = model_role(mname, targets)
        print(f"  [{role}] {mname:<24} clean={c_ap:.4f} -> adv={ap['AP']:.4f}"
              f"  AP50={ap['AP50']:.4f}  drop={c_ap - ap['AP']:.4f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    tgt_names = list(targets.keys())
    header    = f"  {'':30}  {'WB':>6}  " + "  ".join(f"{n:>18}" for n in tgt_names)
    print(f"\n{'═'*max(len(header), 76)}")
    print("  ASR — Object Disappearance Rate (higher = attack more effective)")
    print(header)
    row = f"  {'this run':<30}  {mwb:>6.3f}  " + "  ".join(
        f"{trf_finals.get(n, 0):>18.3f}" for n in tgt_names
    )
    print(row)
    print(f"{'═'*max(len(header), 76)}")

    trf_summary = "  ".join(f"TRF[{n}]={trf_finals.get(n, 0):.3f}" for n in tgt_names)
    print(f"\n  {len(wb_list)} images  {elapsed:.0f}s  WB-ASR={mwb:.3f}  {trf_summary}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    output = {
        "meta": {
            "n_images":      len(image_ids),
            "loss":          cfg.loss_type,
            "osfd_k":        cfg.osfd_k,
            "epsilon_px":    cfg.epsilon_px,
            "n_iters":       cfg.n_iters,
            "step_size_px":  cfg.step_size_px,
            "momentum":      cfg.momentum,
            "n_masks":       cfg.n_masks,
            "pruning_scope": cfg.pruning_scope,
            "pruning_rate":  cfg.pruning_rate,
            "pruning_types": cfg.pruning_types,
            "aux_model":     str(args.aux_config) if args.aux_config else None,
        },
        "clean_mAP":    clean_map,
        "adv_mAP":      adv_map,
        "wb_asr":       mwb,
        "trf_asr":      trf_finals,
        "wb_per_image": wb_list,
    }
    args.out.write_text(json.dumps(output, indent=2))
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
