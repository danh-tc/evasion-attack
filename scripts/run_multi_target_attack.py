#!/usr/bin/env python3
"""Multi-target feasibility: craft on Faster R-CNN, evaluate on 4 target architectures.

Reports both ASR (per-object disappearance) and mAP (COCO standard) for each config.

Usage:
    python scripts/run_multi_target_attack.py \
        --n-images 300 --n-iters 40 --epsilon 8 --step-size 2 \
        --out results/multi_target/results_dev300.json \
        2>&1 | tee results/multi_target/run_dev300.log
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

from assignment_stable_od.attack import load_image_bgr, pgd_attack

DEVICE        = "cuda:0"
SCORE_THRESH  = 0.3    # for ASR matching
MATCH_IOU     = 0.5
COCO_SCORE_TH = 0.05   # low threshold for COCO mAP collection

CONFIGS = [
    dict(name="pgd_baseline",      label="PGD baseline",
         n_masks=1, scope=None,       rate=0.0,  cross_backbone=False, use_osfd=False),
    dict(name="rapa_backbone_005", label="RaPA-OD backbone p=0.05",
         n_masks=2, scope="backbone", rate=0.05, cross_backbone=False, use_osfd=False),
    dict(name="rapa_osfd_005",     label="RaPA+OSFD backbone p=0.05",
         n_masks=2, scope="backbone", rate=0.05, cross_backbone=False, use_osfd=True),
    dict(name="rapa_cb_005",       label="RaPA-CB (R50+Swin) p=0.05",
         n_masks=1, scope="backbone", rate=0.05, cross_backbone=True,  use_osfd=False),
]

# Auxiliary surrogate for cross-backbone Direction A (loaded lazily in main)
AUX_SURROGATE = dict(
    config="/workspace/evasion-venv/lib/python3.10/site-packages/mmdet/.mim/configs/swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py",
    ckpt="checkpoints/mask_rcnn_swin-t_1x_coco_20210902_120937-9d6b7cfa.pth",
)

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
    dict(name="yolov3_d53",
         config="/workspace/evasion-venv/lib/python3.10/site-packages/mmdet/.mim/configs/yolo/yolov3_d53_8xb8-ms-608-273e_coco.py",
         ckpt="checkpoints/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth"),
    dict(name="yolox_l",
         config="/workspace/evasion-venv/lib/python3.10/site-packages/mmdet/.mim/configs/yolox/yolox_l_8xb8-300e_coco.py",
         ckpt="checkpoints/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth"),
    dict(name="mask_rcnn_swin_t",
         config="/workspace/evasion-venv/lib/python3.10/site-packages/mmdet/.mim/configs/swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py",
         ckpt="checkpoints/mask_rcnn_swin-t_1x_coco_20210902_120937-9d6b7cfa.pth"),
    dict(name="dino_r50",
         config="/workspace/evasion-venv/lib/python3.10/site-packages/mmdet/.mim/configs/dino/dino-4scale_r50_8xb2-12e_coco.py",
         ckpt="checkpoints/dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth"),
    dict(name="dino_swin_l",
         config="/workspace/evasion-venv/lib/python3.10/site-packages/mmdet/.mim/configs/dino/dino-5scale_swin-l_8xb2-12e_coco.py",
         ckpt="checkpoints/dino-5scale_swin-l_8xb2-12e_coco_20230228_072924-a654145f.pth"),
]


# ── helpers ───────────────────────────────────────────────────────────────────

def label_to_cat_id(model, coco_gt: COCO) -> list[int]:
    classes = model.dataset_meta["classes"]
    mapping = []
    for cls in classes:
        ids = coco_gt.getCatIds(catNms=[cls])
        if not ids:
            ids = coco_gt.getCatIds(catNms=[cls.replace("_", " ")])
        mapping.append(ids[0])
    return mapping


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
    return {
        "AP":   float(ev.stats[0]),
        "AP50": float(ev.stats[1]),
        "AP75": float(ev.stats[2]),
    }


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
    p.add_argument("--manifest",   type=Path, default=Path("data/manifests/dev_300.json"))
    p.add_argument("--coco-root",  type=Path, default=Path("data/coco"))
    p.add_argument("--sur-config", type=Path,
                   default=Path("checkpoints/faster-rcnn_r50_fpn_1x_coco.py"))
    p.add_argument("--sur-ckpt",   type=Path,
                   default=Path("checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"))
    p.add_argument("--n-images",   type=int,   default=300)
    p.add_argument("--epsilon",    type=float, default=8.0)
    p.add_argument("--n-iters",    type=int,   default=40)
    p.add_argument("--step-size",  type=float, default=2.0)
    p.add_argument("--momentum",   type=float, default=0.9)
    p.add_argument("--out",        type=Path,
                   default=Path("results/multi_target/results_dev300.json"))
    return p.parse_args()


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
    surrogate = init_detector(str(args.sur_config), str(args.sur_ckpt), device=DEVICE)
    surrogate.eval()
    sur_cat_map = build_cat_map(surrogate, coco_gt)
    sur_cat_ids = label_to_cat_id(surrogate, coco_gt)

    # Load aux surrogate if any config needs cross-backbone
    aux_surrogate = None
    if any(cfg.get("cross_backbone") for cfg in CONFIGS):
        print("Loading aux surrogate (cross-backbone)...")
        aux_surrogate = init_detector(AUX_SURROGATE["config"], AUX_SURROGATE["ckpt"], device=DEVICE)
        aux_surrogate.eval()
        print(f"  ✓ aux: {AUX_SURROGATE['config'].split('/')[-1]}")

    print("Loading targets...")
    targets = {}
    for t in TARGETS:
        m = init_detector(t["config"], t["ckpt"], device=DEVICE)
        m.eval()
        targets[t["name"]] = dict(
            model=m,
            cat=build_cat_map(m, coco_gt),
            cat_ids=label_to_cat_id(m, coco_gt),
        )
        print(f"  ✓ {t['name']}")

    print(f"\nSurrogate : {args.sur_config.stem}")
    print(f"Targets   : {', '.join(targets)}")
    print(f"Images    : {len(image_ids)}")
    print(f"ε={args.epsilon}px  iters={args.n_iters}  step={args.step_size}px")

    bar_fmt = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"

    # ── Pre-compute clean predictions (once, reused across both configs) ───────
    print(f"\n{'═'*76}")
    print(f"  Pre-computing clean predictions (5 models × {len(image_ids)} images)...")
    print(f"{'═'*76}")

    clean_cache: dict[int, dict[str, tuple]] = {}   # img_id → model → (bx, lb, sc) on CPU
    clean_coco:  dict[str, list] = {"surrogate": [], **{n: [] for n in targets}}

    for img_id in tqdm(image_ids, desc="  clean", unit="img",
                       bar_format=bar_fmt, file=sys.stdout, dynamic_ncols=True):
        img_path = img_dir / id2file[img_id]
        img_orig = cv2.imread(str(img_path))
        clean_cache[img_id] = {}

        p = preds_from_bgr(surrogate, img_orig)
        clean_cache[img_id]["surrogate"] = (
            p.bboxes.cpu().numpy(), p.labels.cpu().numpy(), p.scores.cpu().numpy()
        )
        clean_coco["surrogate"].extend(preds_to_coco(p, img_id, sur_cat_ids))

        for tname, tinfo in targets.items():
            p = preds_from_bgr(tinfo["model"], img_orig)
            clean_cache[img_id][tname] = (
                p.bboxes.cpu().numpy(), p.labels.cpu().numpy(), p.scores.cpu().numpy()
            )
            clean_coco[tname].extend(preds_to_coco(p, img_id, tinfo["cat_ids"]))

    print("\nClean mAP:")
    clean_map: dict[str, dict] = {}
    for mname, results in clean_coco.items():
        ap = compute_coco_ap(coco_gt, results, image_ids)
        clean_map[mname] = ap
        role = "WB " if mname == "surrogate" else "TRF"
        print(f"  [{role}] {mname:<22} AP={ap['AP']:.4f}  AP50={ap['AP50']:.4f}  AP75={ap['AP75']:.4f}")

    # ── Per-config attack loop ─────────────────────────────────────────────────
    all_results: dict = {
        "meta": {
            "n_images": len(image_ids), "n_iters": args.n_iters,
            "epsilon_px": args.epsilon, "step_size_px": args.step_size,
        },
        "clean_mAP": clean_map,
    }

    for cfg_idx, cfg in enumerate(CONFIGS):
        label = cfg["label"]
        print(f"\n{'═'*76}")
        print(f"  [{cfg_idx+1}/{len(CONFIGS)}] {label}")
        print(f"{'═'*76}")

        wb_list   = []
        trf_lists = {n: [] for n in targets}
        adv_coco  = {"surrogate": [], **{n: [] for n in targets}}
        t_cfg     = time.perf_counter()

        pbar = tqdm(image_ids, desc="  attack", unit="img",
                    bar_format=bar_fmt, file=sys.stdout, dynamic_ncols=True)

        for img_id in pbar:
            img_path = img_dir / id2file[img_id]
            img_orig = cv2.imread(str(img_path))
            img_bgr  = load_image_bgr(img_path)

            gt_sur, lbl_sur = gt_for_image(coco_gt, img_id, sur_cat_map)
            if gt_sur is None:
                pbar.write(f"  skip {id2file[img_id]} (no GT)")
                continue

            tgt_gts = {
                tname: gt_for_image(coco_gt, img_id, tinfo["cat"])
                for tname, tinfo in targets.items()
            }

            # ── Attack ────────────────────────────────────────────────────────
            img_adv_resized = pgd_attack(
                surrogate, img_bgr,
                epsilon_px=args.epsilon, n_iters=args.n_iters,
                step_size_px=args.step_size, n_masks=cfg["n_masks"],
                pruning_scope=cfg["scope"], pruning_rate=cfg["rate"],
                momentum=args.momentum, device=DEVICE,
                aux_model=aux_surrogate if cfg.get("cross_backbone") else None,
                use_osfd=cfg.get("use_osfd", False),
            )
            img_adv = adversarial_at_orig_scale(img_adv_resized, img_bgr, img_orig)

            # ── White-box ASR (cached clean, new adv) ─────────────────────────
            bx, lb, sc = clean_cache[img_id]["surrogate"]
            mc = count_matched_gt(
                gt_sur, lbl_sur,
                torch.tensor(bx, dtype=torch.float32, device=DEVICE),
                torch.tensor(lb, dtype=torch.long,    device=DEVICE),
                torch.tensor(sc, dtype=torch.float32, device=DEVICE),
            )
            p_adv_sur = preds_from_bgr(surrogate, img_adv)
            ma = count_matched_gt(gt_sur, lbl_sur,
                                  p_adv_sur.bboxes, p_adv_sur.labels, p_adv_sur.scores)
            wb = max(0.0, (mc - ma) / max(mc, 1))
            wb_list.append(wb)
            adv_coco["surrogate"].extend(preds_to_coco(p_adv_sur, img_id, sur_cat_ids))

            # ── Transfer ASR + COCO ────────────────────────────────────────────
            trf_parts = []
            for tname, tinfo in targets.items():
                gt_t, lbl_t = tgt_gts[tname]
                if gt_t is None:
                    trf_parts.append(f"{tname}=N/A")
                    continue
                bx, lb, sc = clean_cache[img_id][tname]
                nc = count_matched_gt(
                    gt_t, lbl_t,
                    torch.tensor(bx, dtype=torch.float32, device=DEVICE),
                    torch.tensor(lb, dtype=torch.long,    device=DEVICE),
                    torch.tensor(sc, dtype=torch.float32, device=DEVICE),
                )
                p_adv_t = preds_from_bgr(tinfo["model"], img_adv)
                na = count_matched_gt(gt_t, lbl_t,
                                      p_adv_t.bboxes, p_adv_t.labels, p_adv_t.scores)
                trf = max(0.0, (nc - na) / max(nc, 1))
                trf_lists[tname].append(trf)
                trf_parts.append(f"{tname}={trf:.2f}")
                adv_coco[tname].extend(preds_to_coco(p_adv_t, img_id, tinfo["cat_ids"]))

            mwb      = float(np.mean(wb_list))
            trf_avgs = {k: float(np.mean(v)) if v else 0.0 for k, v in trf_lists.items()}
            pbar.write(
                f"  {id2file[img_id]}  wb={wb:.2f}  " + "  ".join(trf_parts) +
                f"  | avg WB={mwb:.3f}  " +
                "  ".join(f"{k}={v:.3f}" for k, v in trf_avgs.items())
            )

        elapsed    = time.perf_counter() - t_cfg
        mwb        = float(np.mean(wb_list)) if wb_list else 0.0
        trf_finals = {k: float(np.mean(v)) if v else 0.0 for k, v in trf_lists.items()}

        # ── mAP after attack ──────────────────────────────────────────────────
        print(f"\nComputing adv mAP [{label}]...")
        adv_map: dict[str, dict] = {}
        for mname, results in adv_coco.items():
            ap = compute_coco_ap(coco_gt, results, image_ids)
            adv_map[mname] = ap
            c_ap = clean_map[mname]["AP"]
            role = "WB " if mname == "surrogate" else "TRF"
            print(f"  [{role}] {mname:<22} clean={c_ap:.4f} → adv={ap['AP']:.4f} "
                  f" AP50={ap['AP50']:.4f}  drop={c_ap - ap['AP']:.4f}")

        print(f"{'─'*76}")
        print(f"  FINAL  WB-ASR={mwb:.3f}  " +
              "  ".join(f"TRF-{k}={v:.3f}" for k, v in trf_finals.items()) +
              f"  ({len(wb_list)} images, {elapsed:.0f}s)")

        all_results[cfg["name"]] = dict(
            label=label, wb_asr=mwb, trf_asr=trf_finals,
            wb_mAP=adv_map.get("surrogate", {}),
            trf_mAP={k: adv_map.get(k, {}) for k in targets},
            wb_per_image=wb_list,
        )

    # ── Summary tables ────────────────────────────────────────────────────────
    tgt_names = [t["name"] for t in TARGETS]
    col_w  = 36
    header = f"  {'Config':<{col_w}}  {'WB':>6}  " + "  ".join(f"{n:>16}" for n in tgt_names)

    print(f"\n{'═'*len(header)}")
    print("  ASR — Object Disappearance Rate (↑ better)")
    print(header)
    print(f"{'─'*len(header)}")
    for cfg in CONFIGS:
        r   = all_results[cfg["name"]]
        row = f"  {cfg['label']:<{col_w}}  {r['wb_asr']:>6.3f}  "
        row += "  ".join(f"{r['trf_asr'].get(n, 0):>16.3f}" for n in tgt_names)
        print(row)
    print(f"{'═'*len(header)}")

    print(f"\n  mAP after attack — COCO AP (↓ better attack)")
    clean_line = "  Clean AP:  " + "  ".join(
        f"{n}={clean_map.get(n, {}).get('AP', 0):.4f}"
        for n in ["surrogate"] + tgt_names
    )
    print(clean_line)
    print(f"{'─'*len(header)}")
    for cfg in CONFIGS:
        r    = all_results[cfg["name"]]
        wb_ap = r["wb_mAP"].get("AP", 0.0)
        row  = f"  {cfg['label']:<{col_w}}  {wb_ap:>6.4f}  "
        row += "  ".join(f"{r['trf_mAP'].get(n, {}).get('AP', 0):>16.4f}" for n in tgt_names)
        print(row)
    print(f"{'═'*len(header)}")

    args.out.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
