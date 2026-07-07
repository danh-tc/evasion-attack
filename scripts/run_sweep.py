#!/usr/bin/env python
"""Sweep drop_prob x S (masks/iteration) against one or more targets, reusing
loaded models across the whole grid (unlike calling run_attack.py repeatedly,
which would reload every target model per cell).

This is the Stage 1 probe tool from PLAN_EXPERIMENTS.md: compare method A
(drop_prob=0, implicit baseline, always included) against a grid of
DropConnect configs to answer (1) does masking help, especially cross-family,
and (2) does S=1 land close to S=5.

Example (setup_env.sh's smoke test):
    python scripts/run_sweep.py --n-images 5 --rates 0.05 --masks 2 --n-iters 5 --out results/e0_smoke.json

Example (Stage 1 probe, method B vs C vs A baseline):
    python scripts/run_sweep.py --rates 0.05 --masks 1,5 --n-images 70 --seeds 0,1 \\
        --targets fcos_r50,yolox_l,mask_rcnn_swin_t --out results/stage1_probe.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="faster_rcnn_r50")
    p.add_argument("--targets", default="fcos_r50,yolox_l,mask_rcnn_swin_t")
    p.add_argument("--manifest", default="dev_300", choices=["dev_300", "val_100"])
    p.add_argument("--n-images", type=int, default=None)
    p.add_argument("--n-iters", type=int, default=40)
    p.add_argument("--epsilon", type=float, default=8.0)
    p.add_argument("--alpha", type=float, default=2.0)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--rates", default="0.05", help="comma-separated drop_prob values to sweep")
    p.add_argument("--masks", default="1", help="comma-separated S (masks/iteration) values to sweep")
    p.add_argument("--seeds", default="0", help="comma-separated seeds")
    p.add_argument("--skip-baseline", action="store_true", help="don't run the drop_prob=0 (method A) control")
    p.add_argument("--iou-thr", type=float, default=0.5)
    p.add_argument("--score-thr", type=float, default=0.3)
    p.add_argument("--checkpoint-dir", default=os.path.join(PROJECT_DIR, "checkpoints"))
    p.add_argument("--data-dir", default=os.path.join(PROJECT_DIR, "data", "coco"))
    p.add_argument("--manifest-dir", default=os.path.join(PROJECT_DIR, "data", "manifests"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", required=True)
    return p.parse_args()


def run_one_cell(surrogate, targets, dataloader, method, attack_cfg, n_images, iou_thr, score_thr, seed):
    import torch  # noqa: F401  (ensures torch initialized before CUDA calls below)

    from evasion_attack.attack.mi_fgsm import run_mi_fgsm_attack
    from evasion_attack.eval.metrics import (
        ASRAccumulator,
        CocoApAccumulator,
        build_category_id_map,
        img_tensor_to_bgr_uint8,
        run_target_inference,
    )

    asr_acc = {name: ASRAccumulator(iou_thr=iou_thr, score_thr=score_thr) for name in targets}
    ap_acc_adv = {
        name: CocoApAccumulator(coco_gt=dataloader.dataset.coco, category_ids=build_category_id_map(m, dataloader.dataset.coco))
        for name, m in targets.items()
    }
    ap_acc_clean = {
        name: CocoApAccumulator(coco_gt=dataloader.dataset.coco, category_ids=build_category_id_map(m, dataloader.dataset.coco))
        for name, m in targets.items()
    }

    n_images = n_images or len(dataloader.dataset)
    n_attacked = 0
    t_start = time.time()
    for idx, sample in enumerate(dataloader):
        if idx >= n_images:
            break
        if sample.gt_bboxes.shape[0] == 0:
            continue

        img = sample.img.unsqueeze(0).to(attack_cfg_device(surrogate))
        gt_bboxes = [sample.gt_bboxes.to(img.device)]

        cfg = attack_cfg.__class__(**{**vars(attack_cfg), "seed": seed})
        result = run_mi_fgsm_attack(surrogate, img, gt_bboxes, method, cfg)
        n_attacked += 1

        clean_bgr = img_tensor_to_bgr_uint8(img[0])
        adv_bgr = img_tensor_to_bgr_uint8((img + result.delta)[0])

        for name, target_model in targets.items():
            clean_result = run_target_inference(target_model, clean_bgr)
            adv_result = run_target_inference(target_model, adv_bgr)
            asr_acc[name].update(sample.gt_bboxes, sample.gt_labels, clean_result, adv_result)
            ap_acc_adv[name].update(sample.image_id, adv_result)
            ap_acc_clean[name].update(sample.image_id, clean_result)

    elapsed = time.time() - t_start
    per_target = {}
    for name in targets:
        clean_ap = ap_acc_clean[name].evaluate()
        adv_ap = ap_acc_adv[name].evaluate()
        per_target[name] = {
            "asr": asr_acc[name].asr,
            "n_baseline_detected": asr_acc[name].n_baseline_detected,
            "n_disappeared": asr_acc[name].n_disappeared,
            "clean_ap": clean_ap,
            "adv_ap": adv_ap,
            "delta_ap": clean_ap - adv_ap,
        }
    return {
        "n_images_attacked": n_attacked,
        "elapsed_seconds": elapsed,
        "seconds_per_image": elapsed / max(n_attacked, 1),
        "targets": per_target,
    }


def attack_cfg_device(model):
    return next(model.parameters()).device


def main():
    args = parse_args()

    from evasion_attack.attack.methods import MethodConfig
    from evasion_attack.attack.mi_fgsm import AttackConfig
    from evasion_attack.data.manifest import build_dataloader
    from evasion_attack.utils.model_zoo import load_model

    rates = [float(x) for x in args.rates.split(",")]
    masks = [int(x) for x in args.masks.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    target_names = args.targets.split(",")

    print(f"[run_sweep] source={args.source} targets={target_names} rates={rates} masks={masks} seeds={seeds}")

    print("[run_sweep] loading surrogate + targets...")
    surrogate = load_model(args.source, args.checkpoint_dir, device=args.device)
    targets = {name: load_model(name, args.checkpoint_dir, device=args.device) for name in target_names}

    ann_file = os.path.join(args.data_dir, "annotations", "instances_val2017.json")
    img_dir = os.path.join(args.data_dir, "val2017")
    manifest_path = os.path.join(args.manifest_dir, f"{args.manifest}.json")
    dataloader = build_dataloader(manifest_path, ann_file, img_dir)

    attack_cfg = AttackConfig(epsilon=args.epsilon, alpha=args.alpha, n_iters=args.n_iters, momentum=args.momentum)

    cells = []
    if not args.skip_baseline:
        cells.append(("A_baseline", MethodConfig("A_baseline", "drop_prob=0 control", drop_prob=0.0, s_masks=1)))
    for rate in rates:
        for s in masks:
            cell_id = f"rate{rate}_S{s}"
            cells.append((cell_id, MethodConfig(cell_id, f"drop_prob={rate}, S={s}", drop_prob=rate, s_masks=s)))

    results = {"config": vars(args), "cells": {}}
    for cell_id, method in cells:
        for seed in seeds:
            key = f"{cell_id}_seed{seed}"
            print(f"\n[run_sweep] === {key} ===")
            cell_result = run_one_cell(
                surrogate, targets, dataloader, method, attack_cfg,
                args.n_images, args.iou_thr, args.score_thr, seed,
            )
            results["cells"][key] = {
                "method": cell_id, "description": method.description,
                "drop_prob": method.drop_prob, "s_masks": method.s_masks, "seed": seed,
                **cell_result,
            }
            for name, m in cell_result["targets"].items():
                print(f"  {name}: ASR={m['asr']:.3f} dAP={m['delta_ap']:.3f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[run_sweep] wrote {args.out}")


if __name__ == "__main__":
    main()
