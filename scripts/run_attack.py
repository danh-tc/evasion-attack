#!/usr/bin/env python
"""Run one method against one surrogate on N images, evaluate ASR + AP on
one or more target models. This is the unit of work `run_sweep.py` calls
repeatedly for Stage 1-4's method x target grid.

Example (per setup_env.sh's suggested smoke test):
    python scripts/run_attack.py --n-images 5 --n-iters 5 --out results/smoke.json

Example (one real probe cell):
    python scripts/run_attack.py --method B --targets fcos_r50,yolox_l,mask_rcnn_swin_t \\
        --n-images 70 --seed 0 --out results/stage1_B_seed0.json
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
    p.add_argument("--method", default="A", help="method id from METHOD_REGISTRY (A/B/C)")
    p.add_argument("--source", default="faster_rcnn_r50", help="surrogate model short name")
    p.add_argument("--targets", default="fcos_r50,yolox_l,mask_rcnn_swin_t",
                    help="comma-separated target model short names")
    p.add_argument("--manifest", default="dev_300", choices=["dev_300", "val_100"],
                    help="which data/manifests/*.json to use")
    p.add_argument("--n-images", type=int, default=None, help="cap number of images (default: all in manifest)")
    p.add_argument("--n-iters", type=int, default=40, help="attack iterations (overrides AttackConfig default)")
    p.add_argument("--epsilon", type=float, default=8.0)
    p.add_argument("--alpha", type=float, default=2.0)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iou-thr", type=float, default=0.5)
    p.add_argument("--score-thr", type=float, default=0.3)
    p.add_argument("--checkpoint-dir", default=os.path.join(PROJECT_DIR, "checkpoints"))
    p.add_argument("--data-dir", default=os.path.join(PROJECT_DIR, "data", "coco"))
    p.add_argument("--manifest-dir", default=os.path.join(PROJECT_DIR, "data", "manifests"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", required=True, help="output JSON path")
    return p.parse_args()


def main():
    args = parse_args()

    import torch

    from evasion_attack.attack.methods import get_method
    from evasion_attack.attack.mi_fgsm import AttackConfig, run_mi_fgsm_attack
    from evasion_attack.data.manifest import build_dataloader
    from evasion_attack.eval.metrics import (
        ASRAccumulator,
        CocoApAccumulator,
        build_category_id_map,
        img_tensor_to_bgr_uint8,
        run_target_inference,
    )
    from evasion_attack.utils.model_zoo import load_model

    method = get_method(args.method)
    attack_cfg = AttackConfig(epsilon=args.epsilon, alpha=args.alpha, n_iters=args.n_iters,
                               momentum=args.momentum, seed=args.seed)

    print(f"[run_attack] method={args.method} source={args.source} targets={args.targets} "
          f"manifest={args.manifest} n_images={args.n_images} seed={args.seed}")

    print("[run_attack] loading surrogate...")
    surrogate = load_model(args.source, args.checkpoint_dir, device=args.device)

    target_names = args.targets.split(",")
    print(f"[run_attack] loading {len(target_names)} target model(s)...")
    targets = {name: load_model(name, args.checkpoint_dir, device=args.device) for name in target_names}

    ann_file = os.path.join(args.data_dir, "annotations", "instances_val2017.json")
    img_dir = os.path.join(args.data_dir, "val2017")
    manifest_path = os.path.join(args.manifest_dir, f"{args.manifest}.json")
    dataloader = build_dataloader(manifest_path, ann_file, img_dir)

    asr_acc = {name: ASRAccumulator(iou_thr=args.iou_thr, score_thr=args.score_thr) for name in target_names}
    ap_acc = {
        name: CocoApAccumulator(coco_gt=dataloader.dataset.coco, category_ids=build_category_id_map(m, dataloader.dataset.coco))
        for name, m in targets.items()
    }
    ap_acc_clean = {
        name: CocoApAccumulator(coco_gt=dataloader.dataset.coco, category_ids=build_category_id_map(m, dataloader.dataset.coco))
        for name, m in targets.items()
    }

    n_images = args.n_images or len(dataloader.dataset)
    loss_first_last = []
    t_start = time.time()

    for idx, sample in enumerate(dataloader):
        if idx >= n_images:
            break
        if sample.gt_bboxes.shape[0] == 0:
            continue  # nothing to attack toward on this image

        img = sample.img.unsqueeze(0).to(args.device)
        gt_bboxes = [sample.gt_bboxes.to(args.device)]

        result = run_mi_fgsm_attack(surrogate, img, gt_bboxes, method, attack_cfg)
        loss_first_last.append((result.loss_history[0], result.loss_history[-1]))

        clean_bgr = img_tensor_to_bgr_uint8(img[0])
        adv_img = (img + result.delta)[0]
        adv_bgr = img_tensor_to_bgr_uint8(adv_img)

        for name, target_model in targets.items():
            clean_result = run_target_inference(target_model, clean_bgr)
            adv_result = run_target_inference(target_model, adv_bgr)
            asr_acc[name].update(sample.gt_bboxes, sample.gt_labels, clean_result, adv_result)
            ap_acc[name].update(sample.image_id, adv_result)
            ap_acc_clean[name].update(sample.image_id, clean_result)

        if (idx + 1) % 10 == 0:
            elapsed = time.time() - t_start
            print(f"[run_attack] {idx + 1}/{n_images} images, {elapsed:.1f}s elapsed "
                  f"({elapsed / (idx + 1):.2f}s/img)")

    elapsed_total = time.time() - t_start

    out = {
        "method": args.method,
        "method_description": method.description,
        "source": args.source,
        "manifest": args.manifest,
        "n_images_requested": n_images,
        "n_images_attacked": len(loss_first_last),
        "seed": args.seed,
        "attack_cfg": vars(attack_cfg),
        "elapsed_seconds": elapsed_total,
        "seconds_per_image": elapsed_total / max(len(loss_first_last), 1),
        "targets": {},
    }
    for name in target_names:
        clean_ap = ap_acc_clean[name].evaluate()
        adv_ap = ap_acc[name].evaluate()
        out["targets"][name] = {
            "asr": asr_acc[name].asr,
            "n_baseline_detected": asr_acc[name].n_baseline_detected,
            "n_disappeared": asr_acc[name].n_disappeared,
            "clean_ap": clean_ap,
            "adv_ap": adv_ap,
            "delta_ap": clean_ap - adv_ap,
        }
        print(f"[run_attack] target={name}: ASR={out['targets'][name]['asr']:.3f} "
              f"clean_AP={clean_ap:.3f} adv_AP={adv_ap:.3f} dAP={out['targets'][name]['delta_ap']:.3f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[run_attack] wrote {args.out}")


if __name__ == "__main__":
    main()
