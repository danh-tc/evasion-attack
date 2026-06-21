#!/usr/bin/env python3
"""Evaluate a pretrained MMDetection model on COCO val2017 (or a manifest subset).

Reports AP, AP50, AP75, AP_s/m/l and logs peak VRAM + runtime.
Results are saved to JSON for later comparison.

Usage:
    # Full val2017
    python scripts/eval_clean.py \\
        --config checkpoints/faster-rcnn_r50_fpn_1x_coco.py \\
        --checkpoint checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth

    # Development subset only
    python scripts/eval_clean.py \\
        --config checkpoints/faster-rcnn_r50_fpn_1x_coco.py \\
        --checkpoint checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \\
        --manifest data/manifests/dev_300.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from mmdet.apis import inference_detector, init_detector
from mmdet.utils import register_all_modules
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


STAT_NAMES = [
    "AP", "AP50", "AP75", "AP_s", "AP_m", "AP_l",
    "AR1", "AR10", "AR100", "AR_s", "AR_m", "AR_l",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--coco-root",
        type=Path,
        default=Path("data/coco"),
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="JSON manifest with image_ids field; omit to eval full val2017",
    )
    p.add_argument(
        "--score-threshold",
        type=float,
        default=0.05,
        help="Min score to include in COCO result file (COCO eval uses its own threshold)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write metrics JSON here (default: results/clean_eval/<config_stem>.json)",
    )
    return p.parse_args()


def label_to_cat_id(model: object, coco_gt: COCO) -> list[int]:
    """Map MMDetection 0-based label indices to COCO category IDs.

    MMDetection stores names with underscores ('traffic_light') while COCO
    annotations use spaces ('traffic light'); try both variants.
    """
    classes: tuple[str, ...] = model.dataset_meta["classes"]  # type: ignore[attr-defined]
    mapping: list[int] = []
    for cls in classes:
        ids = coco_gt.getCatIds(catNms=[cls])
        if not ids:
            ids = coco_gt.getCatIds(catNms=[cls.replace("_", " ")])
        if not ids:
            raise ValueError(f"Class {cls!r} not found in COCO annotations (tried with spaces too)")
        mapping.append(ids[0])
    return mapping


def main() -> None:
    args = parse_args()

    ann_file = args.coco_root / "annotations/instances_val2017.json"
    img_dir = args.coco_root / "val2017"

    if not ann_file.exists():
        raise SystemExit(f"Annotation file not found: {ann_file}\nRun scripts/download_coco.sh first.")
    if not img_dir.exists():
        raise SystemExit(f"Image directory not found: {img_dir}\nRun scripts/download_coco.sh first.")

    coco_gt = COCO(str(ann_file))

    if args.manifest:
        manifest = json.loads(args.manifest.read_text())
        image_ids: list[int] = manifest["image_ids"]
        subset_tag = f"manifest:{args.manifest.stem}"
    else:
        image_ids = sorted(coco_gt.getImgIds())
        subset_tag = "full_val2017"

    print(f"config    : {args.config}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"subset    : {subset_tag} ({len(image_ids)} images)")

    register_all_modules()
    torch.cuda.reset_peak_memory_stats()
    model = init_detector(str(args.config), str(args.checkpoint), device="cuda:0")
    cat_ids = label_to_cat_id(model, coco_gt)

    coco_results: list[dict] = []
    t0 = time.perf_counter()

    for i, img_id in enumerate(image_ids, 1):
        info = coco_gt.loadImgs(img_id)[0]
        img_path = img_dir / info["file_name"]

        result = inference_detector(model, str(img_path))
        preds = result.pred_instances

        keep = preds.scores >= args.score_threshold
        boxes = preds.bboxes[keep].cpu().numpy()
        scores = preds.scores[keep].cpu().numpy()
        labels = preds.labels[keep].cpu().numpy()

        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            coco_results.append({
                "image_id": img_id,
                "category_id": cat_ids[int(label)],
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score),
            })

        if i % 50 == 0 or i == len(image_ids):
            elapsed = time.perf_counter() - t0
            fps = i / elapsed
            print(f"  [{i:4d}/{len(image_ids)}] {elapsed:6.1f}s  {fps:.1f} img/s")

    elapsed_total = time.perf_counter() - t0
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    print(f"\npeak VRAM  : {peak_gib:.2f} GiB")
    print(f"total time : {elapsed_total:.1f}s  ({elapsed_total / len(image_ids):.2f}s/img)")

    if not coco_results:
        raise SystemExit("No predictions above score threshold — check model/config.")

    # COCO evaluation
    coco_dt = coco_gt.loadRes(coco_results)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.imgIds = image_ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    metrics = {k: float(v) for k, v in zip(STAT_NAMES, evaluator.stats)}

    print("\n── Metrics ──────────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:<8}: {v:.4f}")

    # Save
    out = args.out
    if out is None:
        out = Path("results/clean_eval") / f"{args.config.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "subset": subset_tag,
        "n_images": len(image_ids),
        "runtime_s": round(elapsed_total, 2),
        "peak_vram_gib": round(peak_gib, 3),
        "metrics": metrics,
    }
    out.write_text(json.dumps(record, indent=2))
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
