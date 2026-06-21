#!/usr/bin/env python3
"""Create fixed development and validation subsets from COCO val2017.

Outputs two JSON manifests that pin the exact image IDs used for all
experiments.  Manifests are written once and must not be regenerated after
any model or attack results have been inspected (selection bias).

Usage:
    python scripts/make_subsets.py
    python scripts/make_subsets.py --dev-size 500 --val-size 150
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ann",
        type=Path,
        default=Path("data/coco/annotations/instances_val2017.json"),
    )
    p.add_argument("--dev-size", type=int, default=300,
                   help="Number of images in the development subset")
    p.add_argument("--val-size", type=int, default=100,
                   help="Number of images in the validation subset (no overlap with dev)")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed — freeze this permanently")
    p.add_argument("--out-dir", type=Path, default=Path("data/manifests"))
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.ann.exists():
        raise SystemExit(
            f"Annotation file not found: {args.ann}\n"
            "Run scripts/download_coco.sh first."
        )

    with open(args.ann) as f:
        coco = json.load(f)

    # Only include images that have at least one instance annotation.
    annotated_ids: set[int] = {ann["image_id"] for ann in coco["annotations"]}
    images = [img for img in coco["images"] if img["id"] in annotated_ids]
    print(f"Total annotated images in val2017: {len(images)}")

    rng = random.Random(args.seed)
    pool = images.copy()
    rng.shuffle(pool)

    total = args.dev_size + args.val_size
    if total > len(pool):
        raise SystemExit(
            f"Requested {total} images but only {len(pool)} annotated images available."
        )

    dev = pool[: args.dev_size]
    val = pool[args.dev_size : args.dev_size + args.val_size]

    # Sanity check
    dev_ids = {img["id"] for img in dev}
    val_ids = {img["id"] for img in val}
    assert not (dev_ids & val_ids), "BUG: dev and val sets overlap"

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, subset in [("dev", dev), ("val", val)]:
        ids = sorted(img["id"] for img in subset)
        manifest = {
            "seed": args.seed,
            "split": split_name,
            "size": len(ids),
            "image_ids": ids,
        }
        path = args.out_dir / f"{split_name}_{len(ids)}.json"
        # Refuse to overwrite an existing manifest to prevent accidental re-sampling.
        if path.exists():
            print(f"[SKIP] {path} already exists — not overwriting")
            continue
        path.write_text(json.dumps(manifest, indent=2))
        print(f"{split_name}: {len(ids)} images → {path}")

    print("No overlap between dev and val: OK")


if __name__ == "__main__":
    main()
