#!/usr/bin/env python3
"""Run one MMDetection inference and report peak allocated VRAM."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from mmdet.apis import inference_detector, init_detector
from mmdet.utils import register_all_modules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--score-threshold", type=float, default=0.3)
    return parser.parse_args()


def find_one(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no file matching {pattern!r} in {directory}")
    return matches[0]


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    checkpoint_dir = root / "checkpoints"
    config = args.config or find_one(checkpoint_dir, "faster*rcnn*.py")
    checkpoint = args.checkpoint or find_one(checkpoint_dir, "faster*rcnn*.pth")
    if not args.image.is_file():
        raise FileNotFoundError(args.image)

    register_all_modules()
    torch.cuda.reset_peak_memory_stats()
    model = init_detector(str(config), str(checkpoint), device="cuda:0")
    result = inference_detector(model, str(args.image))
    predictions = result.pred_instances
    detected = int((predictions.scores >= args.score_threshold).sum())
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    print(f"config: {config}")
    print(f"checkpoint: {checkpoint}")
    print(f"detections@{args.score_threshold}: {detected}")
    print(f"peak allocated VRAM: {peak_gib:.2f} GiB")


if __name__ == "__main__":
    main()

