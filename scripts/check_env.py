#!/usr/bin/env python3
"""Fail-fast verification of the CUDA and OpenMMLab stack."""

import platform
import sys

import mmcv
import mmdet
import mmengine
import torch


def main() -> None:
    print(f"Python:       {sys.version.split()[0]}")
    print(f"Platform:     {platform.platform()}")
    print(f"PyTorch:      {torch.__version__}")
    print(f"Torch CUDA:   {torch.version.cuda}")
    print(f"MMCV:         {mmcv.__version__}")
    print(f"MMEngine:     {mmengine.__version__}")
    print(f"MMDetection:  {mmdet.__version__}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available to PyTorch")
    print(f"GPU:          {torch.cuda.get_device_name(0)}")
    print(f"VRAM:         {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")

    from mmcv.ops import nms

    boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11]], device="cuda", dtype=torch.float32)
    scores = torch.tensor([0.9, 0.8], device="cuda")
    _, keep = nms(boxes, scores, 0.5)
    if keep.numel() != 1:
        raise SystemExit("MMCV CUDA NMS returned an unexpected result")
    print("MMCV CUDA ops: OK")


if __name__ == "__main__":
    main()

