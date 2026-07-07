#!/usr/bin/env python
"""Environment + pipeline sanity check (Stage 0 exit criterion).

Run after setup_env.sh finishes. Checks, in order:
  1. torch/CUDA/mmcv/mmengine/mmdet versions and CUDA availability.
  2. The surrogate (faster_rcnn_r50) checkpoint loads via our model_zoo.
  3. DropConnectMasker finds >0 maskable (BatchNorm2d) layers in its backbone.
  4. A tiny end-to-end forward+backward through osfd_loss produces a finite,
     non-zero gradient w.r.t. a dummy perturbation — i.e. the whole
     attack graph (masking -> backbone -> OSFD loss -> autograd) is wired
     correctly before spending any GPU time on real images.

Exits non-zero on the first failure, printing which check failed.
"""
from __future__ import annotations

import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)


def check_versions():
    import torch
    print(f"  torch     : {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise SystemExit("[ERROR] PyTorch cannot access CUDA")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

    import mmcv
    import mmdet
    import mmengine
    print(f"  mmcv      : {mmcv.__version__}")
    print(f"  mmengine  : {mmengine.__version__}")
    print(f"  mmdet     : {mmdet.__version__}")


def check_surrogate_loads():
    from evasion_attack.utils.model_zoo import load_model
    checkpoint_dir = os.path.join(PROJECT_DIR, "checkpoints")
    model = load_model("faster_rcnn_r50", checkpoint_dir, device="cuda:0")
    print(f"  surrogate loaded: {type(model).__name__}, backbone: {type(model.backbone).__name__}")
    return model


def check_masking(model):
    from evasion_attack.hooks.dropconnect import DropConnectMasker
    masker = DropConnectMasker(model.backbone, drop_prob=0.05, type_list=("Normalization", "Linear"))
    n = masker.attach()
    print(f"  maskable layers found in backbone: {n}")
    if n == 0:
        masker.detach()
        raise SystemExit("[ERROR] 0 maskable layers — expected >0 BatchNorm2d in a ResNet-50 backbone")
    masker.resample()
    masker.detach()
    print("  DropConnectMasker: OK")


def check_attack_smoke(model):
    import torch

    from evasion_attack.attack.methods import get_method
    from evasion_attack.attack.mi_fgsm import AttackConfig, run_mi_fgsm_attack

    device = "cuda:0"
    img = torch.randint(0, 256, (3, 256, 256), dtype=torch.float32, device=device)
    gt_bboxes = [torch.tensor([[50.0, 50.0, 150.0, 150.0]], device=device)]

    method = get_method("B")
    attack_cfg = AttackConfig(epsilon=8.0, alpha=2.0, n_iters=3, momentum=0.9, seed=0)
    result = run_mi_fgsm_attack(model, img.unsqueeze(0), gt_bboxes, method, attack_cfg)

    assert result.delta.shape == img.unsqueeze(0).shape, "delta shape mismatch"
    assert torch.isfinite(result.delta).all(), "delta contains non-finite values"
    assert result.delta.abs().max() <= attack_cfg.epsilon + 1e-3, "epsilon budget violated"
    assert all(torch.isfinite(torch.tensor(l)) for l in result.loss_history), "non-finite loss encountered"
    print(f"  attack smoke test OK: {len(result.loss_history)} iters, "
          f"loss {result.loss_history[0]:.4f} -> {result.loss_history[-1]:.4f}, "
          f"{result.n_masked_layers} masked layers")


def main():
    print("===== Versions =====")
    check_versions()

    print("\n===== Surrogate load =====")
    model = check_surrogate_loads()

    print("\n===== DropConnect masking =====")
    check_masking(model)

    print("\n===== Attack smoke test (method B, 3 iters, random image) =====")
    check_attack_smoke(model)

    print("\n===== ALL CHECKS PASSED =====")


if __name__ == "__main__":
    main()
