# Experiment Plan — RaPA on Object Detection (COCO)

## Setup

**Dataset:** COCO val2017
- Dev set: `data/manifests/dev_300.json` (300 ảnh, seed=42)
- Held-out: `data/manifests/val_100.json` (100 ảnh — chỉ chạy 1 lần sau khi freeze config)

**Surrogate:** Faster R-CNN ResNet-50-FPN
- Config: `checkpoints/faster-rcnn_r50_fpn_1x_coco.py`
- Checkpoint: `checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth`

**Attack hyperparams (mặc định):** ε=8px, 40 iters, step=2px, momentum=0.9

**Metrics:** ASR (object disappearance rate) + ∆AP (COCO AP@[.5:.95] drop)

---

## Target Models (Cross-family)

| Group | Name | Backbone | Paradigm |
|---|---|---|---|
| **A — In-family (ResNet-50)** | fcos_r50 | ResNet-50 | anchor-free |
| **A — In-family (ResNet-50)** | deformable_detr | ResNet-50 | transformer |
| **B — Near-family (non-ResNet CNN)** | yolov3_d53 | Darknet-53 | anchor |
| **B — Near-family (non-ResNet CNN)** | yolox_l | CSPNet | anchor-free |
| **C — Cross-family (Swin ViT)** | mask_rcnn_swin_t | Swin-T | two-stage |
| **C — Cross-family (Swin ViT)** | dino_swin_l | Swin-L | full-transformer |

**Expected transfer difficulty:** A (easy) → B (medium) → C (hard)

---

## Nhóm 0 — Hyperparameter Sweep (E0)

### E0 — Rate × n_masks sweep
- Loss: OSFD (k=3), Norm pruning, scope=backbone
- Grid: `rate ∈ [0.01, 0.05, 0.10, 0.20, 0.50]` × `n_masks ∈ [1, 2, 3, 5]`
- Scale: 20 ảnh, 3 representative targets (1 per group)
  - Group A: fcos_r50 | Group B: yolov3_d53 | Group C: mask_rcnn_swin_t
- Script: `python scripts/run_sweep.py --loss osfd --k 3.0 --prune-types norm --n-images 20`
- Output: bảng (rate, n_masks) → ASR per group → chọn sweet spot

---

## Nhóm 1 — Baselines

### E1a — PGD thuần
- Loss: RPN suppression (minimize sigmoid objectness)
- Pruning: không
- n_masks: 1

---

## Nhóm 2 — Fix RaPA đúng theo gốc

### E2a — RaPA (Norm pruning) + RPN loss
- Loss: RPN suppression
- Pruning: **BatchNorm2d + LayerNorm** (đúng theo RaPA gốc, bỏ Conv2d)
- n_masks: 2, rate: 0.05
- So sánh với E1a: kiểm tra xem Norm pruning có giúp RPN loss không

### E2b — RaPA (Norm pruning) + OSFD (k=3)  ← improved baseline chính
- Loss: `MSE(k·f_clean, f_adv)` với **k=3**, maximize, tất cả backbone stages
- Pruning: **BatchNorm2d + LayerNorm**, n_masks=2, rate=0.05
- So sánh với E2a: feature distortion vs RPN suppression
- So sánh với E1a: full improvement stack

---

## Nhóm 3 — Cross-family Extensions (từ E2b)

### E3a — E2b + Low-frequency constraint
- Tất cả như E2b
- Thêm: low-pass filter lên gradient trước khi update delta (hoặc constrain delta vào low-freq space qua DCT)
- Mục tiêu: cải thiện Group C (Swin) — low-freq = global/shape patterns, ít texture-specific hơn

### E3b — E2b + Patch-masking
- Tất cả như E2b
- Thêm: randomly mask NxN patches của adversarial image trước khi extract features
- Mục tiêu: spatial diversity → perturbation robust ở patch level → transfer tốt hơn sang Swin (patch-based)
- Cần xác định: mask ở image space hay feature space, kích thước patch

### E3c — E2b + Dual surrogate (R50 + Swin-T)
- Tất cả như E2b
- Thêm: mỗi iter average gradient từ cả R50 surrogate và Swin-T surrogate
- Swin-T surrogate: `checkpoints/mask_rcnn_swin-t_1x_coco_20210902_120937-9d6b7cfa.pth`
- Mục tiêu: trực tiếp bridge gap Group C bằng cách craft perturbation fool cả 2 backbone families

---

## Nhóm 4 — Best Combo

### E4 — Top-2 từ E3 kết hợp
- Chọn 2 E3 configs có gain rõ nhất trên Group C mà không sacrifice Group A
- Combine và chạy full

---

## Logic đọc kết quả

```
E1a → E2a:   Norm pruning có giúp so với PGD không?
E2a → E2b:   Feature distortion (k=3) vs RPN suppression?
E2b → E3a:   Low-frequency lift Group C bao nhiêu?
E2b → E3b:   Patch-masking thêm được gì?
E2b → E3c:   Dual surrogate giải quyết Group C đến đâu?
E3x → E4:    Combination tốt nhất là gì?
```

---

## Chạy theo scale

1. **Smoke test** (20 ảnh): lọc E3 candidates
2. **Dev run** (300 ảnh): E1a, E2a, E2b + các E3 qua filter
3. **Held-out** (100 ảnh): 1 lần duy nhất sau khi freeze final config

---

## Status

- [ ] E1a — PGD baseline
- [ ] E2a — RaPA (Norm) + RPN
- [ ] E2b — RaPA (Norm) + OSFD k=3
- [ ] E3a — E2b + Low-frequency
- [ ] E3b — E2b + Patch-masking
- [ ] E3c — E2b + Dual surrogate
- [ ] E4  — Best combo
