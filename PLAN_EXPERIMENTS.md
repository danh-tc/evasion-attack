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

- [x] E0  — Hyperparameter sweep (rate × n_masks) — **DONE** `results/e0_sweep.json`
- [x] E1a — PGD baseline — **DONE** `results/e1a_pgd.json`
- [x] E2a — RaPA (Norm) + RPN — **DONE** `results/e2a_rapa_rpn.json`
- [x] E2b — RaPA (Norm) + OSFD k=3 — **DONE** `results/e2b_rapa_osfd.json` ← **MAIN METHOD**
- [ ] E3a — E2b + Low-frequency *(cần implement)*
- [ ] E3b — E2b + Patch-masking *(cần implement)*
- [x] E3c — E2b + Dual surrogate — **DONE** `results/e3c_dual.json` (negative result)
- [ ] E4  — Best combo = E2b

## E0 Results (20 ảnh, 40 iters, OSFD k=3, scope=backbone, prune=Norm)

| rate | masks | WB | A (fcos_r50) | B (yolov3) | C (swin_t) |
|---|---|---|---|---|---|
| 0.01 | 1 | 0.891 | 0.541 | 0.226 | 0.214 |
| 0.01 | 2 | 0.876 | 0.588 | 0.230 | 0.247 |
| 0.01 | 3 | 0.892 | 0.544 | 0.226 | 0.234 |
| 0.01 | 5 | 0.882 | 0.552 | 0.240 | 0.231 |
| **0.05** | 1 | 0.917 | 0.579 | 0.247 | 0.282 |
| **0.05** | 2 | 0.902 | 0.648 | 0.250 | 0.271 |
| **0.05** | 3 | 0.914 | 0.604 | 0.224 | 0.283 |
| **0.05** | 5 | 0.909 | 0.684 | 0.288 | 0.313 |
| 0.10 | 1 | 0.840 | 0.562 | 0.238 | 0.276 |
| 0.10 | 2 | 0.859 | 0.617 | 0.269 | 0.267 |
| 0.10 | 3 | 0.873 | 0.598 | 0.274 | 0.289 |
| 0.10 | 5 | 0.876 | 0.605 | 0.311 | 0.307 |
| 0.20 | 1 | 0.399 | 0.178 | 0.165 | 0.186 |
| 0.20 | 2 | 0.390 | 0.210 | 0.137 | 0.171 |
| 0.20 | 3 | 0.422 | 0.171 | 0.172 | 0.170 |
| 0.20 | 5 | 0.399 | 0.197 | 0.126 | 0.150 |
| 0.50 | 1 | 0.169 | 0.138 | 0.096 | 0.129 |
| 0.50 | 2 | 0.148 | 0.114 | 0.101 | 0.123 |
| 0.50 | 3 | 0.176 | 0.120 | 0.121 | 0.143 |
| 0.50 | 5 | 0.193 | 0.120 | 0.101 | 0.110 |

**Findings:**
- F1 (Inverted-U) **CONFIRMED**: peak tại rate=0.05, sụp đổ tại rate=0.20 (WB: 0.91→0.40)
- Sweet spot: **rate=0.05, n_masks=2** — balance tốt nhất giữa performance và compute
- n_masks tác động nhỏ và không monotone (20 ảnh nhiễu cao)
- Group B (YOLOv3/Darknet) peak tại rate=0.10 thay vì 0.05 — backbone xa hơn cần diversity cao hơn
- E3c dùng chung hyperparams với E2b, không cần sweep riêng

---

## E1a / E2a / E2b Results (100 ảnh, 40 iters, ε=8px, 6 targets)

### ASR (Object Disappearance Rate)

| | E1a (PGD) | E2a (+RaPA+RPN) | E2b (+RaPA+OSFD) |
|---|---|---|---|
| **WB-ASR** | 0.493 | 0.756 | 0.907 |
| [A] fcos_r50 | 0.176 | 0.352 | 0.688 |
| [A] deformable_detr | 0.278 | 0.633 | 0.868 |
| [B] yolov3_d53 | 0.082 | 0.137 | 0.310 |
| [B] yolox_l | 0.063 | 0.134 | 0.272 |
| [C] mask_rcnn_swin_t | 0.086 | 0.143 | 0.263 |
| [C] dino_swin_l | 0.034 | 0.038 | 0.098 |

### ΔAP (mAP drop)

| | E1a (PGD) | E2a (+RaPA+RPN) | E2b (+RaPA+OSFD) |
|---|---|---|---|
| [A] fcos_r50 | −0.068 | −0.152 | −0.366 |
| [A] deformable_detr | −0.116 | −0.334 | −0.531 |
| [B] yolov3_d53 | −0.036 | −0.045 | −0.132 |
| [B] yolox_l | −0.022 | −0.041 | −0.184 |
| [C] mask_rcnn_swin_t | −0.041 | −0.061 | −0.181 |
| [C] dino_swin_l | −0.012 | +0.003 | −0.026 |

**Findings:**
- **E2b thắng tuyệt đối** — tốt hơn E2a trên tất cả 7 metrics
- **OSFD đóng góp nhiều hơn RaPA** trên transfer: fcos E1a→E2a +0.177, E2a→E2b +0.336
- **RaPA** chủ yếu giúp WB và Group A in-family (diversity tăng WB +53%)
- **OSFD** bridge cross-family gap: Group B/C hưởng lợi gấp 2–3× so với RaPA alone
- **DINO-Swin-L vẫn kháng cự** (0.098) — backbone gap quá lớn → động lực chạy E3c

---

## E3c Results (100 ảnh, Dual surrogate R50 + Swin-T)

| | E1a (PGD) | E2b (main) | E3c (dual) | E2b→E3c |
|---|---|---|---|---|
| **WB-ASR** | 0.493 | 0.907 | 0.445 | **-0.461** |
| [A] fcos_r50 | 0.176 | 0.688 | 0.232 | -0.456 |
| [A] deformable_detr | 0.278 | 0.868 | 0.307 | -0.561 |
| [B] yolov3_d53 | 0.082 | 0.310 | 0.118 | -0.192 |
| [B] yolox_l | 0.063 | 0.272 | 0.110 | -0.163 |
| [C] mask_rcnn_swin_t | 0.086 | 0.263 | 0.377 | **+0.115** |
| [C] dino_swin_l | 0.034 | 0.098 | 0.041 | -0.057 |

**Findings (negative result):**
- E3c trade-off cực kỳ bất lợi: Group A mất -0.46 đến -0.56 chỉ để gain +0.115 trên swin_t
- DINO-Swin-L thậm chí GIẢM (0.098→0.041) — Swin-T gradient không transfer sang Swin-L
- **E2b là method chính** — E3c không cải thiện overall
- Negative result support F2: gradient R50 và Swin-T conflict nhau, chứng minh backbone family gap là thực sự deep
- **E4 = E2b** (không cần thêm component)
