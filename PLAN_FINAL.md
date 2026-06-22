# AS-RaPA-OD — Final Research Plan
> Thesis cao học + CVPR 2027 paper
> Cập nhật: 2026-06-22

---

## TÓM TẮT NHANH (đọc phần này trước)

**Bài toán:** Transfer adversarial attacks cho Object Detection — craft perturbation trên 1 model (surrogate), fool nhiều model khác (black-box targets).

**Phương pháp:** AS-RaPA-OD — mỗi PGD iteration, tạm thời prune ngẫu nhiên 5% backbone weights → tạo gradient diversity → perturbation transfer tốt hơn. Phiên bản gốc RaPA (CVPR 2026) chỉ áp dụng cho classification.

**Surrogate:** Faster R-CNN R50-FPN | **Metric:** ASR + COCO mAP | **Dataset:** COCO val2017

**4 Findings đã confirm (smoke test + 300 ảnh):**

| # | Finding | Status |
|---|---|---|
| F1 | Inverted-U: pruning rate vs transfer ASR trong OD | ✅ confirmed |
| F2 | Backbone family drives transfer — NOT detection paradigm | ✅ confirmed (10 targets) |
| F3 | DINO-Swin-L gần như immune từ R50 surrogate (∆AP 0.014) | ✅ confirmed |
| F4 | RaPA-CB (cross-backbone joint gradient) bridges gap một phần | ✅ smoke test |

**6 Target models chính (align với OSFD + Benchmark):**

| Target | Backbone | OSFD | Benchmark | RaPA-OD ASR |
|---|---|---|---|---|
| FCOS-R50 | ResNet-50 | ✅ | ✅ | ~0.41 |
| Def-DETR | ResNet-50 | ✅ | ✅ | ~0.64 |
| YOLOv3-D53 | Darknet-53 | ✅ | ✅ | ~0.20 |
| YOLOX-l | CSPNet | ✅ | ✅ | ~0.14 |
| Mask R-CNN Swin-T | Swin-T | — | (surrogate) | ~0.20 |
| DINO-Swin-L | Swin-L | — | ✅ | ~0.06 |

**Backbone-family transfer tiers (core finding của paper):**
```
Tier 1 — ResNet-50 (= surrogate):  ASR 0.65–0.79  ∆AP 0.34–0.45
Tier 2 — ResNet-101 (close):       ASR ~0.31      ∆AP ~0.18
Tier 3 — Swin-T / Darknet / CSP:   ASR 0.14–0.20  ∆AP 0.03–0.09
Tier 4 — Swin-L (DINO-Swin-L):     ASR ~0.06      ∆AP ~0.01  ← near-immune
```

**Next steps ưu tiên:** Ablation n_masks → Pruning rate 300 ảnh → val_100 held-out → Viết paper

---

## 1. Confirmed Findings

### Setup chính
```
Surrogate:      Faster R-CNN R50-FPN (COCO pretrained)
Pruning scope:  backbone only (ResNet stages 1-4)
Pruning rate:   p = 0.05
n_masks:        2  (averaged per PGD step)
epsilon:        8px (L∞)
Step size:      2px
Iterations:     40
Momentum:       0.9 (MI-FGSM)
Dataset:        COCO val2017, dev_300.json (seed=42)
```

---

### F1 — Inverted-U: pruning rate vs transfer ASR

Không tồn tại trong classification (RaPA gốc monotone). Đặc thù của OD do spatial assignment:

| Pruning rate | Assignment stability | WB-ASR | TRF-ASR | Lý do |
|---|---|---|---|---|
| p = 0.0 (PGD) | 1.0 | 0.52 | 0.40 | Không diverse |
| p = 0.02 | 0.71 | 0.71 | 0.52 | Quá ổn định, ít diversity |
| **p = 0.05** | **0.47** | **0.84** | **0.75** | **Sweet spot** |
| p = 0.10 | 0.15 | 0.82 | 0.68 | Assignment bắt đầu corrupt |
| p = 0.20+ | ~0 | giảm | giảm | Gradient noise |

---

### F2 — Backbone family drives transfer (NOT detection paradigm)

**Key experiment:** DINO-R50 (full transformer, không có RPN, không có anchor) transfer TỐT hơn Mask R-CNN Swin-T (two-stage CNN giống surrogate) vì DINO-R50 share R50 backbone.

**Full results — 20 ảnh, 20 iters, 3 methods:**

| Target | Backbone | Paradigm | PGD | RaPA-OD | RaPA-CB | ∆AP (RaPA-OD) |
|---|---|---|---|---|---|---|
| WB (Faster-RCNN) | ResNet-50 | two-stage | 0.460 | **0.791** | 0.749 | −0.446 |
| DINO-R50 | ResNet-50 | full-transformer | 0.289 | **0.711** | 0.626 | −0.379 |
| Def-DETR | ResNet-50 | transformer | 0.295 | **0.649** | 0.548 | −0.407 |
| RetinaNet-R50 | ResNet-50 | anchor | 0.338 | **0.653** | 0.605 | −0.344 |
| FCOS-R50 | ResNet-50 | anchor-free | 0.140 | **0.311** | 0.277 | −0.113 |
| RetinaNet-R101 | ResNet-101 | anchor | 0.167 | **0.314** | 0.309 | −0.177 |
| Mask R-CNN Swin-T | **Swin-T** | two-stage | 0.091 | 0.199 | **0.307** | −0.088 |
| YOLOv3-D53 | **Darknet** | anchor | 0.102 | 0.204 | 0.177 | −0.067 |
| YOLOX-l | **CSPNet** | anchor-free | 0.128 | 0.138 | 0.106 | −0.026 |
| **DINO-Swin-L** | **Swin-L** | full-transformer | 0.061 | 0.057 | 0.072 | **−0.014** |

> **Takeaway:** Paradigm (anchor/anchor-free/transformer/two-stage) không quyết định transfer. Backbone family mới quyết định — 4 tiers rõ ràng từ ResNet → Swin-T/Darknet → Swin-L.

---

### F3 — DINO-Swin-L gần như immune

- Clean AP: **0.6748** (model mạnh nhất trong toàn bộ target set)
- RaPA-OD adv AP: 0.6612 → ∆AP = **−0.014** (gần như không bị ảnh hưởng)
- Consistent với benchmark paper (arXiv:2602.16494): FRC surrogate → DINO = 15.4% drop trên VOC
- Lý do: Swin-L backbone hoàn toàn khác ResNet-50, capacity lớn hơn nhiều

---

### F4 — RaPA-CB: cross-backbone joint gradient

**Idea:** Mỗi PGD step lấy gradient từ BOTH R50 surrogate + Swin-T aux surrogate → average → single perturbation cover 2 backbone family.

**Novelty vs OSFD/Benchmark:** Họ generate perturbation riêng per surrogate. Mình integrate cross-backbone diversity vào optimization loop → 1 perturbation duy nhất.

| Target | RaPA-OD | RaPA-CB | Δ |
|---|---|---|---|
| Mask R-CNN Swin-T | 0.199 | **0.307** | **+54%** ✅ |
| DINO-Swin-L | 0.057 | 0.072 | +26% (still poor) |
| RetinaNet-R50 | **0.653** | 0.605 | −7% (trade-off) |
| Def-DETR | **0.649** | 0.548 | −16% (trade-off) |

> **Trade-off:** Swin-T target +54%, nhưng ResNet targets giảm ~7–16%. Lý do: chia budget gradient cho 2 backbone → mỗi family nhận ít hơn.

> **DINO-Swin-L không được improve bởi RaPA-CB** vì aux model là Swin-**T** (tiny), không đủ mạnh để bridge Swin-**L**. Cần Swin-L aux surrogate → future work.

---

### [Ref] 300 ảnh × 40 iters — 4 targets gốc (main confirmed numbers)

| Target | Clean AP | PGD | **RaPA-OD** | ∆AP |
|---|---|---|---|---|
| WB (surrogate) | 0.4287 | 0.2266 | **0.0899** | −0.339 |
| RetinaNet-R50 | 0.4283 | 0.2826 | **0.1361** | −0.292 |
| RetinaNet-R101 | 0.4516 | 0.3903 | **0.3020** | −0.150 |
| FCOS-R50 | 0.4334 | 0.3753 | **0.2709** | −0.162 |
| Def-DETR | 0.5018 | 0.3683 | **0.1944** | −0.307 |

---

## 2. Core Claims

### Claim 1 — Finding: Inverted-U trong OD pruning
> OD transfer attacks có **inverted-U relationship** giữa pruning rate và transfer ASR. Không tồn tại trong classification (RaPA gốc — monotone). Nguyên nhân: OD có spatial assignment mechanism — pruning quá cao làm corrupt assignment → gradient mất signal.

### Claim 2 — Mechanism: Assignment Stability
> **Assignment Stability** là yếu tố trung gian quyết định sweet spot. p=0.05 đủ để tạo gradient diversity nhưng chưa corrupt assignment → perturbation vừa diverse vừa aligned với object suppression goal.

### Claim 3 — Method: AS-RaPA-OD
> Backbone-scoped temporary pruning với n_masks=2: cải thiện ASR **+96% đến +185%**, hạ mAP **−0.15 đến −0.34** nhất quán trên ResNet-based targets (anchor, anchor-free, transformer paradigm). 80 FP total — 2.5× ít hơn OSFD (200+ FP).

### Claim 4 — Finding: Backbone-Family Drives Transfer
> Transfer effectiveness phụ thuộc **backbone family**, không phải detection paradigm. DINO (full transformer, zero anchor) với R50 backbone transfer tốt như RetinaNet (anchor-based) cùng backbone. Mask R-CNN (CNN two-stage) với Swin-T backbone transfer kém như YOLO family. DINO-Swin-L gần như immune (∆AP 0.014).

### Claim 5 — Extension: RaPA-CB Cross-Backbone
> Joint per-iteration gradient averaging từ R50 + Swin-T surrogate tạo **single perturbation** cover 2 backbone family: Swin-T target +54% vs RaPA-OD đơn thuần, với trade-off nhẹ trên ResNet targets. Khác OSFD/Benchmark (separate perturbations per surrogate).

---

## 3. Target Models (Final)

### 6 targets chính cho main paper

| Target | Backbone | Paradigm | OSFD | Benchmark | Tier | Status |
|---|---|---|---|---|---|---|
| FCOS-R50 | ResNet-50 | anchor-free 1-stage | ✅ | ✅ | 1 (R50) | ✅ done |
| Def-DETR | ResNet-50 | transformer | ✅ | ✅ | 1 (R50) | ✅ done |
| YOLOv3-D53 | Darknet-53 | anchor 1-stage | ✅ | ✅ | 3 (Darknet) | ✅ done |
| YOLOX-l | CSPNet | anchor-free 1-stage | ✅ | ✅ | 3 (CSP) | ✅ done |
| Mask R-CNN Swin-T | Swin-T | two-stage | — | (surrogate) | 3 (Swin-T) | ✅ done |
| **DINO-Swin-L** | **Swin-L** | **full transformer** | — | ✅ | **4 (Swin-L)** | ✅ done |

### Supplementary / Ablation only

| Target | Backbone | Lý do |
|---|---|---|
| DINO-R50 | ResNet-50 | Chứng minh paradigm không quan trọng (F2) — cần label rõ vs DINO-Swin-L |
| RetinaNet-R50 | ResNet-50 | Baseline ResNet, không có trong OSFD/Benchmark |
| RetinaNet-R101 | ResNet-101 | Tier 2 evidence, supplementary |

### Surrogate models

| Role | Model | Backbone | Status |
|---|---|---|---|
| Primary | Faster R-CNN R50-FPN | ResNet-50 | ✅ confirmed |
| Aux (RaPA-CB) | Mask R-CNN Swin-T | Swin-T | ✅ loaded |
| Future | Mask R-CNN Swin-L | Swin-L | → improve DINO-Swin-L transfer |

---

## 4. Remaining Experiments

### Phase 1 — Ablations (ưu tiên cao, cần trước val_100)

#### 1A. n_masks sweep ← QUAN TRỌNG NHẤT
Justify n_masks=2 là minimum viable diversity.
```bash
source /workspace/evasion-venv/bin/activate && cd /workspace/evasion-attack
# n_masks = 1 (≈ PGD), 2 (current), 3, 4
python scripts/run_multi_target_attack.py --n-images 300 --n-iters 40 --n-masks 1 \
    --out results/ablation/nmasks_1.json
python scripts/run_multi_target_attack.py --n-images 300 --n-iters 40 --n-masks 3 \
    --out results/ablation/nmasks_3.json
python scripts/run_multi_target_attack.py --n-images 300 --n-iters 40 --n-masks 4 \
    --out results/ablation/nmasks_4.json
```
Kỳ vọng: n=1 ≈ PGD baseline, n=2 jump lớn, n=3–4 diminishing returns.

#### 1B. Pruning rate sweep trên 300 ảnh
Confirm inverted-U ở scale lớn (pilot chỉ 15 ảnh).
```bash
python scripts/run_pilot.py --n-images 300 \
    --rates 0.0 0.02 0.05 0.10 0.20 0.50 --scope backbone
```

#### 1C. Pruning scope ablation
Confirm backbone scope tốt nhất (vs neck, rpn_head, all).
```bash
# scope: backbone / neck / rpn_head / all  (giữ p=0.05, n_masks=2, 300 ảnh)
```

#### 1D. Equal-compute (optional, strengthen vs reviewer)
```bash
# PGD @ 40 iters × 1 mask = 40 FP  (đã có)
# RaPA @ 20 iters × 2 masks = 40 FP  (cần chạy)
python scripts/run_multi_target_attack.py --n-images 300 --n-iters 20 --n-masks 2 \
    --out results/ablation/equal_compute.json
```

### Phase 2 — Analysis

#### 2E. Scatter plot: Assignment Stability vs Transfer ASR
Figure quan trọng nhất trong paper — visualize inverted-U ở image level.
```bash
python scripts/run_pilot.py --n-images 300 \
    --rates 0.02 0.05 0.10 --scope backbone
```

#### 2F. Gradient diversity analysis
Cosine similarity giữa gradients qua các masks → chứng minh n_masks=2 tạo real diversity.
```bash
python scripts/pilot_grad_diversity.py --n-images 50 --scope backbone
```

#### 2G. RaPA-CB full run 300 ảnh (nếu commit Direction A)
```bash
python scripts/run_multi_target_attack.py --n-images 300 --n-iters 40 \
    --out results/multi_target/results_rapa_cb_dev300.json
```

### Phase 3 — Final numbers (CHỈ 1 LẦN)

#### 3H. val_100 held-out ← SAU KHI FREEZE HOÀN TOÀN CONFIG
```bash
# Chỉ chạy sau khi ablations xong và config đã frozen
python scripts/run_multi_target_attack.py \
    --manifest data/manifests/val_100.json \
    --n-images 100 --n-iters 40 \
    --out results/multi_target/results_val100_FINAL.json
```
> ⚠ Không được chạy sớm — đây là held-out set, chỉ dùng 1 lần cho final paper numbers.

---

## 5. Paper Outline (CVPR 2027)

```
Abstract
  - Vấn đề: OD transfer attacks kém hơn classification do assignment problem
  - Finding 1: Inverted-U giữa pruning rate và transfer ASR trong OD
  - Finding 2: Backbone family — không phải paradigm — quyết định transfer
  - Method: AS-RaPA-OD + RaPA-CB extension
  - Results: +96–185% ASR, 4-tier backbone hierarchy, DINO-Swin-L near-immune

1. Introduction
   - Transferability gap trong adversarial OD attacks
   - Tại sao OD khó hơn classification (spatial assignment)
   - RaPA (CVPR 2026) không trivially extend sang OD
   - Contributions: 2 findings + method + extension

2. Related Work
   - Transfer attacks: MI-FGSM, DI, RaPA (CVPR 2026)
   - OD attacks: TOG, NumbOD, OSFD (AAAI 2024), T-SEA, CWA
   - Object detection: two-stage, anchor-free, transformer, YOLO family

3. Observation & Motivation
   3.1  Assignment problem trong OD
   3.2  Pilot study: pruning rate × stability × transfer ASR
   3.3  Inverted-U finding (figure chính)
   3.4  Gradient diversity mechanism

4. Method
   4.1  AS-RaPA-OD: temporary backbone pruning + multi-mask averaging
   4.2  RPN suppression loss
   4.3  Algorithm + complexity analysis
   4.4  RaPA-CB extension: cross-backbone joint gradient (per-iteration)

5. Experiments
   5.1  Setup: 6 target models, COCO val2017, metrics (ASR + mAP)
   5.2  Main results: 4-tier backbone hierarchy table
   5.3  Comparison vs OSFD (FCOS overlap, khác dataset — dùng ∆AP ratio)
   5.4  Ablations: n_masks / scope / rate / equal-compute
   5.5  Analysis: scatter plot stability vs ASR, gradient diversity
   5.6  RaPA-CB: cross-backbone coverage vs single-surrogate trade-off

6. Conclusion & Limitations
   - Single-surrogate ceiling: transfer bị giới hạn bởi backbone family
   - DINO-Swin-L near-immune: open problem cho OD robustness
   - Future: Swin-L aux surrogate, adaptive backbone weighting
```

---

## 6. Thesis Outline (Cao học)

```
Chương 1 — Giới thiệu
   1.1  Bối cảnh: adversarial attacks và an toàn AI
   1.2  Phát biểu bài toán: transfer OD attacks
   1.3  Đóng góp: 2 findings + 2 methods
   1.4  Cấu trúc luận văn

Chương 2 — Cơ sở lý thuyết
   2.1  Object Detection: Faster RCNN, FCOS, Def-DETR, YOLO, DINO
   2.2  Adversarial Attacks: FGSM, PGD, MI-FGSM, transfer attacks
   2.3  RaPA (CVPR 2026): ý tưởng gốc cho classification
   2.4  Độ đo: ASR, COCO mAP, Assignment Stability Score

Chương 3 — Công trình liên quan
   3.1  Taxonomy OD attacks (white-box vs transfer)
   3.2  So sánh với OSFD (AAAI 2024) và benchmark (arXiv:2602.16494)
   3.3  Gap hiện tại và động lực nghiên cứu

Chương 4 — Quan sát và Động lực
   4.1  Assignment problem trong OD (khác classification)
   4.2  Tại sao RaPA gốc không work trực tiếp trong OD
   4.3  Pilot study: inverted-U finding
   4.4  Backbone-family hypothesis: paradigm không quan trọng

Chương 5 — Phương pháp
   5.1  AS-RaPA-OD: temporary backbone pruning + multi-mask
   5.2  RPN suppression loss
   5.3  Algorithm và phân tích độ phức tạp
   5.4  RaPA-CB: cross-backbone extension (Direction A)

Chương 6 — Thực nghiệm
   6.1  Thiết lập (6 targets, COCO, hyperparameters)
   6.2  Kết quả chính: 4-tier backbone table (ASR + mAP)
   6.3  So sánh với baseline và OSFD
   6.4  Ablation studies (n_masks, scope, rate, equal-compute)
   6.5  Phân tích định tính (scatter plot, gradient diversity)
   6.6  RaPA-CB results và trade-off discussion
   6.7  Thảo luận và hạn chế

Chương 7 — Kết luận
   7.1  Tóm tắt đóng góp
   7.2  Hạn chế
   7.3  Hướng phát triển: Swin-L surrogate, adaptive weighting, physical-world
```

---

## 7. Comparison với OSFD & Benchmark

### vs OSFD (AAAI 2024)

| | **Ours** | **OSFD** |
|---|---|---|
| Surrogate | Faster R-CNN R50 (1 model) | YOLOv3 / VFNet / FRC-R101 / MRC-Swin |
| Dataset | COCO val2017 | VOC2012 |
| Metric | COCO AP + ASR | VOC mAP@50 |
| ε | 8px | 5px |
| Iterations | 40 | 200 |
| Total FP | 80 (RaPA-OD) / 40 per model (RaPA-CB) | 200+ |
| FCOS ∆AP | −0.162 (clean 0.433) | −92.6% drop (clean 0.789 VOC) |
| YOLO ∆AP | −0.057 (cross-family) | −91.3% (white-box same-family) |

> ⚠ **Không so số tuyệt đối** (khác dataset + metric). So bằng ∆AP ratio hoặc % drop.
> OSFD tốt trên YOLO vì dùng YOLOv3 làm surrogate (within-family). Mình là cross-family → harder setting → backbone-family finding là novel contribution.

### vs Benchmark (arXiv:2602.16494)

| | **Ours** | **Benchmark** |
|---|---|---|
| Surrogate | R50 only (+ Swin-T aux) | YOLOv3 + FRC + MRC-Swin |
| Dataset | COCO val2017 | VOC2007 |
| DINO target | Swin-L ≈ immune (∆AP 0.014) | FRC→DINO = 15.4% drop ✅ consistent |
| Swin target | ✅ (Mask RCNN Swin-T) | (used as surrogate) |

> ⚠ **DINO của benchmark là DINO-Swin-L** — consistent với finding của mình (immune từ R50).
> Paper cần label rõ "DINO-R50" vs "DINO-Swin-L" để tránh confusion.

---

## 8. Timeline

| Tuần | Việc | Output | Priority |
|---|---|---|---|
| 1 | Ablation n_masks (1/2/3/4), 300 ảnh | Table n_masks | 🔴 critical |
| 2 | Pruning rate sweep 300 ảnh | Inverted-U figure | 🔴 critical |
| 2–3 | Scope ablation + equal-compute | Ablation table | 🟡 important |
| 3–4 | Scatter plot stability vs ASR | Figure cơ chế | 🟡 important |
| 4 | Gradient diversity analysis | Supporting figure | 🟡 important |
| 5 | **val_100 held-out** (1 lần duy nhất!) | Final paper numbers | 🔴 critical |
| 5–6 | RaPA-CB full run 300 ảnh (nếu commit) | Extension results | 🟢 optional |
| 6–7 | Viết paper draft | Submission draft | — |
| 8+ | Viết thesis | Luận văn | — |

---

## 9. Known Limitations

1. **Single-surrogate ceiling:** Transfer bị giới hạn bởi backbone family của surrogate (R50 → ResNet family tốt, non-ResNet kém). Địa chỉ: RaPA-CB (partial), Swin-L surrogate (future).
2. **DINO-Swin-L near-immune:** ∆AP chỉ 0.014 từ R50 surrogate — open problem. Cần Swin-L surrogate để bridge.
3. **Shared training data:** Tất cả models train trên COCO train2017 → shared feature bias chồng lên backbone similarity.
4. **Dataset/metric gap vs OSFD:** COCO AP ≠ VOC mAP@50 → không so số tuyệt đối.
5. **ε=8px vs OSFD 5px:** Perturbation budget lớn hơn → phải note khi compare.
6. **val_100 chưa chạy:** Số cuối cùng trong paper phải từ val_100 (freeze config trước).
7. **Chưa test physical-world:** Perturbation chỉ digital.
