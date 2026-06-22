# CLAUDE.md — AS-RaPA-OD Project Context

## Research Goal
CVPR 2027 paper + master's thesis:
**"Assignment-Stable Random Parameter Pruning for Transferable Object Detection Attacks"** (AS-RaPA-OD)

**One-line pitch:** Extend RaPA (CVPR 2026) from classification to OD.
Discover 2 novel findings: (1) inverted-U relationship between pruning rate and
transfer ASR in OD — does NOT exist in classification; (2) backbone family, not
detection paradigm, determines cross-model transfer effectiveness.

---

## Environment
- **Venv:** `/workspace/evasion-venv` (Python 3.10, torch 2.1.2+cu121, mmdet 3.3.0)
- **Activate:** `source /workspace/evasion-venv/bin/activate`
- **MMDET configs:** `/workspace/evasion-venv/lib/python3.10/site-packages/mmdet/.mim/configs/`

---

## Models

### Surrogate
| Model | Config | Checkpoint |
|---|---|---|
| Faster R-CNN R50-FPN (primary) | `checkpoints/faster-rcnn_r50_fpn_1x_coco.py` | `checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth` |
| Mask R-CNN Swin-T (aux, RaPA-CB) | mmdet .mim configs/swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py | `checkpoints/mask_rcnn_swin-t_1x_coco_20210902_120937-9d6b7cfa.pth` |

### 6 Target Models (final paper set)

| Name | Backbone | Paradigm | OSFD | Benchmark | Checkpoint |
|---|---|---|---|---|---|
| fcos_r50 | ResNet-50 | anchor-free | ✅ | ✅ | `checkpoints/fcos_r50_caffe_fpn_gn-head_1x_coco-821213aa.pth` |
| deformable_detr | ResNet-50 | transformer | ✅ | ✅ | `checkpoints/deformable-detr_r50_16xb2-50e_coco_20221029_210934-6bc7d21b.pth` |
| yolov3_d53 | Darknet-53 | anchor | ✅ | ✅ | `checkpoints/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth` |
| yolox_l | CSPNet | anchor-free | ✅ | ✅ | `checkpoints/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth` |
| mask_rcnn_swin_t | Swin-T | two-stage | — | (surrogate) | `checkpoints/mask_rcnn_swin-t_1x_coco_20210902_120937-9d6b7cfa.pth` |
| dino_swin_l | Swin-L | full-transformer | — | ✅ | `checkpoints/dino-5scale_swin-l_8xb2-12e_coco_20230228_072924-a654145f.pth` |

### Supplementary / Ablation targets (downloaded, not in main table)
| Name | Checkpoint |
|---|---|
| retinanet_r50 | `checkpoints/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth` |
| retinanet_r101 | `checkpoints/retinanet_r101_fpn_1x_coco_20200130-7a93545f.pth` |
| dino_r50 | `checkpoints/dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth` |

> All models pretrained on COCO train2017. YOLO/Swin configs: use full mmdet .mim paths,
> NOT copied local configs (relative path resolution breaks outside mmdet config tree).

---

## Data
- COCO val2017 at `data/coco/` (images + annotations)
- `data/manifests/dev_300.json` — 300 images, seed=42 (main dev set)
- `data/manifests/val_100.json` — 100 images, non-overlapping (**HELD-OUT — do not touch until config frozen**)

---

## Codebase

```
src/assignment_stable_od/
  attack.py      — pgd_attack(), rpn_suppression_loss(), bgr↔tensor helpers
                   pgd_attack() has aux_model param for RaPA-CB cross-backbone
  pruning.py     — temporary_random_pruning() context manager
  matching.py    — match_predictions() for assignment stability score

scripts/
  run_multi_target_attack.py  ← MAIN SCRIPT (all experiments go here)
  run_mini_attack.py          — old single-target script (kept for reference)
  eval_clean.py               — COCO AP eval on any model/manifest
  run_pilot.py                — pruning rate sweep + assignment stability
  pilot_grad_diversity.py     — gradient cosine similarity across masks

results/
  multi_target/results_dev300.json        ← 300 imgs, 4 targets, main numbers
  multi_target/results_yolo_smoke.json    ← 20 imgs, 6 targets (+ YOLO)
  multi_target/results_swin_dino_smoke.json ← 20 imgs, 8 targets (+ Swin-T, DINO-R50)
  multi_target/results_dino_swinl_smoke.json ← 20 imgs, 10 targets (+ DINO-Swin-L), FINAL smoke
  multi_target/results_rapa_cb_smoke.json ← 20 imgs, RaPA-CB Direction A
  multi_target/results_swin_surrogate_smoke.json ← 20 imgs, Swin-T AS SURROGATE
```

---

## CONFIGS in run_multi_target_attack.py

```python
CONFIGS = [
    dict(name="pgd_baseline",      label="PGD baseline",
         n_masks=1, scope=None,       rate=0.0,  cross_backbone=False),
    dict(name="rapa_backbone_005", label="RaPA-OD backbone p=0.05",
         n_masks=2, scope="backbone", rate=0.05, cross_backbone=False),
    dict(name="rapa_cb_005",       label="RaPA-CB (R50+Swin) p=0.05",
         n_masks=1, scope="backbone", rate=0.05, cross_backbone=True),
]
AUX_SURROGATE = Mask R-CNN Swin-T  # loaded only when cross_backbone=True config exists
```

---

## Key Implementation Details

### attack.py
- `epsilon_px`, `step_size_px` in **pixel units [0–255]** (e.g. `epsilon_px=8`)
- Internally converts via `px_to_norm(v) = v / 57.6` (mean std)
- `pgd_attack(model, img_bgr, ..., aux_model=None)`:
  - `aux_model`: second surrogate for RaPA-CB — each step averages gradients from both
  - `n_sources = 2` when aux_model provided → grad divided by `n_masks * n_sources`
- `rpn_suppression_loss`: backbone → neck → rpn_head (minimise sigmoid objectness)
  - Works for Faster R-CNN AND Mask R-CNN Swin-T (both have rpn_head)
  - Does NOT work for DINO-R50 / DINO-Swin-L (no rpn_head) → these are targets only

### Critical bugs fixed (do NOT regress)
1. **Epsilon units:** epsilon must be in pixel units, not normalised space
2. **Coordinate mismatch:** `inference_detector` returns coords in *original* image space.
   Attack runs on pre-resized image → must scale delta via `adversarial_at_orig_scale()`

### pruning.py
- `eligible_modules(model, scope)`: finds Conv2d + Linear where `name.startswith(scope)`
- scope `"backbone"` catches both ResNet (Conv2d) AND Swin-T (Linear Q/K/V + FFN)
- Swin-T backbone has 52 eligible modules (vs ~50+ for ResNet)
- Pruning is reversible: saves `weight_orig`, removes mask, restores

---

## Confirmed Results

### Main run — 300 ảnh, 40 iters, ε=8px, 4 targets (results_dev300.json)

| Target | Clean AP | PGD | RaPA-OD | ∆AP |
|---|---|---|---|---|
| WB (surrogate) | 0.4287 | 0.2266 | 0.0899 | −0.339 |
| RetinaNet-R50 | 0.4283 | 0.2826 | 0.1361 | −0.292 |
| RetinaNet-R101 | 0.4516 | 0.3903 | 0.3020 | −0.150 |
| FCOS-R50 | 0.4334 | 0.3753 | 0.2709 | −0.162 |
| Def-DETR | 0.5018 | 0.3683 | 0.1944 | −0.307 |

### Backbone-family tiers — 20 ảnh, 20 iters (results_dino_swinl_smoke.json)

| Tier | Target | Backbone | RaPA-OD ASR | RaPA-OD ∆AP |
|---|---|---|---|---|
| 1 | WB / DINO-R50 / Def-DETR / RetinaNet-R50 | ResNet-50 | 0.65–0.79 | 0.34–0.45 |
| 2 | RetinaNet-R101 / FCOS-R50 | ResNet-101/50 | 0.31–0.31 | 0.11–0.18 |
| 3 | Mask-RCNN-Swin-T / YOLOv3 / YOLOX-l | Swin-T/Darknet/CSP | 0.14–0.20 | 0.03–0.09 |
| 4 | DINO-Swin-L | Swin-L | 0.057 | 0.014 |

**Key finding:** DINO-R50 (full transformer, no anchor) = Tier 1 (same as RetinaNet-R50).
Mask-RCNN-Swin-T (CNN two-stage) = Tier 3. Paradigm doesn't matter — backbone does.

### RaPA-CB (cross-backbone) — results_rapa_cb_smoke.json
- Swin-T target: 0.199 → **0.307 (+54%)** vs RaPA-OD single surrogate
- Trade-off: ResNet targets −7 to −16%
- DINO-Swin-L: 0.057 → 0.072 (aux Swin-T too small to bridge Swin-L)

### Swin-T as surrogate (inverse experiment) — results_swin_surrogate_smoke.json
- Swin surrogate → Swin-T target: ∆AP −0.199 (strong, as expected)
- Swin surrogate → ResNet targets: ∆AP −0.023 to −0.070 (weak)
- Confirms backbone hypothesis from both directions

---

## 4 Core Findings (confirmed)

**F1 — Inverted-U:** Pruning rate vs transfer ASR in OD has inverted-U shape.
Sweet spot: p=0.05, stability=0.47. Too high (p>0.10): assignment corruption.
Does NOT exist in classification (RaPA gốc is monotone).

**F2 — Backbone-family drives transfer:** Detection paradigm does not matter.
DINO-R50 (full transformer) ≈ RetinaNet-R50 (anchor). Mask-RCNN-Swin-T (two-stage CNN) ≈ YOLOv3 (anchor). Backbone family creates 4 clear transfer tiers.

**F3 — DINO-Swin-L near-immune:** ∆AP = 0.014 from R50 surrogate.
Consistent with benchmark paper (FRC→DINO = 15.4% drop on VOC).

**F4 — RaPA-CB:** Per-iteration cross-backbone gradient averaging.
Single perturbation covers 2 backbone families. Swin-T +54%, ResNet trade-off −7 to −16%.
Novel vs OSFD (separate perturbations); novel vs Benchmark (joint optimization loop).

---

## Comparison with Competitors

### OSFD (AAAI 2024)
- Their dataset: VOC2012 | metric: mAP@50 | ε: 5px | iters: 200
- Surrogates: YOLOv3, VFNet, FRC-R101, MRC-Swin-T
- Targets: YOLOF, YOLOX, FCOS, DETR
- Their YOLO results are high because YOLOv3 IS their surrogate (within-family)
- Cannot compare numbers directly (different dataset/metric)

### Benchmark (arXiv:2602.16494)
- Dataset: VOC2007 | metric: mAP@50 | surrogates: YOLO + FRC + MRC-Swin
- Targets: YOLOv3, FCOS, FRC, DETR, YOLOX-l, **DINO-Swin-L**
- Their DINO = DINO-Swin-L (benign mAP 89.6) — FRC→DINO = 15.4% drop
- Consistent with our DINO-Swin-L (∆AP 0.014 from R50 surrogate)
- Our DINO-R50 ≠ their DINO-Swin-L — must label clearly in paper

---

## What's Next (Priority Order)

1. 🔴 **Ablation n_masks** (1/2/3/4), 300 ảnh — justify core method choice
   ```bash
   source /workspace/evasion-venv/bin/activate && cd /workspace/evasion-attack
   python scripts/run_multi_target_attack.py --n-images 300 --n-iters 40 --n-masks 1 --out results/ablation/nmasks_1.json
   python scripts/run_multi_target_attack.py --n-images 300 --n-iters 40 --n-masks 3 --out results/ablation/nmasks_3.json
   python scripts/run_multi_target_attack.py --n-images 300 --n-iters 40 --n-masks 4 --out results/ablation/nmasks_4.json
   ```
2. 🔴 **Pruning rate sweep** 300 ảnh — confirm inverted-U at scale
   ```bash
   python scripts/run_pilot.py --n-images 300 --rates 0 0.02 0.05 0.10 0.20 0.50 --scope backbone
   ```
3. 🟡 Scope ablation (backbone vs neck vs rpn_head vs all)
4. 🟡 Scatter plot: assignment stability vs transfer ASR (image-level inverted-U)
5. 🟡 Gradient diversity analysis (cosine similarity across masks)
6. 🟢 RaPA-CB full run 300 ảnh (if committing Direction A)
7. 🔴 **val_100 held-out** — ONE TIME ONLY after config frozen (final paper numbers)

---

## Pilot Results (inverted-U, 15 ảnh)
```
backbone p=0.02  → stability=0.705  WB=0.711  TRF=0.517
backbone p=0.05  → stability=0.470  WB=0.842  TRF=0.750  ← SWEET SPOT
backbone p=0.10  → stability=0.150  WB=0.816  TRF=0.683
PGD baseline     →                  WB=0.524  TRF=0.402
```
