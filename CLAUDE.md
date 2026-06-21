# CLAUDE.md — AS-RaPA-OD Project Context

## Research Goal
CVPR 2027 paper: **"Assignment-Stable Random Parameter Pruning for Transferable Object Detection Attacks"** (AS-RaPA-OD).

**One-line pitch:** Apply RaPA (CVPR 2026) to untargeted object detection attacks.
Discover that unlike classification, OD has an **inverted-U relationship** between
pruning rate and transfer ASR — naively applying RaPA fails; backbone-scoped pruning
at low rate (p=0.05) finds the sweet spot.

### What we're attacking
- **Task:** Untargeted object hiding — make ALL GT objects disappear (class-agnostic)
- **Surrogate:** Faster R-CNN R50-FPN (craft perturbation here)
- **Transfer targets:** Zero-query black-box (no access to target model)
- **Metric:** ASR = fraction of GT objects (IoU≥0.5, score≥0.3, same class) that disappear

### What's novel vs RaPA
RaPA claims: pruning → diversity → better transfer (monotone relationship).
We find in OD: inverted-U — too little pruning = no diversity; too much = gradient noise
(assignment corruption). Sweet spot: **backbone p=0.05**.

---

## Environment
- **Venv:** `/workspace/evasion-venv` (Python 3.10, torch 2.1.2+cu121, mmdet 3.3.0)
- **Activate:** `source /workspace/evasion-venv/bin/activate`
- **Install from scratch:** `bash setup_env.sh`
- **Workspace partition:** `/workspace` (50 GB) — pip cache goes here, not `/root`

Key pinned versions (ABI compatibility):
- `numpy==1.26.4` (torch 2.1.x compiled against numpy 1.x)
- `opencv-python==4.8.1.78`
- `mmcv==2.1.0` via `--find-links https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html`

---

## Models (all downloaded to `checkpoints/`)

| Role | Model | Config | Checkpoint |
|---|---|---|---|
| **Surrogate** | Faster R-CNN R50-FPN | `faster-rcnn_r50_fpn_1x_coco.py` | `faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth` |
| Target 1 | RetinaNet R50-FPN | `retinanet_r50_fpn_1x_coco.py` | `retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth` |
| Target 2 | RetinaNet R101-FPN | `retinanet_r101_fpn_1x_coco.py` | `retinanet_r101_fpn_1x_coco_20200130-7a93545f.pth` |
| Target 3 | FCOS R50-FPN | `fcos_r50-caffe_fpn_gn-head_1x_coco.py` | `fcos_r50_caffe_fpn_gn-head_1x_coco-821213aa.pth` |
| Target 4 | Deformable DETR R50 | `deformable-detr_r50_16xb2-50e_coco.py` | `deformable-detr_r50_16xb2-50e_coco_20221029_210934-6bc7d21b.pth` |

All models pretrained on **COCO train2017**, evaluated on **COCO val2017** — no data leakage.

---

## Data
- COCO val2017 at `data/coco/` (images + annotations)
- `data/manifests/dev_300.json` — 300 images, seed=42 (main dev set, first 20 used so far)
- `data/manifests/val_100.json` — 100 images, non-overlapping (held-out, not touched yet)
- Download script: `bash scripts/download_coco.sh`

---

## Codebase
```
src/assignment_stable_od/
  attack.py      — pgd_attack(), rpn_suppression_loss(), bgr↔tensor helpers
  pruning.py     — temporary_random_pruning() context manager
  matching.py    — match_predictions() for assignment stability score

scripts/
  run_multi_target_attack.py  ← MAIN SCRIPT (multi-target, current)
  run_mini_attack.py          — old single-target script (kept for reference)
  eval_clean.py               — COCO AP eval on any model/manifest
  run_pilot.py                — assignment stability sweep
  pilot_grad_diversity.py     — gradient cosine similarity across masks
  download_coco.sh            — COCO val2017 download

results/
  multi_target/results_r101.json  ← LATEST RESULTS (4 targets, 20 images)
  mini_attack/backbone_sweep.log  ← backbone sweep (15 images, inverted-U finding)
  pilot/stability_backbone.csv    ← backbone stability: p=0.02→0.705, p=0.05→0.470
```

---

## Key Implementation Details

### attack.py
- `epsilon_px` and `step_size_px` in **pixel units [0–255]**, e.g. `epsilon_px=8` means 8/255 L∞
- Internally converts via `px_to_norm(v) = v / 57.6` (mean std)
- `pgd_attack()` takes/returns **uint8 BGR numpy arrays**
- `rpn_suppression_loss`: backbone → neck → rpn_head only (minimise sigmoid objectness)

### Critical bugs fixed (do NOT regress)
1. **Epsilon units:** epsilon must be in pixel units, not normalised space
2. **Coordinate mismatch:** `inference_detector` returns coords in *original* image space.
   Attack runs on pre-resized image → must scale delta back via `adversarial_at_orig_scale()`
   before calling `inference_detector` for evaluation.

### run_multi_target_attack.py flow
```python
img_orig = cv2.imread(...)          # original resolution → for inference
img_bgr  = load_image_bgr(...)      # resized to 800px short-side → for attack
img_adv_resized = pgd_attack(surrogate, img_bgr, ...)
img_adv  = adversarial_at_orig_scale(img_adv_resized, img_bgr, img_orig)
# evaluate on img_orig (clean) and img_adv (adversarial) at original resolution
```

### pruning.py scopes
- `"backbone"` — ResNet stages (THE KEY SCOPE)
- `"all"` — all Conv2d/Linear weights
- `"neck"` — FPN
- `"rpn_head"` — RPN conv layers

---

## Experimental Results

### Pilot: Inverted-U finding (backbone sweep, 15 images, 20 iters)
```
backbone p=0.02  → stability=0.705  WB=0.711  TRF=0.517  (too stable, low diversity)
backbone p=0.05  → stability=0.470  WB=0.842  TRF=0.750  ← SWEET SPOT
backbone p=0.10  → stability=0.150  WB=0.816  TRF=0.683  (too corrupt)
PGD baseline     →                  WB=0.524  TRF=0.402
```

### Multi-target confirmation (4 targets, 20 images, 20 iters, ε=8px)
Results in `results/multi_target/results_r101.json`:

```
                          WB-ASR  RetNet-R50  RetNet-R101  FCOS-R50  Def-DETR
PGD baseline               0.517     0.333       0.173      0.084     0.267
RaPA-OD backbone p=0.05    0.786     0.673       0.306      0.340     0.657
─────────────────────────────────────────────────────────────────────────────
Improvement                +52%     +102%        +77%       +305%     +146%
```

**Key interpretation:**
- R101 (+77%) confirms: NOT a shared-R50-backbone artifact
- FCOS (+305% relative, lowest absolute) — most architecturally distant from surrogate
- Deformable DETR (+146%) — transformer, shows cross-paradigm transfer
- Consistent across ALL 4 targets → cross-architecture transferability confirmed (preliminary)

---

## Current Plan / What's Next

### DONE ✅
- [x] Fix epsilon units bug + coordinate mismatch bug
- [x] Backbone sweep → find inverted-U + sweet spot (p=0.05)
- [x] Download 4 target models (RetinaNet R50/R101, FCOS, Deformable DETR)
- [x] Multi-target confirmation on 20 images

### IMMEDIATE NEXT (to confirm feasibility for paper)
- [ ] **Scale to dev_300, 40 iters** — the decisive experiment (~2 hours)
  ```bash
  source /workspace/evasion-venv/bin/activate
  python scripts/run_multi_target_attack.py \
    --n-images 300 --n-iters 40 --epsilon 8 --step-size 2 \
    --out results/multi_target/results_dev300.json \
    2>&1 | tee results/multi_target/run_dev300.log
  ```
- [ ] Equal-compute comparison: PGD with 40 iters vs RaPA with 2 masks × 20 iters
  (both = 40 forward passes total)
- [ ] MIM-FGSM baseline (proper comparison)

### AFTER dev_300 confirms
- [ ] Per-image scatter plot: assignment stability vs TRF-ASR (confirm inverted-U at image level)
- [ ] val_100 held-out evaluation (final numbers for paper)
- [ ] Add ViT-backbone target (e.g. Swin-based model) for stronger cross-architecture claim
- [ ] Ablation: n_masks sensitivity (1 vs 2 vs 3 masks)

---

## Paper Claim (pending dev_300 confirmation)
> Backbone-scoped random pruning at low rate (p=0.05) significantly and consistently
> improves transfer ASR of untargeted OD attacks across anchor-based, anchor-free,
> and transformer-based detectors — enabled by an inverted-U tradeoff between
> gradient diversity and signal quality that does not exist in classification.

## Known Limitations to Acknowledge
- All models share COCO train2017 training data (shared bias, not just shared architecture)
- FCOS absolute TRF still low (0.340) despite large relative gain
- 20 images is preliminary — dev_300 needed for statistical confidence
