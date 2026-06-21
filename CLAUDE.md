# CLAUDE.md — AS-RaPA-OD Project Context

## Research Goal
CVPR 2027 paper: **"Assignment-Stable Random Parameter Pruning for Transferable Object Detection Attacks"** (AS-RaPA-OD).

Apply RaPA (CVPR 2026, random weight pruning ensemble for gradient diversity) to **object detection** adversarial attacks (object-hiding / evasion). Key challenge vs classification: OD has assignment instability — pruned models may detect completely different objects, making gradient signals incoherent.

## Environment
- **Venv:** `/workspace/evasion-venv` (Python 3.10, torch 2.1.2+cu121, mmdet 3.3.0)
- **Activate:** `source /workspace/evasion-venv/bin/activate`
- **Install from scratch:** `bash setup_env.sh` (downloads checkpoints via mim)
- **Workspace partition:** `/workspace` (50 GB) — pip cache goes here, not `/root`

Key pinned versions (ABI compatibility):
- `numpy==1.26.4` (torch 2.1.x compiled against numpy 1.x)
- `opencv-python==4.8.1.78` (newer versions require numpy>=2)
- `mmcv==2.1.0` via `--find-links https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html`

## Models
- **Surrogate:** Faster R-CNN R50-FPN (`checkpoints/faster-rcnn_r50_fpn_1x_coco.py` + `.pth`)
- **Target (zero-query):** RetinaNet R50-FPN (`checkpoints/retinanet_r50_fpn_1x_coco.py` + `.pth`)
- Clean AP on dev_300: Faster R-CNN 0.429, RetinaNet 0.428

## Data
- COCO val2017 at `data/coco/` (images + annotations)
- `data/manifests/dev_300.json` — 300 images, seed=42 (main dev set)
- `data/manifests/val_100.json` — 100 images, non-overlapping (held-out eval)
- Download script: `bash scripts/download_coco.sh`

## Codebase
```
src/assignment_stable_od/
  __init__.py
  attack.py      — pgd_attack(), rpn_suppression_loss(), bgr↔tensor helpers
  pruning.py     — temporary_random_pruning() context manager
  matching.py    — match_predictions() for assignment stability score

scripts/
  eval_clean.py         — COCO AP eval on any model/manifest
  make_subsets.py       — create dev_300 / val_100 manifests
  run_pilot.py          — assignment stability sweep (pruning rate × seeds)
  pilot_grad_diversity.py — gradient cosine similarity across masks
  run_mini_attack.py    — end-to-end feasibility demo (PGD vs RaPA variants)
  download_coco.sh      — COCO val2017 download
```

## Key Implementation Details

### attack.py
- `epsilon` and `step_size` are in **pixel units [0–255]**, e.g. `epsilon=8` means 8/255 L∞
- Internally converts to normalised space via `px_to_norm(v) = v / 57.6`
- `pgd_attack()` takes/returns **uint8 BGR numpy arrays** (not tensors)
- `rpn_suppression_loss`: backbone → neck → rpn_head only (no roi_head in graph)
- Normalisation: RGB mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]

### run_mini_attack.py
- **Critical fix (coordinate mismatch):** `inference_detector` returns predictions in
  *original image* coordinates. GT boxes from COCO are also in original coords.
  Attack runs on pre-resized image (`load_image_bgr`), then `adversarial_at_orig_scale()`
  scales delta back to original resolution before calling `inference_detector`.
- ASR = fraction of GT-matched objects (IoU≥0.5, score≥0.3, same class) that disappear

### pruning.py scopes
- `"all"` — all Conv2d/Linear weights
- `"backbone"` — ResNet stages
- `"neck"` — FPN
- `"rpn_head"` — RPN conv layers
- `"roi_head"` — RoI heads (NOT in rpn_suppression_loss graph)

## Pilot Study Results

### Assignment Stability (`results/pilot/`)
- Backbone at p=0.1 → stability collapses to ~0 (assignment-corrupting)
- rpn_head stable up to p=0.7; neck stable up to ~p=0.5

### Gradient Diversity (`results/pilot/grad_diversity.csv`)
- Global/backbone pruning → gradient diversity high but alignment with clean low
- neck/rpn_head → aligned with clean but low pairwise diversity

## Mini Attack Results (30 images, ε=8px, 40 iters)
```
Config                              WB-ASR   TRF-ASR
PGD (no pruning)                     0.524     0.402
RaPA-OD  all    p=0.1 [corrupting]   0.684     0.666   ← best
RaPA-OD  rpn    p=0.3 [stable]       0.482     0.365   ← below baseline
RaPA-OD  neck   p=0.3 [stable]       0.494     0.360   ← below baseline
```

### Key Finding
Global p=0.1 improves transfer by **+66%** over PGD (0.402→0.666).
Scope-aware stable (rpn/neck only) is *worse* than baseline.

**Why:** rpn/neck pruning leaves backbone unchanged → all 3 masks produce nearly
identical gradients (backbone dominates) → no real diversity, just noise.
Global p=0.1 varies ALL params including backbone → genuine gradient diversity.

**Implication for paper:** The decisive variable is **backbone gradient diversity**,
not "assignment stability at prediction level". Hypothesis needs refinement.

## Immediate Next Experiment
Backbone scope sweep to find the diversity/signal sweet spot:
```bash
python scripts/run_mini_attack.py \
  --n-images 30 --n-iters 40
# After modifying CONFIGS in run_mini_attack.py to:
# - PGD baseline
# - backbone p=0.02
# - backbone p=0.05
# - backbone p=0.10
```
Expected runtime: ~20 min on a single GPU.

## What Still Needs Doing
- [ ] Backbone sweep (p=0.02, 0.05, 0.10) to confirm diversity-signal tradeoff
- [ ] Refine hypothesis: is "backbone at very low rate" the AS-RaPA-OD contribution?
- [ ] Add FCOS + Deformable DETR as additional target models (Section 2 of plan)
- [ ] Formal per-object disappearance metric (Section 3 of plan)
- [ ] Scale to full dev_300 with best config
- [ ] Full baseline comparison (MIM-FGSM, etc.)
