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