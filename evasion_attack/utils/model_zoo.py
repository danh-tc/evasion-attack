"""Surrogate/target model registry, matching the checkpoints setup_env.sh downloads.

`mim download mmdet --config <name> --dest checkpoints/` drops both
`checkpoints/<name>.py` and `checkpoints/<checkpoint_filename>.pth`. The
short names here match PLAN_EXPERIMENTS.md's target-model table.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from mmdet.apis import init_detector


@dataclass
class ModelSpec:
    short_name: str
    config_name: str  # matches the mim --config value, i.e. checkpoints/{config_name}.py
    checkpoint_filename: str
    group: str  # "surrogate" | "A" | "B" | "C" | "supplementary"


MODEL_ZOO: dict[str, ModelSpec] = {
    "faster_rcnn_r50": ModelSpec(
        "faster_rcnn_r50", "faster-rcnn_r50_fpn_1x_coco",
        "faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth", "surrogate",
    ),
    "fcos_r50": ModelSpec(
        "fcos_r50", "fcos_r50-caffe_fpn_gn-head_1x_coco",
        "fcos_r50_caffe_fpn_gn-head_1x_coco-821213aa.pth", "A",
    ),
    "deformable_detr": ModelSpec(
        "deformable_detr", "deformable-detr_r50_16xb2-50e_coco",
        "deformable-detr_r50_16xb2-50e_coco_20221029_210934-6bc7d21b.pth", "A",
    ),
    "yolov3_d53": ModelSpec(
        "yolov3_d53", "yolov3_d53_mstrain-608_273e_coco",
        "yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth", "B",
    ),
    "yolox_l": ModelSpec(
        "yolox_l", "yolox_l_8x8_300e_coco",
        "yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth", "B",
    ),
    "mask_rcnn_swin_t": ModelSpec(
        "mask_rcnn_swin_t", "mask-rcnn_swin-t-p4-w7_fpn_1x_coco",
        "mask_rcnn_swin-t-p4-w7_fpn_1x_coco_20210902_120937-9d6b7cfa.pth", "C",
    ),
    "dino_swin_l": ModelSpec(
        "dino_swin_l", "dino-5scale_swin-l_8xb2-12e_coco",
        "dino-5scale_swin-l_8xb2-12e_coco_20230228_072924-a654145f.pth", "C",
    ),
    "retinanet_r50": ModelSpec(
        "retinanet_r50", "retinanet_r50_fpn_1x_coco",
        "retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth", "supplementary",
    ),
    "retinanet_r101": ModelSpec(
        "retinanet_r101", "retinanet_r101_fpn_1x_coco",
        "retinanet_r101_fpn_1x_coco_20200130-7a93545f.pth", "supplementary",
    ),
    "dino_r50": ModelSpec(
        "dino_r50", "dino-4scale_r50_8xb2-12e_coco",
        "dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth", "supplementary",
    ),
}

# Stage 1-4 probe set (PLAN_EXPERIMENTS.md "Probe set"): one representative
# target per group, cheap enough to run repeatedly with 2 seeds.
PROBE_TARGETS = ["fcos_r50", "yolox_l", "mask_rcnn_swin_t"]

# Full target set for Stage 5 (both models in each group).
FULL_TARGETS = ["fcos_r50", "deformable_detr", "yolov3_d53", "yolox_l", "mask_rcnn_swin_t", "dino_swin_l"]


def load_model(short_name: str, checkpoint_dir: str, device: str = "cuda:0"):
    spec = MODEL_ZOO[short_name]
    config_path = os.path.join(checkpoint_dir, f"{spec.config_name}.py")
    checkpoint_path = os.path.join(checkpoint_dir, spec.checkpoint_filename)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path} — did setup_env.sh's mim download step run?")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path} — did setup_env.sh's mim download step run?")
    return init_detector(config_path, checkpoint_path, device=device)
