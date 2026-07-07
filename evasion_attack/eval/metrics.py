"""ASR (object disappearance rate) and COCO AP evaluation against target models.

Unlike the attack itself (which only ever touches the surrogate's backbone
and needs differentiable preprocessing, see evasion_attack.utils.preprocess),
evaluation against a target model needs no gradients, so we use mmdet's own
`inference_detector`, which applies that target's own config-defined
preprocessing (each target may normalize differently) — no need to
replicate it manually here.

ASR definition (object disappearance rate, matching OSFD/TOG-style
untargeted-attack evaluation): of the GT objects the target model correctly
detects on the CLEAN image (IoU >= iou_thr, same label, score >= score_thr),
what fraction are no longer detected at the same location/label on the
ADVERSARIAL image. This isolates "did the attack make objects vanish" from
"was the target model already missing this object" — a target that never
saw the object clean gets no credit either way.
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field

import numpy as np
import torch
from mmdet.apis import inference_detector
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.ops import box_iou


def img_tensor_to_bgr_uint8(img: torch.Tensor) -> np.ndarray:
    """[3, H, W] float RGB in [0, 255] -> [H, W, 3] uint8 BGR (mmdet/cv2 convention)."""
    rgb = img.detach().clamp(0, 255).round().byte().permute(1, 2, 0).cpu().numpy()
    return rgb[:, :, ::-1].copy()


def extract_predictions(result, score_thr: float = 0.0):
    """mmdet 3.x DetDataSample -> (boxes [N,4] xyxy, scores [N], labels [N] int)."""
    inst = result.pred_instances
    boxes = inst.bboxes.detach().cpu()
    scores = inst.scores.detach().cpu()
    labels = inst.labels.detach().cpu()
    if score_thr > 0.0:
        keep = scores >= score_thr
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    return boxes, scores, labels


def _match_gt_to_predictions(
    gt_boxes: torch.Tensor, gt_labels: torch.Tensor,
    pred_boxes: torch.Tensor, pred_scores: torch.Tensor, pred_labels: torch.Tensor,
    iou_thr: float, score_thr: float,
) -> torch.Tensor:
    """Greedy matching (highest score first). Returns bool mask [N_gt], True if detected."""
    n_gt = gt_boxes.shape[0]
    detected = torch.zeros(n_gt, dtype=torch.bool)
    if n_gt == 0 or pred_boxes.shape[0] == 0:
        return detected

    keep = pred_scores >= score_thr
    pred_boxes, pred_scores, pred_labels = pred_boxes[keep], pred_scores[keep], pred_labels[keep]
    if pred_boxes.shape[0] == 0:
        return detected

    order = torch.argsort(pred_scores, descending=True)
    used_gt = torch.zeros(n_gt, dtype=torch.bool)
    ious = box_iou(pred_boxes, gt_boxes)  # [N_pred, N_gt]
    for p in order.tolist():
        candidate_gt = (gt_labels == pred_labels[p]) & (~used_gt) & (ious[p] >= iou_thr)
        if candidate_gt.any():
            best_gt = torch.argmax(torch.where(candidate_gt, ious[p], torch.full_like(ious[p], -1.0)))
            used_gt[best_gt] = True
            detected[best_gt] = True
    return detected


@dataclass
class ASRAccumulator:
    iou_thr: float = 0.5
    score_thr: float = 0.3
    n_baseline_detected: int = 0
    n_disappeared: int = 0
    per_image: list = field(default_factory=list)  # for debugging/inspection

    def update(self, gt_boxes: torch.Tensor, gt_labels: torch.Tensor,
               clean_result, adv_result) -> None:
        clean_boxes, clean_scores, clean_labels = extract_predictions(clean_result)
        adv_boxes, adv_scores, adv_labels = extract_predictions(adv_result)

        clean_detected = _match_gt_to_predictions(
            gt_boxes, gt_labels, clean_boxes, clean_scores, clean_labels, self.iou_thr, self.score_thr
        )
        adv_detected = _match_gt_to_predictions(
            gt_boxes, gt_labels, adv_boxes, adv_scores, adv_labels, self.iou_thr, self.score_thr
        )

        baseline_idx = clean_detected.nonzero(as_tuple=True)[0]
        disappeared = (~adv_detected[baseline_idx]).sum().item()

        self.n_baseline_detected += baseline_idx.numel()
        self.n_disappeared += disappeared
        self.per_image.append({
            "n_baseline_detected": baseline_idx.numel(),
            "n_disappeared": disappeared,
        })

    @property
    def asr(self) -> float:
        if self.n_baseline_detected == 0:
            return float("nan")
        return self.n_disappeared / self.n_baseline_detected


@dataclass
class CocoApAccumulator:
    """Accumulates predictions in COCO-result format, computes AP@[.5:.95]
    via pycocotools once `evaluate()` is called against the manifest's GT
    subset (mirrors PLAN_EXPERIMENTS.md's ∆AP metric)."""
    coco_gt: COCO
    category_ids: list  # ordered: category_ids[model_label_index] -> COCO category_id
    results: list = field(default_factory=list)
    image_ids: set = field(default_factory=set)

    def update(self, image_id: int, result) -> None:
        boxes, scores, labels = extract_predictions(result)
        self.image_ids.add(image_id)
        for box, score, label in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
            x1, y1, x2, y2 = box
            self.results.append({
                "image_id": image_id,
                "category_id": self.category_ids[label],
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": score,
            })

    def evaluate(self) -> float:
        """Returns COCO AP@[.5:.95] (the 'AP' summary metric) over the
        accumulated image ids. Returns 0.0 if there are no predictions at
        all (pycocotools would otherwise error)."""
        if not self.results:
            return 0.0
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(self.results, f)
            res_path = f.name
        coco_dt = self.coco_gt.loadRes(res_path)
        coco_eval = COCOeval(self.coco_gt, coco_dt, iouType="bbox")
        coco_eval.params.imgIds = sorted(self.image_ids)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return float(coco_eval.stats[0])  # stats[0] == AP@[.5:.95]


def build_category_id_map(model, coco_gt: COCO) -> list:
    """model label index i -> COCO category_id, via class name lookup.

    mmdet's CocoDataset maps its fixed METAINFO['classes'] name order to COCO
    category ids by name (`coco.getCatIds(catNms=[name])`), NOT by sorting
    raw category ids — category ids in COCO are not contiguous, so this
    indirection matters for the AP/ASR label matching to be correct.
    """
    classes = model.dataset_meta["classes"]
    return [coco_gt.getCatIds(catNms=[name])[0] for name in classes]


def run_target_inference(model, img_bgr_uint8: np.ndarray):
    return inference_detector(model, img_bgr_uint8)
