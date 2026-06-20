# Assignment-Stable OD Attack

Early research code for testing whether random parameter pruning corrupts object
correspondence and harms transfer-based object-hiding attacks.

## Bootstrap on Ubuntu + RTX 4000 Ada

```bash
bash setup_env.sh
conda activate mmdet-evasion
python scripts/check_env.py
```

The setup downloads Faster R-CNN and RetinaNet R50-FPN configs/checkpoints into
`checkpoints/`. Use any local JPEG for the first smoke test:

```bash
python scripts/smoke_inference.py --image /path/to/image.jpg
```

## First pruning pilot

This is a pseudo-label smoke experiment, not the final GT-based metric. It compares
the original Faster R-CNN predictions with predictions under temporary random
unstructured pruning:

```bash
python scripts/pilot_pruning_stability.py \
  --image /path/to/image.jpg \
  --config checkpoints/faster-rcnn_r50_fpn_1x_coco.py \
  --checkpoint checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
  --rates 0 0.1 0.3 0.5 \
  --seeds 0 1 2
```

Results are written to `results/pilot_stability.csv`. The next milestone replaces
pseudo labels with COCO GT matching and runs the pilot on 10-20 fixed images.
