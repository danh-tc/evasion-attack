#!/usr/bin/env bash
# Download COCO val2017 images and annotations into data/coco/.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$PROJECT_DIR/data/coco"
mkdir -p "$DATA_DIR"

echo "===== Downloading COCO val2017 images (~6.3 GB) ====="
if [ -d "$DATA_DIR/val2017" ] && [ "$(ls -1 "$DATA_DIR/val2017" | wc -l)" -ge 5000 ]; then
    echo "val2017/ already complete — skipping"
else
    wget --show-progress -q \
        "http://images.cocodataset.org/zips/val2017.zip" \
        -O "$DATA_DIR/val2017.zip"
    unzip -q "$DATA_DIR/val2017.zip" -d "$DATA_DIR"
    rm "$DATA_DIR/val2017.zip"
    echo "Images: $(ls -1 "$DATA_DIR/val2017" | wc -l) files"
fi

echo ""
echo "===== Downloading COCO annotations (~241 MB) ====="
if [ -f "$DATA_DIR/annotations/instances_val2017.json" ]; then
    echo "annotations already exist — skipping"
else
    wget --show-progress -q \
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" \
        -O "$DATA_DIR/annotations.zip"
    unzip -q "$DATA_DIR/annotations.zip" -d "$DATA_DIR"
    rm "$DATA_DIR/annotations.zip"
fi

echo ""
echo "===== Done ====="
echo "Images : $DATA_DIR/val2017"
echo "Annots : $DATA_DIR/annotations/instances_val2017.json"
echo ""
echo "Next: python scripts/make_subsets.py"
