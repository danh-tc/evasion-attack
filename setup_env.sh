#!/usr/bin/env bash
# Bootstrap the research environment on Ubuntu + RTX 4000 Ada using Python venv.
set -euo pipefail

PYTHON="/usr/bin/python3.10"
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Place venv in /workspace (large partition) to avoid filling the root overlay.
VENV_DIR="/workspace/evasion-venv"

TORCH_INDEX="https://download.pytorch.org/whl/cu121"
MMCV_INDEX="https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html"

# ── 0. Ensure Python 3.10 is available ───────────────────────────────────────
echo "===== Checking Python 3.10 ====="
if ! command -v "$PYTHON" &>/dev/null; then
    echo "python3.10 not found — installing..."
    apt-get update -qq
    apt-get install -y software-properties-common
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update -qq
    apt-get install -y python3.10 python3.10-dev python3.10-venv
    echo "python3.10 installed"
else
    echo "python3.10 found: $($PYTHON --version)"
fi

# ── 1. Diagnostics ────────────────────────────────────────────────────────────
echo ""
echo "===== System diagnostics ====="
nvidia-smi
echo ""
nvcc --version 2>/dev/null || echo "[WARN] nvcc not found"
"$PYTHON" --version

# ── 2. Create venv ────────────────────────────────────────────────────────────
echo ""
echo "===== Creating venv at $VENV_DIR ====="
if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
    echo "Venv already exists — skipping creation"
else
    # Remove any partial/broken venv before recreating
    [ -d "$VENV_DIR" ] && echo "Removing incomplete venv..." && rm -rf "$VENV_DIR"
    # Use --without-pip because ensurepip may not be available on this system;
    # we bootstrap pip manually via get-pip.py below.
    "$PYTHON" -m venv --without-pip "$VENV_DIR"
    echo "Venv created (no pip yet)"

    # shellcheck source=/dev/null
    source "$VENV_DIR/bin/activate"

    echo "Bootstrapping pip via get-pip.py..."
    curl -fsSL https://bootstrap.pypa.io/get-pip.py | python
    echo "pip bootstrapped: $(pip --version)"
fi
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
echo "Active venv: $VIRTUAL_ENV"
python -m pip install --upgrade pip --quiet

# ── 3. PyTorch + CUDA ────────────────────────────────────────────────────────
echo ""
echo "===== Installing PyTorch 2.1.2 + CUDA 12.1 ====="
pip install --no-cache-dir \
    torch==2.1.2+cu121 \
    torchvision==0.16.2+cu121 \
    --index-url "$TORCH_INDEX"

echo "Verifying PyTorch + CUDA:"
python - <<'PYEOF'
import torch
print(f"  PyTorch : {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("[ERROR] PyTorch cannot access CUDA")
print(f"  GPU: {torch.cuda.get_device_name(0)}")
print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")
PYEOF

# ── 4. numpy (pin before mmcv/mmdet can pull in numpy 2.x) ───────────────────
echo ""
echo "===== Pinning numpy==1.26.4 ====="
pip install --no-cache-dir "numpy==1.26.4"

# ── 5. openmim + mmengine ─────────────────────────────────────────────────────
echo ""
echo "===== Installing openmim + mmengine 0.10.4 ====="
pip install --no-cache-dir openmim
pip install --no-cache-dir "mmengine==0.10.4"

# ── 6. mmcv (prebuilt wheel for cu121 / torch 2.1.0) ─────────────────────────
echo ""
echo "===== Installing mmcv 2.1.0 ====="
pip install --no-cache-dir "mmcv==2.1.0" \
    --find-links "$MMCV_INDEX" \
    --no-deps
# Install mmcv runtime deps separately so numpy stays pinned
pip install --no-cache-dir "addict" "yapf" "packaging" "Pillow" "opencv-python==4.8.1.78"
# Re-pin numpy in case the above pulled in 2.x
pip install --no-cache-dir "numpy==1.26.4"

# ── 7. mmdet ──────────────────────────────────────────────────────────────────
echo ""
echo "===== Installing mmdet 3.3.0 ====="
pip install --no-cache-dir "mmdet==3.3.0" --no-deps
# mmdet runtime deps (excluding mmcv/mmengine already installed)
pip install --no-cache-dir \
    "pycocotools>=2.0.7" \
    "scipy>=1.12" \
    "shapely>=2.0" \
    "terminaltables" \
    "tqdm>=4.66"
# Re-pin numpy again
pip install --no-cache-dir "numpy==1.26.4"

# ── 8. Verify full stack ──────────────────────────────────────────────────────
echo ""
echo "===== Verifying MMDetection stack ====="
python - <<'PYEOF'
import torch, mmcv, mmengine, mmdet
print(f"  torch     : {torch.__version__}")
print(f"  mmcv      : {mmcv.__version__}")
print(f"  mmengine  : {mmengine.__version__}")
print(f"  mmdet     : {mmdet.__version__}")
from mmdet.utils import register_all_modules
register_all_modules()
print("  register_all_modules: OK")
from mmcv.ops import nms
import torch as T
boxes = T.tensor([[0,0,10,10],[1,1,11,11]], device="cuda", dtype=T.float32)
scores = T.tensor([0.9, 0.8], device="cuda")
_, keep = nms(boxes, scores, 0.5)
assert keep.numel() == 1, "MMCV CUDA NMS unexpected result"
print("  MMCV CUDA ops: OK")
PYEOF

# ── 9. Research utilities ─────────────────────────────────────────────────────
echo ""
echo "===== Installing research utilities ====="
pip install --no-cache-dir \
    "matplotlib>=3.8" \
    "pandas>=2.2" \
    "seaborn>=0.13" \
    "tqdm>=4.66" \
    "ipython>=8.20" \
    "jupyter>=1.0"
# Final numpy pin
pip install --no-cache-dir "numpy==1.26.4"

# ── 10. Project directories ───────────────────────────────────────────────────
echo ""
echo "===== Setting up project directories ====="
mkdir -p "$PROJECT_DIR"/{checkpoints,data,logs,results}

# ── 11. Local package ─────────────────────────────────────────────────────────
echo ""
echo "===== Installing local package ====="
pip install --no-cache-dir -e "$PROJECT_DIR"

# ── 12. check_env ─────────────────────────────────────────────────────────────
echo ""
echo "===== Running check_env.py ====="
python "$PROJECT_DIR/scripts/check_env.py"

# ── 13. Download all model checkpoints ───────────────────────────────────────
echo ""
echo "===== Downloading model checkpoints ====="
CKPT_DIR="$PROJECT_DIR/checkpoints"

download_ckpt() {
    local config="$1"
    local file="$2"
    if [ -f "$CKPT_DIR/$file" ]; then
        echo "  [SKIP] $file"
    else
        echo "  [DOWN] $config"
        mim download mmdet --config "$config" --dest "$CKPT_DIR"
    fi
}

# Surrogate
download_ckpt faster-rcnn_r50_fpn_1x_coco    "faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"

# Group A — ResNet-50 targets
download_ckpt fcos_r50-caffe_fpn_gn-head_1x_coco   "fcos_r50_caffe_fpn_gn-head_1x_coco-821213aa.pth"
download_ckpt deformable-detr_r50_16xb2-50e_coco    "deformable-detr_r50_16xb2-50e_coco_20221029_210934-6bc7d21b.pth"

# Group B — Non-ResNet CNN targets
download_ckpt yolov3_d53_mstrain-608_273e_coco      "yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth"
download_ckpt yolox_l_8x8_300e_coco                 "yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth"

# Group C — Swin ViT targets + aux surrogate
download_ckpt mask-rcnn_swin-t-p4-w7_fpn_1x_coco   "mask_rcnn_swin-t-p4-w7_fpn_1x_coco_20210902_120937-9d6b7cfa.pth"
download_ckpt dino-5scale_swin-l_8xb2-12e_coco      "dino-5scale_swin-l_8xb2-12e_coco_20230228_072924-a654145f.pth"

# Supplementary / ablation
download_ckpt retinanet_r50_fpn_1x_coco             "retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth"
download_ckpt retinanet_r101_fpn_1x_coco            "retinanet_r101_fpn_1x_coco_20200130-7a93545f.pth"
download_ckpt dino-4scale_r50_8xb2-12e_coco         "dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth"

echo "All checkpoints done."

# ── 14. Download COCO val2017 dataset ────────────────────────────────────────
echo ""
echo "===== Downloading COCO val2017 ====="
DATA_DIR="$PROJECT_DIR/data/coco"
mkdir -p "$DATA_DIR"

if [ -d "$DATA_DIR/val2017" ] && [ "$(ls -A "$DATA_DIR/val2017" 2>/dev/null | wc -l)" -gt 1000 ]; then
    echo "  [SKIP] val2017 images already exist"
else
    echo "  Downloading val2017 images (~780 MB)..."
    curl -fL "http://images.cocodataset.org/zips/val2017.zip" -o /tmp/val2017.zip
    echo "  Extracting..."
    python3 -c "import zipfile; zipfile.ZipFile('/tmp/val2017.zip').extractall('$DATA_DIR')"
    rm /tmp/val2017.zip
    echo "  val2017 extracted."
fi

if [ -f "$DATA_DIR/annotations/instances_val2017.json" ]; then
    echo "  [SKIP] annotations already exist"
else
    echo "  Downloading annotations (~241 MB)..."
    curl -fL "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" -o /tmp/annotations.zip
    echo "  Extracting..."
    python3 -c "import zipfile; zipfile.ZipFile('/tmp/annotations.zip').extractall('$DATA_DIR')"
    rm /tmp/annotations.zip
    echo "  Annotations extracted."
fi

# ── 15. Generate data manifests ───────────────────────────────────────────────
echo ""
echo "===== Generating data manifests ====="
MANIFEST_DIR="$PROJECT_DIR/data/manifests"
mkdir -p "$MANIFEST_DIR"

if [ -f "$MANIFEST_DIR/dev_300.json" ] && [ -f "$MANIFEST_DIR/val_100.json" ]; then
    echo "  [SKIP] Manifests already exist"
else
    python3 - <<PYEOF
import json, random
from pycocotools.coco import COCO

coco = COCO("$DATA_DIR/annotations/instances_val2017.json")
all_ids = sorted(coco.getImgIds())

rng = random.Random(42)
rng.shuffle(all_ids)

dev_300 = all_ids[:300]
val_100 = all_ids[300:400]  # non-overlapping held-out

with open("$MANIFEST_DIR/dev_300.json", "w") as f:
    json.dump({"seed": 42, "split": "dev", "size": 300, "image_ids": dev_300}, f, indent=2)

with open("$MANIFEST_DIR/val_100.json", "w") as f:
    json.dump({"seed": 42, "split": "val_held_out", "size": 100, "image_ids": val_100}, f, indent=2)

print(f"  dev_300.json: {len(dev_300)} images")
print(f"  val_100.json: {len(val_100)} images (HELD-OUT — do not touch until config frozen)")
PYEOF
fi

# ── 16. Lock environment ──────────────────────────────────────────────────────
echo ""
echo "===== Saving requirements snapshot ====="
pip freeze > "$PROJECT_DIR/requirements_locked.txt"
echo "Saved to: $PROJECT_DIR/requirements_locked.txt"

echo ""
echo "===== DONE ====="
echo "Activate with: source $VENV_DIR/bin/activate"
echo "Project dir:   $PROJECT_DIR"
echo "Next: python scripts/check_env.py"
echo "      python scripts/run_attack.py --n-images 5 --n-iters 5 --out results/smoke.json"
echo "      python scripts/run_sweep.py --n-images 5 --rates 0.05 --masks 2 --n-iters 5 --out results/e0_smoke.json"
