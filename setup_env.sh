#!/usr/bin/env bash
# Bootstrap the research environment on Ubuntu + RTX 4000 Ada using Python venv.
set -euo pipefail

PYTHON="/usr/bin/python3.10"
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Place venv in /workspace (large partition) to avoid filling the root overlay.
VENV_DIR="/workspace/evasion-venv"

TORCH_INDEX="https://download.pytorch.org/whl/cu121"
MMCV_INDEX="https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html"

# ── 1. Diagnostics ────────────────────────────────────────────────────────────
echo "===== System diagnostics ====="
nvidia-smi
echo ""
nvcc --version 2>/dev/null || echo "[WARN] nvcc not found"
"$PYTHON" --version

# ── 2. Create venv ────────────────────────────────────────────────────────────
echo ""
echo "===== Creating venv at $VENV_DIR ====="
if [ -d "$VENV_DIR" ]; then
    echo "Venv already exists — skipping creation"
else
    "$PYTHON" -m venv "$VENV_DIR"
    echo "Venv created"
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
# (mmcv ships its own cv2 — no need for opencv-python-headless)
pip install --no-cache-dir "addict" "yapf" "packaging" "Pillow"
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

# ── 13. Download Faster R-CNN R50-FPN checkpoint ─────────────────────────────
echo ""
echo "===== Downloading Faster R-CNN R50-FPN checkpoint ====="
CKPT_DIR="$PROJECT_DIR/checkpoints"
CKPT_FILE="$CKPT_DIR/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
if [ -f "$CKPT_FILE" ]; then
    echo "Checkpoint already exists — skipping"
else
    mim download mmdet --config faster-rcnn_r50_fpn_1x_coco --dest "$CKPT_DIR"
fi

# ── 14. Download RetinaNet R50-FPN checkpoint ────────────────────────────────
echo ""
echo "===== Downloading RetinaNet R50-FPN checkpoint ====="
if ls "$CKPT_DIR"/retinanet*.pth 1>/dev/null 2>&1; then
    echo "RetinaNet checkpoint already exists — skipping"
else
    mim download mmdet --config retinanet_r50_fpn_1x_coco --dest "$CKPT_DIR"
fi

# ── 15. Lock environment ──────────────────────────────────────────────────────
echo ""
echo "===== Saving requirements snapshot ====="
pip freeze > "$PROJECT_DIR/requirements_locked.txt"
echo "Saved to: $PROJECT_DIR/requirements_locked.txt"

echo ""
echo "===== DONE ====="
echo "Activate with: source $VENV_DIR/bin/activate"
echo "Project dir:   $PROJECT_DIR"
echo "Next: python scripts/smoke_inference.py --image /path/to/image.jpg"
