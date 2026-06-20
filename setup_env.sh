#!/usr/bin/env bash
# Bootstrap the research environment on Ubuntu + RTX 4000 Ada.
set -euo pipefail

ENV_NAME="mmdet-evasion"
PYTHON_VERSION="3.10"
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# ── 1. Diagnostics ────────────────────────────────────────────────────────────
echo "===== System diagnostics ====="
nvidia-smi
echo ""
nvcc --version 2>/dev/null || echo "[WARN] nvcc not found — CUDA toolkit may not be in PATH"

# ── 2. Conda ──────────────────────────────────────────────────────────────────
echo ""
echo "===== Conda ====="
if ! command -v conda &>/dev/null; then
    echo "Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
    conda init bash
    echo "Miniconda installed"
else
    echo "Conda found: $(conda --version)"
    eval "$(conda shell.bash hook)"
fi

# ── 3. Create environment ─────────────────────────────────────────────────────
echo ""
echo "===== Creating conda env: $ENV_NAME ====="
if conda env list | grep -q "^$ENV_NAME "; then
    echo "Env $ENV_NAME already exists — skipping creation"
else
    conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
fi
conda activate "$ENV_NAME"
echo "Active env: $CONDA_DEFAULT_ENV"

# ── 4. PyTorch with CUDA ──────────────────────────────────────────────────────
echo ""
echo "===== Installing PyTorch ====="
# PyTorch 2.2 + CUDA 12.1 — works on RTX 4000 Ada (sm_89)
# Adjust cu121 -> cu118 if server CUDA < 12.0
python -m pip install --upgrade pip
python -m pip install torch==2.2.2 torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cu121

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

# ── 5. MMDetection stack ──────────────────────────────────────────────────────
echo ""
echo "===== Installing MMDetection ====="
python -m pip install -U openmim
mim install "mmengine==0.10.4"
mim install "mmcv==2.1.0"
mim install "mmdet==3.3.0"

echo "Verifying mmdet:"
python -c "import mmdet; print(f'  mmdet: {mmdet.__version__}')"

# ── 6. Research dependencies ──────────────────────────────────────────────────
echo ""
echo "===== Installing research deps ====="
python -m pip install \
    numpy==1.26.4 \
    scipy \
    matplotlib \
    pandas \
    tqdm \
    pycocotools \
    seaborn \
    ipython \
    jupyter

# ── 7. Project directories ────────────────────────────────────────────────────
echo ""
echo "===== Setting up project directory ====="
mkdir -p "$PROJECT_DIR"/{checkpoints,data,logs,results}

# ── 8. Download Faster R-CNN R50-FPN checkpoint ───────────────────────────────
echo ""
echo "===== Downloading Faster R-CNN R50-FPN checkpoint ====="
CKPT_DIR="$PROJECT_DIR/checkpoints"
CKPT_FILE="$CKPT_DIR/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
if [ -f "$CKPT_FILE" ]; then
    echo "Checkpoint already exists — skipping"
else
    mim download mmdet --config faster-rcnn_r50_fpn_1x_coco --dest "$CKPT_DIR"
fi

# ── 9. Download RetinaNet R50-FPN checkpoint (first target) ──────────────────
echo ""
echo "===== Downloading RetinaNet R50-FPN checkpoint ====="
CKPT_RETINA="$CKPT_DIR/retinanet_r50_fpn_1x_coco"
if ls "$CKPT_DIR"/retinanet*.pth 1>/dev/null 2>&1; then
    echo "RetinaNet checkpoint already exists — skipping"
else
    mim download mmdet --config retinanet_r50_fpn_1x_coco --dest "$CKPT_DIR"
fi

# ── 10. Project installation and environment check ───────────────────────────
echo ""
echo "===== Installing local package ====="
python -m pip install -e "$PROJECT_DIR"
python "$PROJECT_DIR/scripts/check_env.py"

# ── 11. Lock environment ──────────────────────────────────────────────────────
echo ""
echo "===== Saving environment snapshot ====="
conda env export -n "$ENV_NAME" > "$PROJECT_DIR/environment_locked.yml"
python -m pip freeze > "$PROJECT_DIR/requirements_locked.txt"
echo "Saved to: $PROJECT_DIR/environment_locked.yml"
echo "          $PROJECT_DIR/requirements_locked.txt"

echo ""
echo "===== DONE ====="
echo "Activate with: conda activate $ENV_NAME"
echo "Project dir:   $PROJECT_DIR"
echo "Next: python scripts/smoke_inference.py --image /path/to/image.jpg"
